from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.common import idempotent, json_body, jresp
from app.approvals import ApprovalError
from app.errors import ApiError
from app.state import ctx
from decisions.policy import PolicyError

router = APIRouter(prefix="/api")

ENDPOINTS = ["GET /api/health", "GET /api/state", "GET /api/events (SSE)", "GET /api/scenarios",
             "POST /api/sim/scenario|speed|pause|resume|next-cdm", "GET /api/conjunctions/{key}",
             "GET /api/conjunctions/{key}/geometry", "GET /api/dashboard?window=24h", "GET /api/eval",
             "GET|PUT /api/policy", "GET /api/approvals", "POST /api/approvals/{id}", "POST /api/assess",
             "POST /api/ask", "GET /api/ledger", "GET /api/ledger/verify", "GET /api/ledger.jsonl",
             "GET /api/models", "GET /api/models/{name}/card", "GET /api/predictions/{file}.csv",
             "GET /api/livesky", "GET /api/futures/{key}"]


@router.get("/health")
async def health(request: Request):
    c = ctx(request)
    s = c.settings
    eng = c.engine.status()
    prov = dict(c.decisions.health)
    prov["anthropic"] = (c.explainer.status if c.explainer.client is not None
                         else ("configured (disabled)" if s.anthropic_api_key else "unconfigured"))
    if s.decisions_mode == "mock":
        prov = {k: ("configured (disabled)" if v == "ok" else v) for k, v in prov.items()}
    return jresp({
        "ok": True, "product": "ORBITAL DODGE",
        "mode": {"decisions": s.decisions_mode, "llm": s.llm_mode, "engine": eng["data_source"],
                 "decisions_chain": c.decisions.chain_names()},
        "model": {"name": "tier_forecaster", "version": eng["forecaster"], "sha_ok": eng["model_sha_ok"],
                  "status": "ok" if eng["forecaster"] else "degraded (persistence-empirical)"},
        "providers": prov,
        "ledger": {"seq": c.ledger.seq, "verified": c.ledger.verified},
        "engine": eng, "sse_clients": len(c.bus.clients), "bus_seq": c.bus.seq,
        "started": c.started_wall, "endpoints": ENDPOINTS,
        "stats": {"decisions": {k: v for k, v in c.decisions.stats.items() if k != "latencies_ms"},
                  "llm": c.explainer.stats},
    })


@router.get("/events")
async def events(request: Request):
    bus = ctx(request).bus
    last = request.headers.get("Last-Event-ID") or request.query_params.get("lastEventId")
    return StreamingResponse(bus.stream(last, request.is_disconnected), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


def _follows(sim) -> bool:
    thr = sim.served_alert_threshold()
    return thr is not None and abs(thr - sim.policy["tau_watch"]) < 1e-6


def _watch_pct(sim) -> str:
    v = sim.policy["tau_watch"] * 100
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def _watch_src(sim) -> str:
    return (" (the served forecaster's own alert threshold)" if _follows(sim)
            else " (desk τ_watch)")


@router.get("/policy")
async def get_policy(request: Request):
    sim = ctx(request).sim
    return jresp({"policy": sim.policy, "version": sim.policy_version,
                  "sentence": "The system logs and keeps monitoring on its own only when an event is SAFE or ELEVATED, "
                              f"the decision model gives its chosen action at least {sim.policy['confidence_bar'] * 100:.0f}% "
                              "and agrees with the cost rule. Critical events, default covariances and disagreements always "
                              "come to you. An event goes on WATCH when the forecast's chance it ends CRITICAL is at least "
                              f"{_watch_pct(sim)}{_watch_src(sim)}.",
                  "watch": {"tau_watch": sim.policy["tau_watch"], "served_alert_threshold": sim.served_alert_threshold(),
                            "follows_served_model": _follows(sim)},
                  "assumptions": "loss values are stated assumptions (analyst-hour equivalents), not measured costs"})


@router.put("/policy")
async def put_policy(request: Request):
    c = ctx(request)
    b = await json_body(request)

    def run():
        try:
            return 200, c.sim.update_policy(b, actor=request.headers.get("X-Actor", "analyst"))
        except PolicyError as e:
            raise ApiError(400, "BAD_REQUEST", str(e))
    return idempotent(c.ledger, request, "PUT /policy", b, run)


@router.get("/approvals")
async def list_approvals(request: Request):
    sim = ctx(request).sim
    return jresp({"approvals": [a.to_dict() for a in sim.approvals.items.values()]})


@router.post("/approvals/{approval_id}")
async def decide(approval_id: str, request: Request):
    c = ctx(request)
    b = await json_body(request)
    action = b.get("action")
    note = b.get("note")
    if not isinstance(action, str):
        raise ApiError(400, "BAD_REQUEST", "action required", "one of the approval's options or REJECT")
    if note is not None and not isinstance(note, str):
        raise ApiError(400, "BAD_REQUEST", "note must be a string")

    def run():
        try:
            return 200, c.sim.decide_approval(approval_id, action, (note or None) and note[:500],
                                              request.headers.get("X-Actor", "analyst"))
        except ApprovalError as e:
            raise ApiError(e.status, e.code, e.message, extra=e.extra)
    return idempotent(c.ledger, request, f"POST /approvals/{approval_id}", b, run)


@router.get("/ledger")
async def ledger(request: Request, after: int = 0, limit: int = 200, event_key: str | None = None):
    c = ctx(request)
    rows = c.ledger.entries(after=after, limit=limit, event_key=event_key)
    return jresp({"entries": rows, "seq": c.ledger.seq, "next_after": rows[-1]["seq"] if rows else after})


@router.get("/ledger/verify")
async def ledger_verify(request: Request):
    import asyncio
    return jresp(await asyncio.to_thread(ctx(request).ledger.verify))


@router.get("/ledger.jsonl")
async def ledger_jsonl(request: Request):
    return StreamingResponse(ctx(request).ledger.export_jsonl(), media_type="application/x-ndjson",
                             headers={"Content-Disposition": "attachment; filename=ledger.jsonl"})


@router.post("/ask")
async def ask(request: Request):
    from llm.ask import answer
    b = await json_body(request)
    q = b.get("question")
    if not isinstance(q, str) or not q.strip():
        raise ApiError(400, "BAD_REQUEST", "question (string) required")
    return jresp(await answer(ctx(request), q.strip()[:500]))
