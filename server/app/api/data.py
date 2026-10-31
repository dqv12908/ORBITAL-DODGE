from __future__ import annotations

import io
import json
import re
from datetime import timedelta

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.api.common import json_body, jresp
from app.errors import ApiError
from app.state import ctx
from app.util import clean_json, fnum, hhmm, iso
from llm.factsheet import fmt_miss, fmt_sci
from registry.registry import collect_eval, list_models, read_card

router = APIRouter(prefix="/api")
KEY_RX = re.compile(r"^(train|test|adhoc):[0-9]+$")


def _key(event_key: str) -> str:
    if not KEY_RX.match(event_key):
        raise ApiError(400, "BAD_REQUEST", f"bad event key {event_key!r}", "format split:event_id, e.g. train:2563")
    return event_key


def _event(request, key):
    sim = ctx(request).sim
    ev = sim.events.get(key)
    if ev is None:
        raise ApiError(404, "NOT_FOUND", f"{key} is not in the current scenario cast",
                       "only released CDMs of cast events are served", extra={"cast": list(sim.events)})
    return sim, ev


@router.get("/conjunctions/{event_key}")
async def conjunction(event_key: str, request: Request):
    sim, ev = _event(request, _key(event_key))
    cdms = []
    for r in ev.released:
        row = r.get("row") or {}
        cdms.append({
            "cdm_index": r["cdm_index"], "cdm_id": r["cdm_id"], "created_sim": r["created_sim"], "ttca_d": r["ttca_d"],
            "miss_m": r["miss_m"], "rel_speed_ms": fnum(row.get("relative_speed")),
            "rtn_m": [fnum(row.get(f"relative_position_{a}")) for a in "rtn"],
            "sigma_r_m": {"t": fnum(row.get("t_sigma_r")), "c": fnum(row.get("c_sigma_r"))},
            "c_object_type": row.get("c_object_type"), "reported_log10": r.get("reported_log10"),
            "max_risk_estimate": fnum(row.get("max_risk_estimate")), "F10": fnum(row.get("F10")), "AP": fnum(row.get("AP")),
            "pc": r.get("pc"), "tier": r.get("tier"), "forecast": r.get("forecast"), "gate": r.get("gate"),
            "decision": r.get("decision"), "verdict": r.get("verdict"), "explanation": r.get("explanation"),
            "approval_id": r.get("approval_id"), "ledger_seqs": r.get("ledger_seqs"), "preloaded": r.get("preloaded")})
    return jresp(clean_json({"event_key": ev.key, "summary": sim.event_summary(ev), "meta": ev.meta,
                             "cdms": cdms, "approvals": [a.to_dict() for a in sim.approvals.for_event(ev.key)],
                             "outcome": ev.outcome,
                             "pc_series": [{"ttca_d": c["ttca_d"], "log10_pc": (c["pc"] or {}).get("log10"),
                                            "status": (c["pc"] or {}).get("status"), "miss_m": c["miss_m"],
                                            "tier": (c["tier"] or {}).get("tier")} for c in cdms]}))


@router.get("/conjunctions/{event_key}/geometry")
async def geometry(event_key: str, request: Request):
    sim, ev = _event(request, _key(event_key))
    import asyncio
    g = await asyncio.to_thread(sim.engine.geometry, ev.key) or {}
    released = {r["cdm_index"] for r in ev.released}
    planes = g.pop("encounter_plane", None)
    if planes is None:
        planes = [{"cdm_index": r["cdm_index"], **{k: r["pc"].get(k) for k in
                                                     ("sigma_plane_m", "theta_deg", "hbr_m", "miss_xy_m", "mahalanobis")}}
                  for r in ev.released if "pc" in r]
    else:
        planes = [p for p in planes if p.get("cdm_index") in released]  # never send unreleased CDMs
    return jresp(clean_json({"event_key": ev.key, "label": "RECONSTRUCTED GEOMETRY", "tca_sim": iso(ev.tca),
                             "elements": g, "encounter_plane": planes}))


@router.get("/dashboard")
async def dashboard(request: Request, window: str = "24h"):
    m = re.match(r"^(\d+)h$", window)
    if not m:
        raise ApiError(400, "BAD_REQUEST", "window must look like 24h", extra={"valid": ["24h", "48h", "72h"]})
    hours = int(m.group(1))
    c = ctx(request)
    sim = c.sim
    now = sim.sim_time
    k = sim.kpis()
    active = [e for e in sim.events.values() if e.released and e.outcome is None and "tier" in e.released[-1]]
    win = [e for e in active if timedelta(0) <= e.tca - now <= timedelta(hours=hours)]

    def row(e):
        s = sim.event_summary(e)
        return {**s, "pc_fmt": "not computable" if (s["latest"] or {}).get("log10_pc") is None
                else fmt_sci(10 ** s["latest"]["log10_pc"]), "miss_fmt": fmt_miss((s["latest"] or {}).get("miss_m"))}
    next_w = sorted([row(e) for e in win], key=lambda r: (-((r["forecast"] or {}).get("p_critical") or 0), r["tca_sim"]))
    alerts = [l for l in sim.logs if l["level"] in ("CRIT", "WARN", "REFUSE", "ERROR")][-50:]
    refusals = [l for l in sim.logs if l["level"] == "REFUSE"][-50:]
    tiles = [
        {"id": "within_1km", "label": f"Within 1 km in next {hours} h", "value": sum(
            1 for e in win if (e.released[-1]["miss_m"] or 1e12) < 1000), "tone": "crit",
         "query": f"cast events with 0 ≤ TCA − now ≤ {hours} h and latest miss < 1 km"},
        {"id": "within_5km", "label": f"Within 5 km in next {hours} h", "value": sum(
            1 for e in win if (e.released[-1]["miss_m"] or 1e12) < 5000), "tone": "warn",
         "query": f"cast events with 0 ≤ TCA − now ≤ {hours} h and latest miss < 5 km"},
        {"id": "critical", "label": "Critical", "value": k["critical"], "tone": "crit",
         "detail": f"{k['critical_by_pc']} by Pc · {k['critical_miss_only']} miss only"},
        {"id": "elevated", "label": "Elevated", "value": k["elevated"], "tone": "warn"},
        {"id": "refused", "label": "Refused", "value": k["refused"], "tone": "refuse",
         "detail": "default covariance or decision model declined"},
        {"id": "awaiting", "label": "Awaiting you", "value": k["awaiting"], "tone": "warn",
         "detail": None if k["awaiting_oldest_sim_min"] is None else f"oldest {k['awaiting_oldest_sim_min']:.0f} sim-min"},
        {"id": "scoreboard", "label": "Forecast scoreboard", "value": k["scoreboard"], "tone": "info"},
        {"id": "decisions", "label": "Decisions", "value": k["decisions"], "tone": "info"},
        {"id": "ai", "label": "AI calls & fallbacks", "value": {"calls": k["ai_calls"], "fallbacks": k["fallbacks"],
                                                               "llm_verified": k["llm_verified"],
                                                               "llm_rejected": k["llm_rejected"]}, "tone": "info"},
    ]
    return jresp(clean_json({"window": window, "sim_time": iso(now), "tiles": tiles, "next": next_w,
                             "alerts": alerts, "decisions": sim.decisions_log[-100:], "refusals": refusals,
                             "outcomes": sim.outcomes, "insights": insights(sim, active, win, hours),
                             "ledger": {"seq": c.ledger.seq, "verified": c.ledger.verified}}))


def insights(sim, active, win, hours) -> list[dict]:
    out = []
    crit = [e for e in active if e.released[-1]["tier"]["tier"] == "CRITICAL"]
    miss_only = [e for e in crit if e.released[-1]["tier"]["pc_tier"] != "CRITICAL"]
    if crit:
        lps = [e.released[-1]["pc"].get("log10") for e in miss_only if e.released[-1]["pc"].get("log10") is not None]
        tail = f"; their Pc ≤ {fmt_sci(10 ** max(lps))}" if lps else ""
        out.append({"id": "miss_only", "text": f"{len(miss_only)} of {len(crit)} critical events are critical only by "
                                               f"miss distance{tail}",
                    "query": "events_by_rule: latest tier CRITICAL and pc_tier != CRITICAL"})
    cov = [e for e in active if e.released[-1]["pc"]["status"] == "COV_UNKNOWN"]
    if cov:
        out.append({"id": "cov_unknown", "text": f"{len(cov)} event{'s' if len(cov) != 1 else ''} cannot be assessed: "
                                                 f"default covariance ({', '.join(e.key for e in cov[:3])})",
                    "query": "events_by_rule: Q-COV-DEFAULT on the latest CDM"})
    flagged = [e for e in active if e.released[-1]["tier"]["tier"] != "SAFE" and e.tca - timedelta(hours=24) > sim.sim_time]
    if flagged:
        e = min(flagged, key=lambda e: e.tca)
        out.append({"id": "next_mcp", "text": f"Next commitment deadline: {e.key} at {hhmm(e.tca - timedelta(hours=24))} SIM",
                    "query": "min(TCA − 24 h) over flagged events whose commitment point is still ahead"})
    close = [e for e in win if (e.released[-1]["miss_m"] or 1e12) < 1000]
    out.append({"id": "within_1km", "text": f"{len(close)} event{'s' if len(close) != 1 else ''} within 1 km in the next {hours} h",
                "query": f"events_within(km=1, hours={hours})"})
    aps = [fnum((r.get("row") or {}).get("AP")) for e in active for r in e.released]
    aps = [a for a in aps if a is not None]
    if aps:
        out.append({"id": "ap_context", "text": f"AP peaked at {max(aps):.0f} in the released CDMs — context only: across "
                                                "8,293 historic events AP did not predict risk change (corr 0.001, offline analysis)",
                    "query": "max(AP) over released CDMs; correlation from docs/design/11 (offline)"})
    sb = sim.kpis()["scoreboard"]
    n = sum(sb["persistence"].values())
    if n:
        out.append({"id": "scoreboard", "text": f"Revealed so far: persistence {sb['persistence']['TP'] + sb['persistence']['TN']}/{n} "
                                                f"correct, model {sb['model']['TP'] + sb['model']['TN']}/{n} correct at T−2 d",
                    "query": "outcome.revealed verdicts in this run"})
    return out


@router.get("/eval")
async def eval_(request: Request):
    c = ctx(request)
    from app.simloop import Sim  # noqa: F401
    e = collect_eval(c.settings.repo_dir)
    e["failure_cases"] = FAILURE_CASES
    e["decision_layer"] = {"mode": c.decisions.mode,
                           "status": "not measured" if c.decisions.mode == "mock" else "see registry/decisions if present",
                           "note": "mock agrees with the cost rule by construction; comparison hidden in mock mode"}
    return jresp(clean_json(e))


FAILURE_CASES = [
    {"id": "F1", "case": "Late escalation not foreseeable at T−2 d", "event": "train:2563", "scenario": "late-cdm"},
    {"id": "F2", "case": "Physics cannot reproduce ESA's risk", "event": "train:4200", "scenario": "failure"},
    {"id": "F3", "case": "Default covariance", "event": "train:2299", "scenario": "cov-unknown"},
    {"id": "F4", "case": "Brief rule over-flags (miss screen ignores covariance)", "event": "train:8756",
     "scenario": "rules-disagree"},
    {"id": "F5", "case": "ML misses more criticals than persistence at argmax", "event": None, "scenario": None},
    {"id": "F6", "case": "Labels are biased (Kelvins pre-filtering)", "event": None, "scenario": None},
    {"id": "F7", "case": "Far-tail Pc (Gaussian tails)", "event": None, "scenario": None},
]


@router.post("/assess")
async def assess(request: Request):
    c = ctx(request)
    ctype = request.headers.get("content-type", "")
    rows: list[dict]
    mode = request.query_params.get("mode") or c.sim.policy["tier_mode"]
    if mode not in ("BRIEF", "CARA_PC"):
        raise ApiError(400, "BAD_REQUEST", "mode must be BRIEF or CARA_PC")
    if "csv" in ctype or "multipart" in ctype:
        import pandas as pd
        if "multipart" in ctype:
            form = await request.form()
            f = next((v for v in form.values() if hasattr(v, "read")), None)
            if f is None:
                raise ApiError(400, "BAD_REQUEST", "multipart upload needs a CSV file field")
            raw = await f.read()
        else:
            raw = await request.body()
        try:
            rows = pd.read_csv(io.BytesIO(raw)).to_dict("records")
        except Exception as e:
            raise ApiError(400, "BAD_REQUEST", f"unreadable CSV: {e}")
    else:
        b = await json_body(request)
        rows = b.get("rows")
        mode = b.get("mode", mode)
        if not isinstance(rows, list):
            raise ApiError(400, "BAD_REQUEST", "body must be {rows: [CDM rows in Kelvins schema]}")
    if not 1 <= len(rows) <= 500:
        raise ApiError(400, "BAD_REQUEST", "1–500 rows per request")
    out = []
    by_event: dict[str, list[dict]] = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            out.append({"i": i, "error": "row must be an object"})
            continue
        r = clean_json(r)
        for side in ("t", "c"):
            r.setdefault(f"cov_unknown_{side}", int(any((fnum(r.get(f"{side}_sigma_{a}")) or 0) >= 1e7
                                                        for a in "rtn")))
        if fnum(r.get("miss_distance")) is None or fnum(r.get("time_to_tca")) is None:
            out.append({"i": i, "error": "miss_distance and time_to_tca are required numbers"})
            continue
        try:
            pc, pc_src = c.engine.pc_live(r)
            tier, src = c.engine.assess(r, pc, mode)
            ek = f"adhoc:{int(fnum(r.get('event_id')) or 0)}"
            r["_tier"] = tier["tier"]
            by_event.setdefault(ek, []).append(r)
            out.append({"i": i, "event_key": ek, "pc": pc, "tier": tier, "pc_fmt": "not computable" if pc.get("log10") is None
                        else fmt_sci(10 ** pc["log10"]), "source": {"pc": pc_src, "rules": src}})
        except Exception as e:
            out.append({"i": i, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    forecasts = {}
    for ek, rs in by_event.items():
        rs = sorted(rs, key=lambda r: -(fnum(r.get("time_to_tca")) or 0))
        if (fnum(rs[0].get("time_to_tca")) or 0) >= 2:
            fc = c.engine.forecast(ek, rs, rs[-1]["_tier"], c.sim.policy["tau_watch"])
            if fc.get("fallback") or not c.engine.forecaster:
                fc["error"] = {"code": "MODEL_DEGRADED", "message": "forecast model unavailable; persistence-empirical shown"}
            forecasts[ek] = fc
    return jresp(clean_json({"mode": mode, "rows": out, "forecasts": forecasts,
                             "note": "Pc is physics; tiers are rules; forecast is statistics (only if a CDM ≥ 2 d before TCA)"}))


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise ApiError(500, "ARTIFACT_UNREADABLE", f"{path.name}: {e}", retryable=True)


@router.get("/livesky")
async def livesky(request: Request):
    p = ctx(request).settings.repo_dir / "data" / "livesky" / "latest.json"
    if not p.exists():
        raise ApiError(404, "NOT_FOUND", "live sky snapshot not built yet", "data/livesky/latest.json is produced by the livesky pipeline")
    return jresp(_read_json(p))


@router.get("/futures/{event_key}")
async def futures(event_key: str, request: Request):
    key = _key(event_key)
    base = ctx(request).settings.repo_dir / "registry" / "futures" / "fans"
    for name in (f"{key}.json", f"{key.replace(':', '_')}.json"):
        p = base / name
        if p.exists():
            return jresp(_read_json(p))
    raise ApiError(404, "NOT_FOUND", f"no futures fan for {key}")


@router.get("/models")
async def models(request: Request):
    c = ctx(request)
    return jresp({"models": list_models(c.settings.repo_dir), "engine": c.engine.status()})


@router.get("/models/{name}/card")
async def model_card(name: str, request: Request):
    card = read_card(ctx(request).settings.repo_dir, name)
    if card is None:
        raise ApiError(404, "NOT_FOUND", f"no model card for {name}")
    return jresp(card)


@router.get("/predictions/{fname}")
async def predictions(fname: str, request: Request):
    if fname not in ("test_events.csv", "test_rows.csv"):
        raise ApiError(404, "NOT_FOUND", f"unknown file {fname}", extra={"valid": ["test_events.csv", "test_rows.csv"]})
    p = ctx(request).settings.repo_dir / "predictions" / fname
    if not p.exists():
        raise ApiError(404, "NOT_FOUND", f"{fname} not generated yet", "engine: predict_test pipeline")
    return FileResponse(p, media_type="text/csv", filename=fname)
