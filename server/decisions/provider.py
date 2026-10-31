"""Decision provider chain (design §4.3): typesafe-jev (official) -> openai-decisions -> jev-router (OpenRouter) -> mock.

- Only the rule-ALLOWED actions are offered, as opaque choice values o1..oN.
- The expected-cost baseline's choice is withheld from the request.
- Responses are cached on disk by sha256(provider, model, prompt_version, canonical request); REPLAY serves only cache.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Protocol

import structlog

from app.util import canonical, dumps, sha256
from decisions import mock as mock_mod
from decisions.policy import MONITOR, PREPARE, TRACK
from llm.factsheet import fmt_miss, fmt_sci, fmt_ttca, history_line, trend_line

log = structlog.get_logger("decisions")
PROMPT_VERSION = "decide-v1"

CHOICE_TEXT = {
    MONITOR: "Keep monitoring: wait for the next CDM (about 8 h) and reassess",
    TRACK: "Request extra tracking and an orbit-determination update before deciding",
    PREPARE: "Hand to flight dynamics now to prepare manoeuvre options for a human go/no-go",
}


class ProviderError(Exception):
    def __init__(self, outcome: str, msg: str):
        super().__init__(msg)
        self.outcome = outcome  # timeout | error | refusal | unconfigured | cache_miss


class DecisionProvider(Protocol):
    name: str
    model: str

    async def decide(self, request: dict, timeout: float) -> dict: ...


def build_input(facts: dict, sim_time: str) -> str:
    """Plain text input, one fact per line (reference contextText style). Facts to read, never to compute."""
    ttca = facts.get("ttca")
    mcp = "commitment window (T-24 h) already open" if facts.get("mcp_open") else "before the commitment window (T-24 h)"
    if facts.get("pc_status") == "COV_UNKNOWN":
        phys = "PHYSICS: Pc not computable (default covariance on at least one object)"
    else:
        phys = f"PHYSICS: Pc {fmt_sci(facts.get('pc'))} (2D, status {facts.get('pc_status')})"
    rules = f"RULES: tier {facts.get('tier')} (Pc tier {facts.get('pc_tier')}, miss tier {facts.get('miss_tier')}, mode {facts.get('tier_mode')}); " + \
        "; ".join(facts.get("reasons") or [])
    if facts.get("forecast_valid"):
        fc = f"FORECAST: chance this ends CRITICAL {facts['p_critical'] * 100:.1f}% (calibrated statistics; baseline tier {facts.get('persistence_tier')})"
    else:
        fc = "FORECAST: not applicable inside 2 d (model trained at T-2 d)"
    q = []
    if facts.get("pc_status") == "COV_UNKNOWN":
        q.append("default covariance")
    if "Q-PHYS-DISAGREE" in (facts.get("rule_ids") or []):
        q.append("our Pc and the reported risk disagree by more than 1 order of magnitude")
    if "Q-LOW-VREL" in (facts.get("rule_ids") or []):
        q.append("low relative speed: 2D Pc assumption invalid")
    dq = "DATA QUALITY: " + ("; ".join(q) if q else "no default covariance; relative speed normal")
    lines = [
        "ROLE: decision support for a satellite conjunction analyst. A human approves every action.",
        f"EVENT {facts['event_key']} · mission {int(facts['mission_id']) if facts.get('mission_id') is not None else 'n/a'} · chaser {facts.get('chaser_type')}",
        f"TIME: SIM {sim_time} · TCA in {fmt_ttca(ttca)} · {mcp}",
        f"LATEST CDM {facts['cdm_number']} of {facts['n_cdm_known']} (T-{fmt_ttca(ttca)}): miss {fmt_miss(facts.get('miss_m'))}",
        phys, rules,
        f"HISTORY: {history_line(facts)}",
        f"TREND: {trend_line(facts)}",
        fc, dq,
        "CONSTRAINT: choose only among the listed options.",
    ]
    return "\n".join(lines)


def build_request(facts: dict, allowed: list[str], sim_time: str, model: str) -> tuple[dict, dict]:
    mapping = {f"o{i + 1}": a for i, a in enumerate(allowed)}
    req = {
        "model": model,
        "input": build_input(facts, sim_time),
        "questions": [
            {"type": "choice", "name": "next_action",
             "instructions": "Which next action best balances collision risk against the cost of acting early? Use only the evidence above.",
             "choices": [{"value": k, "description": CHOICE_TEXT[a]} for k, a in mapping.items()]},
            {"type": "predicate", "name": "evidence_sufficient",
             "instructions": "Is the evidence above sufficient and internally consistent to act on without new tracking data?"},
        ],
    }
    return req, mapping


def finish(raw: dict, mapping: dict) -> dict:
    """Normalise a provider answer: map opaque values back, clamp, renormalise to 4 dp, argmax = answer."""
    probs_raw = raw.get("probabilities") or {}
    probs = {}
    for k, a in mapping.items():
        v = probs_raw.get(k, probs_raw.get(a, 0.0))
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        probs[a] = min(max(v, 0.0), 1.0)
    choice = raw.get("choice")
    ans = mapping.get(choice, choice)  # an unknown value stays as-is so policy can flag it as disallowed
    s = sum(probs.values())
    if s > 0:
        probs = {a: round(p / s, 4) for a, p in probs.items()}
    elif ans in probs:
        probs[ans] = 1.0
    if ans is None and probs:
        ans = max(probs, key=probs.get)
    # ONE certainty number everywhere: `confidence` = the probability of the chosen action (capped when a vote had
    # too few samples). The provider's own confidence field (e.g. TypeSafe's) is kept as `model_confidence`.
    conf = probs.get(ans) if ans in probs else None
    if conf is None and raw.get("confidence") is not None and not probs:
        conf = raw.get("confidence")
    cap = raw.get("confidence_cap")
    if conf is not None and cap is not None:
        conf = min(float(conf), float(cap))
    mconf = raw.get("confidence")
    ev = raw.get("evidence_sufficient_p")
    return {"answer": ans, "probabilities": probs, "confidence": None if conf is None else round(float(conf), 4),
            "model_confidence": None if mconf is None else round(float(mconf), 4),
            "confidence_cap": None if cap is None else float(cap),
            "evidence_sufficient_p": None if ev is None else round(float(ev), 4)}


class DecisionService:
    def __init__(self, settings, providers: list | None = None):
        self.s = settings
        self.sem = asyncio.Semaphore(2)  # bounded concurrency for outbound decision calls
        self.cache_dir: Path = settings.cache_root / "decisions"
        if providers is None:
            from decisions.jev import JevProvider
            from decisions.openai_decisions import OpenAIDecisions
            from decisions.typesafe import TypeSafeJev
            providers = []
            if settings.typesafe_api_key:  # Jev itself, official API: real per-action probabilities
                providers.append(TypeSafeJev(settings))
            if settings.openai_api_key:
                providers.append(OpenAIDecisions(settings))
            if settings.jev_key:
                providers.append(JevProvider(settings))
        self.providers = providers
        self.health = {"typesafe": "ok" if settings.typesafe_api_key else "unconfigured",
                       "openai": "ok" if settings.openai_api_key else "unconfigured",
                       "jev": "ok" if settings.jev_key else "unconfigured"}
        self.stats = {"calls": 0, "fallbacks": 0, "refusals": 0, "cache_hits": 0, "latencies_ms": []}

    @property
    def mode(self) -> str:
        return self.s.decisions_mode

    def chain_names(self) -> list[str]:
        if self.mode == "mock":
            return ["mock"]
        if self.mode == "replay":
            return [f"{p.name} (cache)" for p in self.providers] + ["mock"]
        return [p.name for p in self.providers] + ["mock"]

    def request_hash(self, provider: str, model: str, req: dict) -> str:
        return sha256(canonical({"provider": provider, "model": model, "v": PROMPT_VERSION, "req": req}))

    def _cache_get(self, h: str) -> dict | None:
        p = self.cache_dir / f"{h}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _cache_put(self, h: str, raw: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{h}.json").write_text(dumps(raw))
        except Exception as e:
            log.warning("cache_write_failed", error=str(e))

    def prepare(self, facts: dict, allowed: list[str], sim_time: str) -> dict:
        model = self.providers[0].model if self.providers else "mock"
        req, mapping = build_request(facts, allowed, sim_time, model)
        return {"request": req, "mapping": mapping,
                "request_hash": self.request_hash(self.providers[0].name if self.providers and self.mode != "mock" else "mock",
                                                  model, req)}

    async def decide(self, facts: dict, allowed: list[str], baseline: dict, loss: dict, seed: int, prepared: dict) -> dict:
        """Run the chain. Never raises. Returns normalised decision + `degraded` list for system.degraded events."""
        req, mapping = prepared["request"], prepared["mapping"]
        degraded: list[dict] = []
        attempts: list[dict] = []
        refusals = 0
        t_start = time.monotonic()
        self.stats["calls"] += 1
        if self.mode != "mock":
            for p in self.providers:
                remaining = self.s.decisions_budget_s - (time.monotonic() - t_start)
                if remaining <= 0.2:
                    degraded.append({"component": f"decisions.{p.name}", "reason": "decision budget exhausted",
                                     "fallback": "mock"})
                    break
                preq = dict(req, model=p.model)
                h = self.request_hash(p.name, p.model, preq)
                t0 = time.monotonic()
                try:
                    raw = self._cache_get(h)
                    cached = raw is not None
                    if raw is None:
                        if self.mode == "replay":
                            raise ProviderError("cache_miss", "replay mode: no cached response")
                        async with self.sem:
                            raw = await asyncio.wait_for(p.decide(dict(preq, _mapping=mapping), min(self.s.decisions_timeout_s, remaining)),
                                                         timeout=min(self.s.decisions_timeout_s, remaining) + 0.5)
                        if raw.get("refused"):
                            raise ProviderError("refusal", "provider refused")
                        self._cache_put(h, raw)
                    else:
                        self.stats["cache_hits"] += 1
                    out = finish(raw, mapping)
                    lat = int(raw.get("latency_ms", (time.monotonic() - t0) * 1000)) if cached else int((time.monotonic() - t0) * 1000)
                    attempts.append({"provider": p.name, "outcome": "ok", "latency_ms": lat, "cached": cached})
                    self._health(p.name, "ok")
                    self.stats["latencies_ms"].append(lat)
                    return {**out, "provider": raw.get("provider_label", p.name), "model": raw.get("routed_model", p.model),
                            "probability_kind": raw.get("probability_kind", "model probabilities"),
                            "routed": raw.get("routed"), "refused": False, "latency_ms": lat,
                            "request_hash": h, "cached": cached, "attempts": attempts, "degraded": degraded}
                except (asyncio.TimeoutError, ProviderError, Exception) as e:
                    outcome = e.outcome if isinstance(e, ProviderError) else ("timeout" if isinstance(e, asyncio.TimeoutError) else "error")
                    lat = int((time.monotonic() - t0) * 1000)
                    attempts.append({"provider": p.name, "outcome": outcome, "latency_ms": lat, "error": str(e)[:160]})
                    if outcome == "refusal":
                        refusals += 1
                        self.stats["refusals"] += 1
                    else:
                        if outcome != "cache_miss":
                            self._health(p.name, "down")
                        degraded.append({"component": f"decisions.{p.name}", "reason": f"{outcome}: {str(e)[:120]}",
                                         "fallback": "next provider" if p is not self.providers[-1] else "mock"})
            if self.providers and refusals == len(self.providers):
                # every real model declined: do not substitute the mock's rule as if it were a model
                return {"answer": None, "probabilities": {}, "confidence": None, "model_confidence": None,
                        "evidence_sufficient_p": None,
                        "provider": attempts[-1]["provider"], "model": None, "refused": True,
                        "probability_kind": None, "routed": None,
                        "latency_ms": int((time.monotonic() - t_start) * 1000), "request_hash": prepared["request_hash"],
                        "cached": False, "attempts": attempts, "degraded": degraded}
            if self.providers:
                self.stats["fallbacks"] += 1
        raw = mock_mod.decide(facts, allowed, baseline, loss, seed, prepared["request_hash"])
        out = finish(raw, mapping_identity(allowed, raw))
        attempts.append({"provider": "mock", "outcome": "ok", "latency_ms": 0})
        return {**out, "provider": "mock", "model": "mock (cost rule)", "refused": False, "latency_ms": 0,
                "probability_kind": "mock: softmax(-E[loss]/2) of the cost rule", "routed": None,
                "request_hash": prepared["request_hash"], "cached": False, "attempts": attempts, "degraded": degraded}

    def _health(self, name: str, status: str) -> None:
        key = ("openai" if name.startswith("openai") else "typesafe" if name.startswith("typesafe")
               else "jev" if name.startswith("jev") else name)
        self.health[key] = status


def mapping_identity(allowed: list[str], raw: dict) -> dict:
    return {a: a for a in allowed}
