"""SSE envelope v1 as pydantic models (discriminated union on `type`) -> shared/schema/events.schema.json.

Run: cd server && .venv/bin/python -m app.schema   (writes the file; tests check it is up to date)
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class PcObj(Open):
    status: Literal["OK", "COV_UNKNOWN", "LOW_VREL", "NPD_CLIPPED"]
    log10: float | None = None
    method: str | None = None
    hbr_m: float | None = None
    sigma_plane_m: list[float] | None = Field(default=None, min_length=2, max_length=2)
    theta_deg: float | None = None
    mahalanobis: float | None = None
    k_star: float | None = None
    log10_pc_max: float | None = None
    dilution: bool | None = None


class TierObj(Open):
    tier: Literal["SAFE", "ELEVATED", "CRITICAL"]
    pc_tier: Literal["SAFE", "ELEVATED", "CRITICAL", "UNKNOWN"]
    miss_tier: Literal["SAFE", "ELEVATED", "CRITICAL"]
    mode: Literal["BRIEF", "CARA_PC"]
    rule_ids: list[str]
    reasons: list[str]
    citations: list[str] = []


class RiskAssessed(Open):
    cdm_id: str
    pc: PcObj
    reported_log10: float | None
    tier: TierObj
    disagree_dex: float | None = None


class ClockTick(Open):
    sim_time: str
    speed: int
    paused: bool
    held: bool
    hold_reason: str | None = None


class ScenarioLoaded(Open):
    id: str
    seed: int
    d0: str
    cast: list[str]
    preloaded_cdms: int


class CdmReceived(Open):
    cdm_id: str
    cdm_index: int
    n_known: int
    ttca_d: float | None
    created_sim: str
    miss_m: float | None
    rel_speed_ms: float | None
    rtn_m: list[float | None]
    c_object_type: str | None
    flags: list[str]
    source: dict


class ForecastUpdated(Open):
    model_version: str | None
    valid: bool
    p_tiers: dict | None
    p_critical: float | None
    calib_bin: dict | None
    baseline: dict | None
    watch: bool
    rank: int
    model: str | None = None
    p_source: str | None = None
    alert_threshold: float | None = None
    cost_ratio_R: float | None = None
    alert: bool | None = None


class PolicyGated(Open):
    allowed: list[Literal["MONITOR", "REQUEST_TRACKING", "PREPARE_MANOEUVRE"]]
    must_escalate: bool
    blocked: bool
    reasons: list[str]


class DecisionRequested(Open):
    decision_id: str
    provider_chain: list[str]
    request_hash: str


class DecisionMade(Open):
    decision_id: str
    provider: str
    answer: str | None
    probabilities: dict
    confidence: float | None
    evidence_sufficient_p: float | None
    refused: bool
    latency_ms: int | None
    baseline: dict
    agree: bool | None
    probability_kind: str | None = None


class PolicyVerdict(Open):
    decision_id: str | None
    verdict: Literal["AUTO", "ESCALATE", "BLOCK"]
    action: str | None
    reasons: list[str]


class ApprovalRequired(Open):
    approval_id: str
    decision_id: str
    options: list[str]
    suggested: str | None
    ttca_h: float | None
    mcp_passed: bool
    reasons: list[str]


class HumanAction(Open):
    approval_id: str
    action: str
    actor: str
    note: str | None
    wait_sim_min: float


class ExplanationReady(Open):
    cdm_id: str
    source: Literal["template", "llm"]
    verified: bool
    text: str
    rejected_reason: str | None = None


class LogEntry(Open):
    level: Literal["INFO", "CDM", "AGENT", "WARN", "CRIT", "REFUSE", "ERROR"]
    code: str
    msg: str


class OutcomeRevealed(Open):
    final: dict | None
    forecast_was: dict | None
    verdict: Literal["TP", "FP", "FN", "TN"] | None
    hidden: bool = False


class PolicyUpdated(Open):
    version: str
    diff: dict
    actor: str


class LedgerAppended(Open):
    seq: int
    hash: str
    kind: str


class SystemDegraded(Open):
    component: str
    reason: str
    fallback: str


def _env(t: str, data_model: type) -> type:
    return type(f"Env_{t.replace('.', '_')}", (BaseModel,), {
        "__annotations__": {"v": Literal[1], "seq": int, "type": Literal[t], "at": str, "sim_time": str,
                            "event_key": str | None, "ledger_seq": int | None, "data": data_model},
        "model_config": ConfigDict(extra="forbid"),
        "seq": Field(ge=1),
        "event_key": Field(default=None, pattern=r"^(train|test|adhoc):[0-9]+$"),
        "ledger_seq": None,
    })


DATA = {
    "state.snapshot": Open, "clock.tick": ClockTick, "scenario.loaded": ScenarioLoaded, "cdm.received": CdmReceived,
    "risk.assessed": RiskAssessed, "forecast.updated": ForecastUpdated, "policy.gated": PolicyGated,
    "decision.requested": DecisionRequested, "decision.made": DecisionMade, "policy.verdict": PolicyVerdict,
    "approval.required": ApprovalRequired, "human.action": HumanAction, "explanation.ready": ExplanationReady,
    "log.entry": LogEntry, "outcome.revealed": OutcomeRevealed, "policy.updated": PolicyUpdated,
    "ledger.appended": LedgerAppended, "system.degraded": SystemDegraded,
}
ENVELOPES = [_env(t, m) for t, m in DATA.items()]
Envelope = Annotated[Union[tuple(ENVELOPES)], Field(discriminator="type")]
ADAPTER = TypeAdapter(Envelope)


def json_schema() -> dict:
    s = ADAPTER.json_schema()
    s["$id"] = "orbitaldodge/sse-envelope/v1"
    s["title"] = "ORBITAL DODGE SSE envelope v1"
    s["description"] = ("SSE `data:` payload. Wire: `id: <seq>` + `data: <json>` (no `event:` field; switch on .type). "
                        "Additive within v1; clients ignore unknown types/fields; on a seq gap call GET /api/state.")
    return s


def validate(env: dict) -> Any:
    return ADAPTER.validate_python(env)


def write(path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_schema(), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    from app.settings import get_settings
    out = get_settings().schema_dir / "events.schema.json"
    write(out)
    print(f"wrote {out}")
