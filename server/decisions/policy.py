"""Code-enforced policy (never the model): allowed-action gating (design §4.2) and the verdict (§4.5).

Reason strings are user-facing and shown verbatim in ops.log, the approval card and the dashboard.
"""
from __future__ import annotations

import copy

from app.util import canonical, sha256

MONITOR, TRACK, PREPARE = "MONITOR", "REQUEST_TRACKING", "PREPARE_MANOEUVRE"
ACTIONS = [MONITOR, TRACK, PREPARE]
SHORT = {MONITOR: "MONITOR", TRACK: "TRACK", PREPARE: "PREPARE"}

DEFAULT_POLICY = {
    "tier_mode": "BRIEF",
    "tau_watch": 0.20,
    "confidence_bar": 0.75,
    "loss": {MONITOR: {"critical": 100.0, "not_critical": 0.0},
             TRACK: {"critical": 30.0, "not_critical": 1.0},
             PREPARE: {"critical": 5.0, "not_critical": 8.0}},
    "hold_on_approval": True,
}
EDITABLE = {"tier_mode", "tau_watch", "confidence_bar", "loss", "hold_on_approval"}

BLOCK_REASON = "Covariance is a default: not actionable (CARA App. P.7)"
NO_PREPARE_COV = "PREPARE not offered: cannot plan on a default covariance"


class PolicyError(ValueError):
    pass


def policy_version(p: dict) -> str:
    return sha256(canonical(p))[:12]


def clean_patch(patch: dict, current: dict) -> tuple[dict, dict]:
    """Validate a partial policy. Returns (new_policy, diff). Raises PolicyError listing valid values."""
    if not isinstance(patch, dict):
        raise PolicyError("body must be a JSON object")
    unknown = set(patch) - EDITABLE
    if unknown:
        raise PolicyError(f"unknown policy fields {sorted(unknown)}; editable: {sorted(EDITABLE)}")
    new = copy.deepcopy(current)
    diff = {}
    if "tier_mode" in patch:
        v = patch["tier_mode"]
        if v not in ("BRIEF", "CARA_PC"):
            raise PolicyError("tier_mode must be one of ['BRIEF', 'CARA_PC']")
        new["tier_mode"] = v
    if "tau_watch" in patch:
        v = patch["tau_watch"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.01 <= v <= 0.9):
            raise PolicyError("tau_watch must be a number in [0.01, 0.9] (desk range 0.05–0.50)")
        new["tau_watch"] = float(v)
    if "confidence_bar" in patch:
        v = patch["confidence_bar"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.5 <= v <= 0.99):
            raise PolicyError("confidence_bar must be a number in [0.5, 0.99]")
        new["confidence_bar"] = float(v)
    if "hold_on_approval" in patch:
        if not isinstance(patch["hold_on_approval"], bool):
            raise PolicyError("hold_on_approval must be a boolean")
        new["hold_on_approval"] = patch["hold_on_approval"]
    if "loss" in patch:
        lp = patch["loss"]
        if not isinstance(lp, dict) or set(lp) - set(ACTIONS):
            raise PolicyError(f"loss keys must be among {ACTIONS}")
        for a, row in lp.items():
            if not isinstance(row, dict) or set(row) - {"critical", "not_critical"}:
                raise PolicyError("loss rows must be {critical: number, not_critical: number}")
            for k, v in row.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= v <= 1000):
                    raise PolicyError("loss values must be numbers in [0, 1000]")
                new["loss"][a][k] = float(v)
    for k in EDITABLE:
        if new.get(k) != current.get(k):
            diff[k] = {"from": current.get(k), "to": new.get(k)}
    return new, diff


def gate(tier: dict, pc: dict, ttca_d: float | None, forecast: dict | None, policy: dict) -> dict:
    """Allowed actions for this CDM. Pure function of rule outputs + forecast watch flag."""
    t = tier.get("tier", "ELEVATED")
    flags = set(tier.get("flags", [])) | {r for r in tier.get("rule_ids", []) if r.startswith(("Q-", "T-"))}
    cov = "Q-COV-DEFAULT" in flags or pc.get("status") == "COV_UNKNOWN"
    disagree = "Q-PHYS-DISAGREE" in flags
    lowv = "Q-LOW-VREL" in flags or pc.get("status") == "LOW_VREL"
    mcp = "T-MCP" in flags or (ttca_d is not None and ttca_d <= 1.0)
    p = (forecast or {}).get("p_critical") if (forecast or {}).get("valid") else None
    fcw = (forecast or {}).get("watch") if (forecast or {}).get("valid") else None
    watch = bool(p is not None and t != "CRITICAL" and (fcw if fcw is not None else p >= policy.get("tau_watch", 0.20)))
    any_q = cov or disagree or lowv
    reasons: list[str] = []
    rule_reasons = list(tier.get("reasons", []))

    if t == "CRITICAL":
        allowed = [TRACK, PREPARE]
    elif t == "ELEVATED":
        allowed = [MONITOR, TRACK, PREPARE]
    elif watch:  # SAFE on watch (ML can only add attention)
        allowed = [MONITOR, TRACK]
    else:
        allowed = [MONITOR]
    if disagree or lowv:
        allowed = [a for a in ACTIONS if not (t == "CRITICAL" and a == MONITOR)]
    must = t == "CRITICAL" or any_q
    blocked = False
    if cov:
        blocked = True
        allowed = [a for a in allowed if a != PREPARE]
        if TRACK not in allowed:
            allowed.append(TRACK)
        if t != "CRITICAL" and MONITOR not in allowed:
            allowed.insert(0, MONITOR)
        reasons.append(BLOCK_REASON)
        reasons.append(NO_PREPARE_COV)
    allowed = [a for a in ACTIONS if a in allowed]
    if must:
        # the rule reasons that made it escalate, verbatim
        for r in rule_reasons:
            if r not in reasons and not r.startswith("Pc < 1e-6 and miss > 5 km"):
                reasons.append(r)
    if watch:
        reasons.append(f"WATCH: chance this ends CRITICAL {p * 100:.1f}% ≥ τ_watch {policy.get('tau_watch', 0.20) * 100:.0f}%")
    flagged = t != "SAFE" or watch or any_q
    return {"allowed": allowed, "must_escalate": bool(must), "blocked": blocked, "reasons": reasons,
            "flagged": bool(flagged), "watch": watch, "mcp_passed": bool(mcp), "critical": t == "CRITICAL"}


def unflagged_verdict() -> dict:
    return {"verdict": "AUTO", "action": MONITOR,
            "reasons": ["Within policy: SAFE and not on watch — log and keep monitoring (no model call)"]}


def verdict(g: dict, decision: dict, baseline: dict, policy: dict, *, mock: bool = False) -> dict:
    """AUTO / ESCALATE / BLOCK. Checks in design order; ESCALATE collects every applicable reason."""
    allowed = g["allowed"]
    bar = float(policy.get("confidence_bar", 0.75))
    ans = decision.get("answer")
    conf = decision.get("confidence")
    base_action = baseline.get("action")
    suggested = ans if ans in allowed else base_action
    if g.get("blocked"):
        return {"verdict": "BLOCK", "action": suggested if suggested in allowed else TRACK,
                "reasons": [BLOCK_REASON] + [r for r in g["reasons"] if r != BLOCK_REASON]}
    reasons: list[str] = []
    if decision.get("refused"):
        reasons.append("Decision model declined")
        suggested = base_action
    elif ans not in allowed:
        reasons.append("Model chose a disallowed action")
        suggested = base_action
    else:
        if g.get("must_escalate"):
            reasons += list(g["reasons"]) or ["Rules require a human decision"]
        if conf is not None and conf < bar:
            reasons.append(f"Not sure enough ({conf * 100:.0f}% < {bar * 100:.0f}%)")
        if base_action and ans != base_action and not mock:
            reasons.append(f"Model and cost rule disagree ({SHORT[ans]} vs {SHORT.get(base_action, base_action)})")
        ev = decision.get("evidence_sufficient_p")
        if ev is not None and ev < 0.5 and ans == PREPARE:
            reasons.append("Model says evidence is thin")
        if ans == PREPARE and not reasons:
            reasons.append("PREPARE is never automatic: flight dynamics needs a human go/no-go")
    if g.get("critical") and not reasons:  # defence in depth: CRITICAL is never AUTO
        reasons.append("CRITICAL events always come to you")
    if reasons:
        return {"verdict": "ESCALATE", "action": suggested, "reasons": reasons}
    return {"verdict": "AUTO", "action": ans,
            "reasons": [f"Within policy: {SHORT[ans]} at {conf * 100:.0f}% (bar {bar * 100:.0f}%)"]}
