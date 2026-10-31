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
    "confidence_bar": 0.70,
    "loss": {MONITOR: {"critical": 100.0, "not_critical": 0.0},
             TRACK: {"critical": 30.0, "not_critical": 1.0},
             PREPARE: {"critical": 5.0, "not_critical": 8.0}},
    "hold_on_approval": True,
}
EDITABLE = {"tier_mode", "tau_watch", "confidence_bar", "loss", "hold_on_approval"}

BLOCK_REASON = "Covariance is a default: not actionable (CARA App. P.7)"
NO_PREPARE_COV = "PREPARE not offered: cannot plan on a default covariance"


def served_policy(alert_threshold: float | None) -> dict:
    """DEFAULT_POLICY with the desk's τ_watch set to the served forecaster's alert threshold (its evaluated operating point)."""
    p = copy.deepcopy(DEFAULT_POLICY)
    if alert_threshold is not None and 0.01 <= float(alert_threshold) <= 0.9:
        p["tau_watch"] = float(alert_threshold)
    return p


def watch_rule(forecast: dict | None, policy: dict) -> tuple[float | None, str]:
    """(threshold, source) the WATCH flag uses for this forecast.

    The served forecaster's own alert flag governs whenever the forecast carries an alert_threshold and the desk has
    not moved τ_watch away from it ("served"); otherwise p_critical >= τ_watch ("desk").
    """
    fc = forecast or {}
    tau = float(policy.get("tau_watch", 0.20))
    thr = fc.get("alert_threshold")
    if thr is not None and fc.get("alert") is not None and abs(float(thr) - tau) < 1e-6:
        return float(thr), "served"
    return tau, "desk"


def watch_flag(forecast: dict | None, tier: str, policy: dict) -> bool:
    """ML can only ADD attention: WATCH is never set on a CRITICAL event and never lowers a tier."""
    fc = forecast or {}
    if not fc.get("valid") or fc.get("p_critical") is None or tier == "CRITICAL":
        return False
    if ((fc.get("baseline") or {}).get("persistence_tier")) == "CRITICAL":
        return False
    thr, src = watch_rule(fc, policy)
    if src == "served":
        return bool(fc["alert"])
    return float(fc["p_critical"]) >= thr


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
    watch = watch_flag(forecast, t, policy)
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
        thr, src = watch_rule(forecast, policy)
        name = "the forecaster's alert threshold" if src == "served" else "τ_watch"
        reasons.append(f"WATCH: chance this ends CRITICAL {_pct(p)} ≥ {name} {_pct(thr)}")
    flagged = t != "SAFE" or watch or any_q
    reason_rules = {r: rid for rid, r in zip(tier.get("rule_ids", []), rule_reasons)}
    return {"allowed": allowed, "must_escalate": bool(must), "blocked": blocked, "reasons": reasons,
            "reason_rules": {r: reason_rules[r] for r in reasons if r in reason_rules},
            "flagged": bool(flagged), "watch": watch, "mcp_passed": bool(mcp), "critical": t == "CRITICAL"}


# Reason ranking for `primary_reasons` (lower = more important): data quality / BLOCK > physics-vs-reported
# disagreement > CRITICAL rule > commitment window > model uncertainty > model-vs-cost-rule disagreement > the rest.
RANK_DATA, RANK_PHYS, RANK_CRIT, RANK_MCP, RANK_MODEL, RANK_DISAGREE, RANK_OTHER = range(7)
RULE_RANK = {"Q-COV-DEFAULT": RANK_DATA, "Q-LOW-VREL": RANK_DATA, "Q-PHYS-DISAGREE": RANK_PHYS,
             "R-PC-CRIT": RANK_CRIT, "R-MISS-CRIT": RANK_CRIT, "T-MCP": RANK_MCP}
MODEL_UNSURE_PREFIXES = ("Decision model only", "Decision model declined", "Model chose a disallowed action",
                         "Model says evidence is thin")
CRITICAL_ALWAYS = "CRITICAL events always come to you"


def reason_rank(reason: str, reason_rules: dict | None = None) -> int:
    rid = (reason_rules or {}).get(reason)
    if rid in RULE_RANK:
        return RULE_RANK[rid]
    if reason in (BLOCK_REASON, NO_PREPARE_COV):
        return RANK_DATA
    if reason == CRITICAL_ALWAYS:
        return RANK_CRIT
    if reason.startswith(MODEL_UNSURE_PREFIXES):
        return RANK_MODEL
    if reason.startswith("Model and cost rule disagree"):
        return RANK_DISAGREE
    return RANK_OTHER


def primary_reasons(reasons: list[str], reason_rules: dict | None = None, n: int = 2) -> list[str]:
    """The (at most) n most important reasons, most important first; ties keep their original order."""
    order = sorted(range(len(reasons)), key=lambda i: (reason_rank(reasons[i], reason_rules), i))
    return [reasons[i] for i in order[:n]]


def chosen_p(decision: dict) -> float | None:
    """The decision model's certainty: the probability of the action it chose (capped like `confidence` when the
    provider flagged too few samples). Falls back to `confidence` when no per-action probabilities exist."""
    ans = decision.get("answer")
    p = (decision.get("probabilities") or {}).get(ans)
    if p is None:
        p = decision.get("confidence")
    if p is None:
        return None
    cap = decision.get("confidence_cap")
    return min(float(p), float(cap)) if cap is not None else float(p)


def unflagged_verdict() -> dict:
    r = ["Within policy: SAFE and not on watch — log and keep monitoring (no model call)"]
    return {"verdict": "AUTO", "action": MONITOR, "reasons": r, "primary_reasons": list(r)}


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    v = x * 100
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def verdict(g: dict, decision: dict, baseline: dict, policy: dict, *, mock: bool = False) -> dict:
    """AUTO / ESCALATE / BLOCK. Checks in design order; ESCALATE collects every applicable reason.

    Certainty is ONE number: the probability of the chosen action (chosen_p) against confidence_bar. The provider's own
    confidence field (decision["model_confidence"]) is provenance only and never gates.
    Returns {verdict, action, reasons (all), primary_reasons (max 2, most important first)}.
    """
    allowed = g["allowed"]
    bar = float(policy.get("confidence_bar", 0.70))
    ans = decision.get("answer")
    conf = chosen_p(decision)
    base_action = baseline.get("action")
    suggested = ans if ans in allowed else base_action
    rr = g.get("reason_rules") or {}
    if g.get("blocked"):
        reasons = [BLOCK_REASON] + [r for r in g["reasons"] if r != BLOCK_REASON]
        return {"verdict": "BLOCK", "action": suggested if suggested in allowed else TRACK,
                "reasons": reasons, "primary_reasons": primary_reasons(reasons, rr)}
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
            reasons.append(f"Decision model only {conf * 100:.0f}% sure of {SHORT[ans]} (bar {bar * 100:.0f}%)")
        if base_action and ans != base_action and not mock:
            reasons.append(f"Model and cost rule disagree ({SHORT[ans]} vs {SHORT.get(base_action, base_action)})")
        ev = decision.get("evidence_sufficient_p")
        if ev is not None and ev < 0.5 and ans == PREPARE:
            reasons.append("Model says evidence is thin")
        if ans == PREPARE and not reasons:
            reasons.append("PREPARE is never automatic: flight dynamics needs a human go/no-go")
    if g.get("critical") and not reasons:  # defence in depth: CRITICAL is never AUTO
        reasons.append(CRITICAL_ALWAYS)
    if reasons:
        return {"verdict": "ESCALATE", "action": suggested, "reasons": reasons,
                "primary_reasons": primary_reasons(reasons, rr)}
    r = [f"Within policy: {SHORT[ans]} at {conf * 100:.0f}% (bar {bar * 100:.0f}%)"]
    return {"verdict": "AUTO", "action": ans, "reasons": r, "primary_reasons": list(r)}
