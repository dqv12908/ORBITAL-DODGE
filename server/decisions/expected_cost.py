"""Expected-cost baseline (design §4.4): argmin over allowed actions of E[loss] under p = P(final CRITICAL).

Time-aware tracking value. REQUEST_TRACKING only lowers risk if the new tracking + orbit-determination update arrives
in time to act on it, i.e. at least TRACK_MIN_LEAD_D (about 1 day) before TCA. Inside that, tracking still costs its own
price but buys nothing, so when ttca_d < TRACK_MIN_LEAD_D its loss if the event ends critical becomes

    loss[TRACK]["critical"] = loss[MONITOR]["critical"] + loss[TRACK]["not_critical"]

(an unmitigated critical event, the same as MONITOR, plus tracking's own cost). With the default table (MONITOR
100 / 0, TRACK 30 / 1, PREPARE 5 / 8) late TRACK is 101 / 1, which MONITOR (100 / 0) dominates, so the cost rule never
prefers TRACK late; PREPARE still wins once p is high enough. Before the cut-off, or when ttca_d is unknown, the
policy's loss table is used unchanged.
"""
from __future__ import annotations

from decisions.policy import MONITOR, TRACK

P_CRIT_EMPIRICAL = {"SAFE": 0.002, "ELEVATED": 0.053, "CRITICAL": 0.800}
TRACK_MIN_LEAD_D = 1.0  # tracking must land at least ~1 day before TCA to change anything


def p_for(forecast: dict | None, tier_now: str) -> tuple[float, str]:
    if forecast and forecast.get("valid") and forecast.get("p_critical") is not None:
        src = "persistence-empirical" if forecast.get("fallback") else "ml-calibrated"
        return float(forecast["p_critical"]), src
    return P_CRIT_EMPIRICAL.get(tier_now, 0.053), "persistence-empirical (approximate inside 2 d)"


def track_too_late(ttca_d: float | None) -> bool:
    return ttca_d is not None and ttca_d < TRACK_MIN_LEAD_D


def effective_loss(loss: dict, ttca_d: float | None) -> dict:
    """The policy loss table with TRACK's critical loss raised to MONITOR's (+ its own cost) when tracking is too late."""
    if not track_too_late(ttca_d) or TRACK not in loss or MONITOR not in loss:
        return loss
    out = {a: dict(row) for a, row in loss.items()}
    out[TRACK]["critical"] = float(loss[MONITOR]["critical"]) + float(loss[TRACK]["not_critical"])
    return out


def expected_losses(p: float, allowed: list[str], loss: dict, ttca_d: float | None = None) -> dict[str, float]:
    lt = effective_loss(loss, ttca_d)
    return {a: round(p * lt[a]["critical"] + (1 - p) * lt[a]["not_critical"], 4) for a in allowed}


def baseline(forecast: dict | None, tier_now: str, allowed: list[str], loss: dict, ttca_d: float | None = None) -> dict:
    p, src = p_for(forecast, tier_now)
    el = expected_losses(p, allowed, loss, ttca_d)
    action = min(allowed, key=lambda a: (el[a], allowed.index(a))) if allowed else None
    out = {"action": action, "expected_loss": el, "p_critical": round(p, 6), "p_source": src}
    if track_too_late(ttca_d) and TRACK in allowed:
        out["track_note"] = "tracking too late to help (< 1 d to TCA): costed like MONITOR plus its own cost"
    return out
