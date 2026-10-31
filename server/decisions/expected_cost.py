"""Expected-cost baseline (design §4.4): argmin over allowed actions of E[loss] under p = P(final CRITICAL)."""
from __future__ import annotations

P_CRIT_EMPIRICAL = {"SAFE": 0.002, "ELEVATED": 0.053, "CRITICAL": 0.800}


def p_for(forecast: dict | None, tier_now: str) -> tuple[float, str]:
    if forecast and forecast.get("valid") and forecast.get("p_critical") is not None:
        src = "persistence-empirical" if forecast.get("fallback") else "ml-calibrated"
        return float(forecast["p_critical"]), src
    return P_CRIT_EMPIRICAL.get(tier_now, 0.053), "persistence-empirical (approximate inside 2 d)"


def expected_losses(p: float, allowed: list[str], loss: dict) -> dict[str, float]:
    return {a: round(p * loss[a]["critical"] + (1 - p) * loss[a]["not_critical"], 4) for a in allowed}


def baseline(forecast: dict | None, tier_now: str, allowed: list[str], loss: dict) -> dict:
    p, src = p_for(forecast, tier_now)
    el = expected_losses(p, allowed, loss)
    action = min(allowed, key=lambda a: (el[a], allowed.index(a))) if allowed else None
    return {"action": action, "expected_loss": el, "p_critical": round(p, 6), "p_source": src}
