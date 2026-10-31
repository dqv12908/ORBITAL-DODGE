"""Deterministic template text (design §5.6). Always rendered first; the LLM may replace it only if verified."""
from __future__ import annotations

from llm.factsheet import ACTION_TEXT, render


def forecast_sentence(facts: dict) -> str:
    if not facts.get("forecast_valid"):
        if (facts.get("ttca") or 0) < 2:
            return "Forecast not applicable inside 2 d (physics on the latest CDM governs)"
        return "No forecast available"
    s = f"Chance this ends CRITICAL: {render(facts, 'p_critical')} (statistics, not a collision probability)"
    if facts.get("persistence_tier"):
        s += f"; persistence says {facts['persistence_tier']}"
    return s


def pc_phrase(facts: dict) -> str:
    if facts.get("pc_status") == "COV_UNKNOWN":
        return "Pc not computable (default covariance)"
    return f"Pc {render(facts, 'pc')} ({facts.get('pc_status')})"


def explain_template(facts: dict) -> dict:
    tier = facts.get("tier")
    reasons = "; ".join(facts.get("reasons") or []) or "no rule fired"
    headline = f"{tier} — {facts['event_key']}"
    parts = [f"{headline}: {reasons}.", f"{pc_phrase(facts)}.", f"{forecast_sentence(facts)}."]
    if facts.get("verdict"):
        vr = "; ".join(facts.get("verdict_reasons") or [])
        act = facts.get("verdict_action")
        parts.append(f"Policy: {facts['verdict']} — {vr}." if vr else f"Policy: {facts['verdict']}.")
        if act and facts["verdict"] != "AUTO":
            parts.append(f"Suggested next step for your decision: {ACTION_TEXT.get(act, act)}.")
    parts.append(f"Next CDM expected in ~{render(facts, 'next_cdm_h')}; TCA in {render(facts, 'ttca')}.")
    uncertainty = ("Covariance is a default: the collision probability cannot be computed."
                   if facts.get("pc_status") == "COV_UNKNOWN"
                   else "Pc is a 2D estimate from the CDM covariance; the miss-distance screen ignores covariance.")
    return {"headline": headline, "body": " ".join(parts), "uncertainty": uncertainty, "text": " ".join(parts)}
