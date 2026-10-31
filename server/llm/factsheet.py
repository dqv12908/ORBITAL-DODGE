"""FactSheet: the only input the decision model and the LLM ever see (design §5.2).

`facts` is a flat dict of key -> raw value. `render(key)` turns a value into the exact on-screen string with a fixed
formatter per key kind; the verifier checks every number in generated text against these rendered strings.
"""
from __future__ import annotations

import math
from typing import Any

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

PC_KEYS = {"pc", "reported_pc", "pc_red", "pc_yellow", "pc_max"}
PCT_KEYS = {"p_critical", "p_critical_empirical", "decision_confidence", "evidence_sufficient_p", "confidence_bar",
            "tau_watch", "calib_observed", "calib_lo", "calib_hi"}
MISS_KEYS = {"miss_m"}
KM_KEYS = {"miss_red_km", "miss_yellow_km"}
DUR_KEYS = {"ttca"}
WORDS_DUR_KEYS = {"ttca_words"}  # the same time to TCA, rounded for plain-language sentences ("11 hours", "2.4 days")
HOUR_KEYS = {"mcp_h", "next_cdm_h"}
DAY_KEYS = {"forecast_min_ttca_d"}
SPEED_KEYS = {"rel_speed_ms"}
COUNT_KEYS = {"cdm_number", "n_cdm_known", "calib_n", "mission_id"}

CONSTANTS = {"pc_red": 1e-4, "pc_yellow": 1e-6, "miss_red_km": 1.0, "miss_yellow_km": 5.0, "forecast_min_ttca_d": 2.0,
             "mcp_h": 24.0, "next_cdm_h": 8.0}

ACTION_TEXT = {
    "MONITOR": "keep monitoring and wait for the next CDM",
    "REQUEST_TRACKING": "request extra tracking and an orbit-determination update",
    "PREPARE_MANOEUVRE": "hand to flight dynamics to prepare manoeuvre options for a human go/no-go",
}


def fmt_sci(p: float | None) -> str:
    if p is None:
        return "not computable"
    if p <= 0:
        return "< 1e-300"
    lg = math.log10(p)
    if lg < -10:
        return "< 1e-10"
    exp = math.floor(lg)
    mant = round(10 ** (lg - exp), 1)
    if mant >= 10:
        mant, exp = 1.0, exp + 1
    return f"{mant:.1f}×10{str(exp).translate(SUP)}"


def fmt_pct(p: float | None) -> str:
    if p is None:
        return "n/a"
    v = p * 100
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def fmt_miss(m: float | None) -> str:
    if m is None:
        return "n/a"
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.2f} km"


def fmt_ttca(d: float | None) -> str:
    if d is None:
        return "n/a"
    if d < 0:
        return f"{-d * 24:.1f} h after TCA"
    if d < 1:
        return f"{d * 24:.1f} h"
    return f"{d:.2f} d"


def fmt_ttca_words(d: float | None) -> str:
    """Time to TCA for a sentence: whole hours inside a day, one-decimal days otherwise."""
    if d is None:
        return "n/a"
    if d < 0:
        h = round(-d * 24)
        return f"{h} hour{'s' if h != 1 else ''} after TCA"
    if d < 1:
        h = max(1, round(d * 24))
        return f"{h} hour{'s' if h != 1 else ''}"
    return f"{d:.1f} days"


def render(facts: dict, key: str) -> str:
    v = facts.get(key)
    if key in WORDS_DUR_KEYS:
        return fmt_ttca_words(v)
    if key in PC_KEYS:
        return fmt_sci(v)
    if key in PCT_KEYS:
        return fmt_pct(v)
    if key in MISS_KEYS:
        return fmt_miss(v)
    if key in DUR_KEYS:
        return fmt_ttca(v)
    if key in KM_KEYS:
        return f"{v:g} km"
    if key in HOUR_KEYS:
        return f"{v:g} h"
    if key in DAY_KEYS:
        return f"{v:g} d"
    if key in SPEED_KEYS:
        return "n/a" if v is None else f"{v:,.0f} m/s"
    if key in COUNT_KEYS:
        return "n/a" if v is None else f"{int(v)}"
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "n/a"
    return str(v)


def build_facts(*, event_key: str, row: dict, cdm_index: int, n_known: int, pc: dict, tier: dict,
                forecast: dict | None, gate: dict | None = None, decision: dict | None = None,
                verdict: dict | None = None, history: list[dict] | None = None, policy: dict | None = None) -> dict:
    def f(x):
        try:
            x = float(x)
            return None if math.isnan(x) or math.isinf(x) else x
        except (TypeError, ValueError):
            return None

    lp = pc.get("log10")
    reported = f(row.get("risk"))
    if pc.get("status") == "COV_UNKNOWN":
        reported = None
    ttca = f(row.get("time_to_tca"))
    facts: dict[str, Any] = {
        "event_key": event_key,
        "mission_id": f(row.get("mission_id")),
        "chaser_type": str(row.get("c_object_type") or "UNKNOWN"),
        "cdm_number": cdm_index + 1,
        "n_cdm_known": n_known,
        "ttca": ttca,
        "ttca_words": ttca,
        "miss_m": f(row.get("miss_distance")),
        "rel_speed_ms": f(row.get("relative_speed")),
        "pc": None if lp is None else 10 ** lp,
        "pc_status": pc.get("status"),
        "reported_pc": None if reported is None else 10 ** reported,
        "tier": tier.get("tier"),
        "pc_tier": tier.get("pc_tier"),
        "miss_tier": tier.get("miss_tier"),
        "tier_mode": tier.get("mode"),
        "rule_ids": list(tier.get("rule_ids", [])),
        "reasons": list(tier.get("reasons", [])),
        "citations": list(tier.get("citations", [])),
        "mcp_open": bool(ttca is not None and ttca <= 1.0),
        **CONSTANTS,
    }
    fc = forecast or {}
    facts["forecast_valid"] = bool(fc.get("valid"))
    if fc.get("valid"):
        facts["p_critical"] = f(fc.get("p_critical"))
        base = fc.get("baseline") or {}
        facts["persistence_tier"] = base.get("persistence_tier")
        facts["p_critical_empirical"] = f(base.get("p_critical_empirical"))
        cb = fc.get("calib_bin") or {}
        if cb:
            facts["calib_lo"], facts["calib_hi"] = f(cb.get("lo")), f(cb.get("hi"))
            facts["calib_observed"], facts["calib_n"] = f(cb.get("observed")), cb.get("n")
    if gate:
        facts["allowed"] = list(gate.get("allowed", []))
        facts["gate_reasons"] = list(gate.get("reasons", []))
    if decision:
        facts["decision_provider"] = decision.get("provider")
        facts["decision_answer"] = decision.get("answer")
        facts["decision_confidence"] = f(decision.get("confidence"))
        facts["evidence_sufficient_p"] = f(decision.get("evidence_sufficient_p"))
        facts["baseline_action"] = (decision.get("baseline") or {}).get("action")
    if verdict:
        facts["verdict"] = verdict.get("verdict")
        facts["verdict_action"] = verdict.get("action")
        facts["verdict_reasons"] = list(verdict.get("reasons", []))
    if policy:
        facts["confidence_bar"] = f(policy.get("confidence_bar"))
        facts["tau_watch"] = f(policy.get("tau_watch"))
    facts["history"] = [
        {"ttca": h.get("ttca"), "miss_m": h.get("miss_m"), "pc": h.get("pc"), "pc_status": h.get("pc_status")}
        for h in (history or [])
    ]
    return facts


def history_line(facts: dict) -> str:
    parts = []
    for h in facts.get("history", []):
        pc = "Pc not computable" if h.get("pc_status") == "COV_UNKNOWN" else f"Pc {fmt_sci(h.get('pc'))}"
        parts.append(f"T-{fmt_ttca(h.get('ttca'))} miss {fmt_miss(h.get('miss_m'))} {pc}")
    return " | ".join(parts)


def trend_line(facts: dict) -> str:
    h = facts.get("history", [])
    if len(h) < 2:
        return "first CDM for this event"
    a, b = h[-2], h[-1]
    out = []
    if a.get("miss_m") and b.get("miss_m"):
        out.append("miss shrank" if b["miss_m"] < a["miss_m"] else "miss grew")
    if a.get("pc") and b.get("pc"):
        out.append("Pc rose" if b["pc"] > a["pc"] else "Pc fell")
    return "; ".join(out) + " since the previous CDM" if out else "no comparable previous Pc"
