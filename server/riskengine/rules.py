"""Tier rules engine (design 3.4). Pure function, table-driven.

assess(row, pc, mode="BRIEF") -> {tier, pc_tier, miss_tier, mode, rule_ids, reasons, citations, flags,
                                  escalate, automation, mcp, reported_log10, disagree_dex}

* BRIEF (default, compliant): tier = max(pc_tier, miss_tier)
* CARA_PC: tier = pc_tier; miss rules become the advisory flag MISS-SCREEN
* Q-COV-DEFAULT: pc_tier UNKNOWN, tier = max(miss_tier, ELEVATED) (both modes), automation BLOCKED
* Q-PHYS-DISAGREE: |ours - reported| > 1 dex and max(both) >= -6 -> pc_tier from the higher value, escalate
* T-MCP: TCA - now <= 24 h -> label "AT/AFTER MCP"
"""
from __future__ import annotations

import math

PC_CRIT_LOG10 = -4.0
PC_ELEV_LOG10 = -6.0
MISS_CRIT_M = 1000.0
MISS_ELEV_M = 5000.0
MCP_H = 24.0
LOW_VREL_MS = 100.0
DISAGREE_DEX = 1.0
REPORTED_SENTINEL_BAND = (-16.0, -13.4)   # dataset risk on default-covariance rows is meaningless

TIERS = ["SAFE", "ELEVATED", "CRITICAL"]
RANK = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2, "UNKNOWN": -1}

# rule id -> (condition text, result, citation)
RULES = {
    "R-PC-CRIT": ("Pc >= 1e-4", "pc_tier CRITICAL", "brief; CARA Vol 2 App. F (red)"),
    "R-PC-ELEV": ("1e-6 <= Pc < 1e-4", "pc_tier ELEVATED", "brief"),
    "R-PC-SAFE": ("Pc < 1e-6", "pc_tier SAFE", "brief"),
    "R-MISS-CRIT": ("miss < 1.0 km", "miss_tier CRITICAL", "brief; caution CARA App. M.2"),
    "R-MISS-ELEV": ("1.0 <= miss <= 5.0 km", "miss_tier ELEVATED", "brief"),
    "R-MISS-SAFE": ("miss > 5.0 km", "miss_tier SAFE", "brief"),
    "R-SAFE": ("Pc < 1e-6 and miss > 5 km", "SAFE", "brief"),
    "Q-COV-DEFAULT": ("default covariance on either object", "pc_tier UNKNOWN; tier >= ELEVATED; automation BLOCKED",
                      "CARA Vol 2 App. P.7"),
    "Q-LOW-VREL": ("v_rel < 100 m/s", "2D Pc invalid; ESCALATE", "CARA App. N.3 (threshold is ours)"),
    "Q-PHYS-DISAGREE": ("|log10 Pc_ours - risk| > 1 dex and max >= 1e-6", "tier from higher; ESCALATE", "ours"),
    "Q-DILUTION": ("k* > 1", "badge only", "CARA App. M.4"),
    "T-MCP": ("TCA - now <= 24 h", "AT/AFTER MCP; MONITOR not allowed for CRITICAL", "CARA App. F"),
    "MISS-SCREEN": ("CARA_PC mode, miss tier above Pc tier", "advisory flag", "brief; CARA App. M.2"),
}


def fmt_pc(log10: float) -> str:
    """3.1e-4 style; below 1e-10 only the exponent ('1e-479')."""
    if log10 is None or not math.isfinite(log10):
        return "n/a"
    if log10 >= 0:
        return "1"
    if log10 < -10:
        return f"1e{int(round(log10))}"
    e = math.floor(log10)
    m = 10 ** (log10 - e)
    if round(m, 1) >= 10:
        m, e = 1.0, e + 1
    return f"{m:.1f}e{e}"


def fmt_reported(log10: float) -> str:
    e = math.floor(log10)
    m = 10 ** (log10 - e)
    if round(m, 1) >= 10:
        m, e = 1.0, e + 1
    return f"{m:.1f}e{e}"


def fmt_miss(m: float) -> str:
    if m < 1000:
        return f"{m:.0f} m"
    return f"{m / 1000:.2f} km"


def _f(row, k):
    v = row.get(k) if isinstance(row, dict) else getattr(row, k, None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pc_tier_of(log10):
    if log10 is None:
        return "UNKNOWN"
    if log10 >= PC_CRIT_LOG10:
        return "CRITICAL"
    if log10 >= PC_ELEV_LOG10:
        return "ELEVATED"
    return "SAFE"


def miss_tier_of(miss_m):
    if miss_m is None:
        return "SAFE"
    if miss_m < MISS_CRIT_M:
        return "CRITICAL"
    if miss_m <= MISS_ELEV_M:
        return "ELEVATED"
    return "SAFE"


def _max(*ts):
    return max(ts, key=lambda t: RANK[t])


def assess(row: dict, pc: dict, mode: str = "BRIEF", ttca_now_d: float | None = None) -> dict:
    """Tier object for one CDM.

    row: CDM row (needs miss_distance, relative_speed, time_to_tca; optional risk, cov_unknown_t/c)
    pc:  pc object from riskengine.pc.pc_for_row (or a dict with 'status', 'log10', 'k_star')
    ttca_now_d: time from *now* to TCA in days; default = the row's time_to_tca (now = CDM creation)
    """
    if mode not in ("BRIEF", "CARA_PC"):
        raise ValueError(f"mode must be BRIEF or CARA_PC, got {mode!r}")
    rule_ids, reasons, flags = [], [], []

    def fire(rid, reason, flag=None):
        rule_ids.append(rid)
        reasons.append(reason)
        if flag:
            flags.append(flag)

    status = pc.get("status") or pc.get("pc_status")
    ours = pc.get("log10", pc.get("log10_pc"))
    ours = None if ours is None or not math.isfinite(float(ours)) else float(ours)
    miss = _f(row, "miss_distance")
    vrel = _f(row, "relative_speed")
    if vrel is None:
        vrel = pc.get("v_rel_ms")
    ttca = ttca_now_d if ttca_now_d is not None else _f(row, "time_to_tca")
    cov_t = bool(_f(row, "cov_unknown_t") or 0)
    cov_c = bool(_f(row, "cov_unknown_c") or 0)
    cov_unknown = status == "COV_UNKNOWN" or cov_t or cov_c
    if cov_unknown and not (cov_t or cov_c):     # raw row: infer which side from the sigmas
        def bad(side):
            vals = [_f(row, f"{side}_sigma_{a}") for a in "rtn"]
            return any(v is None or v >= 1e7 for v in vals)
        cov_t, cov_c = bad("t"), bad("c")
        if not (cov_t or cov_c):
            cov_c = True
    reported = _f(row, "risk")
    reported_valid = reported if (reported is not None and not cov_unknown) else None

    # --- miss axis
    miss_tier = miss_tier_of(miss)
    # --- Pc axis
    disagree_dex = None
    if cov_unknown:
        pc_tier = "UNKNOWN"
        who = "Target and chaser covariances are" if (cov_t and cov_c) else (
            "Target covariance is" if cov_t and not cov_c else "Chaser covariance is")
        fire("Q-COV-DEFAULT", f"{who} a default (10 Earth radii): not actionable", "COV_UNKNOWN")
    else:
        pc_val = ours
        if ours is not None and reported_valid is not None:
            rep = max(reported_valid, -30.0)
            disagree_dex = abs(ours - rep)
            if disagree_dex > DISAGREE_DEX and max(ours, rep) >= PC_ELEV_LOG10:
                fire("Q-PHYS-DISAGREE", f"Our Pc {fmt_pc(ours)} vs reported {fmt_reported(rep)}: sources disagree",
                     "PHYS_DISAGREE")
                pc_val = max(ours, rep)
        pc_tier = pc_tier_of(pc_val)
        if pc_tier == "CRITICAL":
            fire("R-PC-CRIT", f"Pc {fmt_pc(pc_val)} ≥ 1e-4")
        elif pc_tier == "ELEVATED":
            fire("R-PC-ELEV", f"Pc {fmt_pc(pc_val)} in [1e-6, 1e-4)")

    # --- combine
    if mode == "BRIEF":
        if miss_tier == "CRITICAL":
            fire("R-MISS-CRIT", f"Miss {fmt_miss(miss)} < 1 km (miss-distance screen)")
        elif miss_tier == "ELEVATED":
            fire("R-MISS-ELEV", f"Miss {fmt_miss(miss)} in [1, 5] km")
        tier = _max(pc_tier if pc_tier != "UNKNOWN" else "SAFE", miss_tier)
    else:
        tier = pc_tier if pc_tier != "UNKNOWN" else "SAFE"
        if RANK[miss_tier] > RANK[tier]:
            word = "CRITICAL" if miss_tier == "CRITICAL" else "ELEVATED"
            fire("MISS-SCREEN", f"{word} by miss screen ({fmt_miss(miss)}) · Pc tier {pc_tier}", "MISS-SCREEN")
    if cov_unknown:
        tier = _max(tier, miss_tier, "ELEVATED")
    if tier == "SAFE" and not cov_unknown:
        fire("R-SAFE", "Pc < 1e-6 and miss > 5 km" if mode == "BRIEF" else "Pc < 1e-6")
    elif not cov_unknown:
        if pc_tier == "SAFE" and ours is not None:
            fire("R-PC-SAFE", f"Pc {fmt_pc(ours)} < 1e-6")
        if mode == "BRIEF" and miss_tier == "SAFE" and miss is not None:
            fire("R-MISS-SAFE", f"Miss {fmt_miss(miss)} > 5 km")

    # --- quality gates and badges
    if vrel is not None and vrel < LOW_VREL_MS:
        fire("Q-LOW-VREL", f"Relative speed {vrel:.0f} m/s: 2D Pc assumption invalid", "LOW_VREL")
    ks = pc.get("k_star")
    if (not cov_unknown) and ks is not None and math.isfinite(float(ks)) and float(ks) > 1 and pc_tier != "CRITICAL":
        fire("Q-DILUTION", f"Low Pc may be dilution: covariance larger than worst case (k* {float(ks):.1f})", "DILUTION")
    if status == "NPD_CLIPPED":
        flags.append("NPD_CLIPPED")
    mcp = ttca is not None and ttca * 24 <= MCP_H
    if mcp:
        h = ttca * 24
        where = "inside commitment window" if h >= 0 else "after TCA"
        fire("T-MCP", f"T−{h:.1f} h: {where}" if h >= 0 else f"T+{-h:.1f} h: {where}", "MCP")

    escalate = any(r in rule_ids for r in ("Q-COV-DEFAULT", "Q-LOW-VREL", "Q-PHYS-DISAGREE")) or tier == "CRITICAL"
    return {
        "tier": tier,
        "pc_tier": pc_tier,
        "miss_tier": miss_tier,
        "mode": mode,
        "rule_ids": rule_ids,
        "reasons": reasons,
        "citations": [RULES[r][2] for r in rule_ids],
        "flags": flags,
        "escalate": bool(escalate),
        "automation": "BLOCKED" if cov_unknown else "ALLOWED",
        "mcp": bool(mcp),
        "mcp_label": "AT/AFTER MCP" if mcp else None,
        "reported_log10": reported_valid,
        "disagree_dex": disagree_dex,
    }


def assess_row(row: dict, mode: str = "BRIEF") -> dict:
    """Convenience: physics + rules for one raw/cleaned row."""
    from .pc import pc_for_row
    pc = pc_for_row(row)
    return {"pc": pc, "tier": assess(row, pc, mode=mode)}
