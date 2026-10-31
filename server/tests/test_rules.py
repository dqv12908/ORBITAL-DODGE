"""Rules table tests (design 3.4): one per row, boundaries 1e-6, 1e-4, 1.0 km, 5.0 km, both tier modes."""
import math

import pytest

from riskengine.rules import assess, fmt_pc


def row(miss=10000.0, vrel=10000.0, ttca=3.0, risk=None, cov_t=0, cov_c=0):
    r = {"miss_distance": miss, "relative_speed": vrel, "time_to_tca": ttca, "cov_unknown_t": cov_t, "cov_unknown_c": cov_c}
    if risk is not None:
        r["risk"] = risk
    return r


def pc(lp, status="OK", k=0.5):
    return {"status": status, "log10": lp, "k_star": k}


@pytest.mark.parametrize("lp,tier", [(-3.9, "CRITICAL"), (-4.0, "CRITICAL"), (math.log10(0.99e-4), "ELEVATED"),
                                     (-6.0, "ELEVATED"), (math.log10(0.99e-6), "SAFE"), (-30, "SAFE")])
def test_pc_boundaries(lp, tier):
    a = assess(row(), pc(lp))
    assert a["pc_tier"] == tier and a["tier"] == tier


@pytest.mark.parametrize("miss,tier", [(999.9, "CRITICAL"), (1000.0, "ELEVATED"), (5000.0, "ELEVATED"),
                                       (5000.1, "SAFE"), (50, "CRITICAL")])
def test_miss_boundaries(miss, tier):
    a = assess(row(miss=miss), pc(-20))
    assert a["miss_tier"] == tier and a["tier"] == tier


def test_rule_ids_and_reasons():
    a = assess(row(), pc(math.log10(3.1e-4)))
    assert a["rule_ids"][0] == "R-PC-CRIT" and a["reasons"][0] == "Pc 3.1e-4 ≥ 1e-4"
    a = assess(row(), pc(math.log10(4.9e-6)))
    assert a["reasons"][0] == "Pc 4.9e-6 in [1e-6, 1e-4)"
    a = assess(row(miss=140), pc(-20))
    assert "Miss 140 m < 1 km (miss-distance screen)" in a["reasons"]
    a = assess(row(miss=1630), pc(-20))
    assert "Miss 1.63 km in [1, 5] km" in a["reasons"]
    a = assess(row(), pc(-20))
    assert a["rule_ids"] == ["R-SAFE"] and a["reasons"] == ["Pc < 1e-6 and miss > 5 km"]
    assert len(a["citations"]) == len(a["rule_ids"])


def test_cov_default():
    a = assess(row(miss=30000, cov_c=1, risk=-14.3), pc(None, "COV_UNKNOWN"))
    assert a["pc_tier"] == "UNKNOWN" and a["tier"] == "ELEVATED" and a["automation"] == "BLOCKED" and a["escalate"]
    assert a["reasons"][0] == "Chaser covariance is a default (10 Earth radii): not actionable"
    assert a["reported_log10"] is None
    a = assess(row(miss=300, cov_t=1), pc(None, "COV_UNKNOWN"))
    assert a["tier"] == "CRITICAL" and a["reasons"][0].startswith("Target covariance")
    a = assess(row(miss=30000, cov_c=1), pc(None, "COV_UNKNOWN"), mode="CARA_PC")
    assert a["tier"] == "ELEVATED"


def test_low_vrel():
    a = assess(row(vrel=63), pc(-8, "LOW_VREL"))
    assert "Q-LOW-VREL" in a["rule_ids"] and a["escalate"]
    assert "Relative speed 63 m/s: 2D Pc assumption invalid" in a["reasons"]
    assert "Q-LOW-VREL" not in assess(row(vrel=100), pc(-8))["rule_ids"]


def test_phys_disagree():
    a = assess(row(miss=56878, risk=-4.99), pc(-478.0))
    assert "Q-PHYS-DISAGREE" in a["rule_ids"] and a["pc_tier"] == "ELEVATED" and a["escalate"]
    assert "Our Pc 1e-478 vs reported 1.0e-5: sources disagree" in a["reasons"]
    # both tiny -> no disagreement flag even if far apart
    assert "Q-PHYS-DISAGREE" not in assess(row(risk=-20), pc(-9))["rule_ids"]
    # exactly 1 dex apart -> not fired (strict >)
    assert "Q-PHYS-DISAGREE" not in assess(row(risk=-5), pc(-6))["rule_ids"]


def test_dilution_badge_only():
    a = assess(row(), pc(-7, k=1.9))
    assert "Q-DILUTION" in a["rule_ids"] and a["tier"] == "SAFE" and not a["escalate"]
    assert "Low Pc may be dilution: covariance larger than worst case (k* 1.9)" in a["reasons"]
    assert "Q-DILUTION" not in assess(row(), pc(-7, k=1.0))["rule_ids"]


def test_mcp():
    a = assess(row(ttca=10.6 / 24, miss=136), pc(-4.19))
    assert a["mcp"] and a["mcp_label"] == "AT/AFTER MCP" and "T−10.6 h: inside commitment window" in a["reasons"]
    assert not assess(row(ttca=1.01), pc(-9))["mcp"]
    assert assess(row(ttca=1.0), pc(-9))["mcp"]
    assert assess(row(ttca=0.5), pc(-9), ttca_now_d=2.0)["mcp"] is False


def test_tier_modes():
    a = assess(row(miss=80), pc(-16.3), mode="BRIEF")
    assert a["tier"] == "CRITICAL" and a["pc_tier"] == "SAFE" and a["miss_tier"] == "CRITICAL"
    b = assess(row(miss=80), pc(-16.3), mode="CARA_PC")
    assert b["tier"] == "SAFE" and b["miss_tier"] == "CRITICAL" and "MISS-SCREEN" in b["flags"]
    assert any(r.startswith("CRITICAL by miss screen (80 m) · Pc tier SAFE") for r in b["reasons"])
    c = assess(row(miss=3930), pc(math.log10(1.2e-4)), mode="CARA_PC")
    assert c["tier"] == "CRITICAL"
    with pytest.raises(ValueError):
        assess(row(), pc(-9), mode="NOPE")


def test_fmt_pc():
    assert fmt_pc(math.log10(6.5e-5)) == "6.5e-5"
    assert fmt_pc(-479.2) == "1e-479"
    assert fmt_pc(-4.0) == "1.0e-4"
