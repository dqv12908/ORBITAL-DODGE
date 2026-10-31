"""Policy overrides (design §6.9): CRITICAL never AUTO; disallowed answer -> ESCALATE; COV_UNKNOWN -> BLOCK."""
import itertools

import pytest

from decisions import expected_cost
from decisions import policy as pol
from decisions.policy import MONITOR, PREPARE, TRACK

P = dict(pol.DEFAULT_POLICY)


def tier(t, pc_tier=None, rules=(), reasons=None):
    return {"tier": t, "pc_tier": pc_tier or t, "miss_tier": t, "mode": "BRIEF", "rule_ids": list(rules),
            "reasons": reasons or [f"reason for {t}"], "flags": [r for r in rules if r[:2] in ("Q-", "T-")]}


def dec(answer, conf=0.95, ev=0.9, refused=False):
    probs = {a: (conf if a == answer else (1 - conf) / 2) for a in (MONITOR, TRACK, PREPARE)}
    return {"answer": answer, "confidence": conf, "probabilities": probs, "evidence_sufficient_p": ev, "refused": refused}


OK_PC = {"status": "OK", "log10": -5.0}


@pytest.mark.parametrize("answer,conf,ev,mock", list(itertools.product([MONITOR, TRACK, PREPARE, "BURN", None],
                                                                     [0.3, 0.8, 0.999], [0.1, 0.9], [True, False])))
def test_critical_never_auto(answer, conf, ev, mock):
    g = pol.gate(tier("CRITICAL"), OK_PC, 3.0, None, P)
    assert MONITOR not in g["allowed"] and g["must_escalate"]
    base = expected_cost.baseline(None, "CRITICAL", g["allowed"], P["loss"])
    v = pol.verdict(g, dec(answer, conf, ev), base, P, mock=mock)
    assert v["verdict"] != "AUTO"


def test_critical_inside_mcp_never_auto_even_if_model_agrees():
    g = pol.gate(tier("CRITICAL", rules=["R-MISS-CRIT", "T-MCP"]), OK_PC, 0.4, None, P)
    base = expected_cost.baseline(None, "CRITICAL", g["allowed"], P["loss"])
    v = pol.verdict(g, dec(base["action"], 0.99), base, P)
    assert v["verdict"] == "ESCALATE" and g["mcp_passed"]


def test_disallowed_answer_escalates_and_is_discarded():
    g = pol.gate(tier("CRITICAL"), OK_PC, 3.0, None, P)
    base = expected_cost.baseline(None, "CRITICAL", g["allowed"], P["loss"])
    v = pol.verdict(g, dec(MONITOR, 0.99), base, P)
    assert v["verdict"] == "ESCALATE"
    assert "Model chose a disallowed action" in v["reasons"]
    assert v["action"] in g["allowed"]


def test_cov_unknown_blocks_and_never_offers_prepare():
    t = tier("ELEVATED", pc_tier="UNKNOWN", rules=["Q-COV-DEFAULT"])
    g = pol.gate(t, {"status": "COV_UNKNOWN", "log10": None}, 4.0, None, P)
    assert g["blocked"] and PREPARE not in g["allowed"] and TRACK in g["allowed"]
    for a in (MONITOR, TRACK, PREPARE):
        v = pol.verdict(g, dec(a, 0.99), {"action": TRACK}, P)
        assert v["verdict"] == "BLOCK"
        assert v["reasons"][0] == pol.BLOCK_REASON


def test_low_confidence_escalates_with_verbatim_reason():
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    v = pol.verdict(g, dec(TRACK, 0.62), {"action": TRACK}, P)
    assert v["verdict"] == "ESCALATE" and "Not sure enough (62% < 75%)" in v["reasons"]


def test_disagreement_with_cost_rule_escalates():
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    v = pol.verdict(g, dec(TRACK, 0.9), {"action": PREPARE}, P)
    assert "Model and cost rule disagree (TRACK vs PREPARE)" in v["reasons"]


def test_refusal_escalates():
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    v = pol.verdict(g, {"answer": None, "confidence": None, "refused": True}, {"action": TRACK}, P)
    assert v["verdict"] == "ESCALATE" and v["reasons"][0] == "Decision model declined"


def test_prepare_never_auto():
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    v = pol.verdict(g, dec(PREPARE, 0.99, 0.9), {"action": PREPARE}, P)
    assert v["verdict"] == "ESCALATE"


def test_auto_within_policy():
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    v = pol.verdict(g, dec(TRACK, 0.88), {"action": TRACK}, P)
    assert v == {"verdict": "AUTO", "action": TRACK, "reasons": ["Within policy: TRACK at 88% (bar 75%)"]}


def test_watch_only_adds_attention():
    fc = {"valid": True, "p_critical": 0.4, "watch": True}
    g = pol.gate(tier("SAFE"), OK_PC, 3.0, fc, P)
    assert g["flagged"] and g["watch"] and g["allowed"] == [MONITOR, TRACK]
    # forecast can never remove an escalation or lower a tier
    g2 = pol.gate(tier("CRITICAL"), OK_PC, 3.0, {"valid": True, "p_critical": 0.0}, P)
    assert g2["must_escalate"] and g2["critical"]


def test_expected_cost_thresholds():
    L = P["loss"]
    assert expected_cost.baseline({"valid": True, "p_critical": 0.01}, "SAFE", [MONITOR, TRACK, PREPARE], L)["action"] == MONITOR
    assert expected_cost.baseline({"valid": True, "p_critical": 0.10}, "SAFE", [MONITOR, TRACK, PREPARE], L)["action"] == TRACK
    assert expected_cost.baseline({"valid": True, "p_critical": 0.30}, "SAFE", [MONITOR, TRACK, PREPARE], L)["action"] == PREPARE


def test_policy_patch_validation():
    new, diff = pol.clean_patch({"tier_mode": "CARA_PC", "tau_watch": 0.3}, P)
    assert new["tier_mode"] == "CARA_PC" and set(diff) == {"tier_mode", "tau_watch"}
    with pytest.raises(pol.PolicyError):
        pol.clean_patch({"tier_mode": "YOLO"}, P)
    with pytest.raises(pol.PolicyError):
        pol.clean_patch({"nope": 1}, P)
