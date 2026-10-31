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
    assert v["verdict"] == "ESCALATE" and "Decision model only 62% sure of TRACK (bar 70%)" in v["reasons"]


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
    assert v == {"verdict": "AUTO", "action": TRACK, "reasons": ["Within policy: TRACK at 88% (bar 70%)"],
                 "primary_reasons": ["Within policy: TRACK at 88% (bar 70%)"]}


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


# ---------------------------------------------------------------- one certainty number: p(chosen action) vs the bar
def test_bar_uses_probability_of_chosen_action_not_provider_confidence():
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    assert P["confidence_bar"] == 0.70
    # Jev: TRACK bar at 80 % but its own confidence field says 33 % -> AUTO (p(chosen) clears the bar)
    d = {"answer": TRACK, "probabilities": {MONITOR: 0.1, TRACK: 0.8, PREPARE: 0.1}, "confidence": 0.8,
         "model_confidence": 0.33, "evidence_sufficient_p": 0.9}
    v = pol.verdict(g, d, {"action": TRACK}, P)
    assert v["verdict"] == "AUTO" and v["reasons"] == ["Within policy: TRACK at 80% (bar 70%)"]
    # top bar 56 % with a high provider confidence -> still escalates, quoting the 56 %
    d = {"answer": TRACK, "probabilities": {MONITOR: 0.3, TRACK: 0.56, PREPARE: 0.14}, "confidence": 0.56,
         "model_confidence": 0.95, "evidence_sufficient_p": 0.9}
    v = pol.verdict(g, d, {"action": TRACK}, P)
    assert v["verdict"] == "ESCALATE" and v["reasons"] == ["Decision model only 56% sure of TRACK (bar 70%)"]


def test_provider_finish_keeps_one_confidence_and_model_confidence():
    from decisions.provider import finish
    m = {"o1": MONITOR, "o2": TRACK}
    out = finish({"choice": "o2", "probabilities": {"o1": 0.45, "o2": 0.55}, "confidence": 0.33}, m)
    assert out["answer"] == TRACK and out["confidence"] == 0.55 and out["model_confidence"] == 0.33


# ---------------------------------------------------------------- time-aware tracking value
def test_track_not_preferred_inside_one_day():
    L, A = P["loss"], [MONITOR, TRACK, PREPARE]
    early = expected_cost.baseline(None, "ELEVATED", A, L, ttca_d=3.0)
    late = expected_cost.baseline(None, "ELEVATED", A, L, ttca_d=0.6)
    assert early["action"] == TRACK and late["action"] == MONITOR
    assert late["expected_loss"][TRACK] > late["expected_loss"][MONITOR] and "track_note" in late
    # late TRACK is costed as MONITOR's critical loss + TRACK's own cost
    eff = expected_cost.effective_loss(L, 0.6)
    assert eff[TRACK]["critical"] == L[MONITOR]["critical"] + L[TRACK]["not_critical"]
    assert eff[MONITOR] == L[MONITOR] and eff[PREPARE] == L[PREPARE] and L[TRACK]["critical"] == 30.0  # not mutated
    # a likely-critical event late still goes to PREPARE; unknown time keeps the table unchanged
    assert expected_cost.baseline({"valid": True, "p_critical": 0.8}, "CRITICAL", [TRACK, PREPARE], L, ttca_d=0.4)["action"] == PREPARE
    assert expected_cost.baseline(None, "ELEVATED", A, L)["action"] == TRACK
    # inside a day, the model saying MONITOR on a tiny-risk ELEVATED event is no longer a "disagreement"
    g = pol.gate(tier("ELEVATED"), OK_PC, 0.6, None, P)
    v = pol.verdict(g, dec(MONITOR, 0.9), late, P)
    assert v["verdict"] == "AUTO"


# ---------------------------------------------------------------- primary reasons
def test_primary_reasons_ranked_and_capped():
    t = tier("CRITICAL", rules=["R-PC-ELEV", "R-MISS-CRIT", "T-MCP"],
             reasons=["Pc 6.5e-5 in [1e-6, 1e-4)", "Miss 136 m < 1 km (miss-distance screen)", "T−10.6 h: inside commitment window"])
    g = pol.gate(t, OK_PC, 0.44, None, P)
    v = pol.verdict(g, dec(TRACK, 0.5), {"action": PREPARE}, P)
    assert len(v["primary_reasons"]) == 2 and len(v["reasons"]) > 2
    assert v["primary_reasons"] == ["Miss 136 m < 1 km (miss-distance screen)", "T−10.6 h: inside commitment window"]
    # physics-vs-reported disagreement outranks the CRITICAL rule
    t = tier("CRITICAL", rules=["Q-PHYS-DISAGREE", "R-PC-CRIT"],
             reasons=["Our Pc 1e-7 vs reported 2.0e-4: sources disagree", "Pc 2.0e-4 ≥ 1e-4"])
    g = pol.gate(t, OK_PC, 3.0, None, P)
    v = pol.verdict(g, dec(PREPARE, 0.9), {"action": PREPARE}, P)
    assert v["primary_reasons"][0] == "Our Pc 1e-7 vs reported 2.0e-4: sources disagree"
    # data quality / BLOCK first
    t = tier("ELEVATED", pc_tier="UNKNOWN", rules=["Q-COV-DEFAULT"], reasons=["Chaser covariance is a default"])
    g = pol.gate(t, {"status": "COV_UNKNOWN", "log10": None}, 4.0, None, P)
    v = pol.verdict(g, dec(TRACK, 0.9), {"action": TRACK}, P)
    assert v["verdict"] == "BLOCK" and v["primary_reasons"][0] == pol.BLOCK_REASON
    # model uncertainty outranks the model-vs-cost-rule disagreement
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, None, P)
    v = pol.verdict(g, dec(MONITOR, 0.5), {"action": TRACK}, P)
    assert v["primary_reasons"] == ["Decision model only 50% sure of MONITOR (bar 70%)",
                                    "Model and cost rule disagree (MONITOR vs TRACK)"]


# ---------------------------------------------------------------- WATCH follows the served model's alert flag
THR = 0.0634


def fc_served(p, alert=None, pers="SAFE"):
    return {"valid": True, "p_critical": p, "alert_threshold": THR, "alert": (p >= THR) if alert is None else alert,
            "baseline": {"persistence_tier": pers}}


def test_watch_follows_served_alert_flag():
    SP = pol.served_policy(THR)
    assert SP["tau_watch"] == THR and P["tau_watch"] == 0.20
    g = pol.gate(tier("SAFE"), OK_PC, 3.0, fc_served(0.07), SP)
    assert g["watch"] and g["flagged"] and g["allowed"] == [MONITOR, TRACK]
    assert any("alert threshold 6.3%" in r for r in g["reasons"])
    assert not pol.gate(tier("SAFE"), OK_PC, 3.0, fc_served(0.05), SP)["watch"]
    # the served flag itself governs (not a recomputation)
    assert pol.watch_flag(fc_served(0.05, alert=True), "SAFE", SP)
    # the desk moving τ takes over; a forecast without an alert threshold falls back to τ
    desk = dict(SP, tau_watch=0.20)
    assert not pol.watch_flag(fc_served(0.07), "SAFE", desk) and pol.watch_flag(fc_served(0.25), "SAFE", desk)
    assert pol.watch_rule(fc_served(0.07), desk) == (0.20, "desk")
    assert pol.watch_flag({"valid": True, "p_critical": 0.07}, "SAFE", SP)


def test_watch_only_adds_attention_never_lowers():
    SP = pol.served_policy(THR)
    # never on a CRITICAL event, and a low forecast never lowers the tier or removes an escalation
    assert not pol.watch_flag(fc_served(0.9), "CRITICAL", SP)
    g = pol.gate(tier("CRITICAL"), OK_PC, 3.0, fc_served(0.0), SP)
    assert g["must_escalate"] and g["critical"] and not g["watch"] and MONITOR not in g["allowed"]
    g = pol.gate(tier("ELEVATED"), OK_PC, 3.0, fc_served(0.0), SP)
    assert g["flagged"] and g["allowed"] == [MONITOR, TRACK, PREPARE]


def test_sim_desk_tau_defaults_to_served_threshold(tmp_path):
    from tests.backend_fixtures import make_sim
    sim, _, _ = make_sim(tmp_path)
    thr = sim.served_alert_threshold()
    assert thr is not None and abs(thr - 0.0634) < 1e-3
    assert sim.policy["tau_watch"] == thr and sim.policy["confidence_bar"] == 0.70
