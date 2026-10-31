"""LLM verifier adversarial cases (design §5.3)."""
from llm.explain import check_llm_output
from llm.factsheet import build_facts
from llm.templates import explain_template
from llm.verify import verify

ROW = {"time_to_tca": 0.44, "miss_distance": 136.0, "relative_speed": 11250.0, "risk": -4.19, "mission_id": 15,
       "c_object_type": "DEBRIS"}
PC = {"status": "OK", "log10": -4.19}
TIER = {"tier": "CRITICAL", "pc_tier": "ELEVATED", "miss_tier": "CRITICAL", "mode": "BRIEF",
        "rule_ids": ["R-PC-ELEV", "R-MISS-CRIT", "T-MCP"],
        "reasons": ["Pc 6.5e-5 in [1e-6, 1e-4)", "Miss 136 m < 1 km (miss-distance screen)", "T−10.6 h: inside commitment window"],
        "citations": ["brief"]}


def facts(action="PREPARE_MANOEUVRE", verdict="ESCALATE", fc=None):
    f = build_facts(event_key="train:2563", row=ROW, cdm_index=2, n_known=3, pc=PC, tier=TIER, forecast=fc,
                    history=[{"ttca": 2.43, "miss_m": 8050.0, "pc": 4.8e-7, "pc_status": "OK"},
                             {"ttca": 0.44, "miss_m": 136.0, "pc": 6.5e-5, "pc_status": "OK"}])
    f.update(verdict=verdict, verdict_action=action, verdict_reasons=["inside commitment window"])
    return f


def out(body, cited=None, headline="CRITICAL — {{event_key}}"):
    return {"headline": headline, "body": body, "uncertainty": "Pc is an estimate.", "cited": cited or [], "refusal": None}


def test_template_passes():
    f = facts()
    t = explain_template(f)
    ok, why = verify(t["text"], f, verdict_action="PREPARE_MANOEUVRE")
    assert ok, why


def test_placeholders_render_and_pass():
    f = facts()
    r = check_llm_output(out("Miss is {{miss_m}} and Pc is {{pc}}; TCA in {{ttca}}.", ["miss_m", "pc", "ttca"]), f)
    assert r["ok"], r["reason"]
    assert "136 m" in r["text"] and "6.5×10⁻⁵" in r["text"]


def test_unit_conversion_within_1pct_passes():
    ok, why = verify("The miss is 0.136 km.", facts(), verdict_action="PREPARE_MANOEUVRE")
    assert ok, why


def test_invented_number_rejected():
    r = check_llm_output(out("Miss is {{miss_m}}, about 40 m smaller than yesterday."), facts())
    assert not r["ok"] and "number not in facts" in r["reason"]


def test_invented_pc_rejected():
    ok, why = verify("Pc is 3.2e-4 now.", facts(), verdict_action="PREPARE_MANOEUVRE")
    assert not ok


def test_pc_as_percent_rejected():
    ok, why = verify("The Pc is 0.0065% which is high.", facts(), verdict_action="PREPARE_MANOEUVRE")
    assert not ok and "percent" in why.lower()
    ok, why = verify("There is a 0.0065% chance of collision.", facts(), verdict_action="PREPARE_MANOEUVRE")
    assert not ok


def test_forecast_in_scientific_notation_rejected():
    f = facts(fc={"valid": True, "p_critical": 0.27, "baseline": {"persistence_tier": "SAFE", "p_critical_empirical": 0.002}})
    ok, why = verify("The forecast is 2.7e-1.", f, verdict_action="PREPARE_MANOEUVRE")
    assert not ok and "scientific" in why


def test_unknown_key_rejected():
    r = check_llm_output(out("Pc is {{pc_after_burn}}."), facts())
    assert not r["ok"] and "unknown fact key" in r["reason"]
    r = check_llm_output(out("Pc is {{pc}}.", cited=["made_up_key"]), facts())
    assert not r["ok"] and "cited unknown" in r["reason"]


def test_manoeuvre_now_with_monitor_verdict_rejected():
    f = facts(action="MONITOR", verdict="AUTO")
    r = check_llm_output(out("You should manoeuvre now."), f)
    assert not r["ok"]
    r = check_llm_output(out("We recommend you perform a manoeuvre."), f)
    assert not r["ok"] and "contradicts" in r["reason"]


def test_dismiss_rejected_and_monitor_phrase_vs_prepare():
    assert not verify("Safe to ignore this.", facts(), verdict_action="PREPARE_MANOEUVRE")[0]
    assert not verify("No action needed here.", facts(), verdict_action="PREPARE_MANOEUVRE")[0]


def test_refusal_is_not_shown():
    r = check_llm_output({"headline": "", "body": "", "uncertainty": "", "cited": [], "refusal": {"reason": "no data"}}, facts())
    assert not r["ok"]
