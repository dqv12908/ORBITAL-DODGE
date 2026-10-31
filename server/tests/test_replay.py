"""Replay determinism: two mock runs of the same scenario + seed give the same ledger content-hash sequence."""
from backend_fixtures import run_scenario


def test_two_mock_runs_same_content_hashes(tmp_path):
    a_sim, _, a_led = run_scenario(tmp_path, "late-cdm", db_name="a.sqlite")
    b_sim, _, b_led = run_scenario(tmp_path, "late-cdm", db_name="b.sqlite")
    ha, hb = a_led.content_hashes(), b_led.content_hashes()
    assert len(ha) > 10
    assert ha == hb
    assert a_led.verify()["ok"] and b_led.verify()["ok"]


def test_ops_day_deterministic_and_complete(tmp_path):
    a_sim, envs, a_led = run_scenario(tmp_path, "ops-day", db_name="a.sqlite")
    b_sim, _, b_led = run_scenario(tmp_path, "ops-day", db_name="b.sqlite")
    assert a_led.content_hashes() == b_led.content_hashes()
    types = {e["type"] for e in envs}
    assert {"cdm.received", "risk.assessed", "policy.gated", "policy.verdict", "outcome.revealed"} <= types
    # every cast event got its outcome at TCA
    assert all(ev.outcome is not None for ev in a_sim.events.values())
