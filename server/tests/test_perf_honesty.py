"""Regression tests: forecast honesty (never in-sample on train events), pluggable forecast sources, tier-mode
re-run, and scenario-load responsiveness (pre-empting an in-flight AI call)."""
from __future__ import annotations

import asyncio
import json
import time

import pandas as pd
import pytest

from backend_fixtures import make_sim, run, run_scenario, shared_engine


def _expected_t2(engine, key):
    """The value the honest chain must serve at the T-2 d CDM of a training event."""
    for name in engine.PRED_SOURCES:
        rec = (engine.pred_sources.get(name) or {}).get("preds", {}).get(key)
        if rec is not None:
            return name, engine._probs_from(rec)["p_critical"]
    meta = engine.events_meta.get(key) or {}
    if meta.get("p_source") == "oof_by_event":
        return "oof", float(meta["p_critical"])
    return "oof", engine.oof[key]["p_critical"]


def test_train_2563_forecast_is_out_of_fold_never_in_sample(tmp_path):
    sim, envs, _ = run_scenario(tmp_path, "late-cdm")
    eng = sim.engine
    fcs = [e for e in envs if e["type"] == "forecast.updated" and e["event_key"] == "train:2563"]
    assert fcs, "late-cdm must emit a forecast at T-2.43 d"
    d = fcs[-1]["data"]
    src, want = _expected_t2(eng, "train:2563")
    assert "out-of-sample" not in (d.get("p_source") or "")
    assert d["p_critical"] == pytest.approx(want, rel=1e-9), (d["p_source"], d["model"])
    assert d["p_source"] and d["model"]
    # the in-sample model (0.88 for this event) must never be what is shown
    if eng.forecaster is not None:
        rows = eng.event_rows("train:2563")
        early = pd.DataFrame([r for r in rows if r["time_to_tca"] >= 2.0])
        insample = eng.forecaster.predict(early, ttca_now_d=float(early["time_to_tca"].iloc[-1]))["p_critical"]
        if abs(insample - want) > 1e-6:
            assert abs(d["p_critical"] - insample) > 1e-6
        # and the outcome scoring uses the same honest value
        assert sim.events["train:2563"].outcome["forecast_was"]["p_critical"] == pytest.approx(want, rel=1e-9)


def test_no_train_event_ever_gets_the_live_model(tmp_path):
    sim, envs, _ = run_scenario(tmp_path, "ops-day")
    fcs = [e for e in envs if e["type"] == "forecast.updated" and e["event_key"].startswith("train:")]
    assert len(fcs) > 20
    for e in fcs:
        assert "out-of-sample" not in (e["data"].get("p_source") or ""), e["event_key"]
        assert e["data"]["p_source"] in ("persistence-empirical", "oof_by_event (out-of-fold)") or \
            e["data"]["p_source"].startswith(("forecaster_cost", "forecaster_bakeoff", "out-of-fold")), e["data"]["p_source"]


def test_forecast_source_order_and_cost_fields():
    eng = shared_engine()
    saved = dict(eng.pred_sources)
    key = "train:2563"
    rows = [dict(r, _tier="SAFE") for r in eng.event_rows(key)[:1]]
    try:
        eng.pred_sources.pop("forecaster_organizer", None)  # this test exercises the cost/bakeoff fallbacks
        eng.pred_sources["forecaster_bakeoff"] = {"preds": {key: {"event_key": key, "p_critical": 0.21, "p_safe": 0.7,
                                                                  "p_elevated": 0.09, "split": "oof"}},
                                                  "model": "bk", "alert_threshold": None, "cost_ratio_R": None}
        fc = eng.forecast(key, rows, "SAFE")
        assert fc["p_critical"] == 0.21 and fc["model"].startswith("forecaster_bakeoff") and "alert_threshold" not in fc
        eng.pred_sources["forecaster_cost"] = {"preds": {key: {"event_key": key, "p_critical": 0.33}},
                                               "model": "cost-lgbm", "alert_threshold": 0.1, "cost_ratio_R": 20.0}
        fc = eng.forecast(key, rows, "SAFE")
        assert fc["p_critical"] == 0.33 and fc["model"] == "forecaster_cost:cost-lgbm"
        assert fc["p_source"].startswith("forecaster_cost") and "nested" in fc["p_source"]
        assert fc["alert_threshold"] == 0.1 and fc["cost_ratio_R"] == 20.0 and fc["alert"] is True
        assert fc["calib_bin"] is None  # tier_forecaster's reliability bins do not describe another model
        # not the T-2 d CDM of a training event -> persistence, never the model
        later = [dict(r, _tier="SAFE") for r in eng.event_rows("train:8767")]
        early = [r for r in later if r["time_to_tca"] >= 2.0]
        if len(early) >= 2:
            fc = eng.forecast("train:8767", early[:1], "SAFE")
            assert fc["p_source"] == "persistence-empirical"
    finally:
        eng.pred_sources.clear()
        eng.pred_sources.update(saved)


def test_forecaster_cost_loader(tmp_path):
    from app.engine_adapter import Engine
    from app.settings import Settings
    d = tmp_path / "registry" / "forecaster_cost" / "1.0.0"
    d.mkdir(parents=True)
    pd.DataFrame({"event_key": ["train:1", "test:2"], "p_critical": [0.4, 0.01]}).to_parquet(d / "predictions.parquet")
    (d / "model_card.json").write_text(json.dumps({"name": "forecaster_cost", "served_model": "lr-cost",
                                                   "alert_threshold": 0.07, "cost_ratio_R": 25}))
    e = Engine(Settings(repo_dir=tmp_path))
    e._load_bakeoff()
    src = e.pred_sources["forecaster_cost"]
    assert src["model"] == "lr-cost" and src["alert_threshold"] == 0.07 and src["cost_ratio_R"] == 25.0
    assert set(src["preds"]) == {"train:1", "test:2"}
    # absent registry -> nothing loaded, nothing raised
    e2 = Engine(Settings(repo_dir=tmp_path / "nowhere"))
    e2._load_bakeoff()
    assert e2.pred_sources == {} and e2.degraded == []


def test_tier_mode_flip_reassesses_and_recolours(tmp_path):
    sim, envs, _ = run_scenario(tmp_path, "rules-disagree")
    assert sim.policy["tier_mode"] == "BRIEF"
    before = {k: ev.released[-1]["tier"]["tier"] for k, ev in sim.events.items()}
    n0 = len(envs)
    out = sim.update_policy({"tier_mode": "CARA_PC"})
    assert out["policy"]["tier_mode"] == "CARA_PC"
    reruns = [e for e in envs[n0:] if e["type"] == "risk.assessed"]
    assert {e["event_key"] for e in reruns} == set(sim.events)
    for e in reruns:
        assert e["data"]["rerun"] is True and e["data"]["tier"]["mode"] == "CARA_PC"
        ev = sim.events[e["event_key"]]
        assert e["data"]["cdm_index"] == ev.released[-1]["cdm_index"]
    # every released CDM was re-assessed, and the CARA_PC tier is the Pc sub-tier (UNKNOWN -> ELEVATED)
    for ev in sim.events.values():
        for rec in ev.released:
            t = rec["tier"]
            assert t["mode"] == "CARA_PC"
            assert t["tier"] == ("ELEVATED" if t["pc_tier"] == "UNKNOWN" else t["pc_tier"])
    after = {k: ev.released[-1]["tier"]["tier"] for k, ev in sim.events.items()}
    # the miss-only critical (train:8756) must re-colour
    if before.get("train:8756") == "CRITICAL" and sim.events["train:8756"].released[-1]["tier"]["pc_tier"] != "CRITICAL":
        assert after["train:8756"] != "CRITICAL"
    # flipping back restores BRIEF
    sim.update_policy({"tier_mode": "BRIEF"})
    assert {k: ev.released[-1]["tier"]["tier"] for k, ev in sim.events.items()} == before


class SlowProvider:
    name = "slow"
    model = "slow"

    async def decide(self, request, timeout):
        await asyncio.sleep(60)
        raise AssertionError("should have been cancelled")


def test_scenario_load_preempts_inflight_ai(tmp_path):
    from app.simloop import Sim
    from app.sse import Bus
    from decisions.provider import DecisionService
    from ledger.ledger import Ledger
    from llm.explain import Explainer
    from backend_fixtures import mock_settings
    s = mock_settings(tmp_path).model_copy(update={"mock_ai": False, "replay": False, "decisions_timeout_s": 60.0,
                                                   "decisions_budget_s": 120.0})
    led = Ledger(tmp_path / "p.sqlite", config_hash="t", model_sha="t")
    dec = DecisionService(s, providers=[SlowProvider()])
    dec.__class__ = type("LiveDS", (DecisionService,), {"mode": property(lambda self: "live")})
    sim = Sim(s, Bus(led), shared_engine(), dec, Explainer(s), led)
    sim.reload_catalogue()

    async def go():
        await sim.load_scenario("late-cdm")
        stepper = asyncio.create_task(sim.next_cdm())
        for _ in range(400):  # until the pipeline is inside the (slow) AI call
            if sim._ai_task is not None:
                break
            if stepper.done():
                stepper = asyncio.create_task(sim.next_cdm())
            await asyncio.sleep(0.01)
        assert sim._ai_task is not None, "late-cdm never reached a flagged CDM"
        t0 = time.monotonic()
        await sim.load_scenario("late-cdm", wait_s=10)
        dt = time.monotonic() - t0
        await asyncio.wait_for(stepper, 5)
        return dt
    dt = run(go())
    assert dt < 5.0, dt
    assert not sim.held and sim.paused and sim._ai_task is None and sim.counters["ai_calls"] == 0
    assert led.verify()["ok"]


def test_ops_day_load_is_fast(tmp_path):
    sim, bus, _ = make_sim(tmp_path)
    lags = []

    async def probe():
        while True:
            t = time.monotonic()
            await asyncio.sleep(0.02)
            lags.append(time.monotonic() - t - 0.02)

    async def go():
        p = asyncio.create_task(probe())
        t0 = time.monotonic()
        await sim.load_scenario("ops-day")
        dt = time.monotonic() - t0
        p.cancel()
        return dt
    dt = run(go())
    # target < 3 s on an idle M2; generous bound so a busy CI box does not flake
    assert dt < 10.0, dt
    assert len(sim.events) >= 30 and sim.ptr > 100
    assert max(lags or [0]) < 2.0, max(lags)
