"""Forecaster tests: valid window, consistency, and the never-suppress invariant (design 3.5, 6.9)."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from riskengine import paths
from riskengine.forecast import Forecaster, attention

RANK = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}
HAVE = (paths.FORECASTER_DIR / "CURRENT").exists() and paths.CDM_CLEAN.exists()


@settings(derandomize=True)
@given(st.sampled_from(list(RANK)), st.booleans(), st.booleans(), st.booleans(), st.floats(0, 1))
def test_attention_never_suppresses(tier, esc, valid, watch, p):
    fc = {"valid": valid, "watch": watch, "p_critical": p}
    a = attention(tier, esc, fc)
    assert RANK[a["tier"]] >= RANK[tier]
    assert a["escalate"] or not esc
    assert a["flagged"] or (tier == "SAFE" and not esc)
    b = attention(tier, esc, None)
    assert b["tier"] == tier and b["escalate"] == esc


@pytest.fixture(scope="module")
def fc():
    if not HAVE:
        pytest.skip("registry / data not built")
    return Forecaster.load()


@pytest.fixture(scope="module")
def train():
    df = pd.read_parquet(paths.CDM_CLEAN)
    return df[df.split == "train"]


def test_valid_window(fc, train):
    ev = train[train.event_key == "train:8767"]
    early = ev[ev.time_to_tca >= 2]
    r = fc.predict(early)
    assert r["valid"] and 0 <= r["p_critical"] <= 1
    assert sum(r["p_tiers"].values()) == pytest.approx(1.0, abs=1e-9)
    assert r["calib_bin"] is not None and r["model_version"] == "1.0.0"
    r2 = fc.predict(ev)  # includes CDMs inside 2 d
    assert not r2["valid"] and r2["p_critical"] is None
    assert r2["baseline"]["persistence_tier"] in RANK and 0 <= r2["baseline"]["p_critical_empirical"] <= 1
    far = ev[ev.time_to_tca >= 2].copy()
    assert not fc.predict(far, ttca_now_d=7.5)["valid"]


def test_event_path_matches_batch(fc, train):
    from riskengine.features import event_features, forecast_inputs
    keys = ["train:2563", "train:8767", "train:2427"]
    sub = forecast_inputs(train[train.event_key.isin(keys)])
    P = fc.predict_features(event_features(sub))
    for k in keys:
        r = fc.predict(sub[sub.event_key == k])
        assert r["p_critical"] == pytest.approx(P.loc[k, "p_critical"], abs=1e-12)


def test_oof_never_suppresses_rule_tier(fc):
    oof = pd.read_parquet(paths.current_forecaster_dir() / "oof.parquet")
    for t, p in zip(oof.persistence_tier, oof.p_critical_event):
        fcd = {"valid": True, "watch": bool(p >= 0.10 and t != "CRITICAL"), "p_critical": p}
        a = attention(t, t == "CRITICAL", fcd)
        assert RANK[a["tier"]] >= RANK[t]
    # watch strictly adds alerts on top of persistence
    watch = (oof.p_critical_event >= 0.10) & (oof.persistence_tier != "CRITICAL")
    alerts = (oof.persistence_tier == "CRITICAL") | watch
    assert alerts.sum() >= (oof.persistence_tier == "CRITICAL").sum()


def test_metrics_artifacts(fc):
    import json
    d = paths.current_forecaster_dir()
    m = json.loads((d / "metrics.json").read_text())
    for s in ("by_event", "by_mission"):
        assert m[s]["n_events"] == 8293
        assert m[s]["hgb_pcrit_calibrated"]["auc"] > 0.9
    card = json.loads((d / "model_card.json").read_text())
    assert len(card["features"]) == 84 and "risk" not in card["features"] and "mission_id" not in card["features"]


def test_events_artifact_alignment():
    if not paths.EVENTS.exists():
        pytest.skip("events not built")
    e = pd.read_parquet(paths.EVENTS).set_index("event_key")
    assert e.loc["train:2563", "final_tier"] == "CRITICAL" and e.loc["train:2563", "persistence_tier"] == "SAFE"
    assert e.loc["train:8756", "final_tier"] == "CRITICAL"
    assert e.loc["train:2299", "n_cov_unknown"] == 14
    assert pd.isna(e.loc["test:891", "final_tier"])
    oof = pd.read_parquet(paths.current_forecaster_dir() / "oof.parquet").set_index("event_key")
    j = e.join(oof[["y_tier"]], how="inner")
    assert (j.final_tier == j.y_tier).all()
