"""Blind test deliverable -> predictions/test_events.csv and predictions/test_rows.csv."""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import time

import numpy as np
import pandas as pd

from riskengine import paths
from riskengine.cdm import TIER_NAMES, brief_tier_codes
from riskengine.features import event_features, forecast_inputs
from riskengine.forecast import Forecaster

RANK = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}


def main():
    t0 = time.time()
    paths.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(paths.CDM_CLEAN)
    te = df[df.split == "test"].reset_index(drop=True)
    pcr = pd.read_parquet(paths.PC_ROWS)
    pcr = pcr[df.split.values == "test"].reset_index(drop=True)
    assert (pcr.event_key.values == te.event_key.values).all()

    rows = pd.DataFrame({
        "event_key": te.event_key, "event_id": te.event_id, "mission_id": te.mission_id, "cdm_index": te.cdm_index,
        "time_to_tca_d": te.time_to_tca, "miss_distance_m": te.miss_distance, "relative_speed_ms": te.relative_speed,
        "pc_status": pcr.pc_status, "log10_pc": pcr.log10_pc.round(4), "hbr_m": pcr.hbr_m,
        "reported_log10_risk": te.risk.where(~(te.cov_unknown_t.astype(bool) | te.cov_unknown_c.astype(bool))),
        "tier": pcr.tier, "pc_tier": pcr.pc_tier, "miss_tier": pcr.miss_tier,
        "rule_ids": [";".join(r) for r in pcr.rule_ids],
    })
    rows.to_csv(paths.PREDICTIONS / "test_rows.csv", index=False)

    fc = Forecaster.load()
    inp = forecast_inputs(te)
    F = event_features(inp)
    P = fc.predict_features(F)
    last = rows.groupby("event_key").tail(1).set_index("event_key")
    early = te[te.time_to_tca >= 2].groupby("event_key").tail(1).set_index("event_key")
    g = te.groupby("event_key")
    ev = pd.DataFrame(index=sorted(te.event_key.unique()))
    ev["event_id"] = g.event_id.first()
    ev["mission_id"] = g.mission_id.first()
    ev["n_cdm"] = g.size()
    ev["latest_ttca_d"] = last.time_to_tca_d
    for c in ("p_safe", "p_elevated", "p_critical"):
        ev[c] = P[c].reindex(ev.index).round(5)
    probs = ev[["p_safe", "p_elevated", "p_critical"]].to_numpy()
    ev["predicted_final_tier"] = [TIER_NAMES[i] if np.isfinite(r).all() else None for i, r in
                                  zip(np.nan_to_num(probs, nan=-1).argmax(1), probs)]
    ev["persistence_tier"] = pd.Series([TIER_NAMES[i] for i in brief_tier_codes(early.risk, early.miss_distance)],
                                       index=early.index).reindex(ev.index)
    ev["recommended_tier"] = [max((a, b), key=lambda t: RANK.get(t, -1)) for a, b in
                              zip(ev.predicted_final_tier, ev.persistence_tier)]
    ev["watch"] = (ev.p_critical >= fc.meta.get("tau_watch", 0.10)) & (ev.persistence_tier != "CRITICAL")
    ev["latest_pc_status"] = last.pc_status
    ev["latest_log10_pc_physics"] = last.log10_pc
    ev["latest_rule_tier"] = last.tier
    ev["latest_reported_log10_risk"] = last.reported_log10_risk
    lr = te.groupby("event_key").tail(1).set_index("event_key").risk.reindex(ev.index)
    ev["kelvins_lrp_log10_risk"] = np.where(lr >= -6, lr, -6.001)
    ev["model_version"] = fc.version
    ev.index.name = "event_key"
    ev.reset_index().to_csv(paths.PREDICTIONS / "test_events.csv", index=False)
    print(f"[predict_test] {len(ev)} events, {len(rows)} rows; predicted tiers "
          f"{ev.predicted_final_tier.value_counts().to_dict()}; persistence {ev.persistence_tier.value_counts().to_dict()}; "
          f"watch {int(ev.watch.sum())} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
