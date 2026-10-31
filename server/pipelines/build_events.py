"""cdm_clean + pc_rows + forecaster OOF -> data/processed/events.parquet and geometry.parquet."""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import time

import numpy as np
import pandas as pd

from riskengine import orbits, paths
from riskengine.cdm import TIER_NAMES, brief_tier_codes
from riskengine.features import event_features, forecast_inputs
from riskengine.forecast import Forecaster

RANKS = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}


def main():
    t0 = time.time()
    df = pd.read_parquet(paths.CDM_CLEAN)
    pcr = pd.read_parquet(paths.PC_ROWS, columns=["event_key", "cdm_index", "pc_status", "log10_pc", "tier", "pc_tier",
                                                    "miss_tier", "escalate"])
    x = pd.concat([df, pcr.drop(columns=["event_key", "cdm_index"])], axis=1)
    g = x.groupby("event_key", sort=True)
    first, last = g.head(1).set_index("event_key"), g.tail(1).set_index("event_key")
    ev = pd.DataFrame({
        "split": first.split, "event_id": first.event_id.astype(int), "mission_id": first.mission_id.astype(int),
        "n_cdm": g.size(), "c_object_type": last.c_object_type.astype(str),
        "first_ttca": first.time_to_tca, "last_ttca": last.time_to_tca,
        "n_cov_unknown": g.apply(lambda d: int((d.cov_unknown_t | d.cov_unknown_c).sum()), include_groups=False),
        "max_rule_tier": g.tier.agg(lambda s: TIER_NAMES[max(RANKS[t] for t in s)]),
        "last_rule_tier": last.tier, "last_pc_status": last.pc_status, "last_log10_pc": last.log10_pc,
        "last_miss_m": last.miss_distance, "last_log10_risk": last.risk,
    })
    is_train = ev.split == "train"
    ft = pd.Series([TIER_NAMES[i] for i in brief_tier_codes(last.risk, last.miss_distance)], index=last.index)
    ev["final_tier"] = ft.reindex(ev.index).where(is_train, None)
    ev["final_log10_risk"] = last.risk.where(is_train)
    ev["final_miss_m"] = last.miss_distance.where(is_train)
    ev["final_ttca"] = last.time_to_tca.where(is_train)
    early = x[x.time_to_tca >= 2].groupby("event_key").tail(1).set_index("event_key")
    ev["latest_ge2_ttca"] = early.time_to_tca.reindex(ev.index)
    ev["latest_ge2_cdm_index"] = early.cdm_index.reindex(ev.index)
    pt = brief_tier_codes(early.risk, early.miss_distance)
    ev["persistence_tier"] = pd.Series([TIER_NAMES[i] for i in pt], index=early.index).reindex(ev.index)
    ev["eligible"] = ev.latest_ge2_ttca.notna() & (~is_train | (ev.last_ttca < 1))

    # forecasts: train eligible -> OOF (by event, calibrated); others with a >=2 d CDM -> full model
    oof = pd.read_parquet(paths.current_forecaster_dir() / "oof.parquet").set_index("event_key")
    for c in ("p_safe", "p_elevated", "p_critical"):
        ev[c] = oof[f"{c}_event"].reindex(ev.index)
    ev["p_critical_by_mission"] = oof["p_critical_mission"].reindex(ev.index)
    ev["p_source"] = np.where(ev.p_critical.notna(), "oof_by_event", None)
    need = ev.index[ev.p_critical.isna() & ev.latest_ge2_ttca.notna() & (ev.latest_ge2_ttca <= 7)]
    fc = Forecaster.load()
    inp = forecast_inputs(df[df.event_key.isin(need)])
    F = event_features(inp)
    P = fc.predict_features(F)
    ev.loc[P.index, "p_safe"] = P.p_safe
    ev.loc[P.index, "p_elevated"] = P.p_elevated
    ev.loc[P.index, "p_critical"] = P.p_critical
    ev.loc[P.index, "p_source"] = "model"
    ev["watch"] = (ev.p_critical >= fc.meta.get("tau_watch", 0.10)) & (ev.persistence_tier != "CRITICAL")
    ev = ev.reset_index().rename(columns={"index": "event_key"})
    ev.to_parquet(paths.EVENTS, index=False)
    print(f"[events] {len(ev)} events ({ev.eligible.sum()} eligible), forecasts {ev.p_critical.notna().sum()} "
          f"-> {paths.EVENTS} ({time.time()-t0:.1f}s)")

    geo = orbits.reconstruct(last.reset_index())
    geo.to_parquet(paths.GEOMETRY, index=False)
    tr = geo.split == "train"
    print(f"[geometry] {len(geo)} events; train chaser inc residual median {geo.inc_res_deg[tr].median():.3f} deg, "
          f"sma residual median {geo.sma_res_km[tr].median():.1f} km -> {paths.GEOMETRY} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
