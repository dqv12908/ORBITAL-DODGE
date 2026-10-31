"""Audit 1: provenance, labels, folds, features recomputed independently from train_data.csv (no harness code paths)."""
import sys, os, hashlib, json
import numpy as np, pandas as pd
BK = "/root/orbital-dodge/research/bakeoff"
CSV = "/root/orbital-dodge/data/raw/Challenge 1 Airbus /train_data.csv"
out = {}
raw = pd.read_csv(CSV)
cache = pd.read_parquet(f"{BK}/data/raw_train.parquet")
out["raw_rows_csv"] = len(raw); out["raw_rows_cache"] = len(cache)
out["cache_equals_csv"] = bool(raw.shape == cache.shape and list(raw.columns) == list(cache.columns) and
                               np.allclose(raw.select_dtypes("number").values, cache.select_dtypes("number").values, equal_nan=True))
# ---- independent labels
r = raw.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
fin = r.groupby("event_id").tail(1).set_index("event_id")
early_ids = set(r.loc[r.time_to_tca >= 2, "event_id"])
elig = sorted(e for e in fin.index if fin.loc[e, "time_to_tca"] < 1 and e in early_ids)
f = fin.loc[elig]
crit = (f.risk >= -4) | (f.miss_distance < 1000)
elevm = (~crit) & ((f.risk >= -6) | (f.miss_distance <= 5000))
ytier = np.where(crit, 2, np.where(elevm, 1, 0))
ev = pd.read_parquet(f"{BK}/data/events.parquet")
out["n_elig_indep"] = len(elig)
out["event_ids_match"] = bool(np.array_equal(ev.event_id.values, np.array(elig)))
out["labels_mismatch"] = int((ev.y_tier.values != ytier).sum())
out["tier_counts"] = np.bincount(ytier).tolist()
# label must be from a CDM < 1 d: check final ttca of every eligible event
out["max_final_ttca"] = float(f.time_to_tca.max())
# persistence latest_tier from latest >=2d CDM
lat = r[r.time_to_tca >= 2].groupby("event_id").tail(1).set_index("event_id").loc[elig]
lt = np.where((lat.risk >= -4) | (lat.miss_distance < 1000), 2, np.where((lat.risk >= -6) | (lat.miss_distance <= 5000), 1, 0))
out["latest_tier_mismatch"] = int((ev.latest_tier.values != lt).sum())
out["latest_ttca_min"] = float(ev.latest_ttca.min())
# ---- folds
fo = pd.read_parquet(f"{BK}/data/folds.parquet"); fo2 = pd.read_parquet(f"{BK}/folds.parquet")
out["folds_two_copies_equal"] = bool(fo.equals(fo2))
out["folds_key_order_ok"] = bool((fo.event_key.values == ev.event_key.values).all())
mm = fo.groupby("mission_id").fold_mission.nunique()
out["missions_in_more_than_one_mission_fold"] = int((mm > 1).sum())
true_mission = f.mission_id.values  # mission of final CDM
allm = r[r.event_id.isin(elig)].groupby("event_id").mission_id.nunique()
out["events_with_multiple_mission_ids"] = int((allm > 1).sum())
out["fold_mission_ids_match_raw"] = bool((fo.mission_id.values == true_mission).all())
# ---- feature file columns: no label-ish names
FA = pd.read_parquet(f"{BK}/data/features_ALL.parquet"); F13 = pd.read_parquet(f"{BK}/data/features_A13.parquet")
sus = [c for c in FA.columns if any(s in c.lower() for s in ["y_", "final", "label", "tier", "mission", "event_id"])]
out["suspicious_feature_names"] = sus
out["feature_index_matches_events"] = bool((FA.index.values == ev.event_key.values).all() and (F13.index.values == ev.event_key.values).all())
# ---- independent per-event recompute for 40 random events (ALL set), only CDMs with ttca >= 2
rng = np.random.default_rng(7); pick = rng.choice(elig, 40, replace=False)
SIG = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]
cats = raw.c_object_type.astype("category").cat.categories
numcols = [c for c in raw.columns if c not in ("event_id", "mission_id") and raw[c].dtype.kind in "if"]
TA = ["miss_distance", "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n", "c_sigma_r", "t_sigma_r",
      "max_risk_estimate", "risk", "mahalanobis_distance", "c_sigma_t", "c_sigma_n", "t_sigma_t", "t_sigma_n",
      "c_position_covariance_det", "t_position_covariance_det", "c_obs_used", "c_weighted_rms", "log_miss", "sig_r_comb", "r_over_sig"]
maxdiff = {}; worst = []
for e in pick:
    d = raw[(raw.event_id == e) & (raw.time_to_tca >= 2)].sort_values("time_to_tca", ascending=False).copy()
    flag = np.zeros(len(d), bool)
    for c in SIG:
        b = (d[c] >= 1e7).values; flag |= b; d.loc[b, c] = np.nan
    d["cov_unknown"] = flag.astype(int)
    d["c_object_type"] = pd.Categorical(d.c_object_type, categories=cats).codes
    d["log_miss"] = np.log10(d.miss_distance.clip(1)); d["sig_r_comb"] = np.sqrt(d.c_sigma_r**2 + d.t_sigma_r**2)
    d["r_over_sig"] = d.relative_position_r.abs() / d.sig_r_comb
    feat = {}
    for c in list(dict.fromkeys(numcols + ["log_miss", "sig_r_comb", "r_over_sig", "cov_unknown"])):
        feat["last_" + c] = d[c].iloc[-1]
    feat["n_cdm"] = len(d); feat["first_ttca"] = d.time_to_tca.iloc[0]
    for c in TA:
        v = d[c].values; t = d.time_to_tca.values
        feat[f"d1_{c}"] = v[-1] - v[-2] if len(v) > 1 else np.nan
        feat[f"dall_{c}"] = v[-1] - v[0]
        feat[f"min_{c}"] = np.nanmin(v) if np.isfinite(v).any() else np.nan
        feat[f"max_{c}"] = np.nanmax(v) if np.isfinite(v).any() else np.nan
        feat[f"std_{c}"] = pd.Series(v).std()
        ok = ~np.isnan(v)
        feat[f"slope_{c}"] = np.polyfit(t[ok], v[ok], 1)[0] if ok.sum() >= 2 and np.ptp(t[ok]) > 0 else np.nan
    row = FA.loc[f"train:{e}"]
    missing = set(FA.columns) - set(feat); extra = set(feat) - set(FA.columns)
    for c in FA.columns:
        if c not in feat: continue
        a, b = row[c], feat[c]
        if np.isnan(a) and np.isnan(b): continue
        rel = abs(a - b) / max(1e-9, abs(b))
        if not (np.isclose(a, b, rtol=1e-6, atol=1e-9)):
            worst.append((int(e), c, float(a), float(b)))
out["recompute_missing_cols"] = sorted(missing); out["recompute_extra_cols"] = sorted(extra)
out["recompute_mismatches"] = len(worst); out["recompute_mismatch_examples"] = worst[:15]
# ---- would features change if <2 d CDMs changed? any feature equal to final-CDM value suspiciously often
fv = f.loc[elig]
for col in ["risk", "miss_distance", "max_risk_estimate"]:
    out[f"frac_last_{col}_equals_final"] = float(np.mean(np.isclose(FA[f"last_{col}"].values, fv[col].values)))
    out[f"frac_last_{col}_equals_latest_ge2d"] = float(np.mean(np.isclose(FA[f"last_{col}"].values, lat[col].values, equal_nan=True)))
out["max_last_time_to_tca_lt2"] = int((FA["last_time_to_tca"] < 2).sum())
# test-split overlap: are any train event_ids also present as features keyed test?
out["feature_keys_all_train"] = bool(pd.Index(FA.index).str.startswith("train:").all())
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(f"{BK}/audit/a1_out.json", "w"), indent=1, default=str)
