"""AUDIT: recompute one test event by hand from the raw CSV (no harness feature code) and compare + predict."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
BK = "/root/orbital-dodge/research/bakeoff"
raw = pd.read_csv("/root/orbital-dodge/data/raw/Challenge 1 Airbus /test_data.csv")
pred = pd.read_parquet(f"{BK}/audit/final/test_predictions.parquet").set_index("event_key")
F = pd.read_parquet(f"{BK}/audit/final/test_features_ALL.parquet")
# pick the event with median-ish high p_critical among those with >=5 CDMs
cand = pred[pred.p_critical.between(0.4, 0.9)].index
eid = int(sys.argv[1]) if len(sys.argv) > 1 else int(sorted(cand)[0].split(":")[1])
e = raw[raw.event_id == eid].sort_values("time_to_tca", ascending=False)
print("event", eid, "CDMs", len(e), "ttca range", e.time_to_tca.max().round(3), e.time_to_tca.min().round(3))
print(e[["time_to_tca", "risk", "max_risk_estimate", "miss_distance", "c_object_type", "t_sigma_r", "c_sigma_r"]].to_string(index=False))
L = e.iloc[-1]; P2 = e.iloc[-2]; F0 = e.iloc[0]
miss = e.miss_distance.values; t = e.time_to_tca.values
hand = {"last_time_to_tca": L.time_to_tca, "last_miss_distance": L.miss_distance, "last_risk": L.risk, "n_cdm": len(e),
        "first_ttca": F0.time_to_tca, "d1_miss_distance": L.miss_distance - P2.miss_distance,
        "dall_miss_distance": L.miss_distance - F0.miss_distance, "min_miss_distance": miss.min(), "max_miss_distance": miss.max(),
        "std_miss_distance": np.std(miss, ddof=1), "slope_miss_distance": np.polyfit(t, miss, 1)[0],
        "last_log_miss": np.log10(max(L.miss_distance, 1)),
        "last_sig_r_comb": np.hypot(L.c_sigma_r, L.t_sigma_r) if max(L.c_sigma_r, L.t_sigma_r) < 1e7 else np.nan,
        "slope_risk": np.polyfit(t, e.risk.values, 1)[0], "max_risk": e.risk.max()}
row = F.loc[f"test:{eid}"]
for k, v in hand.items():
    print(f"{k:24s} hand={v:.6g}  file={row[k]:.6g}  {"OK" if np.isclose(v, row[k], rtol=1e-9, equal_nan=True) else "MISMATCH"}")
cfg = json.load(open(f"{BK}/audit/final/config.json"))
b = lgb.Booster(model_file=f"{BK}/audit/final/lgbm_all.txt")
p = b.predict(row[cfg["feature_columns"]].values.astype(float)[None, :])[0]
print("booster on file row:", p.round(5), " predict.py:", pred.loc[f"test:{eid}"].values.round(5))
crit = (L.risk >= -4) or (L.miss_distance < 1000); elev = (L.risk >= -6) or (L.miss_distance <= 5000)
print("persistence tier of latest CDM:", "critical" if crit else ("elevated" if elev else "safe"))
