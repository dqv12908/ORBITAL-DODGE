"""Served lgbm_all 1.0.0 predictor.
usage: .venv/bin/python predict.py FEATURES_ALL.parquet OUT.parquet
FEATURES_ALL = harness.event_features over CDMs with time_to_tca >= 2 d only (index or column event_key), the 233 columns in config.json.
Output: event_key, p_safe, p_elevated, p_critical, p_critical_calibrated, alert (p_critical >= config alert threshold)."""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))

def predict(F):
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    iso = json.load(open(os.path.join(HERE, "isotonic.json")))
    if "event_key" in F.columns: F = F.set_index("event_key")
    miss = [c for c in cfg["feature_columns"] if c not in F.columns]
    if miss: raise ValueError(f"missing {len(miss)} feature columns, e.g. {miss[:5]}")
    b = lgb.Booster(model_file=os.path.join(HERE, "lgbm_all.txt"))
    P = b.predict(F[cfg["feature_columns"]].values.astype(np.float64))
    out = pd.DataFrame({"event_key": F.index.values, "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2]})
    out["p_critical_calibrated"] = np.interp(out.p_critical.values, iso["x"], iso["y"])
    out["alert"] = out.p_critical.values >= cfg["alert_threshold_p_critical"]
    return out

if __name__ == "__main__":
    o = predict(pd.read_parquet(sys.argv[1])); o.to_parquet(sys.argv[2])
    print(o.shape, int(o.alert.sum()), "alerts")
