"""AUDIT predict for lgbm_all. usage: python predict.py FEATURES_ALL.parquet OUT.parquet"""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
def predict(F):
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    if "event_key" in F.columns: F = F.set_index("event_key")
    miss = [c for c in cfg["feature_columns"] if c not in F.columns]
    if miss: raise ValueError(f"missing {len(miss)} feature columns, e.g. {miss[:5]}")
    b = lgb.Booster(model_file=os.path.join(HERE, "lgbm_all.txt"))
    P = b.predict(F[cfg["feature_columns"]].values.astype(np.float64))
    return pd.DataFrame({"event_key": F.index.values, "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2]})
if __name__ == "__main__":
    out = predict(pd.read_parquet(sys.argv[1])); out.to_parquet(sys.argv[2])
    print(out.shape, out[["p_safe", "p_elevated", "p_critical"]].describe().round(4).to_string())
