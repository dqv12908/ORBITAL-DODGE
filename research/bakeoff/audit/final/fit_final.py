"""AUDIT: build a final lgbm_all model (no such artifact was delivered by the bake-off) + independent test features.
Recipe = refs.py lgbm_all; n_iter chosen by early stopping on a 20% mission-grouped inner split of all 8,293 events,
then refit on all events with that iteration count. seed 0."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
BK = "/root/orbital-dodge/research/bakeoff"; sys.path.insert(0, BK)
import numpy as np, pandas as pd, lightgbm as lgb
import harness as H
from sklearn.model_selection import GroupShuffleSplit
OUT = os.path.dirname(os.path.abspath(__file__))
CATS = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "TBA", "UNKNOWN"]
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values
F = H.load_features("ALL").loc[keys]; cols = F.columns.tolist(); X = F.values
# check train category order is the alphabetical CATS order used for test
tr = pd.read_csv(H.RAW + "train_data.csv", usecols=["c_object_type"])
assert sorted(tr.c_object_type.astype(str).unique()) == CATS, sorted(tr.c_object_type.unique())
P = dict(objective="multiclass", learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=16, verbose=-1)
itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, groups=ev.mission_id.values))
m0 = lgb.LGBMClassifier(n_estimators=3000, **P).fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])],
                                                     callbacks=[lgb.early_stopping(100, verbose=False)])
nit = int(m0.best_iteration_); print("best iter", nit)
m = lgb.LGBMClassifier(n_estimators=nit, **P).fit(X, y)
m.booster_.save_model(os.path.join(OUT, "lgbm_all.txt"))
json.dump({"feature_columns": cols, "n_iter": nit, "params": P, "classes": [int(c) for c in m.classes_]},
          open(os.path.join(OUT, "config.json"), "w"), indent=1)
# independent test features (ttca>=2 d only)
te = pd.read_csv(H.RAW + "test_data.csv")
te["c_object_type"] = pd.Categorical(te.c_object_type.astype(str), categories=CATS)
assert te.c_object_type.notna().all()
te = H.clean(te)
print("test CDMs", len(te), "ttca<2:", int((te.time_to_tca < 2).sum()), "events", te.event_id.nunique())
inp = te[te.time_to_tca >= 2].copy(); eids = np.sort(inp.event_id.unique())
numcols = [c for c in te.columns if c not in ("event_id", "mission_id") and te[c].dtype.kind in "if"]
T = H.event_features(inp, numcols, H.TREND_COLS_ALL).loc[eids]
T.index = [f"test:{e}" for e in eids]; T.index.name = "event_key"; T.columns = [str(c) for c in T.columns]
assert not set(cols) - set(T.columns), set(cols) - set(T.columns)
print("extra test cols ignored:", sorted(set(T.columns) - set(cols)))
T[cols].astype(np.float64).to_parquet(os.path.join(OUT, "test_features_ALL.parquet"))
print("test features", T[cols].shape)
