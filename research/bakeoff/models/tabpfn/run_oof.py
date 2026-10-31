"""TabPFN v2 contender: out-of-fold predictions on the bake-off splits.

usage: HF_HUB_OFFLINE=1 .venv/bin/python models/tabpfn/run_oof.py --fs A13 --split mission --gpu 3 [--diag]
  * fs A13 (pre-declared entry `tabpfn`) or ALL (diagnostic only; written to models/tabpfn/diag/, NOT to oof/)
  * TabPFN's own preprocessing (quantile/power transforms, feature shuffling, NaN handling) is fitted inside
    .fit() on the training folds only. No other preprocessing, no tuning on outer folds.
"""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
BK = "/root/orbital-dodge/research/bakeoff"
sys.path.insert(0, BK)
import numpy as np, pandas as pd
import harness as H
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

N_EST = 8  # fixed a priori (keeps a fold well under 3 min on a V100)

ap = argparse.ArgumentParser()
ap.add_argument("--fs", default="A13"); ap.add_argument("--split", default="mission")
ap.add_argument("--gpu", default="1"); ap.add_argument("--diag", action="store_true")
a = ap.parse_args()
dev = f"cuda:{a.gpu}"
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values
X = H.load_features(a.fs).loc[keys]
assert X.shape[1] <= 500
X = X.values.astype(np.float32)
X[~np.isfinite(X)] = np.nan
P = np.zeros((len(y), 3)); log = []
t0 = time.time()
for k, trn, tst in H.folds(a.split):
    t = time.time()
    m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev, n_estimators=N_EST, random_state=0)
    m.fit(X[trn], y[trn])
    pr = m.predict_proba(X[tst])
    assert list(m.classes_) == [0, 1, 2]
    P[tst] = pr
    dt = time.time() - t
    log.append(dict(fold=int(k), n_train=int(len(trn)), n_test=int(len(tst)), secs=round(dt, 1)))
    print(a.fs, a.split, "fold", k, "secs", round(dt, 1), flush=True)
    del m; torch.cuda.empty_cache()
P = P / P.sum(1, keepdims=True)
name = "tabpfn" if (a.fs == "A13" and not a.diag) else f"tabpfn_{a.fs.lower()}"
if a.diag:
    d = os.path.join(BK, "models/tabpfn/diag"); os.makedirs(d, exist_ok=True)
    assert np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1, atol=1e-4)
    pd.DataFrame({"event_key": keys, "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2]}).to_parquet(
        os.path.join(d, f"{name}_{a.split}.parquet"))
else:
    H.write_oof(name, a.split, keys, P)
tot = time.time() - t0
json.dump(dict(name=name, fs=a.fs, split=a.split, device=dev, gpu_name=torch.cuda.get_device_name(int(a.gpu)),
               n_estimators=N_EST, total_secs=round(tot, 1), folds=log),
          open(os.path.join(BK, f"models/tabpfn/runlog_{name}_{a.split}.json"), "w"), indent=1)
print("done", name, a.split, round(tot, 1), "s")
