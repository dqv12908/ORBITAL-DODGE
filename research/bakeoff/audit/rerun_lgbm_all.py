"""AUDIT: re-run lgbm_all OOF from scratch (same recipe as refs.py) with other seeds. Writes audit/oof/ only.
usage: python audit/rerun_lgbm_all.py SPLIT SEED [SEED ...]   (seed drives LGBM random_state AND the inner early-stop split)"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
BK = "/root/orbital-dodge/research/bakeoff"; sys.path.insert(0, BK)
import numpy as np, pandas as pd
import harness as H
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
import lightgbm as lgb
AOOF = os.path.join(BK, "audit", "oof")
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values
X = H.load_features("ALL").loc[keys].values
split = sys.argv[1]
for seed in map(int, sys.argv[2:]):
    t0 = time.time(); P = np.zeros((len(y), 3)); iters = []
    for _, trn, tst in H.folds(split):
        if split == "event":
            itr, iva = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(trn, y[trn]))
        else:
            itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(trn, groups=ev.mission_id.values[trn]))
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.03, num_leaves=31,
                               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=seed, n_jobs=16, verbose=-1)
        m.fit(X[trn[itr]], y[trn[itr]], eval_set=[(X[trn[iva]], y[trn[iva]])], callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration_); P[tst] = m.predict_proba(X[tst], num_iteration=m.best_iteration_)
    assert np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1)
    pd.DataFrame({"event_key": keys, "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2]}).to_parquet(
        os.path.join(AOOF, f"lgbm_all_s{seed}_{split}.parquet"))
    print(split, "seed", seed, "iters", iters, round(time.time() - t0, 1), "s", flush=True)
