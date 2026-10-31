"""Reference rows: persistence, hgb_a13, lgbm_a13, lgbm_all -> oof/<model>_<split>.parquet (split = event, mission)."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import harness as H
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
import lightgbm as lgb

ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values; lt = ev.latest_tier.values
splits = sys.argv[1:] or ["event", "mission"]
FE = {k: H.load_features(k).loc[keys].values for k in ("A13", "ALL")}

for split in splits:
    t0 = time.time()
    # persistence: P(final tier | latest tier), cross-fitted; argmax == latest tier
    P = np.zeros((len(y), 3))
    for _, trn, tst in H.folds(split):
        for k in range(3):
            m = lt[trn] == k
            cnt = np.bincount(y[trn][m], minlength=3).astype(float)
            P[tst[lt[tst] == k]] = cnt / cnt.sum()
    H.write_oof("persistence", split, keys, P, extra={"tier_pred": lt})
    # hgb_a13 (current baseline, exactly as b_baselines.py)
    P = np.zeros((len(y), 3))
    for _, trn, tst in H.folds(split):
        clf = HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)
        clf.fit(FE["A13"][trn], y[trn]); P[tst] = clf.predict_proba(FE["A13"][tst])
    H.write_oof("hgb_a13", split, keys, P)
    # lightgbm a13 / all, early stopping on an inner split of the training folds (grouped by mission on the mission split)
    for fs in ("A13", "ALL"):
        X = FE[fs]; P = np.zeros((len(y), 3)); iters = []
        for _, trn, tst in H.folds(split):
            if split == "event":
                itr, iva = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=0).split(trn, y[trn]))
            else:
                itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(trn, groups=ev.mission_id.values[trn]))
            m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   reg_lambda=1.0, random_state=0, n_jobs=16, verbose=-1)
            m.fit(X[trn[itr]], y[trn[itr]], eval_set=[(X[trn[iva]], y[trn[iva]])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            iters.append(m.best_iteration_)
            P[tst] = m.predict_proba(X[tst], num_iteration=m.best_iteration_)
        H.write_oof(f"lgbm_{fs.lower()}", split, keys, P)
        print(split, fs, "best iters", iters)
    print(split, "done in", round(time.time() - t0, 1), "s")
