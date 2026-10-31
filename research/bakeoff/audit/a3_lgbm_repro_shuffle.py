"""Audit 3: (i) reproduce lgbm_all OOF exactly with refs.py logic, (ii) shuffle-label sanity (labels permuted in training folds),
(iii) independent rescoring of PRIMARY/AUC for lgbm_all, tabpfn, persistence, (iv) fold-overlap assertions inside the loop."""
import sys, json, time
sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd, lightgbm as lgb
import harness as H
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values; yc = (y == 2).astype(int); mis = ev.mission_id.values
X = H.load_features("ALL").loc[keys].values
out = {}

def run(split, ylab, seed=0):
    P = np.zeros((len(y), 3))
    for k, trn, tst in H.folds(split):
        assert len(np.intersect1d(trn, tst)) == 0
        if split == "mission":
            assert len(np.intersect1d(mis[trn], mis[tst])) == 0
            itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(trn, groups=mis[trn]))
        else:
            itr, iva = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=0).split(trn, ylab[trn]))
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=16, verbose=-1)
        m.fit(X[trn[itr]], ylab[trn[itr]], eval_set=[(X[trn[iva]], ylab[trn[iva]])], callbacks=[lgb.early_stopping(100, verbose=False)])
        P[tst] = m.predict_proba(X[tst], num_iteration=m.best_iteration_)
    return P

for split in ["mission", "event"]:
    t = time.time(); P = run(split, y)
    O = pd.read_parquet(f"{H.OOF}/lgbm_all_{split}.parquet").set_index("event_key").loc[keys][H.PCOLS].values
    out[f"repro_{split}_maxabsdiff"] = float(np.abs(P - O).max()); out[f"repro_{split}_secs"] = round(time.time() - t, 1)
# shuffle-label: permute labels globally (training folds see permuted labels; evaluate vs TRUE labels)
for split in ["mission", "event"]:
    aucs = []
    for s in range(3):
        yp = np.random.default_rng(100 + s).permutation(y)
        P = run(split, yp, seed=s)
        aucs.append(float(roc_auc_score(yc, P[:, 2])))
    out[f"shuffle_{split}_auc_true_labels"] = aucs
# within-mission permutation (keeps mission base rates; detects nothing but mission prior)
yp = y.copy()
for m_ in np.unique(mis):
    i = np.where(mis == m_)[0]; yp[i] = np.random.default_rng(int(m_)).permutation(y[i])
P = run("mission", yp); out["shuffle_within_mission_mission_split_auc"] = float(roc_auc_score(yc, P[:, 2]))
# independent rescoring
def prim(p, budget):
    o = np.argsort(-p, kind="mergesort"); ps = p[o]; tp = np.cumsum(yc[o]); fp = np.cumsum(1 - yc[o])
    ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]; ok = ends[fp[ends] <= budget]
    return tp[ok[-1]] / yc.sum(), int(tp[ok[-1]]), int(fp[ok[-1]])
budget = int(((ev.latest_tier.values == 2) & (yc == 0)).sum()); out["budget"] = budget
for m_ in ["persistence", "lgbm_all", "tabpfn", "lgbm_a13", "futures", "gru", "ebm", "hgb_a13"]:
    for split in ["mission", "event"]:
        O = pd.read_parquet(f"{H.OOF}/{m_}_{split}.parquet").set_index("event_key").loc[keys]
        r, tp, fp = prim(O.p_critical.values, budget)
        out[f"rescore_{m_}_{split}"] = dict(primary=round(float(r), 4), tp=tp, fa=fp, auc=round(float(roc_auc_score(yc, O.p_critical.values)), 4))
print(json.dumps(out, indent=1)); json.dump(out, open("/root/orbital-dodge/research/bakeoff/audit/a3_out.json", "w"), indent=1)
