import os, sys, json, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
ORG = "/root/orbital-dodge/research/organizer"
sys.path.insert(0, ORG); sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
import organizer as O
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold
OUTD = "/root/orbital-dodge/research/fa/ens/work"; os.makedirs(OUTD, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
ev, FE = O.load(); n = len(ev); y = ev.y_tier.values; yc = ev.y_crit.values
for k in FE: FE[k] = np.where(np.isfinite(FE[k]), FE[k], np.nan)
outer = []
for s in range(5):
    trn, tst = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(np.arange(n), groups=ev.event_id.values))
    outer.append((f"8020_s{s}", s, "event", np.sort(trn), np.sort(tst)))
for k, (trn, tst) in enumerate(GroupKFold(5).split(np.arange(n), groups=ev.mission_id.values)):
    outer.append(("mission", k, "mission", trn, tst))

def es_split(m, seed):
    r = np.random.default_rng(seed); i = r.permutation(m); h = int(0.2 * m); return i[h:], i[:h]

def fit_predict(model, Xa, ya_t, Xb, seed=0):
    if model == "lgbm_all":
        return O.fit_pred(O.BEST["ALL"], Xa, ya_t, Xb, seed=0)[:, 2]
    if model == "xgb_a13_crit_w30":
        import xgboost as xgb
        p = dict(grow_policy="lossguide", max_depth=0, max_leaves=38, learning_rate=0.0187, min_child_weight=29.06,
                 subsample=0.5068, colsample_bytree=0.3804, reg_lambda=0.001463, reg_alpha=0.0, gamma=2.581, max_bin=256,
                 n_jobs=1, tree_method="hist", random_state=seed)
        yb = (ya_t == 2).astype(int); w = np.where(yb == 1, 30.0, 1.0)
        a, h = es_split(len(yb), seed)
        m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=30, **p)
        m.fit(Xa[a], yb[a], sample_weight=w[a], eval_set=[(Xa[h], yb[h])], sample_weight_eval_set=[w[h]], verbose=False)
        bi = int(m.best_iteration) + 1
        m = xgb.XGBClassifier(n_estimators=max(1, round(bi * 1.15)), **p); m.fit(Xa, yb, sample_weight=w)
        return m.predict_proba(Xb)[:, 1]
    if model == "hgb_a13_tier_w100":
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        p = dict(max_leaf_nodes=9, max_depth=6, learning_rate=0.03825, l2_regularization=0.0, min_samples_leaf=59,
                 max_features=0.7763, max_bins=255, random_state=seed)
        w = np.where(ya_t == 2, 100.0, 1.0)
        m = HGB(max_iter=2000, early_stopping=True, n_iter_no_change=40, scoring="loss", tol=1e-7, validation_fraction=0.2, **p)
        m.fit(Xa, ya_t, sample_weight=w)
        bi = max(1, m.n_iter_ - 40)
        m = HGB(max_iter=max(1, round(bi * 1.15)), early_stopping=False, **p); m.fit(Xa, ya_t, sample_weight=w)
        return m.predict_proba(Xb)[:, 2]
    if model == "rf_all_crit_bsub":
        from sklearn.ensemble import RandomForestClassifier as RF
        m = RF(n_estimators=200, max_features=0.1, min_samples_leaf=10, max_depth=12, criterion="gini", bootstrap=True,
               class_weight="balanced_subsample", n_jobs=1, random_state=seed)
        Xa2 = np.clip(np.nan_to_num(Xa, nan=-1e30), -1e30, 1e30); Xb2 = np.clip(np.nan_to_num(Xb, nan=-1e30), -1e30, 1e30)
        m.fit(Xa2, (ya_t == 2).astype(int)); return m.predict_proba(Xb2)[:, 1]
    raise ValueError(model)

FEAT = {"lgbm_all": "ALL", "xgb_a13_crit_w30": "A13", "hgb_a13_tier_w100": "A13", "rf_all_crit_bsub": "ALL"}
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FEAT)

def task(model, oi, j):
    fn = f"{OUTD}/t_{model}_{oi}_{j}.joblib"
    if os.path.exists(fn): return joblib.load(fn)
    r = task0(model, oi, j); joblib.dump(r, fn); return r
def task0(model, oi, j):
    sp, s, kind, trn, tst = outer[oi]; X = FE[FEAT[model]]
    if j < 0:
        return model, oi, j, tst, fit_predict(model, X[trn], y[trn], X[tst])
    inner = list(GroupKFold(5).split(trn, groups=ev.mission_id.values[trn])) if kind == "mission" else \
        list(KFold(5, shuffle=True, random_state=100 + s).split(trn))
    ia, ib = inner[j]
    return model, oi, j, trn[ib], fit_predict(model, X[trn[ia]], y[trn[ia]], X[trn[ib]])

jobs = [(m, oi, j) for m in MODELS for oi in range(len(outer)) for j in range(-1, 5)]
# slow ones first
jobs.sort(key=lambda t: {"hgb_a13_tier_w100": 0, "xgb_a13_crit_w30": 1, "rf_all_crit_bsub": 2, "lgbm_all": 3}[t[0]])
log("jobs", len(jobs))
NJ = int(os.environ.get("NJ", "24"))
res = Parallel(n_jobs=NJ, verbose=11)(delayed(task)(*t) for t in jobs)
for m in MODELS:
    rows = []
    for (mm, oi, j, ix, p) in res:
        if mm != m: continue
        sp, s, kind, trn, tst = outer[oi]
        rows.append(pd.DataFrame({"idx": ix, "split": sp, "fold": s if sp == "mission" else 0, "inner_fold": j,
                                  "role": "outer_test" if j < 0 else "inner_oof", "p_critical": p}))
    pd.concat(rows).to_parquet(f"{OUTD}/{m}.parquet")
log("done")
