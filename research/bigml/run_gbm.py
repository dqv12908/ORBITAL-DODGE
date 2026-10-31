"""D: baselines (persistence, HGB A13 as in design) + Optuna-tuned LightGBM on A13 and ALL,
OOF over by-event (10 seeds), by-mission GroupKFold, leave-one-mission-out. Saves out/gbm_oof.joblib."""
import os, sys, time, json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from bm import HERE, ROOT, RAW, folds, tier as btier
os.environ["C1_DATA_DIR"] = RAW
import common as C
import lightgbm as lgb, optuna
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
NTR = int(os.environ.get("NTRIALS", 60))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:.0f}s]", *a, flush=True)

tr = C.clean(C.load("train"))
inp, tgt = C.kelvins_frame(tr)
numcols = [c for c in tr.columns if c not in ("event_id", "mission_id") and tr[c].dtype.kind in "if"]
sets = {"A13": (C.ALLOWED[1:], C.TREND_COLS_A13),
        "ALL": (numcols, C.TREND_COLS_A13 + ["risk", "mahalanobis_distance", "c_sigma_t", "c_sigma_n", "t_sigma_t", "t_sigma_n",
                                              "c_position_covariance_det", "t_position_covariance_det", "c_obs_used", "c_weighted_rms"])}
FE = {k: C.event_features(inp, list(v[0]), list(v[1])).loc[tgt.index] for k, v in sets.items()}
ids = tgt.index.values
yt = C.tier(tgt.y_risk, tgt.y_miss)
mis = tgt.mission_id.values
lat = FE["ALL"]; lat_tier = C.tier(lat.last_risk.values, lat.last_miss_distance.values)
el = pd.DataFrame(dict(event_key=["train:%d" % i for i in ids], mission_id=mis))
log("features", {k: v.shape for k, v in FE.items()}, "crit", (yt == 2).sum())

SPLITS = {"mission": folds(el, "mission"), "lomo": folds(el, "lomo")}
for s in range(10):
    SPLITS[f"event_s{s}"] = folds(el, "event", seed=s)


def oof(fit_pred, X, fl, nj=5):
    P = np.zeros((len(X), 3))
    outs = Parallel(n_jobs=nj)(delayed(fit_pred)(X[a], yt[a], X[b]) for a, b in fl)
    for (a, b), p in zip(fl, outs):
        P[b] = p
    return P


def hgb(Xa, ya, Xb):
    return HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0).fit(Xa, ya).predict_proba(Xb)


def lgb_fp(params, seed=0, threads=8):
    def f(Xa, ya, Xb):
        p = dict(objective="multiclass", num_class=3, verbosity=-1, num_threads=threads, seed=seed, **params)
        nr = p.pop("n_estimators", 400)
        m = lgb.train(p, lgb.Dataset(Xa, ya), num_boost_round=nr)
        return m.predict(Xb)
    return f


res = {"ids": ids, "mission": mis, "y_tier": yt, "lat_tier": lat_tier, "oof": {}}
# persistence-empirical P(tier | latest tier), in-fold
for sp, fl in SPLITS.items():
    P = np.zeros((len(yt), 3))
    for a, b in fl:
        for k in range(3):
            m = lat_tier[a] == k
            for j in range(3):
                P[b[lat_tier[b] == k], j] = (yt[a][m] == j).mean() if m.sum() else (yt[a] == j).mean()
    res["oof"][("persistence_emp", sp)] = P
log("persistence done")

# HGB as designed (A13), seed-0 config
for sp in ["mission", "lomo"] + [f"event_s{s}" for s in range(10)]:
    res["oof"][("hgb_A13", sp)] = oof(hgb, FE["A13"].values, SPLITS[sp], nj=10)
log("HGB done")

# Optuna HPO: tuned on a separate by-event split (seed 99, 4 folds) to avoid tuning on evaluation folds
from sklearn.metrics import log_loss
tune_fl = folds(el, "event", seed=99)[:4]
best = {}
for fs in ["A13", "ALL"]:
    X = FE[fs].values
    def obj(trial):
        params = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                      num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                      min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
                      feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
                      bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0), bagging_freq=1,
                      lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
                      n_estimators=trial.suggest_int("n_estimators", 100, 1200, log=True))
        P = oof(lgb_fp(params, threads=2), X, tune_fl, nj=4)
        m = np.zeros(len(X), bool)
        for a, b in tune_fl: m[b] = True
        return log_loss(yt[m], P[m], labels=[0, 1, 2])
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=NTR, n_jobs=6)
    best[fs] = dict(st.best_params, bagging_freq=1)
    log(fs, "best logloss", round(st.best_value, 4), best[fs])
res["best_params"] = best

for fs in ["A13", "ALL"]:
    X = FE[fs].values
    for sp in ["mission", "lomo"]:
        Ps = [oof(lgb_fp(best[fs], seed=s, threads=4), X, SPLITS[sp], nj=12) for s in range(10)]
        res["oof"][(f"lgbm_{fs}", sp)] = np.mean(Ps, 0)
        res["oof"][(f"lgbm_{fs}_seeds", sp)] = np.stack(Ps)
    for s in range(10):
        res["oof"][(f"lgbm_{fs}", f"event_s{s}")] = oof(lgb_fp(best[fs], seed=s, threads=8), X, SPLITS[f"event_s{s}"], nj=5)
    log("lgbm", fs, "done")
    joblib.dump(res, os.path.join(OUT, "gbm_oof.joblib"))

# final models on all train for test-time use (A13)
evte = None
joblib.dump(res, os.path.join(OUT, "gbm_oof.joblib"))
log("saved")
