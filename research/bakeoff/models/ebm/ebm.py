"""ORBITAL DODGE bake-off contender: Explainable Boosting Machine (interpret 0.7.8), A13 features.

Formulation (fixed before any scoring): ordinal Frank-Hall decomposition into two binary EBMs
  m_crit : P(tier == critical)          (the PRIMARY metric only uses this one -> clean log-odds shapes for the UI)
  m_ge   : P(tier >= elevated)
  p_critical = pc ; p_elevated = max(pge, pc) - pc ; p_safe = 1 - max(pge, pc)
Hyperparameters fixed a priori (no tuning): interactions=10, outer_bags=14, lr=0.015, max_bins=1024,
max_interaction_bins=64, early_stopping_rounds=100, min_samples_leaf=4, smoothing_rounds=75, random_state=0.
Early stopping uses explicit outer bags built from the TRAINING fold only: on the by-mission split each bag holds out
~15% of the training missions (grouped), on the by-event split a stratified 15% of training events.

usage:
  python models/ebm/ebm.py time                       # one-fit timing
  python models/ebm/ebm.py oof <event|mission> [n_parallel]
  python models/ebm/ebm.py final
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BAKE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, BAKE)
import numpy as np, pandas as pd
import harness as H
from interpret.glassbox import ExplainableBoostingClassifier as EBC

SEED = 0
N_BAGS = 14
PARAMS = dict(interactions=10, outer_bags=N_BAGS, inner_bags=0, learning_rate=0.015, max_bins=1024,
              max_interaction_bins=64, early_stopping_rounds=100, min_samples_leaf=4, smoothing_rounds=75,
              interaction_smoothing_rounds=75, max_rounds=50000, random_state=SEED)
TARGETS = {"crit": lambda y: (y == 2).astype(int), "ge": lambda y: (y >= 1).astype(int)}


def feature_types(cols):
    return ["nominal" if c == "last_c_object_type" else "continuous" for c in cols]


def make_bags(y, groups, grouped, n_bags=N_BAGS, val=0.15, seed=SEED):
    """bags (n, n_bags): +1 train, -1 validation. grouped -> hold out whole missions."""
    rng = np.random.default_rng(seed)
    n = len(y); B = np.ones((n, n_bags), np.int8)
    if grouped:
        ug = np.unique(groups); sizes = pd.Series(groups).value_counts()
        for b in range(n_bags):
            for _ in range(200):
                perm = rng.permutation(ug); held = []; cnt = 0
                for g in perm:
                    if cnt >= val * n: break
                    held.append(g); cnt += sizes[g]
                m = np.isin(groups, held)
                if 0.07 * n <= m.sum() <= 0.30 * n and y[m].sum() > 0 and y[~m].sum() > 0: break
            B[m, b] = -1
    else:
        for b in range(n_bags):
            for cls in np.unique(y):
                idx = np.where(y == cls)[0]
                B[rng.choice(idx, int(round(val * len(idx))), replace=False), b] = -1
    return B


def fit_one(X, yb, groups, grouped, n_jobs):
    m = EBC(feature_names=list(X.columns), feature_types=feature_types(X.columns), n_jobs=n_jobs, **PARAMS)
    m.fit(X, yb, bags=make_bags(yb, groups, grouped))
    return m


def combine(pc, pge):
    pge = np.maximum(pge, pc)
    P = np.c_[1 - pge, pge - pc, pc]
    P = np.clip(P, 0, 1); return P / P.sum(1, keepdims=True)


def _task(args):
    split, k, tname, trn, tst, n_jobs = args
    ev = H.load_events(); y = ev.y_tier.values; groups = ev.mission_id.values
    X = H.load_features("A13").loc[ev.event_key.values]
    t0 = time.time()
    m = fit_one(X.iloc[trn], TARGETS[tname](y[trn]), groups[trn], split != "event", n_jobs)
    p = m.predict_proba(X.iloc[tst])[:, 1]
    inter = [m.term_names_[i] for i, f in enumerate(m.term_features_) if len(f) == 2]
    return split, k, tname, tst, p, time.time() - t0, inter


def run_oof(split, n_par):
    from concurrent.futures import ProcessPoolExecutor
    ev = H.load_events(); keys = ev.event_key.values
    n_jobs = max(1, 54 // n_par)
    tasks = [(split, k, t, trn, tst, n_jobs) for k, trn, tst in H.folds(split) for t in TARGETS]
    preds = {t: np.full(len(keys), np.nan) for t in TARGETS}; log = []
    t0 = time.time()
    with ProcessPoolExecutor(n_par) as ex:
        for s, k, t, tst, p, dt, inter in ex.map(_task, tasks):
            preds[t][tst] = p
            log.append(dict(split=s, fold=int(k), target=t, fit_s=round(dt, 1), interactions=inter))
            print(s, k, t, f"{dt:.1f}s", inter, flush=True)
    P = combine(preds["crit"], preds["ge"])
    H.write_oof("ebm", split, keys, P)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump(dict(split=split, wall_s=time.time() - t0, fits=log),
              open(os.path.join(HERE, "logs", f"oof_{split}.json"), "w"), indent=1)
    print(split, "OOF written in", round(time.time() - t0, 1), "s")


def test_features():
    """same recipe as harness.build for A13, applied to test_data.csv (all test CDMs have ttca >= 2 d)."""
    df = H.clean(pd.read_csv(H.RAW + "test_data.csv"))
    inp = df[df.time_to_tca >= 2].copy()
    F = H.event_features(inp, H.ALLOWED[1:], H.TREND_COLS_A13)
    F.index = [f"test:{e}" for e in F.index]; F.index.name = "event_key"
    F.columns = [str(c) for c in F.columns]
    cols = H.load_features("A13").columns
    assert list(F.columns) == list(cols), set(F.columns) ^ set(cols)
    return F.astype(np.float64)


def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o


def export_shapes(models, X, path):
    out = {"formulation": "two binary EBMs (Frank-Hall ordinal); scores are additive log-odds contributions",
           "models": {}}
    for tname, m in models.items():
        g = m.explain_global(); terms = []
        imp = m.term_importances()
        for i, name in enumerate(m.term_names_):
            d = g.data(i); feats = [m.feature_names_in_[j] for j in m.term_features_[i]]
            if len(feats) == 1:
                ft = m.feature_types_in_[m.term_features_[i][0]]
                t = dict(term=name, type="main", feature=feats[0], feature_type=ft, importance=float(imp[i]),
                         bin_edges=d["names"], scores=d["scores"], lower=d.get("lower_bounds"),
                         upper=d.get("upper_bounds"),
                         density=d.get("density"))
                if ft != "nominal":
                    t["note"] = "bin_edges has len(scores)+1 entries; score[i] applies on [edge i, edge i+1)"
                else:
                    t["note"] = "bin_edges are category codes (harness cat codes: 0 DEBRIS, 1 PAYLOAD, 2 ROCKET BODY, 3 TBA, 4 UNKNOWN)"
                t["missing_score"] = float(m.term_scores_[i][0])
            else:
                t = dict(term=name, type="interaction", features=feats, importance=float(imp[i]),
                         left_edges=d.get("left_names"), right_edges=d.get("right_names"), scores=d["scores"])
            terms.append(t)
        terms.sort(key=lambda t: -t["importance"])
        out["models"][tname] = dict(target={"crit": "tier == critical", "ge": "tier >= elevated"}[tname],
                                    intercept=float(np.ravel(m.intercept_)[0]), terms=terms)
    json.dump(_jsonable(out), open(path, "w"))


def export_local(models, X, keys, path):
    out = {"formulation": "per-event additive log-odds: logit(p) = intercept + sum(contributions)", "events": {}}
    for k in keys:
        row = X.loc[[k]]; e = {"features": {c: _jsonable(row[c].iloc[0]) for c in X.columns}}
        pr = {t: float(m.predict_proba(row)[0, 1]) for t, m in models.items()}
        P = combine(np.array([pr["crit"]]), np.array([pr["ge"]]))[0]
        e["p_tiers"] = dict(p_safe=float(P[0]), p_elevated=float(P[1]), p_critical=float(P[2]))
        for t, m in models.items():
            d = m.explain_local(row).data(0)
            contrib = sorted([dict(term=n, value=_jsonable(v), score=float(s))
                              for n, v, s in zip(d["names"], d["values"], d["scores"])], key=lambda c: -abs(c["score"]))
            e[t] = dict(intercept=float(np.ravel(m.intercept_)[0]), prob=pr[t], logit=float(np.ravel(m.intercept_)[0] + sum(c["score"] for c in contrib)),
                        contributions=contrib)
        out["events"][k] = e
    json.dump(_jsonable(out), open(path, "w"), indent=1)


def run_final():
    import joblib
    from concurrent.futures import ProcessPoolExecutor
    ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values; groups = ev.mission_id.values
    X = H.load_features("A13").loc[keys]
    fdir = os.path.join(HERE, "final"); os.makedirs(fdir, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        futs = {t: ex.submit(fit_one, X, TARGETS[t](y), groups, True, 27) for t in TARGETS}
        models = {t: f.result() for t, f in futs.items()}
    print("final fits", round(time.time() - t0, 1), "s")
    for t, m in models.items():
        joblib.dump(m, os.path.join(fdir, f"ebm_{t}.joblib"))
    json.dump(dict(features=list(X.columns), params=PARAMS, formulation="frank-hall ordinal, 2 binary EBMs",
                   fit_on=f"all {len(keys)} eligible train events", bags="14 mission-grouped 15% early-stopping bags",
                   interactions={t: [m.term_names_[i] for i, f in enumerate(m.term_features_) if len(f) == 2]
                                 for t, m in models.items()}),
              open(os.path.join(fdir, "model_meta.json"), "w"), indent=1)
    export_shapes(models, X, os.path.join(fdir, "shapes.json"))
    export_local(models, X, ["train:2563", "train:8767", "train:2427", "train:8756"], os.path.join(fdir, "local_examples.json"))
    Ft = test_features()
    Ft.to_parquet(os.path.join(fdir, "test_features_A13.parquet"))
    print("test features", Ft.shape, "final done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "time":
        ev = H.load_events(); y = ev.y_tier.values
        f = list(H.folds("mission"))[0]
        r = _task(("mission", f[0], "crit", f[1], f[2], int(sys.argv[2]) if len(sys.argv) > 2 else 14))
        print("fit", r[5], "s", r[6])
    elif cmd == "oof":
        run_oof(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    elif cmd == "final":
        run_final()
