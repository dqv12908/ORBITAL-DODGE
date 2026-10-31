"""ORBITAL DODGE - evaluation aligned with the organizer's official label file (train_event_labels.csv).

    .venv python organizer.py build    # features for the 11,942 usable train events + 2,167 test events
    .venv python organizer.py eval     # 80/20 by event (5 seeds) + GroupKFold(5) by mission -> out/eval.joblib, results.json
    .venv python organizer.py final    # fit recommended model on all 11,942, predict test -> out/

Population = organizer usable_for_training (event has >= 1 message with ttca >= 2 d). Target = tier of the event's final
message (smallest ttca, any ttca). Inputs = messages with ttca >= 2 d only, harness recipe (research/bakeoff/harness.py).
Sub-populations by final-message ttca: final_lt1d (= Kelvins-eligible 8,293), final_1to2d (599), final_ge2d (3,050; the
final message IS the latest input, so persistence is exact by construction there).
No "is this the final message" feature is added (unknowable at prediction time).
Alert cut-off for CRITICAL: min 10*missed + false alarms subject to recall >= 90 %, picked on inner training folds only.
Resource cap: <= 8 cores (NJ parallel jobs x THR LightGBM threads).
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
import numpy as np, pandas as pd
import harness as H
import lightgbm as lgb
import joblib
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold

DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "out")
ORG = os.path.join(HERE, "organizer")
R = 10; FLOOR = 0.90; NJ = 4; THR = 2; B = 500
TN = {"Safe": 0, "Elevated": 1, "Critical": 2}; TNAME = np.array(["Safe", "Elevated", "Critical"])
SUBPOPS = ["final_lt1d", "final_1to2d", "final_ge2d"]
# Optuna-tuned bake-off params (research/bigml/results.json best_lgbm_params; tuned on Kelvins pop, seed-99 split)
BEST = json.load(open(os.path.join(HERE, "..", "bigml", "results.json")))["best_lgbm_params"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:.0f}s]", *a, flush=True)


def clean(df, cats):
    df = df.copy(); flag = np.zeros(len(df), bool)
    for c in H.SIGMA_COLS:
        bad = (df[c] >= H.SENT_MIN).values; flag |= bad; df.loc[bad, c] = np.nan
    df["cov_unknown"] = flag.astype(int)
    df["c_object_type"] = pd.Categorical(df["c_object_type"], categories=cats).codes.astype(int)
    return df


def build():
    os.makedirs(DATA, exist_ok=True)
    raw = H.load_raw("train")
    cats = sorted(raw.c_object_type.dropna().unique())
    tr = clean(raw, cats).sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
    last = tr.groupby("event_id").tail(1).set_index("event_id")
    usable = np.sort(tr.loc[tr.time_to_tca >= 2, "event_id"].unique())
    inp = tr[tr.event_id.isin(usable) & (tr.time_to_tca >= 2)]
    yt = H.tier(last.loc[usable, "risk"], last.loc[usable, "miss_distance"]).astype(int)
    fttca = last.loc[usable, "time_to_tca"].values
    sub = np.where(fttca < 1, SUBPOPS[0], np.where(fttca < 2, SUBPOPS[1], SUBPOPS[2]))
    lat = inp.groupby("event_id").tail(1).set_index("event_id").loc[usable]
    org = pd.read_csv(os.path.join(ORG, "train_event_labels.csv"))
    o = org[org.usable_for_training.str.lower() == "yes"].set_index("event_id")
    assert set(o.index) == set(usable), (len(o), len(usable))
    mism = int((o.loc[usable, "final_tier"].map(TN).values != yt).sum())
    ev = pd.DataFrame({"event_id": usable, "mission_id": last.loc[usable, "mission_id"].values.astype(int), "y_tier": yt,
                       "y_crit": (yt == 2).astype(int), "latest_tier": H.tier(lat.risk, lat.miss_distance).astype(int),
                       "latest_ttca": lat.time_to_tca.values, "n_cdm_ge2d": inp.groupby("event_id").size().loc[usable].values,
                       "final_ttca": fttca, "subpop": sub, "kelvins": sub == SUBPOPS[0]})
    # test: every message is >= 2 d out
    te = clean(pd.read_csv(H.RAW + "test_data.csv"), cats)
    te = te.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
    tinp = te[te.time_to_tca >= 2]
    tids = np.sort(tinp.event_id.unique())
    tlat = tinp.groupby("event_id").tail(1).set_index("event_id").loc[tids]
    tev = pd.DataFrame({"event_id": tids, "mission_id": tlat.mission_id.values.astype(int),
                        "latest_tier": H.tier(tlat.risk, tlat.miss_distance).astype(int), "latest_ttca": tlat.time_to_tca.values,
                        "n_cdm": tinp.groupby("event_id").size().loc[tids].values})
    numcols = [c for c in tr.columns if c not in ("event_id", "mission_id") and tr[c].dtype.kind in "if"]
    sets = {"A13": (H.ALLOWED[1:], H.TREND_COLS_A13), "ALL": (numcols, H.TREND_COLS_ALL)}
    shapes = {}
    for name, (cl, tc) in sets.items():
        F = H.event_features(inp, cl, tc).loc[usable]; F.columns = [str(c) for c in F.columns]
        T = H.event_features(tinp, cl, tc).loc[tids]; T.columns = [str(c) for c in T.columns]
        T = T.reindex(columns=F.columns)
        F.astype(np.float64).to_parquet(os.path.join(DATA, f"features_{name}.parquet"))
        T.astype(np.float64).to_parquet(os.path.join(DATA, f"test_features_{name}.parquet"))
        shapes[name] = F.shape[1]
    ev.to_parquet(os.path.join(DATA, "events.parquet")); tev.to_parquet(os.path.join(DATA, "test_events.parquet"))
    summ = {"n_usable": int(len(ev)), "tiers_safe_elev_crit": np.bincount(yt, minlength=3).tolist(),
            "label_mismatches_vs_organizer": mism,
            "subpop_counts": {s: int((sub == s).sum()) for s in SUBPOPS},
            "subpop_crit": {s: int(((sub == s) & (yt == 2)).sum()) for s in SUBPOPS},
            "persistence_exact_on_ge2d": bool(np.all(ev.latest_tier[sub == SUBPOPS[2]] == ev.y_tier[sub == SUBPOPS[2]])),
            "n_missions": int(ev.mission_id.nunique()), "n_test": int(len(tev)),
            "test_missions_seen_in_train": bool(set(tev.mission_id) <= set(ev.mission_id)),
            "test_latest_ttca_quantiles": np.quantile(tev.latest_ttca, [0, .25, .5, .75, 1]).round(3).tolist(),
            "n_features": shapes}
    json.dump(summ, open(os.path.join(DATA, "build_summary.json"), "w"), indent=1)
    log(json.dumps(summ))
    assert mism == 0 and len(ev) == 11942 and summ["tiers_safe_elev_crit"] == [8946, 2162, 834]


# ----------------------------------------------------------------------------- helpers
def load():
    ev = pd.read_parquet(os.path.join(DATA, "events.parquet"))
    FE = {k: pd.read_parquet(os.path.join(DATA, f"features_{k}.parquet")).loc[ev.event_id].values for k in ("A13", "ALL")}
    return ev, FE


def fit_pred(params, Xa, ya, Xb, seed=0, ret_model=False):
    p = dict(objective="multiclass", num_class=3, verbosity=-1, num_threads=THR, seed=seed, **params)
    nr = p.pop("n_estimators")
    m = lgb.train(p, lgb.Dataset(Xa, ya), num_boost_round=nr)
    return (m.predict(Xb), m) if ret_model else m.predict(Xb)


def pick_thr(yc, s):
    """alert if s >= t; t minimises R*FN + FP subject to recall >= FLOOR (on the data given = inner/training folds)."""
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; ys = yc[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    d = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, ts = tp[d], fp[d], ss[d]
    rec = tp / P; cost = R * (P - tp) + fp
    i = int(np.argmin(np.where(rec >= FLOOR, cost, np.inf)))
    return float(ts[i]), {"inner_recall": float(rec[i]), "inner_cost_per_1000": float(cost[i] / len(s) * 1000)}


def inner_folds(split, trn, ev, seed):
    if split == "mission":
        return list(GroupKFold(5).split(trn, groups=ev.mission_id.values[trn]))
    return list(KFold(5, shuffle=True, random_state=100 + seed).split(trn))


def persistence_P(lt, y, trn, tst):
    P = np.zeros((len(tst), 3))
    for k in range(3):
        m = lt[trn] == k
        cnt = np.bincount(y[trn][m], minlength=3).astype(float) + 1e-9
        P[lt[tst] == k] = cnt / cnt.sum()
    return P


def run_lgbm(fs, pop, split, seed, trn, tst, ev, X):
    y = ev.y_tier.values; kel = ev.kelvins.values
    sel = (lambda ix: ix[kel[ix]]) if pop == "kelvins" else (lambda ix: ix)
    a = sel(trn)
    P = fit_pred(BEST[fs], X[a], y[a], X[tst])
    pin = np.zeros(len(trn))
    for ia, ib in inner_folds(split, trn, ev, seed):
        aa = sel(trn[ia])
        pin[ib] = fit_pred(BEST[fs], X[aa], y[aa], X[trn[ib]])[:, 2]
    t, info = pick_thr(ev.y_crit.values[trn], pin)
    return P, t, info


# ----------------------------------------------------------------------------- metrics
def metrics(y, P, alert, tp_):
    yc = (y == 2).astype(int); n = len(y); npos = yc.sum()
    TP = int((alert & (yc == 1)).sum()); FA = int((alert & (yc == 0)).sum()); FN = int(npos - TP)
    cm = np.bincount(3 * y + tp_, minlength=9).reshape(3, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec_t = np.diag(cm) / cm.sum(1); prec_t = np.diag(cm) / cm.sum(0)
        f1 = np.where(rec_t + prec_t > 0, 2 * rec_t * prec_t / (rec_t + prec_t), 0.0)
    present = cm.sum(1) > 0
    if 0 < npos < n:
        r = rankdata(P[:, 2]); auc = float((r[yc == 1].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))
    else:
        auc = np.nan
    return {"n": n, "n_crit": int(npos), "recall": TP / npos if npos else np.nan, "missed": FN, "false_alarms": FA,
            "alerts": int(alert.sum()), "precision": TP / max(alert.sum(), 1), "cost": R * FN + FA,
            "cost_per_1000": (R * FN + FA) / n * 1000, "fa_per_1000": FA / n * 1000,
            "macro_f1": float(np.nanmean(np.where(present, f1, np.nan))),
            "recall_safe": rec_t[0], "recall_elevated": rec_t[1], "recall_critical_tier": rec_t[2], "auc_crit": auc}


def summarise(parts, rng):
    """parts = list of (y, P, alert, tierpred) (one per seed / one pooled). Point = mean over parts;
    CI = event bootstrap (each part resampled independently, metric averaged over parts per replicate)."""
    pt = pd.DataFrame([metrics(*p) for p in parts]).mean().to_dict()
    reps = []
    for _ in range(B):
        rr = []
        for y, P, al, tpz in parts:
            i = rng.integers(0, len(y), len(y))
            rr.append(metrics(y[i], P[i], al[i], tpz[i]))
        reps.append(pd.DataFrame(rr).mean())
    reps = pd.DataFrame(reps)
    keys = ["recall", "missed", "false_alarms", "cost", "cost_per_1000", "fa_per_1000", "macro_f1", "recall_safe",
            "recall_elevated", "recall_critical_tier", "auc_crit"]
    ci = {k: [float(np.nanpercentile(reps[k], 2.5)), float(np.nanpercentile(reps[k], 97.5))] for k in keys}
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)) for k, v in pt.items()}, ci


def eval_():
    os.makedirs(OUT, exist_ok=True)
    ev, FE = load(); y = ev.y_tier.values; lt = ev.latest_tier.values; n = len(ev)
    outer = []
    for s in range(5):
        trn, tst = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(np.arange(n), groups=ev.event_id.values))
        outer.append(("event80_20", s, np.sort(trn), np.sort(tst)))
    for k, (trn, tst) in enumerate(GroupKFold(5).split(np.arange(n), groups=ev.mission_id.values)):
        outer.append(("mission", k, trn, tst))
    jobs = [(fs, pop, sp, s, trn, tst) for (sp, s, trn, tst) in outer for fs in ("ALL", "A13") for pop in ("organizer", "kelvins")]
    log("jobs", len(jobs))
    outs = Parallel(n_jobs=NJ, verbose=0)(delayed(run_lgbm)(fs, pop, sp, s, trn, tst, ev, FE[fs]) for fs, pop, sp, s, trn, tst in jobs)
    log("lgbm done")
    res = {}  # (model, split, s) -> dict(tst, P, thr, info)
    for (fs, pop, sp, s, trn, tst), (P, t, info) in zip(jobs, outs):
        res[(f"lgbm_{fs.lower()}__train_{pop}", sp, s)] = dict(tst=tst, P=P, thr=t, info=info, alert=P[:, 2] >= t,
                                                               tier=P.argmax(1))
    for sp, s, trn, tst in outer:
        P = persistence_P(lt, y, trn, tst)
        res[("persistence", sp, s)] = dict(tst=tst, P=P, thr=2.0, info={}, alert=lt[tst] == 2, tier=lt[tst])
        t, info = pick_thr(ev.y_crit.values[trn], lt[trn].astype(float))
        res[("persistence_costcut", sp, s)] = dict(tst=tst, P=P, thr=t, info=info, alert=lt[tst] >= t, tier=lt[tst])
    joblib.dump({"res": res, "outer": outer}, os.path.join(OUT, "eval.joblib"))
    report(ev, res)


def report(ev, res):
    y = ev.y_tier.values; sub = ev.subpop.values
    models = sorted({k[0] for k in res}, key=lambda m: (not m.startswith("persistence"), m))
    rng = np.random.default_rng(0)
    out = {}
    for sp in ("event80_20", "mission"):
        out[sp] = {}
        for m in models:
            ks = sorted(k for k in res if k[0] == m and k[1] == sp)
            if sp == "mission":  # pool OOF (each fold's own threshold)
                tst = np.concatenate([res[k]["tst"] for k in ks])
                parts_all = [(y[tst], np.concatenate([res[k]["P"] for k in ks]), np.concatenate([res[k]["alert"] for k in ks]),
                              np.concatenate([res[k]["tier"] for k in ks]), tst)]
            else:
                parts_all = [(y[res[k]["tst"]], res[k]["P"], res[k]["alert"], res[k]["tier"], res[k]["tst"]) for k in ks]
            entry = {"thresholds": [round(res[k]["thr"], 4) for k in ks], "inner": [res[k]["info"] for k in ks]}
            for grp in ["all"] + SUBPOPS:
                parts = []
                for yy, P, al, tz, ix in parts_all:
                    mk = np.ones(len(ix), bool) if grp == "all" else (sub[ix] == grp)
                    parts.append((yy[mk], P[mk], al[mk], tz[mk]))
                pt, ci = summarise(parts, rng)
                entry[grp] = {"point": pt, "ci95": ci}
            out[sp][m] = entry
            log(sp, m, {g: (entry[g]["point"]["recall"], entry[g]["point"]["false_alarms"], entry[g]["point"]["cost_per_1000"],
                            entry[g]["point"]["macro_f1"], entry[g]["point"]["auc_crit"]) for g in ["all"] + SUBPOPS})
    json.dump(out, open(os.path.join(OUT, "eval_metrics.json"), "w"), indent=1, default=float)


def final(model="lgbm_all", pop="organizer"):
    ev, FE = load(); y = ev.y_tier.values; n = len(ev)
    fs = model.split("_")[1].upper(); X = FE[fs]
    idx = np.arange(n) if pop == "organizer" else np.where(ev.kelvins.values)[0]
    # by-event 5-fold OOF on all 11,942 (training data only) -> alert threshold for the test-time model
    folds = list(KFold(5, shuffle=True, random_state=7).split(np.arange(n)))
    def one(a, b):
        a = a if pop == "organizer" else a[ev.kelvins.values[a]]
        return fit_pred(BEST[fs], X[a], y[a], X[b])
    Ps = Parallel(n_jobs=NJ)(delayed(one)(a, b) for a, b in folds)
    Poof = np.zeros((n, 3)); fold_ev = np.zeros(n, int)
    for k, ((a, b), P) in enumerate(zip(folds, Ps)): Poof[b] = P; fold_ev[b] = k
    thr, info = pick_thr(ev.y_crit.values, Poof[:, 2])
    log("final threshold (event-5fold OOF)", thr, info)
    THR_ = 8
    p = dict(objective="multiclass", num_class=3, verbosity=-1, num_threads=THR_, seed=0, **BEST[fs]); nr = p.pop("n_estimators")
    mdl = lgb.train(p, lgb.Dataset(X[idx], y[idx]), num_boost_round=nr)
    tev = pd.read_parquet(os.path.join(DATA, "test_events.parquet"))
    TX = pd.read_parquet(os.path.join(DATA, f"test_features_{fs}.parquet")).loc[tev.event_id].values
    PT = mdl.predict(TX)
    pred = pd.DataFrame({"event_id": tev.event_id.values, "mission_id": tev.mission_id.values, "n_cdm": tev.n_cdm.values,
                         "latest_ttca": tev.latest_ttca.values, "p_safe": PT[:, 0], "p_elevated": PT[:, 1], "p_critical": PT[:, 2],
                         "pred_final_tier": TNAME[PT.argmax(1)], "alert_critical": PT[:, 2] >= thr, "alert_threshold": thr,
                         "persistence_tier": TNAME[tev.latest_tier.values]})
    pred.to_parquet(os.path.join(OUT, "predictions_test.parquet"))
    # OOF file: by-mission outer OOF (from eval) + by-event 5-fold OOF used for the threshold
    E = joblib.load(os.path.join(OUT, "eval.joblib"))["res"]
    name = f"{model}__train_{pop}"
    oof = ev[["event_id", "mission_id", "subpop", "final_ttca", "latest_tier", "y_tier"]].copy()
    oof["y_tier"] = TNAME[oof.y_tier.values]; oof["persistence_tier"] = TNAME[oof.pop("latest_tier").values]
    for c in ["p_safe", "p_elevated", "p_critical"]: oof[c + "_mission"] = np.nan
    oof["alert_mission"] = False; oof["fold_mission"] = -1; oof["thr_mission"] = np.nan
    for k in sorted(k for k in E if k[0] == name and k[1] == "mission"):
        r = E[k]; t = r["tst"]
        oof.loc[t, ["p_safe_mission", "p_elevated_mission", "p_critical_mission"]] = r["P"]
        oof.loc[t, "alert_mission"] = r["alert"]; oof.loc[t, "fold_mission"] = k[2]; oof.loc[t, "thr_mission"] = r["thr"]
    oof["p_safe_event5"], oof["p_elevated_event5"], oof["p_critical_event5"] = Poof[:, 0], Poof[:, 1], Poof[:, 2]
    oof["fold_event5"] = fold_ev; oof["alert_event5"] = Poof[:, 2] >= thr
    oof.to_parquet(os.path.join(OUT, "oof_organizer.parquet"))
    mdl.save_model(os.path.join(OUT, "model.txt"))
    fi = pd.Series(mdl.feature_importance("gain"),
                   index=pd.read_parquet(os.path.join(DATA, f"features_{fs}.parquet")).columns).sort_values(ascending=False)
    json.dump({"model": model, "train_pop": pop, "n_train": int(len(idx)), "threshold": thr, "threshold_info": info,
               "params": BEST[fs], "feature_set": fs, "n_features": int(X.shape[1]), "top_features_gain": fi.head(15).round(1).to_dict(),
               "test_pred_counts": pred.pred_final_tier.value_counts().to_dict(), "test_alerts": int(pred.alert_critical.sum()),
               "test_persistence_counts": pred.persistence_tier.value_counts().to_dict()},
              open(os.path.join(OUT, "final_summary.json"), "w"), indent=1, default=float)
    log("final done", pred.pred_final_tier.value_counts().to_dict(), "alerts", int(pred.alert_critical.sum()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build": build()
    elif cmd == "eval": eval_()
    elif cmd == "report":
        ev, _ = load(); report(ev, joblib.load(os.path.join(OUT, "eval.joblib"))["res"])
    elif cmd == "final": final(*sys.argv[2:])
    else: print(__doc__)
