"""A/B/D evaluation: compares MC risk futures vs persistence vs HGB vs tuned LightGBM on by-event, by-mission
(GroupKFold) and leave-one-mission-out OOF predictions; split/Mondrian conformal on P(critical).
Writes out/eval.json (consumed by report.py)."""
import os, json, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
from bm import HERE, tier_metrics, prob_metrics, boot_ci, load_rows, events_table, folds

OUT = os.path.join(HERE, "out")
if os.path.exists(os.path.join(OUT, "gbm_oof.joblib")):
    G = joblib.load(os.path.join(OUT, "gbm_oof.joblib"))
else:  # fallback before the GBM run finishes: persistence only
    _ev = events_table(load_rows()); _el = _ev[(_ev.split == "train") & _ev.eligible].sort_values("event_id")
    G = dict(ids=_el.event_id.values, y_tier=_el.final_tier.values, mission=_el.mission_id.values, lat_tier=_el.start_tier.values, oof={})
    _e = pd.DataFrame(dict(event_key=_el.event_key.values, mission_id=_el.mission_id.values))
    for _sp, _k in [("event_s0", "event"), ("mission", "mission"), ("lomo", "lomo")]:
        _P = np.zeros((len(_el), 3)); _lt = G["lat_tier"]; _yt = G["y_tier"]
        for a, b in folds(_e, _k):
            for k in range(3):
                m = _lt[a] == k
                for j in range(3): _P[b[_lt[b] == k], j] = (_yt[a][m] == j).mean()
        G["oof"][("persistence_emp", _sp)] = _P
keys = np.array(["train:%d" % i for i in G["ids"]])
yt = G["y_tier"]; yc = (yt == 2).astype(int); mis = G["mission"]; lat_tier = G["lat_tier"]
n = len(yt)
el = pd.DataFrame(dict(event_key=keys, mission_id=mis))
FOLDS = {"event": folds(el, "event", 0), "mission": folds(el, "mission"), "lomo": folds(el, "lomo")}
d = load_rows(); ev = events_table(d).set_index("event_key")
objtype = d.loc[ev.loc[keys, "start_idx"].values, "c_object_type"].values


def fut_probs(kind):
    p = os.path.join(OUT, f"futures_{kind}.joblib")
    if not os.path.exists(p):
        return None
    F = joblib.load(p)
    idx = pd.Series(np.arange(len(F["event_key"])), index=F["event_key"])
    have = np.isin(keys, F["event_key"])
    if not have.all():
        return None
    return F["p_tiers"][idx.loc[keys].values]


def iso_crossfit(p, fl):
    """isotonic calibration of each class prob fitted on the other folds (cross-fitted), renormalised."""
    q = np.zeros_like(p)
    for a, b in fl:
        for j in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[a, j], (yt[a] == j).astype(float))
            q[b, j] = ir.predict(p[b, j])
    q = q / np.clip(q.sum(1, keepdims=True), 1e-9, None)
    return q


def methods_for(split):
    gsp = "event_s0" if split == "event" else split
    M = {}
    pe = G["oof"][("persistence_emp", gsp)]
    M["persistence"] = (pe, lat_tier)
    for m in ["hgb_A13", "lgbm_A13", "lgbm_ALL"]:
        if (m, gsp) in G["oof"]:
            P = G["oof"][(m, gsp)]; M[m] = (P, P.argmax(1))
    pf = fut_probs(split)
    if pf is not None:
        M["futures_mc"] = (pf, pf.argmax(1))
        pfc = iso_crossfit(pf, FOLDS[split]); M["futures_mc_iso"] = (pfc, pfc.argmax(1))
        if ("lgbm_A13", gsp) in G["oof"]:
            pa = 0.5 * (pfc + G["oof"][("lgbm_A13", gsp)]); M["ens_futures_lgbmA13"] = (pa, pa.argmax(1))
    return M


def op_points(p):
    out = {}
    for t in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        a = p >= t
        out[str(t)] = dict(recall=float(a[yc == 1].mean()), false_alarms=int((a & (yc == 0)).sum()), alerts=int(a.sum()))
    # false alarms at matched persistence recall
    pr = (lat_tier[yc == 1] == 2).mean()
    s = np.sort(p[yc == 1]); k = int(np.floor((1 - pr) * len(s)))
    thr = s[k] if k < len(s) else 1.0
    a = p >= thr
    out["at_persistence_recall"] = dict(threshold=float(thr), recall=float(a[yc == 1].mean()), false_alarms=int((a & (yc == 0)).sum()))
    return out


def full_metrics(P, pt, groups=None, B=300):
    m = tier_metrics(yt, pt); m.update(prob_metrics(yc, P[:, 2]))
    ci = {}
    for name, fn in [("macro_f1", lambda i: tier_metrics(yt[i], pt[i])["macro_f1"]),
                     ("crit_recall", lambda i: tier_metrics(yt[i], pt[i])["crit_recall"]),
                     ("false_alarms", lambda i: tier_metrics(yt[i], pt[i])["false_alarms"]),
                     ("auc", lambda i: prob_metrics(yc[i], P[i, 2])["auc"]),
                     ("brier", lambda i: prob_metrics(yc[i], P[i, 2])["brier"])]:
        ci[name] = boot_ci(fn, n, B=B, groups=groups)
    m["ci95"] = ci
    m["op_points"] = op_points(P[:, 2])
    return m


R = {"n_events": int(n), "n_crit": int(yc.sum()), "splits": {}}
for split in ["event", "mission", "lomo"]:
    M = methods_for(split)
    R["splits"][split] = {}
    for name, (P, pt) in M.items():
        r = full_metrics(P, pt, groups=(mis if split != "event" else None))
        if split == "lomo":
            per = {}
            for mm in np.unique(mis):
                i = mis == mm
                if yc[i].sum() == 0: continue
                per[int(mm)] = dict(n=int(i.sum()), n_crit=int(yc[i].sum()), crit_recall=float((pt[i][yc[i] == 1] == 2).mean()),
                                    false_alarms=int(((pt[i] == 2) & (yc[i] == 0)).sum()))
            r["per_mission"] = per
        R["splits"][split][name] = r
        print(f"{split:8s} {name:22s} F1 {r['macro_f1']:.3f} rec {r['crit_recall']:.3f} FA {r['false_alarms']:4d} "
              f"AUC {r['auc']:.3f} Brier {r['brier']:.4f} ECE {r['ece']:.4f} | FA@pers-recall {r['op_points']['at_persistence_recall']['false_alarms']}")
    # seed spread for by-event LightGBM
    if split == "event":
        for m in ["lgbm_A13", "lgbm_ALL", "hgb_A13"]:
            vals = [G["oof"][(m, f"event_s{s}")] for s in range(10) if (m, f"event_s{s}") in G["oof"]]
            if vals:
                f1 = [tier_metrics(yt, v.argmax(1))["macro_f1"] for v in vals]
                au = [prob_metrics(yc, v[:, 2])["auc"] for v in vals]
                R["splits"][split][m]["seeds10"] = dict(macro_f1_mean=float(np.mean(f1)), macro_f1_sd=float(np.std(f1)),
                                                        auc_mean=float(np.mean(au)), auc_sd=float(np.std(au)))

# ---------------- B: conformal on P(critical), held-out missions ----------------
def conformal(score, split, alpha, mondrian=None):
    """class-conditional split conformal: per held-out fold, tau = k-th smallest calibration critical score, k=floor(alpha(n+1)).
    Calibration = OOF scores of the other folds (missions never seen by the test fold's model nor by tau)."""
    alert = np.zeros(n, bool); taus = []
    for a, b in FOLDS[split]:
        grp_a = mondrian[a] if mondrian is not None else np.zeros(len(a), int)
        grp_b = mondrian[b] if mondrian is not None else np.zeros(len(b), int)
        for g in np.unique(grp_b):
            cal = np.sort(score[a][(yc[a] == 1) & (grp_a == g)])
            k = int(np.floor(alpha * (len(cal) + 1)))
            tau = -np.inf if k < 1 or len(cal) == 0 else cal[k - 1]
            ib = b[grp_b == g]
            alert[ib] = score[ib] >= tau
            taus.append(float(tau) if np.isfinite(tau) else -1.0)
    return alert, taus


R["conformal"] = {}
for split in ["mission", "lomo"]:
    M = methods_for(split)
    R["conformal"][split] = {}
    for name in ["persistence", "hgb_A13", "lgbm_A13", "futures_mc", "ens_futures_lgbmA13"]:
        if name not in M: continue
        P, pt = M[name]
        score = P[:, 2] + 1e-6 * (pt == 2)
        for alpha in [0.05, 0.10]:
            for mond_name, mond in [("plain", None), ("mondrian_latest_tier", lat_tier)]:
                al, taus = conformal(score, split, alpha, mond)
                rec = float(al[yc == 1].mean()); fa = int((al & (yc == 0)).sum())
                ci_r = boot_ci(lambda i: al[i][yc[i] == 1].mean(), n, B=300, groups=mis)
                ci_f = boot_ci(lambda i: (al[i] & (yc[i] == 0)).sum(), n, B=300, groups=mis)
                per = {int(m): float(al[(mis == m) & (yc == 1)].mean()) for m in np.unique(mis) if yc[mis == m].sum() >= 5}
                key = f"{name}|target_recall_{1-alpha:.2f}|{mond_name}"
                R["conformal"][split][key] = dict(realised_recall=rec, recall_ci95=ci_r, false_alarms=fa, fa_ci95=ci_f,
                                                  alerts=int(al.sum()), alert_rate=float(al.mean()),
                                                  missions_below_target=[m for m, v in per.items() if v < 1 - alpha],
                                                  per_mission_recall=per, median_tau=float(np.median(taus)))
                print(f"conformal {split:7s} {key:55s} recall {rec:.3f} [{ci_r[0]:.3f},{ci_r[1]:.3f}] FA {fa} alerts {int(al.sum())}")
json.dump(R, open(os.path.join(OUT, "eval.json"), "w"), indent=1, default=float)
print("saved eval.json")
