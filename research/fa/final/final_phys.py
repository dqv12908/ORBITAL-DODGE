"""fa/final: fit lgbm_all_phys on all 11,942 organizer events, predict 2,167 test events, write registry 1.1.0."""
import os, sys, json, time, inspect, warnings, shutil
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/orbital-dodge/research/organizer"); sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
sys.path.insert(0, "/root/orbital-dodge/research/fa/physics")
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import organizer as O, phys as PHY
ORGD = "/root/orbital-dodge/research/organizer"; CANDS = "/root/orbital-dodge/research/fa/cands"
REG = "/root/orbital-dodge/registry/forecaster_organizer"; V = "1.1.0"; OUTD = os.path.join(REG, V)
os.makedirs(OUTD, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
TNAME = np.array(["Safe", "Elevated", "Critical"])

# physics features generalised to a key prefix (identical code to phys.build_phys otherwise)
src = inspect.getsource(PHY.build_phys).replace('def build_phys(ev):', 'def build_phys_pfx(ev, PFX):') \
    .replace('pc.event_key.str.startswith("train:")', 'pc.event_key.str.startswith(PFX)') \
    .replace('d.event_key.str.startswith', 'd.event_key.str.startswith')
assert 'startswith(PFX)' in src
ns = dict(PHY.__dict__); exec(src, ns); build_phys_pfx = ns["build_phys_pfx"]
cc_te = pd.read_parquet(os.path.join(PHY.HERE, "cdm_clean.parquet"), columns=["event_key", "time_to_tca"])
cc_te = cc_te[cc_te.event_key.str.startswith("test:")]
log("test cdm min ttca", float(cc_te.time_to_tca.min()))

ev = pd.read_parquet(os.path.join(ORGD, "data", "events.parquet"))
tev = pd.read_parquet(os.path.join(ORGD, "data", "test_events.parquet"))
XA = pd.read_parquet(os.path.join(ORGD, "data", "features_ALL.parquet")).loc[ev.event_id]
TXA = pd.read_parquet(os.path.join(ORGD, "data", "test_features_ALL.parquet")).loc[tev.event_id]
assert list(XA.columns) == list(TXA.columns)
PH, cot, _ = build_phys_pfx(ev, "train:")
TPH, tcot, _ = build_phys_pfx(tev, "test:")
TPH = TPH[PH.columns]
log("phys train", PH.shape, "test", TPH.shape, "test rows w/o phys", int(TPH.isna().all(1).sum()))
# sanity: equals stored training phys features
st = pd.read_parquet(os.path.join(PHY.HERE, "phys_features.parquet"))
assert np.allclose(st.values, PH.values, equal_nan=True), "phys features differ from stored"
X = np.hstack([XA.values, PH.values]); TX = np.hstack([TXA.values, TPH.values])
names = list(XA.columns) + list(PH.columns)
y = ev.y_tier.values; n = len(ev); p = O.BEST["ALL"]

# threshold: by-event 5-fold OOF on all training data (same rule & folds as organizer.final)
folds = list(KFold(5, shuffle=True, random_state=7).split(np.arange(n)))
Ps = Parallel(n_jobs=5)(delayed(O.fit_pred)(p, X[a], y[a], X[b]) for a, b in folds)
Poof = np.zeros((n, 3)); fold_ev = np.zeros(n, int)
for k, ((a, b), P) in enumerate(zip(folds, Ps)): Poof[b] = P; fold_ev[b] = k
thr, info = O.pick_thr((y == 2).astype(int), Poof[:, 2]); log("threshold", thr, info)
pp = dict(objective="multiclass", num_class=3, verbosity=-1, num_threads=16, seed=0, **p); nr = pp.pop("n_estimators")
mdl = lgb.train(pp, lgb.Dataset(X, y, feature_name=[f"f{i}" for i in range(X.shape[1])]), num_boost_round=nr)
PT = mdl.predict(TX)
pred = pd.DataFrame({"event_key": [f"test:{e}" for e in tev.event_id.values], "event_id": tev.event_id.values,
                     "mission_id": tev.mission_id.values, "n_cdm": tev.n_cdm.values, "latest_ttca": tev.latest_ttca.values,
                     "c_object_type": tcot, "p_safe": PT[:, 0], "p_elevated": PT[:, 1], "p_critical": PT[:, 2],
                     "pred_final_tier": TNAME[PT.argmax(1)], "alert": PT[:, 2] >= thr, "alert_threshold": thr,
                     "threshold_source": "by-event 5-fold OOF (KFold rs=7) on all 11,942 train events; argmin 10*FN+FP s.t. recall>=0.90",
                     "persistence_tier": TNAME[tev.latest_tier.values]})
pred.to_parquet(os.path.join(OUTD, "predictions_test.parquet"))
mdl.save_model(os.path.join(OUTD, "model.txt"))
json.dump(names, open(os.path.join(OUTD, "feature_names.json"), "w"))
log("test alerts", int(pred.alert.sum()), pred.pred_final_tier.value_counts().to_dict())

# ---- evaluation from candidate files (nested thresholds from inner_oof) ----
def ek(s):
    s = pd.Series(s).astype(str)
    return np.where(s.str.contains(":"), s, "train:" + s)
def alerts(name):
    D = pd.read_parquet(os.path.join(CANDS, f"{name}.parquet"))
    D["event_key"] = ek(D.event_key)
    D.loc[D.split.str.startswith("8020"), "fold"] = 0
    out = []
    for (sp, k), G in D.groupby(["split", "fold"]):
        T = G[G.role == "outer_test"].copy()
        if "alert" in T.columns and T.alert.notna().all():
            T["thr"] = np.nan; T["alert"] = T.alert.astype(bool)
        else:
            I = G[G.role == "inner_oof"]
            t, _ = O.pick_thr((I.y_tier.values == 2).astype(int), I.p_critical.values)
            T["thr"] = t; T["alert"] = T.p_critical.values >= t
        out.append(T)
    T = pd.concat(out, ignore_index=True)
    T["mission_id"] = T.event_key.map(dict(zip([f"train:{e}" for e in ev.event_id], ev.mission_id)))
    return T
A = {m: alerts(m) for m in ("lgbm_all_phys", "current_lgbm_all", "persistence")}
key = ["split", "fold", "event_key"]
base = A["current_lgbm_all"][key + ["y_tier", "final_sub", "mission_id"]]
M = base.copy()
for m, T in A.items():
    M = M.merge(T[key + ["alert"]].rename(columns={"alert": "al_" + m}), on=key, how="left")
assert M.filter(like="al_").notna().all().all() and len(M) == len(base)
M["yc"] = M.y_tier == 2
rng = np.random.default_rng(0); Bn = 500
def stats(G, m):
    al = G["al_" + m].values.astype(bool); yc = G.yc.values
    tp = (al & yc).sum(); fa = (al & ~yc).sum(); fn = (~al & yc).sum()
    return np.array([tp, fa, fn, len(G)], float)
def summ(v):  # v = [tp, fa, fn, n]
    tp, fa, fn, nn = v
    return dict(recall=tp / max(tp + fn, 1), false_alarms=fa, missed=fn, cost_per_1000=(10 * fn + fa) / nn * 1000, n=nn)
models = ["lgbm_all_phys", "current_lgbm_all", "persistence"]
def evaluate(sub):
    res = {}
    for grp in ("8020", "mission"):
        S = sub[sub.split.str.startswith("8020")] if grp == "8020" else sub[sub.split == "mission"]
        units = S.split.unique() if grp == "8020" else None
        # point estimates: 8020 = mean over seeds; mission = totals
        pt = {}
        for m in models:
            if grp == "8020":
                L = [summ(stats(S[S.split == s], m)) for s in units]
                pt[m] = {k: float(np.mean([d[k] for d in L])) for k in L[0]}
            else:
                pt[m] = {k: float(v) for k, v in summ(stats(S, m)).items()}
        # bootstrap: 8020 resamples events within each seed's test set; mission resamples whole missions
        bs = {m: [] for m in models}
        if grp == "8020":
            per = {s: S[S.split == s] for s in units}
            for _ in range(Bn):
                idx = {s: rng.integers(0, len(per[s]), len(per[s])) for s in units}
                for m in models:
                    L = [summ(stats(per[s].iloc[idx[s]], m)) for s in units]
                    bs[m].append([np.mean([d[k] for d in L]) for k in ("recall", "false_alarms", "missed", "cost_per_1000")])
        else:
            mids = S.mission_id.unique(); byM = {mm: g for mm, g in S.groupby("mission_id")}
            Ms = {m: np.array([stats(byM[mm], m) for mm in mids]) for m in models}
            for _ in range(Bn):
                w = rng.multinomial(len(mids), np.ones(len(mids)) / len(mids))
                for m in models:
                    v = (Ms[m] * w[:, None]).sum(0); s_ = summ(v)
                    bs[m].append([s_[k] for k in ("recall", "false_alarms", "missed", "cost_per_1000")])
        bs = {m: np.array(v) for m, v in bs.items()}
        ci = lambda a: [round(float(np.nanpercentile(a, 2.5)), 4), round(float(np.nanpercentile(a, 97.5)), 4)]
        r = {}
        for m in models:
            r[m] = {k: round(v, 4) for k, v in pt[m].items()}
            for j, k in enumerate(("recall", "false_alarms", "missed", "cost_per_1000")):
                r[m][k + "_ci95"] = ci(bs[m][:, j])
                if m != "current_lgbm_all":
                    r[m]["d_" + k + "_vs_current"] = round(pt[m][k] - pt["current_lgbm_all"][k], 4)
                    r[m]["d_" + k + "_vs_current_ci95"] = ci(bs[m][:, j] - bs["current_lgbm_all"][:, j])
        res[grp] = r
    return res
EV = {"all": evaluate(M)}
for s in ("final_lt1d", "final_1to2d", "final_ge2d"): EV[s] = evaluate(M[M.final_sub == s])
log("eval all", json.dumps(EV["all"], default=float)[:1500])

# oof file: outer_test predictions + alert for both splits
Pp = pd.read_parquet(os.path.join(CANDS, "lgbm_all_phys.parquet")); Pp["event_key"] = ek(Pp.event_key)
Pp.loc[Pp.split.str.startswith("8020"), "fold"] = 0
Pp = Pp[Pp.role == "outer_test"]
oof = Pp[key + ["p_safe", "p_elevated", "p_critical", "y_tier", "latest_tier", "c_object_type", "final_sub"]].merge(
    A["lgbm_all_phys"][key + ["alert", "thr"]], on=key)
oof["y_tier"] = TNAME[oof.y_tier.values.astype(int)]
oof.to_parquet(os.path.join(OUTD, "oof_organizer.parquet"))

fi = pd.Series(mdl.feature_importance("gain"), index=names).sort_values(ascending=False)
c1 = json.load(open(os.path.join(REG, "1.0.0", "model_card.json"))) if os.path.exists(os.path.join(REG, "1.0.0", "model_card.json")) else {}
card = {
 "name": "forecaster_organizer", "version": V, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "previous_version": "1.0.0",
 "task": "Predict each event's final_tier (organizer definition) from CDMs with time_to_tca >= 2 d; CRITICAL alert at R=10",
 "model": "LightGBM multiclass (same params as 1.0.0) on ALL features (233) + 52 physics Pc features (ph_*) from pc_rows.parquet",
 "candidate_name": "lgbm_all_phys", "training_population": "organizer usable_for_training (11,942)",
 "params": p, "n_features": int(X.shape[1]), "feature_groups": {"ALL": len(XA.columns), "physics_ph": len(PH.columns)},
 "physics_features": list(PH.columns), "top_features_gain": fi.head(20).round(1).to_dict(),
 "strategy": "alert if p_critical >= threshold; threshold = argmin 10*missed + FA subject to recall >= 0.90 on out-of-fold predictions",
 "R": 10, "recall_floor": 0.90, "alert_threshold": thr, "threshold_info": info,
 "threshold_source": "by-event 5-fold OOF (KFold shuffle rs=7) over all 11,942 training events (same rule as 1.0.0)",
 "test": {"n": int(len(pred)), "alerts": int(pred.alert.sum()), "pred_counts": pred.pred_final_tier.value_counts().to_dict(),
          "alerts_1.0.0": None},
 "evaluation": {"protocol": "organizer.py splits: 80/20 by event (5 seeds; mean per test set ~2,389) and GroupKFold(5) by mission (totals over 11,942). Outer cut-off chosen on inner OOF of each outer-training set. CIs: 80/20 event bootstrap within each seed's test set; by-mission = MISSION-level bootstrap (whole missions resampled; fixes audit item 2). B=500, paired differences vs current 1.0.0.",
                "results": EV},
 "selection": "Best of 15 candidates in research/fa/select (selected on the same by-mission data reported here, so the reported gain is optimistically biased).",
 "audit": {"ok": True, "leakage_found": False, "summary": "No leakage found. The improvement over 1.0.0 is NOT statistically significant: all difference CIs include 0 or are small; ranking quality at the current model's recall is not better (FA at matched recall 82.6 vs 81.0 on 80/20; 442 vs 423 by mission). The 80/20 FA reduction mostly comes from where the inner-chosen cut-off lands, and varies by seed (s0: 71 vs 113 FA; s3: 92 vs 79).",
           "open_items": ["Provenance of pc_rows.parquet (per-message tier/pc_tier/escalate/rule_ids columns) not confirmed: no .py in /root/orbital-dodge creates it. Rules look like fixed physics thresholds, not label-fit; ph_last_ptier is a top-3 gain feature.",
                          "Earlier select.py CIs used an event bootstrap for the by-mission split (too narrow); this card uses a mission bootstrap."]},
 "known_limits": ["Treat 1.1.0 as on-par with 1.0.0, not a proven false-alarm reduction; keep 1.0.0 available for rollback.",
                  "Depends on an extra input (pc_rows.parquet physics Pc pipeline) that must be available at inference time for every event.",
                  "final_lt1d sub-population carries most misses; recall there is below the 0.90 floor in some splits.",
                  "Threshold chosen on by-event OOF; by-mission generalisation is somewhat worse.",
                  "Never trained/evaluated on the Kelvins Zenodo release."],
 "files": ["predictions_test.parquet", "oof_organizer.parquet", "model.txt", "feature_names.json", "model_card.json"],
}
if c1: card["evaluation"]["v1_0_0_card_metrics_all"] = {k: c1.get("evaluation", {}).get(k, {}).get("all") for k in ("event80_20", "mission")}
p1 = os.path.join(REG, "1.0.0", "predictions_test.parquet")
if os.path.exists(p1):
    q = pd.read_parquet(p1); card["test"]["alerts_1.0.0"] = int(q.alert_critical.sum())
    j = pd.Series(pred.alert.values, index=pred.event_id).reindex(q.event_id).values
    card["test"]["alert_agreement_with_1.0.0"] = float((j == q.alert_critical.values).mean())
json.dump(card, open(os.path.join(OUTD, "model_card.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
open(os.path.join(REG, "CURRENT"), "w").write(V + "\n")
log("done", OUTD)
