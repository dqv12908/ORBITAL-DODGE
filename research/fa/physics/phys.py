"""fa/physics: ALL + physics Pc features (+ risk-futures fan features). Same splits/params/threshold rule as organizer.py."""
import os, sys, json, glob, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ORGD = "/root/orbital-dodge/research/organizer"
sys.path.insert(0, ORGD); sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd
import organizer as O
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
O.THR = 2
CANDS = "/root/orbital-dodge/research/fa/cands"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:.0f}s]", *a, flush=True)

def build_phys(ev):
    pc = pd.read_parquet(os.path.join(HERE, "pc_rows.parquet"))
    pc = pc[pc.event_key.str.startswith("train:")]
    cc = pd.read_parquet(os.path.join(HERE, "cdm_clean.parquet"), columns=["event_key", "cdm_index", "time_to_tca", "c_object_type"])
    d = pc.merge(cc, on=["event_key", "cdm_index"], how="inner")
    d["event_id"] = d.event_key.str.split(":").str[1].astype(int)
    d = d[d.time_to_tca >= 2].sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    d["status_ok"] = (d.pc_status == "OK").astype(float)
    d["lsx"] = np.log10(d.sigma_x.clip(lower=1e-3)); d["lsy"] = np.log10(d.sigma_y.clip(lower=1e-3))
    d["lsr"] = d.lsx - d.lsy
    d["lmah"] = np.log10(d.mahalanobis.clip(lower=1e-6))
    d["lks"] = np.log10(d.k_star.clip(lower=1e-6))
    d["ptier"] = d.tier.map({"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}).astype(float)
    d["pctier"] = d.pc_tier.map({"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}).astype(float)
    d["dd"] = d.disagree_dex.astype(float); d["esc"] = d.escalate.astype(float); d["npd"] = d.npd_clipped.astype(float)
    d["gap_pcmax"] = d.log10_pc_max - d.log10_pc
    for c in ("log10_pc", "log10_pc_max"): d[c] = d[c].clip(lower=-30)
    trend = ["log10_pc", "log10_pc_max", "lmah", "lks", "lsx", "lsy", "lsr", "gap_pcmax"]
    lastonly = ["status_ok", "ptier", "pctier", "dd", "esc", "npd"]
    g = d.groupby("event_id")
    F = g[trend + lastonly].last().add_prefix("ph_last_")
    F = F.join(g[trend].first().rsub(g[trend].last()).add_prefix("ph_d_"))
    F = F.join(g[trend].min().add_prefix("ph_min_")).join(g[trend].max().add_prefix("ph_max_"))
    F = F.join(g[["status_ok", "ptier", "esc"]].mean().add_prefix("ph_mean_"))
    # slope vs ttca (per day, sign: positive = rising as TCA approaches)
    def slopes(x):
        t = -x.time_to_tca.values
        if len(t) < 2 or np.ptp(t) == 0: return pd.Series({f"ph_sl_{c}": np.nan for c in trend})
        tc = t - t.mean(); den = (tc ** 2).sum()
        return pd.Series({f"ph_sl_{c}": np.nansum(tc * (x[c].values - np.nanmean(x[c].values))) / den for c in trend})
    S = g[["time_to_tca"] + trend].apply(slopes)
    F = F.join(S)
    cot = g.c_object_type.last()
    return F.reindex(ev.event_id.values), cot.reindex(ev.event_id.values).values, d

def build_fut(ev, d):
    lastidx = d.groupby("event_id").cdm_index.last()
    rows = []; bad = 0
    for eid in ev.event_id.values:
        j = json.load(open(os.path.join(HERE, "fans", f"train:{eid}.json")))
        grid = np.array(j["ttca_grid_d"]); o = np.argsort(grid)
        r = {"event_id": eid, "fu_pcrit": j["p_critical"], "fu_pelev": j["p_tiers"]["ELEVATED"], "fu_start_ttca": j["start_ttca_d"],
             "fu_start_ok": float(j["start_cdm_index"] == lastidx.get(eid, -1) and j["start_ttca_d"] >= 2)}
        for tt in (1.0, 0.5):
            for q in ("p05", "p25", "p50", "p75", "p95"):
                r[f"fu_pc_{q}_{tt}"] = np.interp(tt, grid[o], np.array(j["quantiles"][q])[o])
                r[f"fu_miss_{q}_{tt}"] = np.log10(max(np.interp(tt, grid[o], np.array(j["miss_quantiles"][q])[o]), 1.0))
        rows.append(r)
    U = pd.DataFrame(rows).set_index("event_id")
    log("fan start check: start_ok frac", U.fu_start_ok.mean(), "start_ttca<2:", int((U.fu_start_ttca < 2).sum()))
    U = U.drop(columns=["fu_start_ok", "fu_start_ttca"])
    return U

def job(name, split, k, trn, tst, ev, X):
    y = ev.y_tier.values; p = O.BEST["ALL"]
    P = O.fit_pred(p, X[trn], y[trn], X[tst])
    pin = np.zeros((len(trn), 3)); ifold = np.zeros(len(trn), int)
    for f, (ia, ib) in enumerate(O.inner_folds("mission" if split == "mission" else "e", trn, ev, k)):
        pin[ib] = O.fit_pred(p, X[trn[ia]], y[trn[ia]], X[trn[ib]]); ifold[ib] = f
    return name, split, k, trn, tst, P, pin, ifold

def main():
    ev = pd.read_parquet(os.path.join(ORGD, "data", "events.parquet"))
    XA = pd.read_parquet(os.path.join(ORGD, "data", "features_ALL.parquet")).loc[ev.event_id]
    PH, cot, d = build_phys(ev); log("phys", PH.shape, "nan frac latest pc", PH.ph_last_log10_pc.isna().mean())
    FU = build_fut(ev, d).reindex(ev.event_id.values); log("fut", FU.shape)
    PH.to_parquet(os.path.join(HERE, "phys_features.parquet")); FU.to_parquet(os.path.join(HERE, "fut_features.parquet"))
    sets = {"lgbm_all_phys": np.hstack([XA.values, PH.values]), "lgbm_all_phys_fut": np.hstack([XA.values, PH.values, FU.values])}
    names = list(XA.columns) + list(PH.columns) + list(FU.columns)
    if not os.path.exists(os.path.join(CANDS, "current_lgbm_all.parquet")): sets["current_lgbm_all"] = XA.values
    n = len(ev); outer = []
    for s in range(5):
        trn, tst = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(np.arange(n), groups=ev.event_id.values))
        outer.append((f"8020_s{s}", s, np.sort(trn), np.sort(tst)))
    for k, (trn, tst) in enumerate(GroupKFold(5).split(np.arange(n), groups=ev.mission_id.values)):
        outer.append(("mission", k, trn, tst))
    jobs = [(nm, sp, k, trn, tst) for nm in sets for (sp, k, trn, tst) in outer]
    log("jobs", len(jobs))
    outs = Parallel(n_jobs=int(os.environ.get("NJ", 24)))(delayed(job)(nm, sp, k, trn, tst, ev, sets[nm]) for nm, sp, k, trn, tst in jobs)
    sub = ev.subpop.values; ekey = np.array([f"train:{e}" for e in ev.event_id.values])
    base = lambda ix: dict(event_key=ekey[ix], event_id=ev.event_id.values[ix], y_tier=ev.y_tier.values[ix],
                           latest_tier=ev.latest_tier.values[ix], c_object_type=cot[ix], final_sub=sub[ix], mission_id=ev.mission_id.values[ix])
    dfs = {}
    for nm, sp, k, trn, tst, P, pin, ifold in outs:
        a = pd.DataFrame({**base(tst), "split": sp, "fold": k, "role": "outer_test", "inner_fold": -1,
                          "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2]})
        b = pd.DataFrame({**base(trn), "split": sp, "fold": k, "role": "inner_oof", "inner_fold": ifold,
                          "p_safe": pin[:, 0], "p_elevated": pin[:, 1], "p_critical": pin[:, 2]})
        dfs.setdefault(nm, []).extend([a, b])
    for nm, L in dfs.items():
        D = pd.concat(L, ignore_index=True)
        out = os.path.join(CANDS, f"{nm}.parquet")
        if nm == "current_lgbm_all" and os.path.exists(out): continue
        D.to_parquet(out); log("wrote", out, D.shape)
    # quick metrics
    res = {}
    for nm, L in dfs.items():
        D = pd.concat(L, ignore_index=True); r = {}
        for grp, sps in (("8020", [f"8020_s{s}" for s in range(5)]), ("mission", ["mission"])):
            rr = []; tot = dict(TP=0, FA=0, FN=0, n=0)
            for sp in sps:
                for k in sorted(D[D.split == sp].fold.unique()):
                    I = D[(D.split == sp) & (D.fold == k) & (D.role == "inner_oof")]
                    T = D[(D.split == sp) & (D.fold == k) & (D.role == "outer_test")]
                    t, _ = O.pick_thr((I.y_tier.values == 2).astype(int), I.p_critical.values)
                    al = T.p_critical.values >= t; yc = T.y_tier.values == 2
                    tp = int((al & yc).sum()); fa = int((al & ~yc).sum()); fn = int((~al & yc).sum())
                    if grp == "8020": rr.append((tp / (tp + fn), fa, (10 * fn + fa) / len(T) * 1000))
                    else: tot["TP"] += tp; tot["FA"] += fa; tot["FN"] += fn; tot["n"] += len(T)
            if grp == "8020": r[grp] = dict(zip(["recall", "FA", "cost_per_1000"], np.round(np.mean(rr, 0), 4).tolist()))
            else: r[grp] = {"recall": round(tot["TP"] / (tot["TP"] + tot["FN"]), 4), "FA": tot["FA"],
                            "cost_per_1000": round((10 * tot["FN"] + tot["FA"]) / tot["n"] * 1000, 2)}
        res[nm] = r; log(nm, r)
    json.dump(res, open(os.path.join(HERE, "quick_results.json"), "w"), indent=1)
    import lightgbm as lgb
    p = dict(objective="multiclass", num_class=3, verbosity=-1, num_threads=8, seed=0, **O.BEST["ALL"]); nr = p.pop("n_estimators")
    m = lgb.train(p, lgb.Dataset(sets["lgbm_all_phys_fut"], ev.y_tier.values), num_boost_round=nr)
    fi = pd.Series(m.feature_importance("gain"), index=names).sort_values(ascending=False)
    fi.head(30).to_json(os.path.join(HERE, "top_gain.json")); log(fi.head(20).round(0).to_dict())

if __name__ == "__main__":
    main()
