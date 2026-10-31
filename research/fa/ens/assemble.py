import os, sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/orbital-dodge/research/organizer"); sys.path.insert(0, "/root/orbital-dodge/research/bakeoff")
import numpy as np, pandas as pd
import organizer as O
from sklearn.isotonic import IsotonicRegression
W = "/root/orbital-dodge/research/fa/ens/work"; CD = "/root/orbital-dodge/research/fa/cands"
ev, FE = O.load(); yc = ev.y_crit.values
cot = pd.read_parquet("/root/orbital-dodge/research/organizer/data/features_ALL.parquet")["last_c_object_type"].loc[ev.event_id].values
MODELS = [m for m in ["lgbm_all", "xgb_a13_crit_w30", "hgb_a13_tier_w100", "rf_all_crit_bsub"] if os.path.exists(f"{W}/{m}.parquet")]
D = {m: pd.read_parquet(f"{W}/{m}.parquet") for m in MODELS}
def export(name, df):
    o = df[["idx", "split", "fold", "role", "p_critical"]].copy()
    o.insert(0, "event_key", "train:" + ev.event_id.values[o.idx].astype(str))
    o["y_tier"] = ev.y_tier.values[o.idx]; o["latest_tier"] = ev.latest_tier.values[o.idx]
    o["c_object_type"] = cot[o.idx]; o["final_sub"] = ev.subpop.values[o.idx]
    o.drop(columns="idx").reset_index(drop=True).to_parquet(f"{CD}/{name}.parquet")
def evaluate(df):
    out = {}
    for sp, g in df.groupby("split"):
        rs = []
        for f, h in g.groupby("fold"):
            io = h[h.role == "inner_oof"]; ot = h[h.role == "outer_test"]
            t, _ = O.pick_thr(yc[io.idx.values], io.p_critical.values)
            al = ot.p_critical.values >= t; yy = yc[ot.idx.values]
            rs.append((yy, al))
        y_ = np.concatenate([r[0] for r in rs]); a_ = np.concatenate([r[1] for r in rs])
        tp = int((a_ & (y_ == 1)).sum()); fa = int((a_ & (y_ == 0)).sum()); P = int(y_.sum()); mi = P - tp
        out[sp] = dict(recall=tp / P, FA=fa, missed=mi, cost_per_1000=(10 * mi + fa) / len(y_) * 1000, n=len(y_))
    e = [out[f"8020_s{s}"] for s in range(5)]
    return {"8020": {k: float(np.mean([x[k] for x in e])) for k in ["recall", "FA", "missed", "cost_per_1000"]}, "mission": out["mission"]}
key = ["split", "fold", "role", "inner_fold", "idx"]
base = D[MODELS[0]][key].sort_values(key).reset_index(drop=True)
P = {m: D[m].sort_values(key).reset_index(drop=True) for m in MODELS}
for m in MODELS: assert (P[m][key].values == base[key].values).all()
res = {}
for m in MODELS:
    res[m] = evaluate(P[m]); export("current_lgbm_all" if m == "lgbm_all" else f"ens_member_{m}", P[m])
# ensembles: per (split, fold): percentile rank vs inner_oof distribution; isotonic cross-fit on inner folds
rank = np.zeros(len(base)); cal = np.zeros(len(base))
for (sp, f), g in base.groupby(["split", "fold"]):
    io = g.index[g.role.values == "inner_oof"]; ot = g.index[g.role.values == "outer_test"]
    rk = []; ca = []
    for m in MODELS:
        p = P[m].p_critical.values; ref = np.sort(p[io])
        r = np.zeros(len(base)); r[io] = np.searchsorted(ref, p[io], "right") / len(ref)
        r[ot] = np.searchsorted(ref, p[ot], "right") / len(ref); rk.append(r)
        c = np.zeros(len(base)); yi = yc[base.idx.values]
        for j in range(5):
            tr_ = io[base.inner_fold.values[io] != j]; te_ = io[base.inner_fold.values[io] == j]
            c[te_] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr_], yi[tr_]).predict(p[te_])
        c[ot] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[io], yi[io]).predict(p[ot]); ca.append(c)
    idx = np.r_[io, ot]
    rank[idx] = np.mean([r[idx] for r in rk], 0); cal[idx] = np.mean([c[idx] for c in ca], 0)
for name, v in [("ens_rank", rank), ("ens_mean", cal)]:
    df = base.copy(); df["p_critical"] = v; res[name] = evaluate(df); export(name, df)
json.dump(res, open("/root/orbital-dodge/research/fa/ens/results.json", "w"), indent=1)
json.dump(res, sys.stdout, indent=0)
