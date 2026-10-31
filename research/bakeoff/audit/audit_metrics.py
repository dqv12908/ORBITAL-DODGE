"""Independent metrics audit. Does NOT import score.py or harness.py. Recomputes labels from raw CSV."""
import numpy as np, pandas as pd, json, os, sys
from scipy.stats import rankdata
B_DIR = "/root/orbital-dodge/research/bakeoff"
RAW = "/root/orbital-dodge/data/raw/Challenge 1 Airbus /train_data.csv"
df = pd.read_csv(RAW, usecols=["event_id","mission_id","time_to_tca","risk","miss_distance"])
def tier(r, m):
    r = np.asarray(r, float); m = np.asarray(m, float)
    return np.where((r >= -4) | (m < 1000), 2, np.where((r >= -6) | (m <= 5000), 1, 0))
df = df.sort_values(["event_id","time_to_tca"], ascending=[True, False], kind="mergesort")
g = df.groupby("event_id")
final = g.tail(1).set_index("event_id")
ge2 = df[df.time_to_tca >= 2]
elig = sorted(set(final.index[final.time_to_tca < 1]) & set(ge2.event_id.unique()))
print("eligible", len(elig))
y = tier(final.loc[elig, "risk"], final.loc[elig, "miss_distance"])
print("tiers", np.bincount(y))
lat = ge2.groupby("event_id").tail(1).set_index("event_id").loc[elig]
# ties in time_to_tca at the latest >=2d CDM?
cnt_tie = ge2.groupby("event_id").apply(lambda d: (d.time_to_tca == d.time_to_tca.min()).sum()).loc[elig]
print("events with duplicated latest ttca:", int((cnt_tie > 1).sum()))
lt = tier(lat.risk, lat.miss_distance)
keys = np.array([f"train:{e}" for e in elig])
yc = (y == 2).astype(int); n = len(y); npos = yc.sum()
pers_fa = ((lt == 2) & (yc == 0)).astype(float)
budget = pers_fa.sum()
print("budget", budget, "pers alerts", (lt == 2).sum(), "pers TP", ((lt == 2) & (yc == 1)).sum(), "pers recall", ((lt==2)&(yc==1)).sum()/npos)
ev = pd.read_parquet(f"{B_DIR}/data/events.parquet")
print("events.parquet key match", (ev.event_key.values == keys).all(), "y match", (ev.y_tier.values == y).all(), "lt match", (ev.latest_tier.values == lt).all())
folds = pd.read_parquet(f"{B_DIR}/folds.parquet").set_index("event_key").loc[keys]
# independent fold check
from sklearn.model_selection import KFold, GroupKFold
fe = np.zeros(n, int); fm = np.zeros(n, int)
mid = final.loc[elig, "mission_id"].values.astype(int)
for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))): fe[te] = k
for k, (_, te) in enumerate(GroupKFold(5).split(np.arange(n), groups=mid)): fm[te] = k
print("folds match", (folds.fold_event.values == fe).all(), (folds.fold_mission.values == fm).all())
# missions not crossing folds
print("mission leak across mission folds:", int(pd.DataFrame({"m": mid, "f": fm}).groupby("m").f.nunique().max()))

def primary(p, w):
    """my own: sweep distinct thresholds descending; feasible if FA(w) <= budget(w); pick lowest feasible threshold."""
    bud = (w * pers_fa).sum(); pos = (w * yc).sum()
    u = np.unique(p)[::-1]
    best = (0.0, np.inf, 0.0)
    # vectorised via sorting
    o = np.argsort(-p); ps = p[o]; fa = np.cumsum(w[o] * (1 - yc[o])); tp = np.cumsum(w[o] * yc[o])
    last = np.r_[ps[1:] != ps[:-1], True]
    idx = np.nonzero(last)[0]
    feas = idx[fa[idx] <= bud + 1e-9]
    if len(feas) == 0: return 0.0, np.inf, 0.0
    e = feas[-1]
    # monotone: check all feasible ends form prefix
    assert np.all(np.diff(fa[idx]) >= -1e-12)
    return tp[e] / pos, ps[e], fa[e]

def primary_brute(p):
    """brute force over unique thresholds (unweighted) for cross-check"""
    best = None
    for t in np.unique(p):
        a = p >= t; fa = (a & (yc == 0)).sum()
        if fa <= budget:
            r = (a & (yc == 1)).sum() / npos
            if best is None or t < best[1]: best = (r, t, fa)
    return best

def auc(p, w):
    # weighted Mann-Whitney with ties = 0.5
    o = np.argsort(p, kind="mergesort"); ps = p[o]; ys = yc[o]; ws = w[o]
    # group ties
    uniq, inv = np.unique(ps, return_inverse=True)
    wpos = np.bincount(inv, weights=ws * ys); wneg = np.bincount(inv, weights=ws * (1 - ys))
    cneg = np.cumsum(wneg) - wneg
    num = (wpos * (cneg + 0.5 * wneg)).sum()
    return num / (wpos.sum() * wneg.sum())

rng = np.random.default_rng(20261031)
B = 1000
W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(B)])
# alternative independent bootstrap (different seed) for robustness of conclusions
rng2 = np.random.default_rng(12345)
W2 = np.stack([np.bincount(rng2.integers(0, n, n), minlength=n).astype(float) for _ in range(B)])
ones = np.ones(n)
models = ["persistence", "lgbm_all", "tabpfn", "futures", "ebm", "gru", "lgbm_a13", "hgb_a13"]
out = {}
for split in ["event", "mission"]:
    out[split] = {}
    bp = {}; bp2 = {}
    for m in models:
        d = pd.read_parquet(f"{B_DIR}/oof/{m}_{split}.parquet")
        assert d.event_key.is_unique and set(d.event_key) == set(keys)
        d = d.set_index("event_key").loc[keys]
        P = d[["p_safe","p_elevated","p_critical"]].values.astype(float)
        p = P[:, 2]
        prim, th, fa = primary(p, ones)
        bf = primary_brute(p)
        assert abs(bf[0] - prim) < 1e-12 and bf[2] == fa, (m, bf, prim, fa)
        a = auc(p, ones); br = np.mean((p - yc) ** 2)
        bpr = np.array([primary(p, W[b])[0] for b in range(B)])
        bau = np.array([auc(p, W[b]) for b in range(B)])
        bbr = np.array([np.average((p - yc) ** 2, weights=W[b]) for b in range(B)])
        bp[m] = bpr
        bp2[m] = np.array([primary(p, W2[b])[0] for b in range(B)])
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        extra = {}
        if "fold" in d.columns or any(c.startswith("fold") for c in d.columns):
            extra["fold_cols"] = [c for c in d.columns if c.startswith("fold")]
        out[split][m] = dict(primary=prim, tp=int(round(prim * npos)), thr=float(th), fa=float(fa), auc=a, auc_ci=ci(bau),
                             brier=br, brier_ci=ci(bbr), primary_ci=ci(bpr), cols=list(d.columns), rowsum_dev=float(np.abs(P.sum(1)-1).max()),
                             pmin=float(P.min()), pmax=float(P.max()), **extra)
    for m in models:
        dd = bp[m] - bp["persistence"]
        out[split][m]["dprimary"] = out[split][m]["primary"] - out[split]["persistence"]["primary"]
        out[split][m]["dprimary_ci"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        out[split][m]["dprimary_ci_sorted_24_25"] = [float(np.sort(dd)[24]), float(np.sort(dd)[25])]
        out[split][m]["frac_d_le0"] = float((dd <= 0).mean())
        dd2 = bp2[m] - bp2["persistence"]
        out[split][m]["dprimary_ci_altseed"] = [float(np.percentile(dd2, 2.5)), float(np.percentile(dd2, 97.5))]
        # unpaired (wrong) CI for contrast
        rng3 = np.random.default_rng(7); perm = rng3.permutation(B)
        out[split][m]["dprimary_ci_UNPAIRED"] = [float(np.percentile(bp[m] - bp["persistence"][perm], 2.5)), float(np.percentile(bp[m] - bp["persistence"][perm], 97.5))]
    # persistence p_critical ordering vs tier rule
    d = pd.read_parquet(f"{B_DIR}/oof/persistence_{split}.parquet").set_index("event_key").loc[keys]
    pc = d.p_critical.values
    print(split, "persistence p_crit min over lt==2:", pc[lt == 2].min(), "max over lt<2:", pc[lt < 2].max())
json.dump(out, open(f"{B_DIR}/audit/audit_metrics.json", "w"), indent=1)
for split in out:
    print("==", split)
    for m, r in sorted(out[split].items(), key=lambda kv: -kv[1]["primary"]):
        print(f"{m:12s} PRIM {r['primary']:.4f} ({r['tp']}/551) thr {r['thr']:.4f} FA {r['fa']:.0f} CI {np.round(r['primary_ci'],4)} AUC {r['auc']:.4f} {np.round(r['auc_ci'],4)} Brier {r['brier']:.5f} {np.round(r['brier_ci'],4)} d {r['dprimary']:+.4f} dCI {np.round(r['dprimary_ci'],4)} s24/25 {r['dprimary_ci_sorted_24_25']} alt {np.round(r['dprimary_ci_altseed'],4)} unpaired {np.round(r['dprimary_ci_UNPAIRED'],4)} le0 {r['frac_d_le0']:.3f} dev {r['rowsum_dev']:.1e} cols {r['cols']}")
