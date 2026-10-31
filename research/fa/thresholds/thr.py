# Post-processing threshold strategies (no retraining). Groups' cut-offs chosen jointly & exactly on inner_oof
# (DP over total TP: min sum FP s.t. overall recall >= 0.90, objective 10*missed + FA), applied to outer_test.
import os, sys, glob, json, math, numpy as np, pandas as pd
R = 10; FLOOR = 0.90; SHR = 30
CANDS = "/root/orbital-dodge/research/fa/cands"; OUTD = "/root/orbital-dodge/research/fa/thresholds"
TN = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}

def toint(s):
    s = pd.Series(s)
    if s.dtype.kind in "iuf": return s.astype(int).values
    return s.astype(str).str.upper().map(TN).astype(int).values

def curve(s, yc):
    """Pareto points (tp, fp, thr) for alert = s >= thr, incl. never-alert (0,0,inf)."""
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; ys = yc[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); d = np.r_[ss[1:] != ss[:-1], True] if len(ss) else np.zeros(0, bool)
    tp, fp, th = np.r_[0, tp[d]], np.r_[0, fp[d]], np.r_[np.inf, ss[d]]
    best = {}
    for a, b, t in zip(tp, fp, th):
        if a not in best or b < best[a][0]: best[a] = (b, t)
    ks = sorted(best); pts = []; mfp = np.inf
    for k in reversed(ks):  # keep only points not dominated (more tp with <= fp)
        if best[k][0] < mfp: pts.append((k, best[k][0], best[k][1])); mfp = best[k][0]
    return pts[::-1]

def eff_curve(s, yc, cand_thr):
    """(tp, fp, thr) for an explicit list of thresholds."""
    pts = [(0, 0, np.inf)]
    for t in cand_thr:
        a = s >= t; pts.append((int((a & (yc == 1)).sum()), int((a & (yc == 0)).sum()), t))
    best = {}
    for a, b, t in pts:
        if a not in best or b < best[a][0]: best[a] = (b, t)
    return [(k, best[k][0], best[k][1]) for k in sorted(best)]

def solve(curves, P):
    """curves: list of point lists. returns chosen thr per group; exact min R*(P-TP)+FP s.t. TP >= ceil(.9P)."""
    INF = 1e18; dp = np.full(P + 1, INF); dp[0] = 0; arg = []
    for pts in curves:
        nd = np.full(P + 1, INF); ch = np.full(P + 1, -1)
        for j, (tp, fp, _) in enumerate(pts):
            cand = np.full(P + 1, INF); cand[tp:] = dp[:P + 1 - tp] + fp
            m = cand < nd; nd[m] = cand[m]; ch[m] = j
        arg.append(ch); dp = nd
    need = math.ceil(FLOOR * P - 1e-9); T = np.arange(P + 1)
    obj = np.where(T >= need, R * (P - T) + dp, INF); Tb = int(np.argmin(obj))
    thr = []
    for g in range(len(curves) - 1, -1, -1):
        j = arg[g][Tb]; tp, fp, t = curves[g][j]; thr.append(t); Tb -= tp
    return thr[::-1]

def grp_thr(inn, keys, extra=None):
    """Mondrian: groups = unique values of keys on inner set. returns dict key->thr."""
    yc = (inn.y_tier.values == 2).astype(int); s = inn.p_critical.values; P = int(yc.sum())
    gs = sorted(set(keys)); curves = []
    for g in gs:
        m = keys == g
        curves.append(extra(g, s[m], yc[m]) if extra else curve(s[m], yc[m]))
    return dict(zip(gs, solve(curves, P)))

def strategies(inn):
    out = {}
    lt = inn.latest_tier.values; ct = inn.c_object_type.values; yc = (inn.y_tier.values == 2).astype(int); s = inn.p_critical.values
    out["global"] = ("all", grp_thr(inn, np.zeros(len(inn), int)))
    tierT = grp_thr(inn, lt); out["mondrian_tier"] = ("tier", tierT)
    out["two_signal"] = ("crit", grp_thr(inn, (lt == 2).astype(int)))
    # tier x object type with shrinkage to tier cut-off (weight n_crit/30 on the logit scale) when cell has < 30 criticals
    cell = lt * 100 + ct
    def ext(g, ss, yy):
        tt = tierT.get(g // 100, np.inf); nc = int(yy.sum())
        if nc >= SHR: return curve(ss, yy)
        w = nc / SHR
        if not np.isfinite(tt): return eff_curve(ss, yy, [np.inf])
        lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        base = lg(tt); cands = np.unique(np.r_[ss, tt])
        eff = 1 / (1 + np.exp(-(base + w * (lg(cands) - base))))
        return eff_curve(ss, yy, np.unique(eff))
    out["mondrian_tier_obj"] = ("cell", grp_thr(inn, cell, ext), tierT)
    return out

def apply(te, kind, T, tierT=None):
    lt = te.latest_tier.values; s = te.p_critical.values
    if kind == "all": t = np.full(len(te), T[0])
    elif kind == "tier": t = np.array([T.get(k, np.inf) for k in lt])
    elif kind == "crit": t = np.array([T.get(int(k == 2), np.inf) for k in lt])
    else:
        cell = lt * 100 + te.c_object_type.values
        t = np.array([T.get(c, tierT.get(c // 100, np.inf)) for c in cell])
    return s >= t

def score(te, alert):
    yc = te.y_tier.values == 2; miss = int((yc & ~alert).sum()); fa = int((~yc & alert).sum())
    return dict(n=len(te), n_crit=int(yc.sum()), recall=1 - miss / max(yc.sum(), 1), missed=miss, fa=fa, cost=R * miss + fa)

def run(path, write=True):
    base = os.path.basename(path)[:-8]
    df = pd.read_parquet(path)
    df["latest_tier"] = toint(df.latest_tier); df["y_tier"] = toint(df.y_tier)
    co = df.c_object_type.astype(str).fillna("NA"); cats = {c: i for i, c in enumerate(sorted(co.unique()))}
    df["c_object_type_orig"] = df.c_object_type; df["c_object_type"] = co.map(cats).astype(int)
    res = {}; outs = {}
    for (sp, fo), g in df.groupby(["split", "fold"]):
        inn = g[g.role == "inner_oof"]; te = g[g.role == "outer_test"].copy()
        for name, st in strategies(inn).items():
            kind, T = st[0], st[1]; tierT = st[2] if len(st) > 2 else None
            al = apply(te, kind, T, tierT)
            res.setdefault(name, {})[f"{sp}|{fo}"] = dict(score(te, al), thr={str(k): float(v) for k, v in T.items()})
            t2 = te.copy(); t2["alert"] = al
            outs.setdefault(name, []).append(pd.concat([inn.assign(alert=np.nan), t2]))
    summ = {}
    for name, r in res.items():
        e8 = [v for k, v in r.items() if k.startswith("8020")]; mi = [v for k, v in r.items() if k.startswith("mission")]
        n8 = np.mean([v["n"] for v in e8]); nm = sum(v["n"] for v in mi)
        summ[name] = {
            "8020": {"recall": float(np.mean([v["recall"] for v in e8])), "fa": float(np.mean([v["fa"] for v in e8])),
                     "missed": float(np.mean([v["missed"] for v in e8])),
                     "cost_per_1000": float(np.mean([v["cost"] / v["n"] * 1000 for v in e8]))},
            "mission": {"recall": 1 - sum(v["missed"] for v in mi) / sum(v["n_crit"] for v in mi), "fa": int(sum(v["fa"] for v in mi)),
                        "missed": int(sum(v["missed"] for v in mi)), "cost_per_1000": sum(v["cost"] for v in mi) / nm * 1000},
            "per_fold": r}
        if write and name != "global":
            o = pd.concat(outs[name], ignore_index=True)
            o["alert"] = o["alert"].astype("boolean"); o["c_object_type"] = o.pop("c_object_type_orig")
            o.to_parquet(os.path.join(CANDS, f"{base}__{name}.parquet"))
    return base, summ

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(CANDS, "*.parquet")) if "__" not in os.path.basename(p))
    allr = {}
    sp = os.path.join(OUTD, "summary.json")
    if os.path.exists(sp): allr = json.load(open(sp))
    for p in paths:
        try:
            b, s = run(p); allr[b] = s
            print(b, {k: (round(v["8020"]["recall"], 3), round(v["8020"]["fa"], 1), round(v["8020"]["cost_per_1000"], 1),
                          "|M", round(v["mission"]["recall"], 3), v["mission"]["fa"], round(v["mission"]["cost_per_1000"], 1)) for k, v in s.items()}, flush=True)
        except Exception as e:
            print("FAIL", p, repr(e), flush=True)
        json.dump(allr, open(sp, "w"), indent=1, default=float)
