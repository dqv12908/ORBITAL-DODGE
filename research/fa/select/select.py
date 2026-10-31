import glob, json, os, numpy as np, pandas as pd
R, FLOOR, B = 10, 0.90, 500
CD = "/root/orbital-dodge/research/fa/cands"; OUT = "/root/orbital-dodge/research/fa/select"
SPL = [f"8020_s{s}" for s in range(5)]

def pick_thr(yc, s):  # same rule as organizer.py
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; ys = yc[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    d = np.r_[ss[1:] != ss[:-1], True]; tp, fp, ts = tp[d], fp[d], ss[d]
    rec = tp / P; cost = R * (P - tp) + fp
    i = int(np.argmin(np.where(rec >= FLOOR, cost, np.inf))); return float(ts[i])

def thr_for_recall(yc, s, target):  # min FA with recall >= target (diagnostic, uses outer labels)
    o = np.argsort(-s, kind="mergesort"); ss = s[o]; ys = yc[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    d = np.r_[ss[1:] != ss[:-1], True]; tp, fp = tp[d], fp[d]
    ok = tp / P >= target - 1e-12; return int(fp[ok].min())

def ek(x):
    x = str(x); return int(x.split(":")[-1])

def load(f):
    d = pd.read_parquet(f); d["ek"] = d.event_key.map(ek); d["yc"] = (d.y_tier == 2).astype(int)
    d.loc[d.split.str.startswith("8020"), "fold"] = 0  # phys cands store seed in fold
    has_alert = "alert" in d.columns
    out = {}
    for (sp, fo), g in d.groupby(["split", "fold"]):
        te = g[g.role == "outer_test"].copy()
        if has_alert:
            te["al"] = te.alert.astype(bool)
        else:
            io = g[g.role == "inner_oof"]
            t = pick_thr(io.yc.values, io.p_critical.values.astype(float)); te["al"] = te.p_critical.values >= t
        te["split"] = sp; out[(sp, fo)] = te
    te = pd.concat(out.values()); te["has_alert"] = has_alert
    if {"p_safe", "p_elevated"} <= set(te.columns) and te.p_safe.notna().all():
        te["pred"] = np.where(te.al, 2, np.where(te.p_elevated > te.p_safe, 1, 0))
    else: te["pred"] = np.nan
    return te[["ek", "split", "fold", "yc", "y_tier", "al", "pred", "p_critical", "final_sub", "has_alert"]]

def mf1(y, p):
    f = []
    for k in range(3):
        tp = ((y == k) & (p == k)).sum(); fp = ((y != k) & (p == k)).sum(); fn = ((y == k) & (p != k)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))

def mets(te):
    y = te.yc.values; a = te.al.values
    tp = int((y & a).sum()); fa = int((~y.astype(bool) & a).sum()); miss = int((y & ~a).sum()); n = len(y)
    return dict(recall=tp / max(y.sum(), 1), fa=fa, missed=miss, cost=(R * miss + fa) / n * 1000,
                precision=tp / max(tp + fa, 1), macro_f1=(mf1(te.y_tier.values, te.pred.values.astype(int)) if te.pred.notna().all() else None), n=n)

files = sorted(glob.glob(f"{CD}/*.parquet")); C = {os.path.basename(f)[:-8]: load(f) for f in files}
REF = "current_lgbm_all"; rng = np.random.default_rng(0)
# aligned arrays per split
def arr(name, sp):
    t = C[name]; t = t[t.split == sp].sort_values("ek"); return t
W = {}
for sp in SPL + ["mission"]:
    n = len(arr(REF, sp)); W[sp] = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float32)
def boot(name, sp):
    t = arr(name, sp); y = t.yc.values.astype(np.float32); a = t.al.values.astype(np.float32); w = W[sp]
    tp = w @ (y * a); fa = w @ ((1 - y) * a); miss = w @ (y * (1 - a)); n = len(y)
    return dict(recall=tp / np.maximum(tp + miss, 1), fa=fa, cost=(R * miss + fa) / n * 1000)
def agg_boot(name, view):
    sps = SPL if view == "8020" else ["mission"]; bs = [boot(name, s) for s in sps]
    return {k: np.mean([b[k] for b in bs], axis=0) for k in ["recall", "fa", "cost"]}
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

res = {}; refb = {v: agg_boot(REF, v) for v in ["8020", "mission"]}
for name, te in C.items():
    r = {"has_alert": bool(te.has_alert.iloc[0])}
    for s_ in SPL + ["mission"]:
        a1, a2 = arr(name, s_), arr(REF, s_); assert (a1.ek.values == a2.ek.values).all() and (a1.yc.values == a2.yc.values).all(), (name, s_)
    # ensure same event sets as ref
    for v in ["8020", "mission"]:
        sps = SPL if v == "8020" else ["mission"]
        ms = [mets(te[te.split == s]) for s in sps]
        m = {k: (float(np.mean([x[k] for x in ms])) if ms[0][k] is not None else None) for k in ms[0]}
        b = agg_boot(name, v)
        m["ci"] = {k: ci(b[k]) for k in ["recall", "fa", "cost"]}
        m["paired_diff_vs_current"] = {k: {"mean": float(np.mean(b[k] - refb[v][k])), "ci": ci(b[k] - refb[v][k])} for k in ["fa", "cost", "recall"]}
        # FA at matched recall (per split/fold, target = current's outer recall there)
        if not r["has_alert"]:
            fas = []
            for s in sps:
                for fo in sorted(te[te.split == s].fold.unique()):
                    g = te[(te.split == s) & (te.fold == fo)]; gr = C[REF][(C[REF].split == s) & (C[REF].fold == fo)]
                    tgt = (gr.yc & gr.al).sum() / gr.yc.sum(); fas.append(thr_for_recall(g.yc.values, g.p_critical.values, tgt))
            m["fa_at_matched_recall"] = float(np.mean(fas)) if v == "8020" else float(np.sum(fas))
        else: m["fa_at_matched_recall"] = None
        sub = {}
        for fs, g in te[te.split.isin(sps)].groupby("final_sub"):
            gs = [mets(g[g.split == s]) for s in sps]
            sub[fs] = {k: float(np.mean([x[k] for x in gs])) for k in ["recall", "fa", "missed", "cost", "n"]}
        m["by_final_sub"] = sub; r[v] = m
    res[name] = r

# winner rule
cur = res[REF]["mission"]
elig = {k: v for k, v in res.items() if v["mission"]["recall"] >= FLOOR}
best = min(elig, key=lambda k: elig[k]["mission"]["cost"])
if best == REF: winner, why = REF, "current model already has the lowest by-mission cost"
else:
    b = res[best]["mission"]; lo, hi = b["paired_diff_vs_current"]["cost"]["ci"]
    if hi < 0: winner, why = best, "paired cost CI excludes 0"
    elif b["fa"] < cur["fa"] and b["recall"] >= cur["recall"] - 0.01: winner, why = best, "cost CI includes 0 but FA lower and recall within 0.01"
    else: winner, why = REF, f"best-cost candidate {best} not significantly better and fails FA/recall tie-break"
f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
rows = ["| candidate | 80/20 recall | 80/20 FA [95% CI] | 80/20 cost | dFA vs cur [CI] | dCost vs cur [CI] | FA@matched rec | mission recall | mission FA [CI] | mission missed | mission cost | dFA [CI] | dCost [CI] | FA@matched | macro-F1 (mission) |", "|" + "---|" * 15]
for name in sorted(res, key=lambda k: res[k]["mission"]["cost"]):
    a, m = res[name]["8020"], res[name]["mission"]
    pd8, pdm = a["paired_diff_vs_current"], m["paired_diff_vs_current"]
    rows.append(f"| {name}{' **(winner)**' if name == winner else ''} | {a['recall']:.3f} | {a['fa']:.1f} [{a['ci']['fa'][0]:.0f},{a['ci']['fa'][1]:.0f}] | {a['cost']:.1f} | {pd8['fa']['mean']:+.1f} [{pd8['fa']['ci'][0]:+.1f},{pd8['fa']['ci'][1]:+.1f}] | {pd8['cost']['mean']:+.1f} [{pd8['cost']['ci'][0]:+.1f},{pd8['cost']['ci'][1]:+.1f}] | {f(a['fa_at_matched_recall'])} | {m['recall']:.3f} | {m['fa']:.0f} [{m['ci']['fa'][0]:.0f},{m['ci']['fa'][1]:.0f}] | {m['missed']:.0f} | {m['cost']:.1f} | {pdm['fa']['mean']:+.0f} [{pdm['fa']['ci'][0]:+.0f},{pdm['fa']['ci'][1]:+.0f}] | {pdm['cost']['mean']:+.1f} [{pdm['cost']['ci'][0]:+.1f},{pdm['cost']['ci'][1]:+.1f}] | {f(m['fa_at_matched_recall'],0)} | {f(m['macro_f1'],3)} |")
tbl = "\n".join(rows)
sub_rows = ["| candidate | split | final_sub | n | recall | FA | missed | cost/1000 |", "|---|---|---|---|---|---|---|---|"]
for name in dict.fromkeys([REF, winner, "lgbm_all_phys", "ens_member_rf_all_crit_bsub"]):
    if name not in res: continue
    for v in ["8020", "mission"]:
        for fs, s in res[name][v]["by_final_sub"].items():
            sub_rows.append(f"| {name} | {v} | {fs} | {s['n']:.0f} | {s['recall']:.3f} | {s['fa']:.1f} | {s['missed']:.1f} | {s['cost']:.1f} |")
out = {"winner": winner, "winner_reason": why, "best_cost_candidate": best, "rule": "lowest by-mission cost (R=10) with recall>=0.90; if paired cost-diff CI includes 0 prefer only if FA lower and recall within 0.01 of current; else keep current",
       "table_markdown": tbl, "subpop_markdown": "\n".join(sub_rows), "results": res}
json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print(tbl); print(); print("\n".join(sub_rows)); print(winner, why, best)
