"""Bake-off scorer (pre-registered; see PREREG.md).

usage: .venv/bin/python score.py [--models m1,m2] [--splits event,mission] [--B 1000] [--out leaderboard.md]
Reads oof/<model>_<split>.parquet (event_key, p_safe, p_elevated, p_critical). persistence must exist for each split.
Writes <out> (markdown) and <out minus .md>.json.
"""
import sys, os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import harness as H
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.isotonic import IsotonicRegression

BOOT_SEED = 20261031


def recall_at_budget(p, yc, budget, w=None):
    """max critical recall s.t. weighted false alarms <= budget, alerting on p >= threshold (tied scores move together).
    returns (recall, threshold, false_alarms, alerts)"""
    w = np.ones(len(p)) if w is None else w
    o = np.argsort(-p, kind="mergesort"); ps = p[o]
    ctp = np.cumsum(w[o] * yc[o]); cfp = np.cumsum(w[o] * (1 - yc[o])); cn = np.cumsum(w[o])
    ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]
    ok = ends[cfp[ends] <= budget + 1e-9]
    pos = (w * yc).sum()
    if len(ok) == 0 or pos == 0:
        return 0.0, np.inf, 0.0, 0.0
    e = ok[-1]
    return ctp[e] / pos, ps[e], cfp[e], cn[e]


def ece(y, p, bins=10, w=None):
    w = np.ones(len(p)) if w is None else w
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    tot = w.sum(); e = 0.0
    for b in range(bins):
        m = idx == b
        if w[m].sum() > 0:
            e += w[m].sum() / tot * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return e


def isotonic_crossfit(p, yc, fold):
    out = np.zeros_like(p)
    for k in np.unique(fold):
        tr = fold != k
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], yc[tr])
        out[fold == k] = ir.predict(p[fold == k])
    return out


def at_threshold(p, yc, th):
    a = p >= th
    tp = int((a & (yc == 1)).sum()); fa = int((a & (yc == 0)).sum())
    return dict(recall=tp / max(yc.sum(), 1), precision=tp / max(a.sum(), 1), false_alarms=fa, alerts=int(a.sum()))


def load_oof(model, split, keys):
    p = os.path.join(H.OOF, f"{model}_{split}.parquet")
    d = pd.read_parquet(p)
    assert d.event_key.is_unique, f"{p}: duplicate event_key"
    missing = set(keys) - set(d.event_key)
    assert not missing, f"{p}: {len(missing)} eligible events missing"
    d = d.set_index("event_key").loc[keys]
    P = d[H.PCOLS].values.astype(float)
    assert np.all(np.isfinite(P)), f"{p}: non-finite probabilities"
    assert np.allclose(P.sum(1), 1, atol=1e-3), f"{p}: rows do not sum to 1"
    return P


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--splits", default="event,mission")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(H.HERE, "leaderboard.md"))
    a = ap.parse_args()
    ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values; yc = (y == 2).astype(int)
    n = len(y)
    pers_fa_ind = ((ev.latest_tier.values == 2) & (yc == 0)).astype(float)   # tier persistence false alarms
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(a.B)])  # shared, paired
    res = {"boot_seed": BOOT_SEED, "B": a.B, "n_events": int(n), "n_critical": int(yc.sum()), "splits": {}}
    for split in a.splits.split(","):
        fold = H.fold_ids(split)
        models = a.models.split(",") if a.models else sorted(
            os.path.basename(f)[: -len(f"_{split}.parquet")] for f in glob.glob(os.path.join(H.OOF, f"*_{split}.parquet")))
        if "persistence" not in models: models = ["persistence"] + models
        budget = float(pers_fa_ind.sum())
        budgets_b = W @ pers_fa_ind
        rows = {}; boot_primary = {}
        for m in models:
            P = load_oof(m, split, keys); pc = P[:, 2]
            prim, th, fa, al = recall_at_budget(pc, yc, budget)
            bp = np.array([recall_at_budget(pc, yc, budgets_b[b], W[b])[0] for b in range(a.B)])
            bauc = np.array([roc_auc_score(yc, pc, sample_weight=W[b]) for b in range(a.B)])
            bbr = np.array([np.average((pc - yc) ** 2, weights=W[b]) for b in range(a.B)])
            boot_primary[m] = bp
            piso = isotonic_crossfit(pc, yc, fold)
            r = dict(primary=prim, primary_ci=ci(bp), primary_threshold=float(th), primary_false_alarms=float(fa),
                     primary_alerts=float(al), auc=float(roc_auc_score(yc, pc)), auc_ci=ci(bauc),
                     ap=float(average_precision_score(yc, pc)), brier=float(np.mean((pc - yc) ** 2)), brier_ci=ci(bbr),
                     ece=float(ece(yc, pc)), ece_iso=float(ece(yc, piso)), brier_iso=float(np.mean((piso - yc) ** 2)),
                     macro_f1=float(f1_score(y, P.argmax(1), average="macro")),
                     at_0_50=at_threshold(pc, yc, 0.5), at_0_10=at_threshold(pc, yc, 0.10))
            rows[m] = r
        bpers = boot_primary["persistence"]
        pa = rows["persistence"]
        for m, r in rows.items():
            d = boot_primary[m] - bpers
            r["delta_primary_vs_persistence"] = r["primary"] - pa["primary"]
            r["delta_primary_ci"] = ci(d)
            r["p_delta_le_0"] = float((d <= 0).mean())
            sig = r["delta_primary_ci"][0] > 0
            noninf = (r["primary"] >= pa["primary"] - 0.01) and (r["auc_ci"][0] > pa["auc"])
            r["gate_pass"] = bool(m != "persistence" and (sig or noninf))
            r["gate_reason"] = "n/a (reference)" if m == "persistence" else (
                "dPRIMARY CI lo > 0" if sig else ("PRIMARY within 0.01 & AUC CI lo > persistence AUC" if noninf else "fails"))
            r["calibration_ok"] = r["ece_iso"] <= 0.03
        res["splits"][split] = {"fa_budget": budget, "models": rows}
    # decision (pre-registered): winner = highest PRIMARY on by-mission among non-persistence rows; tie -> AUC, then Brier
    if "mission" in res["splits"]:
        rows = res["splits"]["mission"]["models"]
        cand = [m for m in rows if m != "persistence"]
        if cand:
            win = sorted(cand, key=lambda m: (-round(rows[m]["primary"], 6), -rows[m]["auc"], rows[m]["brier"]))[0]
            res["decision"] = dict(winner=win, serve=rows[win]["gate_pass"], gate_reason=rows[win]["gate_reason"],
                                   calibration_warning=not rows[win]["calibration_ok"],
                                   verdict=(f"serve {win}" if rows[win]["gate_pass"] else
                                            f"{win} has the highest by-mission PRIMARY but does not pass the gate: "
                                            f"the product ranks with persistence"))
    write_md(res, a.out)
    json.dump(res, open(a.out[:-3] + ".json", "w"), indent=1)
    print(open(a.out).read())


def f3(x): return f"{x:.3f}"
def fci(x, c): return f"{x:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def write_md(res, out):
    L = [f"# Bake-off leaderboard ({os.path.basename(out)})", "",
         f"{res['n_events']} eligible events, {res['n_critical']} critical. Bootstrap B={res['B']} over events "
         f"(seed {res['boot_seed']}, same resamples for every model, so deltas are paired). PRIMARY = critical recall "
         "at <= the tier-persistence false-alarm count on the split (recomputed inside each bootstrap resample).", ""]
    for split, S in res["splits"].items():
        L += [f"## Split: by-{split}  (false-alarm budget = {S['fa_budget']:.0f})", "",
              "| model | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE | ECE iso | macro-F1 | @0.5 rec/prec/FA | @0.10 rec/prec/FA | gate |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in sorted(S["models"].items(), key=lambda kv: -kv[1]["primary"]):
            t5, t1 = r["at_0_50"], r["at_0_10"]
            gate = r["gate_reason"] if split == "mission" else ""
            L.append(f"| {m} | {fci(r['primary'], r['primary_ci'])} | {fci(r['delta_primary_vs_persistence'], r['delta_primary_ci'])} | "
                     f"{fci(r['auc'], r['auc_ci'])} | {f3(r['ap'])} | {fci(r['brier'], r['brier_ci'])} | {f3(r['ece'])} | "
                     f"{f3(r['ece_iso'])} | {f3(r['macro_f1'])} | {t5['recall']:.3f}/{t5['precision']:.3f}/{t5['false_alarms']} | "
                     f"{t1['recall']:.3f}/{t1['precision']:.3f}/{t1['false_alarms']} | {gate} |")
        L.append("")
    if "decision" in res:
        d = res["decision"]
        L += ["## Pre-registered decision (by-mission split)", "",
              f"- Highest by-mission PRIMARY (excluding persistence): **{d['winner']}**",
              f"- Gate: {'PASS' if d['serve'] else 'FAIL'} ({d['gate_reason']})",
              f"- Calibration (ECE after cross-fitted isotonic <= 0.03): {'warning' if d['calibration_warning'] else 'ok'}",
              f"- Verdict: {d['verdict']}", ""]
    open(out, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
