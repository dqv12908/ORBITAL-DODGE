"""Statistics curves for the UI's ggplot-style plots (read-only, additive router).

GET /api/eval/curves
    ROC, precision-recall, reliability (10 bins + Clopper-Pearson 95 % CI), expected cost vs threshold (R = 10),
    density of P(critical) by true tier, and the archive miss-distance histogram for events whose final CDM is
    at most 1 d before TCA. Computed from the served forecaster's OUT-OF-FOLD predictions on train events only:
      registry/forecaster_cost/1.0.0/predictions.parquet  ->  registry/forecaster_bakeoff/predictions.parquet
      ->  registry/tier_forecaster/1.0.0/oof.parquet.
    Labels: data/processed/events.parquet final_tier (CRITICAL vs not).

GET /api/conjunctions/{event_key}/rtn
    Relative position R/T/N of the RELEASED CDMs of a cast event with the combined 1-sigma (target + chaser,
    sqrt(st^2 + sc^2)) per component, for the RTN-vs-time plot. Unreleased CDMs are never sent.
"""
from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path

import numpy as np

from fastapi import APIRouter, Request

from app.api.common import jresp
from app.errors import ApiError
from app.state import ctx
from app.util import clean_json, fnum

router = APIRouter(prefix="/api")
KEY_RX = re.compile(r"^(train|test|adhoc):[0-9]+$")
R_COST = 10.0          # cost ratio: one missed critical = R false alarms
RECALL_FLOOR = 0.90
P_FLOOR = 1e-4         # density / log axis floor for P(critical)

_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────── loading
def _source(repo: Path):
    """First available OOF prediction file, as (name, path, column picker)."""
    cands = [
        ("forecaster_cost", repo / "registry" / "forecaster_cost" / "1.0.0" / "predictions.parquet"),
        ("forecaster_bakeoff", repo / "registry" / "forecaster_bakeoff" / "predictions.parquet"),
        ("tier_forecaster", repo / "registry" / "tier_forecaster" / "1.0.0" / "oof.parquet"),
    ]
    for name, p in cands:
        if p.exists():
            return name, p
    return None, None


def _load(repo: Path, processed: Path):
    import pandas as pd

    name, p = _source(repo)
    if p is None:
        raise ApiError(404, "NOT_FOUND", "no out-of-fold predictions in the registry",
                       "expected registry/forecaster_cost, forecaster_bakeoff or tier_forecaster")
    d = pd.read_parquet(p)
    note = ""
    if name == "tier_forecaster":
        col = "p_critical_event" if "p_critical_event" in d else "p_critical_mission"
        d = d.assign(p=d[col])
        note = f"{col} (OOF by event)"
    else:
        if "split" in d:
            oof = d["split"].astype(str).str.contains("oof|outer|test_fold|nested", regex=True) & \
                d["split"].astype(str).str.startswith("train")
            d = d[oof]
            note = "split " + ", ".join(sorted(d["split"].astype(str).unique()))
        col = "p_critical_calibrated" if "p_critical_calibrated" in d else "p_critical"
        d = d.assign(p=d[col])
        note = f"{col}; {note}".strip("; ")
    d = d[d["event_key"].astype(str).str.startswith("train:")][["event_key", "p"]].dropna()
    ev = pd.read_parquet(processed / "events.parquet",
                         columns=["event_key", "split", "final_tier", "persistence_tier", "final_miss_m", "final_ttca"])
    tr = ev[ev["split"] == "train"]
    m = d.merge(tr[["event_key", "final_tier", "persistence_tier"]], on="event_key", how="inner")
    m = m[m["final_tier"].notna()]
    model = None
    try:
        import json
        card = json.loads((p.parent / "model_card.json").read_text()) if (p.parent / "model_card.json").exists() else {}
        model = card.get("served_model") or card.get("name")
    except Exception:
        pass
    return name, p, note, model, m, tr


def _key(repo: Path, processed: Path):
    _, p = _source(repo)
    files = [p, processed / "events.parquet"]
    return tuple((str(f), f.stat().st_mtime) for f in files if f is not None and f.exists())


# ─────────────────────────────────────────────────────────────── maths
def _roc(y, s):
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    P, N = y.sum(), (1 - y).sum()
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.diff(s) != 0, True]          # distinct thresholds only
    tpr = np.r_[0, tp[last] / P]
    fpr = np.r_[0, fp[last] / N]
    thr = np.r_[np.inf, s[last]]
    prec = np.r_[1.0, tp[last] / np.maximum(tp[last] + fp[last], 1)]
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, prec, thr, auc


def _thin(*cols, n=300):
    k = len(cols[0])
    if k <= n:
        return cols
    idx = np.unique(np.r_[np.linspace(0, k - 1, n).round().astype(int), k - 1])
    return tuple(c[idx] for c in cols)


def _cp(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def _kde(x, grid):
    """Gaussian KDE with R's bw.nrd0 (ggplot2 geom_density default)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.zeros_like(grid), None
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lo = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = 0.9 * (lo if lo > 0 else (abs(x[0]) or 1.0)) * n ** -0.2
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (n * bw * math.sqrt(2 * math.pi)), bw


def _compute(repo: Path, processed: Path) -> dict:
    name, path, note, model, m, tr = _load(repo, processed)
    y = (m["final_tier"] == "CRITICAL").to_numpy().astype(int)
    s = m["p"].to_numpy().astype(float)
    n, npos = len(y), int(y.sum())
    base_rate = npos / n if n else None
    fpr, tpr, prec, thr, auc = _roc(y, s)

    # persistence (tier of the latest CDM at >= 2 d) as one operating point
    pers = (m["persistence_tier"] == "CRITICAL").to_numpy().astype(int)
    ptp, pfp = int((pers & y).sum()), int((pers & (1 - y)).sum())
    persistence = {"fpr": pfp / max(n - npos, 1), "tpr": ptp / max(npos, 1),
                   "precision": ptp / max(ptp + pfp, 1), "recall": ptp / max(npos, 1),
                   "false_alarms": pfp, "missed": npos - ptp, "cost": R_COST * (npos - ptp) + pfp}

    rf, rt, rth = _thin(fpr, tpr, thr)
    roc = [{"fpr": float(a), "tpr": float(b), "threshold": (None if not np.isfinite(c) else float(c))}
           for a, b, c in zip(rf, rt, rth)]
    rec = tpr
    pr_rec, pr_prec = _thin(rec[1:], prec[1:])
    pr = [{"recall": float(a), "precision": float(b)} for a, b in zip(pr_rec, pr_prec)]
    ap = float(np.sum(np.diff(np.r_[0, rec[1:]]) * prec[1:]))

    # reliability: 10 equal-width bins
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(s, edges[1:-1]), 0, 9)
    rel = []
    for i in range(10):
        msk = b == i
        k, nn = int(y[msk].sum()), int(msk.sum())
        if nn == 0:
            rel.append({"bin": i, "lo": edges[i], "hi": edges[i + 1], "n": 0})
            continue
        lo, hi = _cp(k, nn)
        rel.append({"bin": i, "lo": float(edges[i]), "hi": float(edges[i + 1]), "n": nn, "k": k,
                    "mean_pred": float(s[msk].mean()), "obs": k / nn, "ci_lo": lo, "ci_hi": hi})
    brier = float(np.mean((s - y) ** 2))
    ece = float(sum(r["n"] * abs(r["obs"] - r["mean_pred"]) for r in rel if r["n"]) / n)

    # expected cost vs threshold (alert if p >= t)
    ts = np.unique(np.r_[np.linspace(0.0, 1.0, 201), np.quantile(s, np.linspace(0, 1, 101))])
    ts = ts[(ts > 0) & (ts < 1)]
    cost = []
    for t in ts:
        a = s >= t
        tp_, fp_ = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
        fn_ = npos - tp_
        cost.append({"threshold": float(t), "cost": R_COST * fn_ + fp_, "missed": fn_, "false_alarms": fp_,
                     "recall": tp_ / max(npos, 1)})
    ok = [c for c in cost if c["recall"] >= RECALL_FLOOR]
    best_any = min(cost, key=lambda c: c["cost"]) if cost else None
    chosen = min(ok, key=lambda c: c["cost"]) if ok else best_any
    floor_max_t = max((c["threshold"] for c in ok), default=None)

    # density of P(critical) on log10 axis, by true tier
    lx = np.log10(np.clip(s, P_FLOOR, 1.0))
    grid = np.linspace(math.log10(P_FLOOR), 0, 160)
    dens = []
    counts = {}
    for tier in ("SAFE", "ELEVATED", "CRITICAL"):
        msk = (m["final_tier"] == tier).to_numpy()
        counts[tier] = int(msk.sum())
        dd, bw = _kde(lx[msk], grid)
        dens += [{"tier": tier, "log10_p": float(g), "density": float(v)} for g, v in zip(grid, dd)]
    n_floor = int((s < P_FLOOR).sum())

    # prediction histogram (10 bins, matches reliability)
    hist_pred = [{"lo": r["lo"], "hi": r["hi"], "n": r["n"]} for r in rel]

    # archive miss-distance histogram: train events whose final CDM is <= 1 d before TCA
    a = tr[(tr["final_ttca"] <= 1.0) & tr["final_miss_m"].notna() & (tr["final_miss_m"] > 0)]
    lm = np.log10(a["final_miss_m"].to_numpy() / 1000.0)
    he = np.linspace(-2.0, 2.0, 41)
    hc, _ = np.histogram(np.clip(lm, -2.0, 2.0 - 1e-9), bins=he)
    miss_hist = [{"lo_log10_km": float(he[i]), "hi_log10_km": float(he[i + 1]), "n": int(hc[i])} for i in range(len(hc))]
    tiers_final = a["final_tier"].value_counts().to_dict()

    return {
        "source": {"name": name, "file": str(path.relative_to(repo)), "column": note, "model": model,
                   "labels": "data/processed/events.parquet final_tier (CRITICAL = final risk >= -4 or miss < 1 km)",
                   "scope": "train events only, out-of-fold predictions"},
        "n": n, "n_critical": npos, "base_rate": base_rate, "auc": auc, "ap": ap, "brier": brier, "ece": ece,
        "roc": roc, "pr": pr, "persistence": persistence, "reliability": rel, "pred_hist": hist_pred,
        "cost": {"R": R_COST, "recall_floor": RECALL_FLOOR, "curve": cost, "chosen": chosen, "best_unconstrained": best_any,
                 "floor_max_threshold": floor_max_t},
        "density": {"floor": P_FLOOR, "n_below_floor": n_floor, "counts": counts, "points": dens,
                    "bandwidth": "bw.nrd0 on log10 scale"},
        "miss_hist": {"bins": miss_hist, "n": int(len(a)), "window": "final CDM time to TCA <= 1 d (train events)",
                      "final_tiers": {str(k): int(v) for k, v in tiers_final.items()}},
    }


@router.get("/eval/curves")
async def eval_curves(request: Request):
    s = ctx(request).settings
    repo, processed = s.repo_dir, s.processed_dir
    k = _key(repo, processed)
    if _CACHE.get("key") != k:
        _CACHE["val"] = await asyncio.to_thread(_compute, repo, processed)
        _CACHE["key"] = k
    return jresp(clean_json(_CACHE["val"]))


@router.get("/conjunctions/{event_key}/rtn")
async def rtn(event_key: str, request: Request):
    if not KEY_RX.match(event_key):
        raise ApiError(400, "BAD_REQUEST", f"bad event key {event_key!r}", "format split:event_id, e.g. train:2563")
    sim = ctx(request).sim
    ev = sim.events.get(event_key)
    if ev is None:
        raise ApiError(404, "NOT_FOUND", f"{event_key} is not in the current scenario cast")
    out = []
    for r in ev.released:
        row = r.get("row") or (ev.rows[r["cdm_index"]] if r.get("cdm_index") is not None else {})
        pos = [fnum(row.get(f"relative_position_{a}")) for a in "rtn"]
        st = [fnum(row.get(f"t_sigma_{a}")) for a in "rtn"]
        sc = [fnum(row.get(f"c_sigma_{a}")) for a in "rtn"]
        comb = []
        for a, b in zip(st, sc):
            # default covariances (>= 1e7 m) are not sigmas: no error bar
            ok = a is not None and b is not None and a < 1e7 and b < 1e7
            comb.append(math.sqrt(a * a + b * b) if ok else None)
        out.append({"cdm_index": r["cdm_index"], "ttca_d": r.get("ttca_d"), "rtn_m": pos,
                    "sigma_t_m": st, "sigma_c_m": sc, "sigma_comb_m": comb})
    return jresp(clean_json({"event_key": event_key, "frame": "RTN of the target (CDM relative_position_r/t/n)",
                             "sigma": "combined 1-sigma = sqrt(target^2 + chaser^2) per axis (correlations ignored)",
                             "cdms": out}))
