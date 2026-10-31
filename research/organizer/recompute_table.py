"""Recompute the README §6.2 headline table (held-out missions) from the committed OOF file, without retraining.

    server/.venv/bin/python research/organizer/recompute_table.py

Input: registry/forecaster_organizer/1.0.0/oof_organizer.parquet (11,942 organizer-usable events, one row per event,
out-of-fold predictions from GroupKFold(5) over missions; `thr_mission` is the threshold each fold chose inside its own
training data). Point estimates only; the bootstrap CIs in the README come from research/organizer/results.json.
Also prints the threshold variants discussed in README §5.5 / §9 (global 0.0634, pooled optimum, watch tau 0.20).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = 10
ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "registry/forecaster_organizer/1.0.0/oof_organizer.parquet"
TIERS = {"Safe": 0, "Elevated": 1, "Critical": 2}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    npos = int(y.sum()); n = len(y); r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for k in range(3):
        tp = ((p == k) & (y == k)).sum(); pr = tp / max((p == k).sum(), 1); rc = tp / max((y == k).sum(), 1)
        f.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def row(name: str, alert: np.ndarray, yc: np.ndarray, n: int) -> dict:
    tp = int((alert & yc).sum()); fn = int((~alert & yc).sum()); fa = int((alert & ~yc).sum())
    return {"model": name, "caught": tp, "missed": fn, "recall_%": round(100 * tp / yc.sum(), 1), "false_alarms": fa,
            "alerts": int(alert.sum()), "cost_total": R * fn + fa, "cost_per_1000": round((R * fn + fa) / n * 1000, 1)}


def main() -> int:
    o = pd.read_parquet(OOF)
    y = o.y_tier.map(TIERS).values; yc = y == 2; n = len(o)
    lt = o.persistence_tier.map(TIERS).values
    p = o.p_critical_mission.values

    # persistence as a probability: per fold, the empirical final-tier frequencies of each T-2 d tier in the
    # training folds (same construction as research/organizer/organizer.py::persistence_P)
    pers_p = np.zeros(n)
    for f in sorted(o.fold_mission.unique()):
        tst = o.fold_mission.values == f; trn = ~tst
        for k in range(3):
            m = trn & (lt == k)
            cnt = np.bincount(y[m], minlength=3).astype(float) + 1e-9
            pers_p[tst & (lt == k)] = (cnt / cnt.sum())[2]

    P = o[["p_safe_mission", "p_elevated_mission", "p_critical_mission"]].values
    rows = [
        row("persistence (tier at T-2 d)", lt == 2, yc, n) | {"auc": round(auc(yc, pers_p), 3),
                                                            "macro_f1": round(macro_f1(y, lt), 3)},
        row("ORBITAL DODGE served (per-fold thresholds)", o.alert_mission.values, yc, n)
        | {"auc": round(auc(yc, p), 3), "macro_f1": round(macro_f1(y, P.argmax(1)), 3)},
        row("same OOF scores, global threshold 0.0634", p >= 0.0634, yc, n),
        row("same OOF scores, pooled threshold 0.090", p >= 0.090, yc, n),
        row("same OOF scores, watch tau 0.20", p >= 0.20, yc, n),
    ]
    print(f"Held-out missions, organizer population: n = {n:,} events, {int(yc.sum())} critical, R = {R}\n")
    print(pd.DataFrame(rows).astype(object).fillna("").to_string(index=False))
    print("\nPer-fold thresholds chosen inside each training fold:")
    g = o.groupby("fold_mission").agg(threshold=("thr_mission", "first"),
                                      missions=("mission_id", lambda s: sorted(s.unique().tolist())))
    print(g.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
