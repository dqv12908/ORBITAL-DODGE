"""Evaluation helpers (port of research b_baselines.py / b2_kelvins.py)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, fbeta_score,
                             precision_recall_fscore_support, roc_auc_score)

REL_EDGES = (0, .02, .05, .1, .2, .4, .6, .8, 1.0001)
TAUS = (0.05, 0.10, 0.20, 0.30, 0.50)


def tier_report(yt, pt) -> dict:
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=[0, 1, 2], zero_division=0)
    return dict(rec_safe=r[0], rec_elev=r[1], rec_crit=r[2], prec_safe=p[0], prec_elev=p[1], prec_crit=p[2],
                macro_f1=float(f.mean()), crit_alerts=int((pt == 2).sum()),
                false_alarms=int(((pt == 2) & (yt != 2)).sum()), missed_crit=int(((pt != 2) & (yt == 2)).sum()))


def ece(y, p, bins=10) -> float:
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i])
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability(y, p, edges=REL_EDGES) -> list[dict]:
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            rows.append({"lo": float(a), "hi": float(min(b, 1.0)), "n": int(m.sum()), "pred": float(p[m].mean()),
                         "observed": float(y[m].mean())})
    return rows


def prob_report(y, p) -> dict:
    return {"brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p))}


def operating_curve(y, p, taus=TAUS) -> list[dict]:
    out = []
    for t in taus:
        a = p >= t
        out.append({"tau": t, "alerts": int(a.sum()), "crit_recall": float((a & (y == 1)).sum() / max(y.sum(), 1)),
                    "precision": float((a & (y == 1)).sum() / max(a.sum(), 1)), "false_alarms": int((a & (y == 0)).sum()),
                    "missed_crit": int((~a & (y == 1)).sum())})
    return out


def kelvins(y_risk, pred) -> dict:
    yc = np.maximum(y_risk, -6.001)
    pc = np.maximum(pred, -6.001)
    hr = y_risk >= -6
    f2 = fbeta_score(hr, pred >= -6, beta=2)
    mse = float(np.mean((yc[hr] - pc[hr]) ** 2))
    return {"f2": float(f2), "mse_hr": mse, "L": float(mse / f2) if f2 > 0 else None,
            "recall": float(np.mean(pred[hr] >= -6)), "alerts": int((pred >= -6).sum())}
