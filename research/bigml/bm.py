"""BIG-ML shared data layer: clean rows, CDM states, transition pairs, folds, metrics.
Labels follow docs/design/11 (brief tiers on the final CDM's dataset `risk` and `miss_distance`)."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "data", "raw", "Challenge 1 Airbus ")
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments"))

SIG = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]
OBJ = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA"]


def tier(risk, miss):
    risk = np.asarray(risk); miss = np.asarray(miss)
    crit = (risk >= -4) | (miss < 1000)
    elev = ((risk >= -6) & (risk < -4)) | ((miss >= 1000) & (miss <= 5000))
    return np.where(crit, 2, np.where(elev, 1, 0))


def load_rows():
    p = os.path.join(CACHE, "rows.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    out = []
    for split in ["train", "test"]:
        d = pd.read_csv(os.path.join(RAW, f"{split}_data.csv"))
        d["split"] = split
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["event_key"] = d.split + ":" + d.event_id.astype(str)
    for c in SIG:
        bad = d[c] >= 1e7
        d.loc[bad, c] = np.nan
    d["cov_unknown_t"] = d[["t_sigma_r", "t_sigma_t", "t_sigma_n"]].isna().any(axis=1).astype(int)
    d["cov_unknown_c"] = d[["c_sigma_r", "c_sigma_t", "c_sigma_n"]].isna().any(axis=1).astype(int)
    d = d.sort_values(["split", "event_id", "time_to_tca"], ascending=[True, True, False]).reset_index(drop=True)
    d["cdm_index"] = d.groupby("event_key").cumcount()
    # state columns
    d["lm"] = np.log10(d.miss_distance.clip(lower=1.0))
    d["rk"] = d.risk.clip(-30, 0)
    d["floor"] = (d.risk <= -29.99).astype(int)
    for c in SIG:
        d["ls_" + c[0] + c[-1]] = np.log10(d[c].clip(lower=1e-2))
    g = d.groupby("event_key")
    d["prev_dlm"] = (d.lm - g.lm.shift(1)).fillna(0)
    d["prev_drk"] = (d.rk - g.rk.shift(1)).fillna(0)
    d["prev_dt"] = (g.time_to_tca.shift(1) - d.time_to_tca).fillna(0)
    d["has_prev"] = (d.cdm_index > 0).astype(int)
    d["rk_max"] = g.rk.cummax()
    d["lm_min"] = g.lm.cummin()
    d["lm_first"] = g.lm.transform("first")
    d["rk_first"] = g.rk.transform("first")
    d["lspeed"] = np.log10(d.relative_speed.clip(lower=1))
    d["lhbr"] = np.log10(((d.t_span.fillna(5) + d.c_span.fillna(2)) / 2).clip(lower=0.05))
    d["rfrac"] = (d.relative_position_r / d.miss_distance.clip(lower=1)).clip(-1, 1)
    for i, o in enumerate(OBJ):
        d["obj_" + str(i)] = (d.c_object_type == o).astype(int)
    d["F10f"] = d.F10.fillna(d.F10.median()); d["APf"] = d.AP.fillna(d.AP.median())
    d.to_parquet(p)
    return d


DYN = ["time_to_tca", "lm", "rk", "floor", "ls_tr", "ls_tt", "ls_tn", "ls_cr", "ls_ct", "ls_cn",
       "prev_dlm", "prev_drk", "prev_dt", "has_prev", "cdm_index", "rk_max", "lm_min"]
STAT = ["lspeed", "lhbr", "rfrac", "F10f", "APf", "cov_unknown_t", "cov_unknown_c", "lm_first", "rk_first"] + \
       ["obj_%d" % i for i in range(len(OBJ))]
XCOLS = DYN + STAT
LS = ["ls_tr", "ls_tt", "ls_tn", "ls_cr", "ls_ct", "ls_cn"]
# targets: log dt, dlm, rk_next, 6 dls ; plus floor_next (bernoulli)
YD = 9


def pairs(d):
    """consecutive CDM pairs within each event (train split only)."""
    g = d.groupby("event_key")
    nxt = g.shift(-1)
    ok = nxt.time_to_tca.notna().values
    a = d[ok]; b = nxt[ok]
    dt = (a.time_to_tca.values - b.time_to_tca.values).clip(1e-3, None)
    Y = np.zeros((ok.sum(), YD), np.float32)
    Y[:, 0] = np.log(dt)
    Y[:, 1] = b.lm.values - a.lm.values
    Y[:, 2] = np.where(b.floor.values == 1, np.nan, b.rk.values)
    for j, c in enumerate(LS):
        Y[:, 3 + j] = b[c].values - a[c].values
    F = b.floor.values.astype(np.float32)
    return a, Y, F


def events_table(d):
    """per event: start row (latest ttca>=2 CDM, else first CDM), eligibility, final tier."""
    p = os.path.join(CACHE, "events.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    rows = []
    for ek, e in d.groupby("event_key", sort=False):
        early = e[e.time_to_tca >= 2]
        last = e.iloc[-1]
        start = early.iloc[-1] if len(early) else e.iloc[0]
        elig = (e.split.iloc[0] == "train") and len(early) > 0 and last.time_to_tca < 1
        rows.append(dict(event_key=ek, split=e.split.iloc[0], event_id=int(e.event_id.iloc[0]),
                         mission_id=int(e.mission_id.iloc[0]), n_cdm=len(e), has_early=len(early) > 0,
                         eligible=bool(elig), start_idx=int(start.name), start_cdm=int(start.cdm_index),
                         start_ttca=float(start.time_to_tca), start_tier=int(tier(start.risk, start.miss_distance)),
                         final_ttca=float(last.time_to_tca), final_risk=float(last.risk), final_miss=float(last.miss_distance),
                         final_tier=int(tier(last.risk, last.miss_distance))))
    ev = pd.DataFrame(rows)
    ev.to_parquet(p)
    return ev


def folds(ev_elig, kind, seed=0):
    """list of (train_mask, test_mask) over ev_elig rows. kind: event | mission | lomo"""
    from sklearn.model_selection import KFold, GroupKFold
    n = len(ev_elig)
    if kind == "event":
        sp = KFold(5, shuffle=True, random_state=seed).split(np.arange(n))
    elif kind == "mission":
        sp = GroupKFold(5).split(np.arange(n), groups=ev_elig.mission_id.values)
    elif kind == "lomo":
        ms = sorted(ev_elig.mission_id.unique())
        sp = [(np.where(ev_elig.mission_id.values != m)[0], np.where(ev_elig.mission_id.values == m)[0]) for m in ms]
    return [(tr, te) for tr, te in sp]


# ---------------- metrics ----------------
def ece(y, p, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)


def tier_metrics(yt, pt):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=[0, 1, 2], zero_division=0)
    return dict(macro_f1=float(f.mean()), crit_recall=float(r[2]), crit_precision=float(p[2]),
                crit_alerts=int((pt == 2).sum()), false_alarms=int(((pt == 2) & (yt != 2)).sum()),
                missed_crit=int(((pt != 2) & (yt == 2)).sum()))


def prob_metrics(yc, p):
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
    return dict(brier=float(brier_score_loss(yc, p)), ece=ece(yc, p), auc=float(roc_auc_score(yc, p)),
                ap=float(average_precision_score(yc, p)))


def boot_ci(fn, n, B=500, seed=0, groups=None):
    """percentile CI of fn(idx) over bootstrap resamples (event-level, or cluster by groups)."""
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(B):
            vals.append(fn(rng.integers(0, n, n)))
    else:
        ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
        for _ in range(B):
            pick = rng.choice(ug, len(ug)); vals.append(fn(np.concatenate([gi[g] for g in pick])))
    vals = np.array(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
