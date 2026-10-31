"""ORBITAL DODGE bake-off harness (shared by every contender).

Build once:   .venv/bin/python harness.py build
Contender API (import harness as H):
    ev  = H.load_events()                  # one row per eligible event, fixed order (sorted by event_id)
    X   = H.load_features("A13"|"ALL")     # DataFrame aligned to ev (index = event_key)
    S   = H.load_sequences()               # dict: X (n,T,F) float32 with NaN, mask (n,T) bool, lengths, event_key, feature_names
    for k, (trn, tst) in H.folds("event"|"mission"|"lomo"): ...   # integer positions into ev
    H.write_oof("mymodel", "mission", ev.event_key.values, P)      # P (n,3): p_safe, p_elevated, p_critical

Rules baked in here (do not change in contenders):
  * Inputs = CDMs with time_to_tca >= 2 d only. Label = brief tier of the FINAL CDM (final ttca < 1 d).
  * Nothing from the final CDM or any CDM with ttca < 2 d is ever a feature.
  * Sentinel position sigmas (>= 1e7 m) -> NaN, cov_unknown = 1.
  * Sequence standardisation must be fitted per fold inside the contender (stats here are NOT global-normalised).
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("C1_DATA_DIR", "/root/orbital-dodge/data/raw/Challenge 1 Airbus ") + "/"
DATA = os.path.join(HERE, "data")
OOF = os.path.join(HERE, "oof")
SENT_MIN = 1e7
ALLOWED = ["event_id", "time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t",
           "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP"]
SIGMA_COLS = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]
TREND_COLS_A13 = ["miss_distance", "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n",
                  "c_sigma_r", "t_sigma_r", "max_risk_estimate"]
TREND_COLS_ALL = TREND_COLS_A13 + ["risk", "mahalanobis_distance", "c_sigma_t", "c_sigma_n", "t_sigma_t", "t_sigma_n",
                                   "c_position_covariance_det", "t_position_covariance_det", "c_obs_used", "c_weighted_rms"]
TIERS = ["safe", "elevated", "critical"]
PCOLS = ["p_safe", "p_elevated", "p_critical"]
SPLITS = ["event", "mission", "lomo"]


# ----------------------------------------------------------------------------- data
def tier(risk, miss):
    """brief tiers: 2=critical (risk>=-4 or miss<1000 m), 1=elevated (risk>=-6 or miss<=5000 m), 0=safe"""
    risk = np.asarray(risk, float); miss = np.asarray(miss, float)
    crit = (risk >= -4) | (miss < 1000)
    elev = (risk >= -6) | (miss <= 5000)
    return np.where(crit, 2, np.where(elev, 1, 0))


def load_raw(split="train"):
    cache = os.path.join(DATA, f"raw_{split}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = pd.read_csv(RAW + f"{split}_data.csv")
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(cache)
    return df


def clean(df):
    df = df.copy()
    flag = np.zeros(len(df), bool)
    for c in SIGMA_COLS:
        bad = (df[c] >= SENT_MIN).values
        flag |= bad
        df.loc[bad, c] = np.nan
    df["cov_unknown"] = flag.astype(int)
    df["c_object_type"] = df["c_object_type"].astype("category").cat.codes.astype(int)
    return df


def eligible_frame(df):
    """returns (inp: CDMs with ttca>=2 of eligible events, tgt: per-event final-CDM targets indexed by event_id)"""
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
    last = df.groupby("event_id").tail(1).set_index("event_id")
    elig = np.intersect1d(last.index[last.time_to_tca < 1], df.loc[df.time_to_tca >= 2, "event_id"].unique())
    inp = df[df.event_id.isin(elig) & (df.time_to_tca >= 2)].copy()
    tgt = last.loc[elig, ["risk", "miss_distance", "time_to_tca", "mission_id"]].rename(
        columns={"risk": "y_risk", "miss_distance": "y_miss", "time_to_tca": "y_ttca"})
    return inp, tgt


def _vec_slope(inp, c):
    """least-squares slope of c vs time_to_tca per event over non-NaN rows (== np.polyfit deg 1); NaN if <2 pts or ptp(t)=0"""
    d = inp[["event_id", "time_to_tca", c]].copy()
    d = d[d[c].notna()]
    g = d.groupby("event_id")
    tm = g.time_to_tca.transform("mean"); ym = g[c].transform("mean")
    d["sxy"] = (d.time_to_tca - tm) * (d[c] - ym); d["sxx"] = (d.time_to_tca - tm) ** 2
    s = d.groupby("event_id")[["sxy", "sxx"]].sum(); n = g.size()
    out = s.sxy / s.sxx
    out[(n < 2) | (s.sxx <= 1e-15)] = np.nan
    return out


def event_features(inp, cols_latest, trend_cols):
    """same feature recipe as research/experiments/common.event_features (slope vectorised)."""
    inp = inp.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort").copy()
    inp["log_miss"] = np.log10(inp.miss_distance.clip(1))
    inp["sig_r_comb"] = np.sqrt(inp.c_sigma_r ** 2 + inp.t_sigma_r ** 2)
    inp["r_over_sig"] = np.abs(inp.relative_position_r) / inp.sig_r_comb
    g = inp.groupby("event_id")
    cols_latest = list(dict.fromkeys(list(cols_latest) + ["log_miss", "sig_r_comb", "r_over_sig", "cov_unknown"]))
    last = g.tail(1).set_index("event_id")[cols_latest].add_prefix("last_")
    prev = g.nth(-2).set_index("event_id")
    first = g.head(1).set_index("event_id")
    out = [last, g.size().rename("n_cdm"), first["time_to_tca"].rename("first_ttca")]
    tcols = list(dict.fromkeys(list(trend_cols) + ["log_miss", "sig_r_comb", "r_over_sig"]))
    for c in tcols:
        lastv = g[c].last()
        out += [(lastv - prev[c].reindex(lastv.index)).rename(f"d1_{c}"),
                (lastv - first[c].reindex(lastv.index)).rename(f"dall_{c}"),
                g[c].min().rename(f"min_{c}"), g[c].max().rename(f"max_{c}"), g[c].std().rename(f"std_{c}"),
                _vec_slope(inp, c).reindex(lastv.index).rename(f"slope_{c}")]
    return pd.concat(out, axis=1)


SEQ_COLS = ALLOWED[1:] + ["risk", "log_c_sigma_t", "log_c_sigma_n", "log_t_sigma_t", "log_t_sigma_n",
                          "mahalanobis_distance", "cov_unknown"]


def build_sequences(inp, keys_order):
    inp = inp.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort").copy()
    for c in ["c_sigma_t", "c_sigma_n", "t_sigma_t", "t_sigma_n"]:
        inp["log_" + c] = np.log10(inp[c].clip(lower=1e-3))
    pos = inp.groupby("event_id").cumcount().values
    T = int(pos.max()) + 1
    eid_to_row = {e: i for i, e in enumerate(keys_order)}
    rows = inp.event_id.map(eid_to_row).values
    X = np.full((len(keys_order), T, len(SEQ_COLS)), np.nan, np.float32)
    M = np.zeros((len(keys_order), T), bool)
    X[rows, pos] = inp[SEQ_COLS].values.astype(np.float32)
    M[rows, pos] = True
    return X, M


def build():
    from sklearn.model_selection import KFold, GroupKFold
    os.makedirs(DATA, exist_ok=True); os.makedirs(OOF, exist_ok=True)
    tr = clean(load_raw("train"))
    inp, tgt = eligible_frame(tr)
    eids = tgt.index.values  # sorted (np.intersect1d)
    assert np.all(np.diff(eids) > 0)
    yt = tier(tgt.y_risk, tgt.y_miss)
    keys = np.array([f"train:{e}" for e in eids])
    # persistence inputs (latest >=2 d CDM, risk + miss -> tier); stored for the reference row only
    lat = inp.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).groupby("event_id").tail(1).set_index("event_id").loc[eids]
    ev = pd.DataFrame({"event_key": keys, "event_id": eids, "mission_id": tgt.mission_id.values.astype(int),
                       "y_tier": yt.astype(int), "y_crit": (yt == 2).astype(int),
                       "latest_tier": tier(lat.risk, lat.miss_distance).astype(int),
                       "latest_ttca": lat.time_to_tca.values, "n_cdm_ge2d": inp.groupby("event_id").size().loc[eids].values})
    # NOTE y_risk / y_miss / y_ttca are label information: kept in labels.parquet only, never in feature files
    lab = pd.DataFrame({"event_key": keys, "y_tier": yt.astype(int), "y_risk": tgt.y_risk.values,
                        "y_miss": tgt.y_miss.values, "y_ttca": tgt.y_ttca.values})
    # folds
    n = len(ev); fe = np.zeros(n, int); fm = np.zeros(n, int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))): fe[te] = k
    for k, (_, te) in enumerate(GroupKFold(5).split(np.arange(n), groups=ev.mission_id.values)): fm[te] = k
    folds = pd.DataFrame({"event_key": keys, "mission_id": ev.mission_id.values, "fold_event": fe, "fold_mission": fm,
                          "fold_lomo": ev.mission_id.values})
    # features
    numcols = [c for c in tr.columns if c not in ("event_id", "mission_id") and tr[c].dtype.kind in "if"]
    sets = {"A13": (ALLOWED[1:], TREND_COLS_A13), "ALL": (numcols, TREND_COLS_ALL)}
    for name, (cl, tc) in sets.items():
        F = event_features(inp, cl, tc).loc[eids]
        F.index = keys; F.index.name = "event_key"
        F.columns = [str(c) for c in F.columns]
        F.astype(np.float64).to_parquet(os.path.join(DATA, f"features_{name}.parquet"))
        print(name, "features:", F.shape)
    X, M = build_sequences(inp, eids)
    np.savez_compressed(os.path.join(DATA, "sequences.npz"), X=X, mask=M, lengths=M.sum(1), event_key=keys,
                        feature_names=np.array(SEQ_COLS))
    print("sequences:", X.shape, "max len", M.sum(1).max())
    ev.to_parquet(os.path.join(DATA, "events.parquet")); lab.to_parquet(os.path.join(DATA, "labels.parquet"))
    folds.to_parquet(os.path.join(DATA, "folds.parquet")); folds.to_parquet(os.path.join(HERE, "folds.parquet"))
    tc = np.bincount(yt, minlength=3)
    summ = {"n_events": int(n), "tiers_safe_elev_crit": tc.tolist(), "n_features_A13": None, "n_missions": int(ev.mission_id.nunique()),
            "fold_sizes_event": np.bincount(fe).tolist(), "fold_sizes_mission": np.bincount(fm).tolist(),
            "crit_per_fold_mission": [int(ev.y_crit[fm == k].sum()) for k in range(5)]}
    summ["n_features_A13"] = pd.read_parquet(os.path.join(DATA, "features_A13.parquet")).shape[1]
    summ["n_features_ALL"] = pd.read_parquet(os.path.join(DATA, "features_ALL.parquet")).shape[1]
    json.dump(summ, open(os.path.join(DATA, "build_summary.json"), "w"), indent=1)
    print(json.dumps(summ))
    assert n == 8293 and tc.tolist() == [6138, 1604, 551], (n, tc)


# ----------------------------------------------------------------------------- contender API
def load_events():
    return pd.read_parquet(os.path.join(DATA, "events.parquet"))


def load_labels():
    """y_tier (0/1/2) is the training target; y_risk/y_miss are allowed as auxiliary TRAINING targets only."""
    return pd.read_parquet(os.path.join(DATA, "labels.parquet"))


def load_features(name="A13"):
    return pd.read_parquet(os.path.join(DATA, f"features_{name}.parquet"))


def load_sequences():
    z = np.load(os.path.join(DATA, "sequences.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def fold_ids(split):
    return pd.read_parquet(os.path.join(DATA, "folds.parquet"))[f"fold_{split}"].values


def folds(split):
    """yields (fold_id, train_positions, test_positions) in events.parquet order"""
    f = fold_ids(split)
    for k in np.unique(f):
        yield k, np.where(f != k)[0], np.where(f == k)[0]


def write_oof(model, split, event_keys, P, extra=None):
    P = np.asarray(P, float)
    assert P.shape == (len(event_keys), 3), P.shape
    assert np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1, atol=1e-4), "rows must be finite and sum to 1"
    df = pd.DataFrame({"event_key": event_keys, "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2]})
    if extra is not None:
        for k, v in extra.items(): df[k] = v
    os.makedirs(OOF, exist_ok=True)
    p = os.path.join(OOF, f"{model}_{split}.parquet")
    df.to_parquet(p)
    return p


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
