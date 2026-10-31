import pandas as pd, numpy as np, os
RAW = os.environ.get("C1_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "Challenge 1 Airbus ")) + "/"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SENT = 63781363.0
ALLOWED = ["event_id","time_to_tca","miss_distance","relative_speed","relative_position_r","relative_position_t",
           "relative_position_n","c_object_type","c_sigma_r","t_sigma_r","max_risk_estimate","F10","AP"]

def load(split="train"):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{split}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    df = pd.read_csv(RAW + f"{split}_data.csv")
    df.to_parquet(p)
    return df

def tier(risk, miss):
    """brief tiers: 2=critical,1=elevated,0=safe"""
    risk = np.asarray(risk); miss = np.asarray(miss)
    crit = (risk >= -4) | (miss < 1000)
    elev = ((risk >= -6) & (risk < -4)) | ((miss >= 1000) & (miss <= 5000))
    return np.where(crit, 2, np.where(elev, 1, 0))

def kelvins_frame(df):
    """returns (inputs: all rows with ttca>=2 for eligible events, target per event)"""
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    last = df.groupby("event_id").tail(1).set_index("event_id")
    elig_final = last.index[last.time_to_tca < 1]
    has_early = df[df.time_to_tca >= 2].event_id.unique()
    elig = np.intersect1d(elig_final, has_early)
    inp = df[(df.event_id.isin(elig)) & (df.time_to_tca >= 2)]
    tgt = last.loc[elig, ["risk", "miss_distance", "time_to_tca", "mission_id"]].rename(
        columns={"risk": "y_risk", "miss_distance": "y_miss", "time_to_tca": "y_ttca"})
    return inp, tgt

SIGMA_COLS = ["t_sigma_r","t_sigma_t","t_sigma_n","c_sigma_r","c_sigma_t","c_sigma_n"]
def clean(df):
    """sentinel cleaning: position sigma >= 1e7 m (incl. 63,781,363 = 10 Earth radii) -> NaN + flag"""
    df = df.copy()
    flag = np.zeros(len(df), bool)
    for c in SIGMA_COLS:
        bad = df[c] >= 1e7
        flag |= bad.values
        df.loc[bad, c] = np.nan
    df["cov_unknown"] = flag.astype(int)
    df["c_object_type"] = df["c_object_type"].astype("category").cat.codes
    return df

TREND_COLS_A13 = ["miss_distance","relative_speed","relative_position_r","relative_position_t","relative_position_n",
                  "c_sigma_r","t_sigma_r","max_risk_estimate"]

def event_features(inp, cols_latest, trend_cols):
    inp = inp.sort_values(["event_id","time_to_tca"], ascending=[True, False])
    inp = inp.copy()
    if "miss_distance" in inp: inp["log_miss"] = np.log10(inp.miss_distance.clip(1))
    if "c_sigma_r" in inp and "t_sigma_r" in inp:
        inp["sig_r_comb"] = np.sqrt(inp.c_sigma_r**2 + inp.t_sigma_r**2)
        inp["r_over_sig"] = np.abs(inp.relative_position_r) / inp.sig_r_comb
    g = inp.groupby("event_id")
    cols_latest = list(dict.fromkeys(cols_latest + [c for c in ["log_miss","sig_r_comb","r_over_sig","cov_unknown"] if c in inp]))
    last = g.tail(1).set_index("event_id")[cols_latest].add_prefix("last_")
    prev = g.nth(-2).set_index("event_id") if hasattr(g.nth(-2),"set_index") else g.nth(-2)
    first = g.head(1).set_index("event_id")
    out = [last]
    out.append(g.size().rename("n_cdm"))
    out.append(first["time_to_tca"].rename("first_ttca"))
    tcols = list(dict.fromkeys(trend_cols + [c for c in ["log_miss","sig_r_comb","r_over_sig"] if c in inp]))
    for c in tcols:
        lastv = g[c].last()
        out += [ (lastv - prev[c].reindex(lastv.index)).rename(f"d1_{c}"),
                 (lastv - first[c]).rename(f"dall_{c}"),
                 g[c].min().rename(f"min_{c}"), g[c].max().rename(f"max_{c}"), g[c].std().rename(f"std_{c}")]
        # slope vs ttca
        def slope(x):
            t = x.time_to_tca.values; y = x[c].values; ok = ~np.isnan(y)
            if ok.sum() < 2 or np.ptp(t[ok]) == 0: return np.nan
            return np.polyfit(t[ok], y[ok], 1)[0]
        out.append(inp.groupby("event_id")[["time_to_tca", c]].apply(slope).rename(f"slope_{c}"))
    return pd.concat(out, axis=1)
