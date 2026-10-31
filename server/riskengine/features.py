"""A13 event features (84) from the CDMs at time_to_tca >= 2 d (port of research common.event_features).

Only the 13 allowed columns (plus derived log-miss, combined radial sigma, |r|/sigma_r and the cov_unknown flag).
No `risk`, no `mission_id`. Vectorised (slopes by closed-form least squares instead of np.polyfit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cdm import ALLOWED_13, OBJECT_TYPE_CODES, brief_tier_codes

INPUT_MIN_TTCA_D = 2.0
TREND_COLS = ["miss_distance", "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n",
              "c_sigma_r", "t_sigma_r", "max_risk_estimate", "log_miss", "sig_r_comb", "r_over_sig"]
LATEST_COLS = ALLOWED_13[1:] + ["log_miss", "sig_r_comb", "r_over_sig", "cov_unknown"]
FEATURES = (["last_" + c for c in LATEST_COLS] + ["n_cdm", "first_ttca"]
            + [f"{p}_{c}" for c in TREND_COLS for p in ("d1", "dall", "min", "max", "std", "slope")])
assert len(FEATURES) == 84


def _prep(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    x = rows.copy()
    if key not in x:
        x[key] = "adhoc:0"
    x = x.sort_values([key, "time_to_tca"], ascending=[True, False], kind="stable")
    if x["c_object_type"].dtype.kind not in "if":
        x["c_object_type"] = x["c_object_type"].astype("object").map(OBJECT_TYPE_CODES).astype(float)
    if "cov_unknown" not in x:
        cu = np.zeros(len(x), int)
        for s in ("cov_unknown_t", "cov_unknown_c"):
            if s in x:
                cu |= x[s].fillna(0).to_numpy().astype(int)
        x["cov_unknown"] = cu
    for c in ALLOWED_13[1:]:
        if c not in x:
            x[c] = np.nan
        if c != "c_object_type":
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    x["log_miss"] = np.log10(x.miss_distance.clip(lower=1))
    x["sig_r_comb"] = np.sqrt(x.c_sigma_r ** 2 + x.t_sigma_r ** 2)
    x["r_over_sig"] = np.abs(x.relative_position_r) / x.sig_r_comb
    return x


def event_features(rows: pd.DataFrame, key: str = "event_key") -> pd.DataFrame:
    """rows: CDMs (already restricted to the forecast inputs, i.e. ttca >= 2 d). Returns FEATURES per event."""
    x = _prep(rows, key)
    g = x.groupby(key, sort=True)
    last = g.tail(1).set_index(key)[LATEST_COLS].add_prefix("last_")
    prev = g.nth(-2).set_index(key)
    first = g.head(1).set_index(key)
    out = {c: last[c] for c in last.columns}
    out["n_cdm"] = g.size()
    out["first_ttca"] = first["time_to_tca"]
    t = x["time_to_tca"].to_numpy(float)
    keys = x[key].to_numpy()
    for c in TREND_COLS:
        lastv = g[c].last()
        out[f"d1_{c}"] = lastv - prev[c].reindex(lastv.index)
        out[f"dall_{c}"] = lastv - first[c].reindex(lastv.index)
        out[f"min_{c}"] = g[c].min()
        out[f"max_{c}"] = g[c].max()
        out[f"std_{c}"] = g[c].std()
        # slope of c vs ttca over rows with c present (least squares)
        y = x[c].to_numpy(float)
        ok = ~np.isnan(y)
        d = pd.DataFrame({"k": keys[ok], "t": t[ok], "y": y[ok]})
        gg = d.groupby("k")
        tm = gg.t.transform("mean").to_numpy()
        ym = gg.y.transform("mean").to_numpy()
        d["tt"] = (d.t.to_numpy() - tm) ** 2
        d["ty"] = (d.t.to_numpy() - tm) * (d.y.to_numpy() - ym)
        agg = d.groupby("k").agg(n=("t", "size"), sxx=("tt", "sum"), sxy=("ty", "sum"), tmin=("t", "min"), tmax=("t", "max"))
        valid = (agg.n >= 2) & (agg.tmax - agg.tmin > 0)
        sl = (agg.sxy / agg.sxx).where(valid)
        out[f"slope_{c}"] = sl.reindex(lastv.index)
    F = pd.DataFrame(out)
    return F[FEATURES].astype(float)


def forecast_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows.time_to_tca >= INPUT_MIN_TTCA_D]


def kelvins_frame(df: pd.DataFrame, key: str = "event_key"):
    """Eligible events (>= 1 CDM at >= 2 d, final CDM < 1 d). Returns (inputs rows, target frame per event)."""
    df = df.sort_values([key, "time_to_tca"], ascending=[True, False], kind="stable")
    last = df.groupby(key).tail(1).set_index(key)
    elig_final = last.index[last.time_to_tca < 1]
    has_early = df.loc[df.time_to_tca >= INPUT_MIN_TTCA_D, key].unique()
    elig = np.intersect1d(elig_final, has_early)
    inp = df[df[key].isin(elig) & (df.time_to_tca >= INPUT_MIN_TTCA_D)]
    tgt = last.loc[elig, ["risk", "miss_distance", "time_to_tca", "mission_id", "event_id"]].rename(
        columns={"risk": "y_risk", "miss_distance": "y_miss", "time_to_tca": "y_ttca"})
    tgt["y_tier"] = brief_tier_codes(tgt.y_risk, tgt.y_miss)
    lat = inp.groupby(key).tail(1).set_index(key).loc[elig]
    tgt["latest_risk"] = lat["risk"]
    tgt["latest_miss"] = lat["miss_distance"]
    tgt["latest_ttca"] = lat["time_to_tca"]
    tgt["persistence_tier"] = brief_tier_codes(lat["risk"], lat["miss_distance"])
    return inp, tgt
