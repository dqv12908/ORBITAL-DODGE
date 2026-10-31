"""CDM loading and cleaning (port of research/experiments/common.py).

* any position sigma >= 1e7 m (incl. the 63,781,363 m = 10 Earth-radii default) -> NaN, cov_unknown_{t,c}=1
* NaN position sigma (11 chaser rows) -> cov_unknown as well (physics cannot run)
* event_key = "{split}:{event_id}"; cdm_index 0-based by descending time_to_tca (0 = first CDM issued)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths

SENTINEL_M = 63781363.0
SENTINEL_THRESHOLD_M = 1e7
T_POS_SIGMA = ["t_sigma_r", "t_sigma_t", "t_sigma_n"]
C_POS_SIGMA = ["c_sigma_r", "c_sigma_t", "c_sigma_n"]
SIGMA_COLS = T_POS_SIGMA + C_POS_SIGMA
ALLOWED_13 = ["event_id", "time_to_tca", "miss_distance", "relative_speed", "relative_position_r",
              "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r",
              "max_risk_estimate", "F10", "AP"]
OBJECT_TYPE_CODES = {"DEBRIS": 0, "PAYLOAD": 1, "ROCKET BODY": 2, "TBA": 3, "UNKNOWN": 4}  # = research category codes
RISK_FLOOR = -30.0


def event_key(split: str, event_id) -> str:
    return f"{split}:{int(event_id)}"


def load_raw(split: str) -> pd.DataFrame:
    return pd.read_csv(paths.RAW_DIR / f"{split}_data.csv")


def clean(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Return a cleaned copy. Works on a whole split or on ad-hoc rows (split may already be a column)."""
    df = df.copy()
    if split is not None:
        df["split"] = split
    elif "split" not in df:
        df["split"] = "adhoc"
    for side, cols in (("t", T_POS_SIGMA), ("c", C_POS_SIGMA)):
        flag = np.zeros(len(df), bool)
        for c in cols:
            if c not in df:
                df[c] = np.nan
            v = pd.to_numeric(df[c], errors="coerce")
            bad = (v >= SENTINEL_THRESHOLD_M).to_numpy()
            flag |= bad | v.isna().to_numpy()
            df[c] = v.where(~bad)
        df[f"cov_unknown_{side}"] = flag.astype("int8")
    df["event_key"] = [f"{s}:{int(e)}" for s, e in zip(df["split"], df["event_id"])]
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["split", "event_id", "time_to_tca", "_ord"], ascending=[True, True, False, True])
    df["cdm_index"] = df.groupby("event_key").cumcount().astype("int32")
    df = df.drop(columns="_ord").reset_index(drop=True)
    front = ["event_key", "split", "event_id", "mission_id", "cdm_index"]
    front = [c for c in front if c in df]
    return df[front + [c for c in df.columns if c not in front]]


def object_type_code(s: pd.Series) -> pd.Series:
    return s.map(OBJECT_TYPE_CODES).astype("float64")


_CACHE: dict[str, pd.DataFrame] = {}


def load_clean(path=None) -> pd.DataFrame:
    """cdm_clean.parquet (both splits). Cached in-process; do not mutate the returned frame."""
    p = str(path or paths.CDM_CLEAN)
    if p not in _CACHE:
        _CACHE[p] = pd.read_parquet(p)
    return _CACHE[p]


def brief_tier_codes(log10_pc, miss_m) -> np.ndarray:
    """Brief tiers from a log10 Pc (or dataset risk) and miss distance: 0 SAFE, 1 ELEVATED, 2 CRITICAL.
    NaN log10 is treated as 'no Pc information' (miss decides). Matches research common.tier()."""
    r = np.asarray(log10_pc, float)
    m = np.asarray(miss_m, float)
    with np.errstate(invalid="ignore"):
        crit = (r >= -4) | (m < 1000)
        elev = ((r >= -6) & (r < -4)) | ((m >= 1000) & (m <= 5000))
    return np.where(crit, 2, np.where(elev, 1, 0))


TIER_NAMES = ["SAFE", "ELEVATED", "CRITICAL"]
