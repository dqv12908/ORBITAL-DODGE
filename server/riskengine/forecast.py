"""Tier forecaster: P(final-CDM tier = CRITICAL) at the latest CDM with ttca in [2, 7] d.

HGB 3-class on the 84 A13 features + isotonic calibration of P(critical). The forecast may only ADD attention
(WATCH flag, queue rank); it never lowers a rule tier or removes an escalation (see attention()).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import paths
from .cdm import TIER_NAMES, brief_tier_codes
from .features import FEATURES, event_features

VALID_MIN_D, VALID_MAX_D = 2.0, 7.0
TAU_WATCH = 0.10
PERSISTENCE_EMPIRICAL_DEFAULT = {"SAFE": 0.002, "ELEVATED": 0.053, "CRITICAL": 0.800}


class Forecaster:
    def __init__(self, model, calibrator, meta: dict, version: str):
        self.model = model
        self.calibrator = calibrator
        self.meta = meta
        self.version = version
        self.persistence_empirical = meta.get("persistence_empirical", PERSISTENCE_EMPIRICAL_DEFAULT)
        self.reliability = meta.get("reliability_calibrated", [])

    @classmethod
    def load(cls, path=None) -> "Forecaster":
        d = Path(path) if path else paths.current_forecaster_dir()
        model = joblib.load(d / "model.joblib")
        cal = joblib.load(d / "calibrator.joblib")
        card = json.loads((d / "model_card.json").read_text())
        meta = card.get("runtime", {})
        assert meta.get("features", FEATURES) == FEATURES, "feature order mismatch"
        return cls(model, cal, meta, card.get("version", d.name))

    # ---- batch API (features already built)
    def predict_features(self, F: pd.DataFrame) -> pd.DataFrame:
        raw = self.model.predict_proba(F[FEATURES].to_numpy(float))
        return pd.DataFrame(calibrate_tiers(raw, self.calibrator), index=F.index, columns=["p_safe", "p_elevated", "p_critical"])\
            .assign(p_critical_raw=raw[:, 2])

    def calib_bin(self, p: float):
        for b in self.reliability:
            if b["lo"] <= p < b["hi"] or (b["hi"] >= 1 and p >= b["lo"]):
                return {"lo": b["lo"], "hi": min(b["hi"], 1.0), "observed": b["observed"], "n": b["n"]}
        return None

    # ---- per-event API used by the backend
    def predict(self, cdm_rows_df: pd.DataFrame, ttca_now_d: float | None = None, tau_watch: float = TAU_WATCH) -> dict:
        """cdm_rows_df: the CDMs of ONE event released so far (raw or cleaned columns)."""
        rows = cdm_rows_df
        if len(rows) == 0:
            raise ValueError("no CDM rows")
        if "cov_unknown_t" not in rows or "cov_unknown_c" not in rows:   # raw rows: apply the same cleaning
            from .cdm import clean
            rows = rows.copy()
            if "event_id" not in rows:
                rows["event_id"] = 0
            rows = clean(rows, split=rows["split"].iloc[0] if "split" in rows else "adhoc")
        latest_ttca = float(rows.time_to_tca.min()) if ttca_now_d is None else float(ttca_now_d)
        early = rows[rows.time_to_tca >= VALID_MIN_D].sort_values("time_to_tca", ascending=False)
        base_row = (early if len(early) else rows.sort_values("time_to_tca", ascending=False)).iloc[-1]
        risk = base_row.get("risk", np.nan)
        pers = TIER_NAMES[int(brief_tier_codes([np.nan if risk is None else risk], [base_row["miss_distance"]])[0])]
        baseline = {"persistence_tier": pers, "p_critical_empirical": float(self.persistence_empirical[pers])}
        valid = VALID_MIN_D <= latest_ttca <= VALID_MAX_D and len(early) > 0
        res = {"valid": bool(valid), "p_tiers": None, "p_critical": None, "calib_bin": None, "baseline": baseline,
               "watch": False, "rank_score": baseline["p_critical_empirical"], "model_version": self.version,
               "reason": None}
        if not valid:
            res["reason"] = ("inside 2 d: physics on the latest CDM governs" if latest_ttca < VALID_MIN_D
                             else "outside the trained window (ttca > 7 d)")
            return res
        e = early.copy()
        e["event_key"] = "x"
        F = event_features(e)
        P = self.predict_features(F).iloc[0]
        p = float(P.p_critical)
        res.update({
            "p_tiers": {"SAFE": float(P.p_safe), "ELEVATED": float(P.p_elevated), "CRITICAL": p},
            "p_critical": p, "p_critical_raw": float(P.p_critical_raw), "calib_bin": self.calib_bin(p),
            "watch": bool(p >= tau_watch and pers != "CRITICAL"), "rank_score": p,
        })
        return res


def calibrate_tiers(raw: np.ndarray, calibrator) -> np.ndarray:
    """Replace P(crit) by its isotonic-calibrated value; rescale SAFE/ELEVATED to keep the sum at 1."""
    pc = np.clip(calibrator.predict(raw[:, 2]), 0, 1)
    rest = raw[:, :2]
    s = rest.sum(1, keepdims=True)
    rest = np.where(s > 0, rest / np.where(s > 0, s, 1), 0.5) * (1 - pc)[:, None]
    return np.column_stack([rest, pc])


def attention(rule_tier: str, escalate: bool, forecast: dict | None) -> dict:
    """Combine the rule result with the forecast. Invariant: never lower the tier, never drop an escalation."""
    watch = bool(forecast and forecast.get("valid") and forecast.get("watch"))
    return {"tier": rule_tier, "escalate": bool(escalate), "watch": watch and rule_tier != "CRITICAL",
            "flagged": rule_tier != "SAFE" or escalate or watch}
