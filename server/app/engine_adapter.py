"""Thin adapter over the engine's `riskengine` package and precomputed artifacts.

Order of preference for every capability (each falls back independently, never raises into the request path):
  data:      riskengine.cdm.load_clean() -> data/processed/cdm_clean.parquet -> raw CSV (cleaned here) -> built-in fixture
  pc:        pc_rows.parquet (precomputed) -> riskengine.pc.pc_for_row -> local 2D Pc port -> reported risk
  rules:     riskengine.rules.assess -> local table (design §3.4)
  forecast:  riskengine.forecast.Forecaster (registry) -> persistence-empirical
  geometry:  riskengine.orbits.geometry -> geometry.parquet -> elements from the CDM row (labelled RECONSTRUCTED)
"""
from __future__ import annotations

import importlib
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.util import fnum

log = structlog.get_logger("engine")

SENT = 1e7
TIERS = ["SAFE", "ELEVATED", "CRITICAL"]
TIER_RANK = {"SAFE": 0, "ELEVATED": 1, "CRITICAL": 2, "UNKNOWN": -1}
P_CRIT_EMPIRICAL = {"SAFE": 0.002, "ELEVATED": 0.053, "CRITICAL": 0.800}
MU = 398600.4418e9

# The late-CDM story (train:2563), used when no data is available at all. Values from the design (§3.7 F1).
FIXTURE = {
    "train:2563": [
        {"time_to_tca": 2.43, "risk": -6.32, "miss_distance": 8050.0, "relative_speed": 11250.0,
         "relative_position_r": 120.0, "relative_position_t": -7900.0, "relative_position_n": 1540.0,
         "c_object_type": "DEBRIS", "mission_id": 15, "t_sigma_r": 9.0, "c_sigma_r": 85.0, "max_risk_estimate": -4.9,
         "F10": 98.0, "AP": 8.0, "t_j2k_sma": 7085.0, "t_j2k_ecc": 0.001, "t_j2k_inc": 98.2,
         "c_j2k_sma": 7091.0, "c_j2k_ecc": 0.004, "c_j2k_inc": 71.0},
        {"time_to_tca": 1.41, "risk": -6.31, "miss_distance": 1630.0, "relative_speed": 11250.0,
         "relative_position_r": 40.0, "relative_position_t": -1500.0, "relative_position_n": 630.0,
         "c_object_type": "DEBRIS", "mission_id": 15, "t_sigma_r": 7.0, "c_sigma_r": 60.0, "max_risk_estimate": -4.6,
         "F10": 99.0, "AP": 9.0, "t_j2k_sma": 7085.0, "t_j2k_ecc": 0.001, "t_j2k_inc": 98.2,
         "c_j2k_sma": 7091.0, "c_j2k_ecc": 0.004, "c_j2k_inc": 71.0},
        {"time_to_tca": 0.44, "risk": -4.19, "miss_distance": 136.0, "relative_speed": 11250.0,
         "relative_position_r": 12.0, "relative_position_t": -120.0, "relative_position_n": 62.0,
         "c_object_type": "DEBRIS", "mission_id": 15, "t_sigma_r": 5.0, "c_sigma_r": 40.0, "max_risk_estimate": -3.9,
         "F10": 99.0, "AP": 7.0, "t_j2k_sma": 7085.0, "t_j2k_ecc": 0.001, "t_j2k_inc": 98.2,
         "c_j2k_sma": 7091.0, "c_j2k_ecc": 0.004, "c_j2k_inc": 71.0},
    ]
}


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # engine still being written, or broken
        log.info("riskengine_unavailable", module=name, error=str(e)[:200])
        return None


def brief_tier(log10_pc: float | None, miss_m: float | None) -> str:
    """Brief label definition (final-CDM tier on dataset risk + miss)."""
    r = -30.0 if log10_pc is None else log10_pc
    m = 1e9 if miss_m is None else miss_m
    if r >= -4 or m < 1000:
        return "CRITICAL"
    if r >= -6 or (1000 <= m <= 5000):
        return "ELEVATED"
    return "SAFE"


# ---------------------------------------------------------------------------
# Local physics fallback (port of research/experiments/pc2d.py for one row)
# ---------------------------------------------------------------------------
def _cov(s_r, s_t, s_n, c_tr, c_nr, c_nt):
    C = np.zeros((3, 3))
    C[0, 0], C[1, 1], C[2, 2] = s_r ** 2, s_t ** 2, s_n ** 2
    C[0, 1] = C[1, 0] = c_tr * s_r * s_t
    C[0, 2] = C[2, 0] = c_nr * s_r * s_n
    C[1, 2] = C[2, 1] = c_nt * s_t * s_n
    return C


def local_pc(row: dict) -> dict:
    g = lambda k: fnum(row.get(k))
    sig = [g(k) for k in ("t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n")]
    reported = g("risk")
    hbr = None
    if g("t_span") is not None and g("c_span") is not None:
        hbr = (g("t_span") + g("c_span")) / 2.0
    vrel = g("relative_speed")
    base = {"method": "2D-Foster-GL24x48", "hbr_m": hbr or 0.0}
    cov_flag = (fnum(row.get("cov_unknown_t")) or 0) > 0 or (fnum(row.get("cov_unknown_c")) or 0) > 0
    if cov_flag or any(s is not None and s >= SENT for s in sig):
        return {**base, "status": "COV_UNKNOWN", "log10": None}
    if any(s is None for s in sig):
        if reported is None:
            return {**base, "status": "COV_UNKNOWN", "log10": None}
        return {**base, "status": "OK", "log10": reported, "method": "reported-risk", "dilution": False}
    corr = [g(k) or 0.0 for k in ("t_ct_r", "t_cn_r", "t_cn_t", "c_ct_r", "c_cn_r", "c_cn_t")]
    rp = np.array([g("relative_position_r") or 0.0, g("relative_position_t") or 0.0, g("relative_position_n") or 0.0])
    v = np.array([g("relative_velocity_r") or 0.0, g("relative_velocity_t") or 0.0, g("relative_velocity_n") or 0.0])
    if hbr is None or np.linalg.norm(v) == 0 or g("t_j2k_sma") is None:
        # not enough columns (fixture / partial row): use the dataset's reported risk, labelled as such
        return {**base, "status": "OK", "log10": reported, "method": "reported-risk", "dilution": False}
    try:
        Ct = _cov(*sig[:3], *corr[:3])
        Cc = _cov(*sig[3:], *corr[3:])
        a = g("t_j2k_sma") * 1e3
        vt = math.sqrt(MU / a)
        vc = np.array([0.0, vt, 0.0]) + v
        R = np.array([1.0, 0, 0])
        N = np.cross(R, vc)
        N /= np.linalg.norm(N)
        T = np.cross(N, R)
        M = np.stack([R, T, N], 1)
        Cc = M @ Cc @ M.T
        C = Ct + Cc
        npd = False
        w, V = np.linalg.eigh(C)
        if w.min() <= 0:
            w = np.clip(w, 1e-6 * w.max(), None)
            C = V @ np.diag(w) @ V.T
            npd = True
        vh = v / np.linalg.norm(v)
        x = rp - rp.dot(vh) * vh
        d = float(np.linalg.norm(x))
        x = x / d if d > 0 else np.array([1.0, 0, 0])
        y = np.cross(vh, x)
        B = np.stack([x, y])
        C2 = B @ C @ B.T
        det = C2[0, 0] * C2[1, 1] - C2[0, 1] ** 2
        ia, ib, ic = C2[1, 1] / det, -C2[0, 1] / det, C2[0, 0] / det
        gx, gw = np.polynomial.legendre.leggauss(24)
        rr = (gx + 1) / 2 * hbr
        wr = gw / 2 * hbr
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        px = (rr[:, None] * np.cos(th)[None, :]).ravel()
        py = (rr[:, None] * np.sin(th)[None, :]).ravel()
        lw = np.log((wr[:, None] * rr[:, None] * (2 * np.pi / 48) * np.ones(48)[None, :]).ravel())
        dx = px - d
        q = ia * dx * dx + 2 * ib * dx * py + ic * py * py
        lp = -0.5 * q - math.log(2 * math.pi * math.sqrt(det)) + lw
        m = lp.max()
        log10 = float((m + math.log(np.exp(lp - m).sum())) / math.log(10))
        ev, evec = np.linalg.eigh(C2)
        sx, sy = float(math.sqrt(max(ev[1], 0))), float(math.sqrt(max(ev[0], 0)))
        theta = float(math.degrees(math.atan2(evec[1, 1], evec[0, 1])))
        maha = float(math.sqrt(max(d * d * ic, 0)))  # miss along x
        k_star = maha / math.sqrt(2) if maha > 0 else 0.0
        if maha > 0:
            log10_max = float(math.log10(hbr ** 2 / (2 * math.sqrt(det))) + math.log10(2 / (maha ** 2)) - 1 / math.log(10))
        else:
            log10_max = log10
        status = "OK"
        if vrel is not None and vrel < 100:
            status = "LOW_VREL"
        elif npd:
            status = "NPD_CLIPPED"
        return {**base, "status": status, "log10": max(log10, -300.0), "sigma_plane_m": [sx, sy], "theta_deg": theta,
                "mahalanobis": maha, "k_star": k_star, "log10_pc_max": min(log10_max, 0.0), "dilution": k_star > 1,
                "miss_xy_m": [d, 0.0]}
    except Exception as e:
        log.warning("local_pc_failed", error=str(e))
        return {**base, "status": "OK", "log10": reported, "method": "reported-risk", "dilution": False}


# ---------------------------------------------------------------------------
# Local rules fallback (design §3.4 table)
# ---------------------------------------------------------------------------
def fmt_pc(log10: float | None) -> str:
    if log10 is None:
        return "n/a"
    if log10 < -10:
        return "< 1e-10" if log10 > -300 else "< 1e-300"
    m, e = math.modf(log10)
    exp = math.floor(log10)
    mant = 10 ** (log10 - exp)
    if round(mant, 1) >= 10:
        mant, exp = 1.0, exp + 1
    return f"{mant:.1f}e{exp}"


def fmt_miss(m: float | None) -> str:
    if m is None:
        return "n/a"
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.2f} km"


def local_assess(row: dict, pc: dict, mode: str = "BRIEF") -> dict:
    miss = fnum(row.get("miss_distance"))
    ttca = fnum(row.get("time_to_tca"))
    reported = fnum(row.get("risk"))
    status = pc.get("status", "OK")
    lp = pc.get("log10")
    rule_ids: list[str] = []
    reasons: list[str] = []
    citations: list[str] = []
    flags: list[str] = []
    # Pc sub-tier
    if status == "COV_UNKNOWN" or lp is None:
        pc_tier = "UNKNOWN"
        rule_ids.append("Q-COV-DEFAULT")
        reasons.append("Chaser covariance is a default (10 Earth radii): not actionable")
        citations.append("CARA Vol 2 App. P.7")
        flags.append("Q-COV-DEFAULT")
    else:
        eff = lp
        if reported is not None and abs(lp - reported) > 1 and max(lp, reported) >= -6:
            rule_ids.append("Q-PHYS-DISAGREE")
            reasons.append(f"Our Pc {fmt_pc(lp)} vs reported {fmt_pc(reported)}: sources disagree")
            flags.append("Q-PHYS-DISAGREE")
            eff = max(lp, reported)
        if eff >= -4:
            pc_tier = "CRITICAL"
            rule_ids.append("R-PC-CRIT")
            reasons.append(f"Pc {fmt_pc(eff)} ≥ 1e-4")
            citations.append("brief; CARA Vol 2 App. F")
        elif eff >= -6:
            pc_tier = "ELEVATED"
            rule_ids.append("R-PC-ELEV")
            reasons.append(f"Pc {fmt_pc(eff)} in [1e-6, 1e-4)")
            citations.append("brief")
        else:
            pc_tier = "SAFE"
        if status == "LOW_VREL":
            v = fnum(row.get("relative_speed")) or 0
            rule_ids.append("Q-LOW-VREL")
            reasons.append(f"Relative speed {v:.0f} m/s: 2D Pc assumption invalid")
            citations.append("CARA App. N.3")
            flags.append("Q-LOW-VREL")
        if pc.get("dilution"):
            flags.append("Q-DILUTION")
            rule_ids.append("Q-DILUTION")
            reasons.append(f"Low Pc may be dilution: covariance larger than worst case (k* {pc.get('k_star', 0):.1f})")
            citations.append("CARA App. M.4")
    # miss sub-tier
    if miss is not None and miss < 1000:
        miss_tier = "CRITICAL"
        rule_ids.append("R-MISS-CRIT")
        reasons.append(f"Miss {fmt_miss(miss)} < 1 km (miss-distance screen)")
        citations.append("brief; caution CARA App. M.2")
    elif miss is not None and miss <= 5000:
        miss_tier = "ELEVATED"
        rule_ids.append("R-MISS-ELEV")
        reasons.append(f"Miss {fmt_miss(miss)} in [1, 5] km")
        citations.append("brief")
    else:
        miss_tier = "SAFE"
    # combine
    pc_eff = "ELEVATED" if pc_tier == "UNKNOWN" else pc_tier
    if mode == "CARA_PC":
        tier = pc_eff
        if miss_tier != "SAFE":
            flags.append("MISS-SCREEN")
    else:
        tier = TIERS[max(TIER_RANK[pc_eff], TIER_RANK[miss_tier])]
    if pc_tier == "UNKNOWN":
        tier = TIERS[max(TIER_RANK[tier], 1)]
    if tier == "SAFE" and not [r for r in rule_ids if r.startswith("Q-")]:
        rule_ids.append("R-SAFE")
        reasons.append("Pc < 1e-6 and miss > 5 km")
        citations.append("brief")
    if ttca is not None and ttca <= 1.0:
        rule_ids.append("T-MCP")
        h = ttca * 24
        reasons.append(f"T−{h:.1f} h: inside commitment window" if h >= 0 else f"T+{-h:.1f} h: after TCA")
        citations.append("CARA App. F")
        flags.append("T-MCP")
    return {"tier": tier, "pc_tier": pc_tier, "miss_tier": miss_tier, "mode": mode, "rule_ids": rule_ids,
            "reasons": reasons, "citations": citations, "flags": flags, "source": "local-rules"}


# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, settings):
        self.s = settings
        self.lock = threading.RLock()
        self.df = None
        self.by_event: dict[str, np.ndarray] = {}
        self.pc_rows: dict[tuple[str, int], dict] = {}
        self.events_meta: dict[str, dict] = {}
        self.geometry_rows: dict[str, dict] = {}
        self.forecaster = None
        self.forecaster_version: str | None = None
        self.model_sha_ok: bool | None = None
        self.mods: dict[str, Any] = {}
        self.data_source = "fixture"
        self.degraded: list[dict] = []
        self.loaded = False
        self.fc_cache: dict[tuple[str, int], dict] = {}
        self.bakeoff: dict[str, dict] = {}
        self.bakeoff_name: str | None = None

    # ---- loading --------------------------------------------------------
    def load(self) -> None:
        with self.lock:
            self.degraded = []
            for m in ("riskengine.cdm", "riskengine.pc", "riskengine.rules", "riskengine.forecast", "riskengine.orbits"):
                self.mods[m] = _try_import(m)
            self._load_data()
            self._load_pc_rows()
            self._load_events()
            self._load_geometry()
            self._load_forecaster()
            self._load_bakeoff()
            self.loaded = True
            log.info("engine_loaded", data=self.data_source, events=len(self.by_event), pc_rows=len(self.pc_rows),
                     forecaster=self.forecaster_version)

    def _degrade(self, component: str, reason: str, fallback: str) -> None:
        self.degraded.append({"component": component, "reason": reason, "fallback": fallback})

    def _load_data(self) -> None:
        import pandas as pd
        df = None
        cdm = self.mods.get("riskengine.cdm")
        pq = self.s.processed_dir / "cdm_clean.parquet"
        if cdm is not None and pq.exists():
            try:
                df = cdm.load_clean()
                self.data_source = "riskengine.cdm"
            except Exception as e:
                self._degrade("engine.cdm", f"load_clean failed: {e}", "parquet")
        if df is None and pq.exists():
            try:
                df = pd.read_parquet(pq)
                self.data_source = "cdm_clean.parquet"
            except Exception as e:
                self._degrade("engine.cdm", f"cdm_clean.parquet unreadable: {e}", "raw csv")
        if df is None:
            df = self._load_raw()
        if df is None:
            self.data_source = "fixture"
            self._degrade("engine.data", "no processed or raw data found", "built-in fixture (train:2563)")
            rows = []
            for k, rs in FIXTURE.items():
                for i, r in enumerate(rs):
                    rows.append({**r, "event_key": k, "split": k.split(":")[0], "event_id": int(k.split(":")[1]),
                                 "cdm_index": i})
            df = pd.DataFrame(rows)
        if "event_key" not in df.columns:
            df["event_key"] = df["split"].astype(str) + ":" + df["event_id"].astype(int).astype(str)
        if "cdm_index" not in df.columns:
            df = df.sort_values(["event_key", "time_to_tca"], ascending=[True, False])
            df["cdm_index"] = df.groupby("event_key").cumcount()
        df = df.sort_values(["event_key", "cdm_index"]).reset_index(drop=True)
        self.df = df
        self.by_event = {k: np.asarray(v) for k, v in df.groupby("event_key", sort=False).indices.items()}

    def _load_raw(self):
        import pandas as pd
        frames = []
        for split in ("train", "test"):
            p = self.s.raw_dir / f"{split}_data.csv"
            if not p.exists():
                continue
            try:
                d = pd.read_csv(p)
                d["split"] = split
                frames.append(d)
            except Exception as e:
                self._degrade("engine.data", f"raw {split} unreadable: {e}", "skip")
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        df["event_key"] = df["split"] + ":" + df["event_id"].astype(int).astype(str)
        for side in ("t", "c"):
            cols = [f"{side}_sigma_{a}" for a in ("r", "t", "n")]
            bad = np.zeros(len(df), bool)
            for c in cols:
                bad |= (df[c] >= SENT).values | df[c].isna().values
            df[f"cov_unknown_{side}"] = bad.astype(int)
            for c in cols:
                df.loc[df[c] >= SENT, c] = np.nan
        self.data_source = "raw-csv (backend fallback cleaning)"
        self._degrade("engine.data", "data/processed/cdm_clean.parquet missing", "raw CSV cleaned by backend adapter")
        return df

    def _load_pc_rows(self) -> None:
        import pandas as pd
        p = self.s.processed_dir / "pc_rows.parquet"
        self.pc_rows = {}
        if not p.exists():
            self._degrade("engine.pc_rows", "pc_rows.parquet missing", "live Pc per CDM")
            return
        try:
            d = pd.read_parquet(p)
            for rec in d.to_dict("records"):
                self.pc_rows[(rec["event_key"], int(rec["cdm_index"]))] = rec
        except Exception as e:
            self._degrade("engine.pc_rows", f"unreadable: {e}", "live Pc per CDM")

    def _load_events(self) -> None:
        import pandas as pd
        p = self.s.processed_dir / "events.parquet"
        self.events_meta = {}
        if p.exists():
            try:
                for rec in pd.read_parquet(p).to_dict("records"):
                    self.events_meta[rec["event_key"]] = rec
            except Exception as e:
                self._degrade("engine.events", f"unreadable: {e}", "derive from CDM rows")

    def _load_geometry(self) -> None:
        import pandas as pd
        p = self.s.processed_dir / "geometry.parquet"
        self.geometry_rows = {}
        if p.exists():
            try:
                for rec in pd.read_parquet(p).to_dict("records"):
                    self.geometry_rows[rec["event_key"]] = rec
            except Exception as e:
                self._degrade("engine.geometry", f"unreadable: {e}", "elements from CDM rows")

    def _load_forecaster(self) -> None:
        self.forecaster = None
        self.forecaster_version = None
        self.model_sha_ok = None
        mod = self.mods.get("riskengine.forecast")
        reg = self.s.repo_dir / "registry" / "tier_forecaster"
        cur = reg / "CURRENT"
        if not cur.exists():
            self._degrade("model", "registry/tier_forecaster/CURRENT missing", "persistence-empirical forecast")
            return
        version = cur.read_text().strip()
        path = reg / version
        try:
            from registry.registry import verify_model_dir
            self.model_sha_ok = verify_model_dir(path)
        except Exception:
            self.model_sha_ok = None
        if self.model_sha_ok is False:
            self._degrade("model", "artifact sha256 mismatch", "persistence-empirical forecast")
            return
        if mod is None or not hasattr(mod, "Forecaster"):
            self._degrade("model", "riskengine.forecast unavailable", "persistence-empirical forecast")
            return
        try:
            self.forecaster = mod.Forecaster.load(path)
            self.forecaster_version = version
        except Exception as e:
            self._degrade("model", f"Forecaster.load failed: {e}", "persistence-empirical forecast")

    # ---- queries ----------------------------------------------------------
    def has_event(self, key: str) -> bool:
        return key in self.by_event

    def event_rows(self, key: str) -> list[dict]:
        idx = self.by_event.get(key)
        if idx is None:
            return []
        recs = self.df.iloc[idx].to_dict("records")
        recs.sort(key=lambda r: int(r.get("cdm_index", 0)))
        return recs

    def event_keys(self) -> list[str]:
        return list(self.by_event.keys())

    def pc_for(self, key: str, cdm_index: int, row: dict) -> tuple[dict, str]:
        rec = self.pc_rows.get((key, int(cdm_index)))
        if rec is not None:
            return self._pc_from_rec(rec), "precomputed"
        mod = self.mods.get("riskengine.pc")
        if mod is not None and hasattr(mod, "pc_for_row"):
            try:
                return self._norm_pc(mod.pc_for_row(row)), "riskengine.pc"
            except Exception as e:
                log.warning("riskengine_pc_failed", error=str(e)[:200])
        return self._norm_pc(local_pc(row)), "local"

    def pc_live(self, row: dict) -> tuple[dict, str]:
        mod = self.mods.get("riskengine.pc")
        if mod is not None and hasattr(mod, "pc_for_row"):
            try:
                return self._norm_pc(mod.pc_for_row(row)), "riskengine.pc"
            except Exception as e:
                log.warning("riskengine_pc_failed", error=str(e)[:200])
        return self._norm_pc(local_pc(row)), "local"

    @staticmethod
    def _pc_from_rec(rec: dict) -> dict:
        k = fnum(rec.get("k_star"))
        pc = {"status": rec.get("pc_status") or "OK", "log10": fnum(rec.get("log10_pc")), "method": "2D-Foster-GL24x48",
              "hbr_m": fnum(rec.get("hbr_m")) or 0.0, "theta_deg": fnum(rec.get("theta_deg")),
              "mahalanobis": fnum(rec.get("mahalanobis")), "k_star": k, "log10_pc_max": fnum(rec.get("log10_pc_max")),
              "dilution": bool(k is not None and k > 1)}
        sx, sy = fnum(rec.get("sigma_x")), fnum(rec.get("sigma_y"))
        if sx is not None and sy is not None:
            pc["sigma_plane_m"] = [sx, sy]
        mx, my = fnum(rec.get("miss_x")), fnum(rec.get("miss_y"))
        if mx is not None and my is not None:
            pc["miss_xy_m"] = [mx, my]
        return Engine._norm_pc(pc)

    @staticmethod
    def _norm_pc(pc: dict) -> dict:
        pc = {k: v for k, v in dict(pc).items()}
        st = pc.get("status") or "OK"
        if st not in ("OK", "COV_UNKNOWN", "LOW_VREL", "NPD_CLIPPED"):
            st = "OK"
        pc["status"] = st
        pc["log10"] = fnum(pc.get("log10"))
        if st == "COV_UNKNOWN":
            pc["log10"] = None
        pc.setdefault("method", "2D-Foster-GL24x48")
        pc["hbr_m"] = fnum(pc.get("hbr_m")) or 0.0
        for k in ("theta_deg", "mahalanobis", "k_star", "log10_pc_max"):
            if k in pc:
                v = fnum(pc[k])
                if v is None:
                    pc.pop(k)
                else:
                    pc[k] = v
        if "sigma_plane_m" in pc:
            sp = [fnum(x) for x in (pc["sigma_plane_m"] or [])]
            if len(sp) != 2 or None in sp:
                pc.pop("sigma_plane_m")
            else:
                pc["sigma_plane_m"] = sp
        pc["dilution"] = bool(pc.get("dilution", False))
        return pc

    def assess(self, row: dict, pc: dict, mode: str = "BRIEF") -> tuple[dict, str]:
        mod = self.mods.get("riskengine.rules")
        if mod is not None and hasattr(mod, "assess"):
            try:
                t = dict(mod.assess(row, pc, mode=mode))
                return self._norm_tier(t, mode), "riskengine.rules"
            except Exception as e:
                log.warning("riskengine_rules_failed", error=str(e)[:200])
        return self._norm_tier(local_assess(row, pc, mode), mode), "local"

    @staticmethod
    def _norm_tier(t: dict, mode: str) -> dict:
        t["tier"] = t.get("tier") if t.get("tier") in TIERS else "ELEVATED"
        t["pc_tier"] = t.get("pc_tier") if t.get("pc_tier") in TIERS + ["UNKNOWN"] else "UNKNOWN"
        t["miss_tier"] = t.get("miss_tier") if t.get("miss_tier") in TIERS else "SAFE"
        t["mode"] = t.get("mode") if t.get("mode") in ("BRIEF", "CARA_PC") else mode
        for k in ("rule_ids", "reasons", "citations", "flags"):
            v = t.get(k)
            t[k] = [str(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else []
        if not t["flags"]:
            t["flags"] = [r for r in t["rule_ids"] if r.startswith("Q-") or r.startswith("T-")]
        return t

    # ---- forecast sources --------------------------------------------------
    def _load_bakeoff(self) -> None:
        """registry/forecaster_bakeoff predictions (per event), preferred at the T-2 d CDM when present."""
        import pandas as pd
        base = self.s.repo_dir / "registry" / "forecaster_bakeoff"
        cands = [base / "predictions.parquet"]
        if (base / "CURRENT").exists():
            cands.insert(0, base / (base / "CURRENT").read_text().strip() / "predictions.parquet")
        for p in cands:
            if p.exists():
                try:
                    mt = p.stat().st_mtime
                    if getattr(self, "_bakeoff_mtime", None) == mt:
                        return
                    d = pd.read_parquet(p)
                    self.bakeoff = {r["event_key"]: r for r in d.to_dict("records")}
                    self._bakeoff_mtime = mt
                    card = p.parent / "model_card.json"
                    name = None
                    if card.exists():
                        import json
                        c = json.loads(card.read_text())
                        name = c.get("name") or c.get("model") or c.get("winner")
                    self.bakeoff_name = name
                    log.info("bakeoff_loaded", n=len(self.bakeoff), model=name)
                except Exception as e:
                    self._degrade("model.bakeoff", f"unreadable: {e}", "tier_forecaster")
                return
        self.bakeoff = {}

    def prepare_forecasts(self, keys: list[str], tau_watch: float = 0.20) -> None:
        """Batch the live model over every CDM prefix that needs it (one feature build + one predict call)."""
        if self.forecaster is None:
            return
        try:
            import pandas as pd
            from riskengine.features import event_features
            frames = []
            for key in keys:
                rows = self.event_rows(key)
                for k, r in enumerate(rows):
                    t = fnum(r.get("time_to_tca"))
                    if t is None or not (2.0 <= t <= 7.0) or (key, k) in self.fc_cache:
                        continue
                    if self._served_source(key, k)[0] != "model":
                        continue
                    early = [x for x in rows[: k + 1] if (fnum(x.get("time_to_tca")) or 0) >= 2.0]
                    df = pd.DataFrame(early)
                    df["event_key"] = f"{key}#{k}"
                    frames.append(df)
            if not frames:
                return
            F = event_features(pd.concat(frames, ignore_index=True))
            P = self.forecaster.predict_features(F)
            for idx, r in P.iterrows():
                key, k = str(idx).rsplit("#", 1)
                self.fc_cache[(key, int(k))] = {"p_safe": float(r.p_safe), "p_elevated": float(r.p_elevated),
                                                "p_critical": float(r.p_critical)}
        except Exception as e:
            log.warning("prepare_forecasts_failed", error=str(e)[:200])

    def _served_source(self, key: str, k: int) -> tuple[str, dict | None]:
        """Which forecast is honest to show at CDM k: bakeoff/OOF at the T-2 d CDM of train events, the model elsewhere."""
        meta = self.events_meta.get(key) or {}
        ge2 = fnum(meta.get("latest_ge2_cdm_index"))
        at_t2 = ge2 is not None and int(ge2) == k
        bo = (getattr(self, "bakeoff", None) or {}).get(key)
        if bo is not None and at_t2:
            return "bakeoff", bo
        if meta.get("p_source") == "oof_by_event":
            if at_t2 and fnum(meta.get("p_critical")) is not None:
                return "oof", meta
            return "persistence", None  # in-sample model values on training events would be overfit
        return "model", None

    def forecast(self, key: str, rows: list[dict], tier_now: str, tau_watch: float = 0.20) -> dict:
        """Forecast at the latest released CDM (only valid for ttca in [2, 7] d). Never raises."""
        ttca_now = fnum(rows[-1].get("time_to_tca")) if rows else None
        early = [r for r in rows if (fnum(r.get("time_to_tca")) or 0) >= 2.0]
        if not early or ttca_now is None or not (2.0 <= ttca_now <= 7.0):
            return {"valid": False, "reason": "inside 2 d: physics on the latest CDM governs"}
        last = early[-1]
        pers = brief_tier(fnum(last.get("risk")), fnum(last.get("miss_distance")))
        base = {"persistence_tier": pers, "p_critical_empirical": P_CRIT_EMPIRICAL.get(pers, 0.053)}
        k = int(rows[-1].get("cdm_index", len(rows) - 1))
        src, rec = self._served_source(key, k) if self.forecaster is not None or key in (getattr(self, "bakeoff", None) or {}) \
            else ("persistence", None)
        probs = None
        model, p_source = None, None
        try:
            if src == "bakeoff":
                pc = fnum(rec.get("p_critical_calibrated"))
                pc = pc if pc is not None else fnum(rec.get("p_critical"))
                probs = {"p_safe": fnum(rec.get("p_safe")), "p_elevated": fnum(rec.get("p_elevated")), "p_critical": pc}
                model = f"forecaster_bakeoff:{rec.get('model') or getattr(self, 'bakeoff_name', None) or 'winner'}"
                p_source = f"bakeoff ({rec.get('split') or 'n/a'})"
            elif src == "oof":
                probs = {"p_safe": fnum(rec.get("p_safe")), "p_elevated": fnum(rec.get("p_elevated")),
                         "p_critical": fnum(rec.get("p_critical"))}
                model, p_source = f"tier_forecaster:{self.forecaster_version}", "oof_by_event (out-of-fold)"
            elif src == "model":
                probs = self.fc_cache.get((key, k))
                if probs is None:
                    import pandas as pd
                    df = pd.DataFrame([{kk: v for kk, v in r.items() if not str(kk).startswith("_")} for r in rows])
                    out = self.forecaster.predict(df, ttca_now_d=ttca_now, tau_watch=tau_watch)
                    probs = {"p_safe": out["p_tiers"]["SAFE"], "p_elevated": out["p_tiers"]["ELEVATED"],
                             "p_critical": out["p_critical"]}
                    self.fc_cache[(key, k)] = probs
                model, p_source = f"tier_forecaster:{self.forecaster_version}", "model (out-of-sample)"
        except Exception as e:
            log.warning("forecast_failed", error=str(e)[:200])
            self._degrade("model", f"predict failed: {e}", "persistence-empirical")
            probs = None
        if probs is None or probs.get("p_critical") is None:
            p = base["p_critical_empirical"]
            rest = 1 - p
            split = {"SAFE": (0.93, 0.07), "ELEVATED": (0.35, 0.65)}.get(pers, (0.3, 0.7))
            note = ("ML value is only shown out-of-fold at the T−2 d CDM of training events"
                    if src == "persistence" and self.forecaster is not None else "forecast model unavailable")
            return {"valid": True, "p_tiers": {"SAFE": rest * split[0], "ELEVATED": rest * split[1], "CRITICAL": p},
                    "p_critical": p, "calib_bin": None, "baseline": base,
                    "watch": bool(p >= tau_watch and pers != "CRITICAL"), "model": "persistence-empirical",
                    "model_version": "persistence-empirical", "p_source": "persistence-empirical", "fallback": True,
                    "note": note}
        p = float(probs["p_critical"])
        cb = None
        if self.forecaster is not None:
            try:
                cb = self.forecaster.calib_bin(p)
            except Exception:
                cb = None
        return {"valid": True, "p_tiers": {"SAFE": probs.get("p_safe"), "ELEVATED": probs.get("p_elevated"), "CRITICAL": p},
                "p_critical": p, "calib_bin": cb, "baseline": base, "watch": bool(p >= tau_watch and pers != "CRITICAL"),
                "model": model, "model_version": model, "p_source": p_source, "fallback": False}

    def final_outcome(self, key: str) -> dict | None:
        if key.startswith("test:") or key.startswith("adhoc:"):
            return None
        meta = self.events_meta.get(key) or {}
        rows = self.event_rows(key)
        if not rows:
            return None
        last = rows[-1]
        lp = fnum(meta.get("final_log10_risk")) if "final_log10_risk" in meta else fnum(last.get("risk"))
        miss = fnum(meta.get("final_miss_m")) if "final_miss_m" in meta else fnum(last.get("miss_distance"))
        tier = meta.get("final_tier") if meta.get("final_tier") in TIERS else brief_tier(lp, miss)
        return {"tier": tier, "log10_pc": lp, "miss_m": miss, "ttca_d": fnum(last.get("time_to_tca"))}

    def event_meta(self, key: str) -> dict:
        rows = self.event_rows(key)
        meta = dict(self.events_meta.get(key) or {})
        first = rows[0] if rows else {}
        return {"event_key": key, "mission_id": fnum(first.get("mission_id")), "n_cdm": len(rows),
                "c_object_type": first.get("c_object_type"), "first_ttca_d": fnum(first.get("time_to_tca")),
                "last_ttca_d": fnum(rows[-1].get("time_to_tca")) if rows else None,
                "p_critical_oof": fnum(meta.get("p_critical"))}

    def geometry(self, key: str) -> dict:
        mod = self.mods.get("riskengine.orbits")
        if mod is not None and hasattr(mod, "geometry"):
            try:
                g = dict(mod.geometry(key))
                g.setdefault("label", "RECONSTRUCTED GEOMETRY")
                g["source"] = "riskengine.orbits"
                return g
            except Exception as e:
                log.warning("geometry_failed", error=str(e)[:200])
        rec = self.geometry_rows.get(key)
        if rec is not None:
            return {"label": "RECONSTRUCTED GEOMETRY", "source": "geometry.parquet", **rec}
        rows = self.event_rows(key)
        if not rows:
            return {}
        r = rows[-1]
        el = lambda s: {"a_km": fnum(r.get(f"{s}_j2k_sma")), "e": fnum(r.get(f"{s}_j2k_ecc")),
                        "i_deg": fnum(r.get(f"{s}_j2k_inc"))}
        return {"label": "RECONSTRUCTED GEOMETRY", "source": "cdm-row elements (partial; RAAN/argp unknown)",
                "target": el("t"), "chaser": el("c"), "lat_deg": fnum(r.get("geocentric_latitude")),
                "azimuth_deg": fnum(r.get("azimuth")), "elevation_deg": fnum(r.get("elevation"))}

    def free_play_candidates(self, n: int, seed: int) -> list[str]:
        keys = [k for k, idx in self.by_event.items() if k.startswith("train:") and len(idx) >= 3]
        if not keys:
            return list(self.by_event.keys())[:n]
        keys.sort()
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
        return [keys[i] for i in sorted(pick)]

    def status(self) -> dict:
        return {"data_source": self.data_source, "n_events": len(self.by_event), "pc_rows": len(self.pc_rows),
                "riskengine": {m.split(".")[1]: bool(v) for m, v in self.mods.items()},
                "forecaster": self.forecaster_version, "model_sha_ok": self.model_sha_ok, "degraded": self.degraded}
