"""Orbit reconstruction for the globe (port of research/experiments/e_orbits.py). DISPLAY ONLY - RECONSTRUCTED GEOMETRY.

Conjunction point P at latitude = geocentric_latitude, longitude SYNTHETIC (seeded by event_key), radius = t_j2k_sma
(circular approximation). Target heading from sin(psi) = cos(i_t)/cos(lat); of the two branches (northbound /
southbound) the one reproducing the chaser's c_j2k_inc / c_j2k_sma is kept. Chaser = target + RTN relative state
(exact relative geometry at TCA). Frame: Earth-centred, x towards longitude 0 at TCA (Earth-fixed at TCA), km, km/s.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

from . import paths

MU_KM = 398600.4418
RE_KM = 6378.137
LABEL = "RECONSTRUCTED GEOMETRY · longitude/RAAN synthetic"


def synthetic_lon_deg(event_key: str) -> float:
    return (zlib.crc32(event_key.encode()) % 36000) / 100.0 - 180.0


def rv2coe(r, v):
    """Vectorised Cartesian -> classical elements (km, km/s). Returns dict of arrays (angles in degrees)."""
    rn = np.linalg.norm(r, axis=1)
    vn = np.linalg.norm(v, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / hn, -1, 1)))
    nvec = np.cross(np.array([0.0, 0, 1]), h)
    nn = np.linalg.norm(nvec, axis=1)
    evec = np.cross(v, h) / MU_KM - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    a = 1 / (2 / rn - vn ** 2 / MU_KM)
    with np.errstate(invalid="ignore", divide="ignore"):
        raan = np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) % 360
        argp = np.degrees(np.arccos(np.clip((nvec * evec).sum(1) / (nn * e), -1, 1)))
        argp = np.where(evec[:, 2] < 0, 360 - argp, argp)
        nu = np.degrees(np.arccos(np.clip((evec * r).sum(1) / (e * rn), -1, 1)))
        nu = np.where((r * v).sum(1) < 0, 360 - nu, nu)
        # argument of latitude (well defined for circular orbits)
        u = np.degrees(np.arccos(np.clip((nvec * r).sum(1) / (nn * rn), -1, 1)))
        u = np.where(r[:, 2] < 0, 360 - u, u)
    small_e = e < 1e-8
    argp = np.where(small_e, 0.0, argp)
    nu = np.where(small_e, u, nu)
    return {"a_km": a, "e": e, "i_deg": inc, "raan_deg": raan, "argp_deg": argp, "nu_deg_at_tca": nu, "u_deg": u}


def _state(phi, lam, a_t, i_t, branch_desc, vrel, rrel):
    n = len(a_t)
    cp, sp, cl, sl = np.cos(phi), np.sin(phi), np.cos(lam), np.sin(lam)
    rhat = np.stack([cp * cl, cp * sl, sp], 1)
    east = np.stack([-sl, cl, np.zeros(n)], 1)
    north = np.stack([-sp * cl, -sp * sl, cp], 1)
    s = np.clip(np.cos(i_t) / np.cos(phi), -1, 1)
    psi = np.arcsin(s)
    psi = np.where(branch_desc, np.pi - psi, psi)
    vmag = np.sqrt(MU_KM / a_t)
    v = vmag[:, None] * (np.cos(psi)[:, None] * north + np.sin(psi)[:, None] * east)
    r = a_t[:, None] * rhat
    R = rhat
    N = np.cross(r, v)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = np.cross(N, R)
    dv = (vrel[:, 0:1] * R + vrel[:, 1:2] * T + vrel[:, 2:3] * N) / 1000.0
    dr = (rrel[:, 0:1] * R + rrel[:, 1:2] * T + rrel[:, 2:3] * N) / 1000.0
    return r, v, r + dr, v + dv


def reconstruct(final: pd.DataFrame) -> pd.DataFrame:
    """final: one CDM per event (the latest/final). Returns the geometry frame."""
    f = final.reset_index(drop=True)
    keys = f.event_key.to_numpy()
    lat = f.geocentric_latitude.fillna(0.0).to_numpy(float)
    lon = np.array([synthetic_lon_deg(k) for k in keys])
    a_t = f.t_j2k_sma.fillna(7000.0).to_numpy(float)
    i_t = np.radians(f.t_j2k_inc.fillna(98.0).to_numpy(float))
    phi, lam = np.radians(lat), np.radians(lon)
    vrel = f[["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]].fillna(0).to_numpy(float)
    rrel = f[["relative_position_r", "relative_position_t", "relative_position_n"]].fillna(0).to_numpy(float)
    res = {}
    for b, desc in (("asc", False), ("desc", True)):
        rt, vt, rc, vc = _state(phi, lam, a_t, i_t, np.full(len(f), desc), vrel, rrel)
        ct = rv2coe(rc, vc)
        di = np.abs(ct["i_deg"] - f.c_j2k_inc.to_numpy(float))
        da = np.abs(ct["a_km"] - f.c_j2k_sma.to_numpy(float))
        res[b] = (rt, vt, rc, vc, di, da)
    score_a = np.nan_to_num(res["asc"][4] + res["asc"][5] / 10, nan=1e9)
    score_d = np.nan_to_num(res["desc"][4] + res["desc"][5] / 10, nan=1e9)
    use_desc = score_d < score_a
    pick = lambda j: np.where(use_desc[:, None] if res["asc"][j].ndim == 2 else use_desc, res["desc"][j], res["asc"][j])
    rt, vt, rc, vc, di, da = (pick(j) for j in range(6))
    et, ec = rv2coe(rt, vt), rv2coe(rc, vc)
    out = pd.DataFrame({"event_key": keys, "split": f.split.to_numpy(), "event_id": f.event_id.to_numpy(),
                        "cdm_index": f.cdm_index.to_numpy(), "ttca_d": f.time_to_tca.to_numpy(),
                        "lat_deg": lat, "lon_deg": lon, "lon_synthetic": True, "alt_km": a_t - RE_KM,
                        "branch": np.where(use_desc, "southbound", "northbound")})
    for pre, el in (("t_", et), ("c_", ec)):
        for k in ("a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg_at_tca"):
            out[pre + k] = el[k]
    out["inc_res_deg"] = di
    out["sma_res_km"] = da
    out["ecc_res"] = np.abs(ec["e"] - f.c_j2k_ecc.to_numpy(float))
    out["t_r_km"] = list(rt)
    out["t_v_kms"] = list(vt)
    out["c_r_km"] = list(rc)
    out["c_v_kms"] = list(vc)
    out["c_object_type"] = f.c_object_type.astype(str).to_numpy()
    out["label"] = LABEL
    return out


_GEO: pd.DataFrame | None = None
_PCR: pd.DataFrame | None = None


def _clean_num(v):
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_clean_num(x) for x in v.tolist()]
    return v


def encounter_planes(event_key: str) -> list[dict]:
    global _PCR
    if _PCR is None:
        cols = ["event_key", "cdm_index", "pc_status", "log10_pc", "hbr_m", "sigma_x", "sigma_y", "theta_deg",
                "mahalanobis", "k_star", "miss_x", "miss_y", "c2_xx", "c2_xy", "c2_yy", "tier"]
        _PCR = pd.read_parquet(paths.PC_ROWS, columns=cols).set_index("event_key").sort_index()
    if event_key not in _PCR.index:
        return []
    x = _PCR.loc[[event_key]].sort_values("cdm_index")
    return [{k: _clean_num(v) for k, v in r.items()} for r in x.to_dict("records")]


def geometry(event_key: str) -> dict:
    """Reconstructed elements for target + chaser, residuals, and the encounter-plane 2x2 per CDM."""
    global _GEO
    if _GEO is None:
        _GEO = pd.read_parquet(paths.GEOMETRY).set_index("event_key")
    if event_key not in _GEO.index:
        raise KeyError(event_key)
    row = {k: _clean_num(v) for k, v in _GEO.loc[event_key].to_dict().items()}
    row["event_key"] = event_key
    row["encounter_plane"] = encounter_planes(event_key)
    return row
