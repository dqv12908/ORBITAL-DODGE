"""2D probability of collision from the CDM covariance (port of research/experiments/pc2d.py).

Method "2D-Foster-GL24x48":
1. per-object RTN covariance  S_ij = rho_ij s_i s_j  (non-PD 3x3 -> eigenvalues clipped, status NPD_CLIPPED)
2. chaser covariance rotated into the target RTN frame: v_c = (0, v_t, 0) + v_rel, v_t = vis-viva on t_j2k_sma,
   R_c ~ R_t, N_c = R x v_c, T_c = N_c x R
3. C = C_t + C_c, encounter plane perpendicular to v_rel; x axis along the miss vector, y = v_hat x x
4. bivariate normal centred at the miss vector integrated over a disc of radius HBR = (t_span + c_span)/2,
   Gauss-Legendre 24 radial x 48 angular, summed with log-sum-exp (log10 Pc exact to about -300 and below)
5. extras: principal sigmas, ellipse angle, Mahalanobis distance of the miss, dilution k* and Pc_max

Units: metres, m/s; t_j2k_sma in km (as in the dataset).
Status: OK | COV_UNKNOWN | LOW_VREL (v_rel < 100 m/s, 2D assumption invalid; Pc still computed) | NPD_CLIPPED.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp

MU_M3S2 = 398600.4418e9
METHOD = "2D-Foster-GL24x48"
LOW_VREL_MS = 100.0
DEFAULT_HBR_M = 5.0
NR, NTH = 24, 48
LN10 = math.log(10.0)

_gx, _gw = np.polynomial.legendre.leggauss(NR)
_U = (_gx + 1) / 2                     # radial nodes on [0, 1]
_WU = _gw / 2
_TH = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
_UX = (_U[:, None] * np.cos(_TH)[None, :]).ravel()
_UY = (_U[:, None] * np.sin(_TH)[None, :]).ravel()
_LWU = np.log((_WU[:, None] * _U[:, None] * (2 * np.pi / NTH) * np.ones(NTH)[None, :]).ravel())

PC_COLS = ["pc_status", "log10_pc", "hbr_m", "sigma_x", "sigma_y", "theta_deg", "mahalanobis", "k_star",
           "log10_pc_max", "miss_x", "miss_y", "c2_xx", "c2_xy", "c2_yy", "npd_clipped", "v_rel_ms"]


def _col(df, name, default=np.nan):
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(float)
    return np.full(len(df), default, float)


def cov_rtn(s_r, s_t, s_n, c_tr, c_nr, c_nt):
    n = len(s_r)
    C = np.zeros((n, 3, 3))
    C[:, 0, 0] = s_r ** 2
    C[:, 1, 1] = s_t ** 2
    C[:, 2, 2] = s_n ** 2
    C[:, 0, 1] = C[:, 1, 0] = c_tr * s_r * s_t
    C[:, 0, 2] = C[:, 2, 0] = c_nr * s_r * s_n
    C[:, 1, 2] = C[:, 2, 1] = c_nt * s_t * s_n
    return C


def clip_npd(C):
    """Clip negative eigenvalues of symmetric (n,k,k) matrices (CARA App. N.8.2 style). Returns (C, clipped)."""
    w, V = np.linalg.eigh(C)
    bad = w[:, 0] < 0
    if bad.any():
        floor = 1e-12 * np.abs(w[bad, -1:]) + 1e-6
        wb = np.maximum(w[bad], floor)
        C = C.copy()
        C[bad] = (V[bad] * wb[:, None, :]) @ np.transpose(V[bad], (0, 2, 1))
    return C, bad


def chaser_to_target_rtn(sma_km, vrel):
    """Rotation matrices M (n,3,3) with v_target_rtn = M @ v_chaser_rtn."""
    a = np.where(np.isfinite(sma_km) & (sma_km > 0), sma_km, 7000.0) * 1e3
    vt = np.sqrt(MU_M3S2 / a)
    vc = np.stack([np.zeros_like(vt), vt, np.zeros_like(vt)], 1) + vrel
    R = np.tile([1.0, 0, 0], (len(a), 1))
    N = np.cross(R, vc)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = np.cross(N, R)
    return np.stack([R, T, N], 2)


def log10_pc_disc(mx, my, cxx, cxy, cyy, hbr, chunk=4000):
    """Vectorised log10 of the 2D Gaussian (mean (mx,my), cov [[cxx,cxy],[cxy,cyy]]) mass inside a disc of
    radius hbr centred at the origin. All inputs 1-D arrays (broadcastable)."""
    mx, my, cxx, cxy, cyy, hbr = np.broadcast_arrays(*(np.asarray(v, float) for v in (mx, my, cxx, cxy, cyy, hbr)))
    out = np.full(mx.shape, np.nan)
    for s in range(0, len(mx), chunk):
        sl = slice(s, s + chunk)
        det = cxx[sl] * cyy[sl] - cxy[sl] ** 2
        ia, ib, ic = cyy[sl] / det, -cxy[sl] / det, cxx[sl] / det
        h = hbr[sl][:, None]
        dx = _UX[None, :] * h - mx[sl][:, None]
        dy = _UY[None, :] * h - my[sl][:, None]
        q = ia[:, None] * dx * dx + 2 * ib[:, None] * dx * dy + ic[:, None] * dy * dy
        lp = -0.5 * q - np.log(2 * np.pi * np.sqrt(det))[:, None] + _LWU[None, :] + 2 * np.log(h)
        out[sl] = logsumexp(lp, axis=1) / LN10
    return np.minimum(out, 0.0)


def hbr_from_spans(t_span, c_span, default=DEFAULT_HBR_M):
    t = np.asarray(t_span, float)
    c = np.asarray(c_span, float)
    h = (t + c) / 2
    h = np.where(np.isfinite(h), h, np.where(np.isfinite(t), t, np.where(np.isfinite(c), c, default)))
    return np.where(h > 0, h, default)


def pc_batch(df: pd.DataFrame, hbr_m=None, chunk: int = 4000) -> pd.DataFrame:
    """Vectorised Pc for every row of a (cleaned) CDM frame. Returns a frame with PC_COLS, same index."""
    n = len(df)
    sig = {k: _col(df, k) for k in ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]}
    cor = {k: _col(df, k) for k in ["t_ct_r", "t_cn_r", "t_cn_t", "c_ct_r", "c_cn_r", "c_cn_t"]}
    rp = np.stack([_col(df, f"relative_position_{a}") for a in "rtn"], 1)
    v = np.stack([_col(df, f"relative_velocity_{a}") for a in "rtn"], 1)
    sma = _col(df, "t_j2k_sma")
    flags = np.zeros(n, bool)
    for side in "tc":
        if f"cov_unknown_{side}" in df:
            flags |= df[f"cov_unknown_{side}"].to_numpy().astype(bool)
    unknown = flags | ~np.all(np.isfinite(np.stack(list(sig.values()) + list(cor.values()), 1)), 1)
    unknown |= np.any(np.stack(list(sig.values()), 1) >= 1e7, 1)
    unknown |= ~np.all(np.isfinite(np.concatenate([rp, v], 1)), 1)
    # placeholders so the maths runs; outputs are masked afterwards
    fill = lambda a, val: np.where(np.isfinite(a), a, val)
    s = {k: fill(np.where(a >= 1e7, np.nan, a), 1.0) for k, a in sig.items()}
    c = {k: np.clip(fill(a, 0.0), -1, 1) for k, a in cor.items()}
    rp = fill(rp, 0.0)
    v = fill(v, 0.0)
    if hbr_m is None:
        hbr = hbr_from_spans(_col(df, "t_span"), _col(df, "c_span"))
    else:
        hbr = np.broadcast_to(np.asarray(hbr_m, float), (n,)).copy()

    Ct, npd_t = clip_npd(cov_rtn(s["t_sigma_r"], s["t_sigma_t"], s["t_sigma_n"], c["t_ct_r"], c["t_cn_r"], c["t_cn_t"]))
    Cc, npd_c = clip_npd(cov_rtn(s["c_sigma_r"], s["c_sigma_t"], s["c_sigma_n"], c["c_ct_r"], c["c_cn_r"], c["c_cn_t"]))
    M = chaser_to_target_rtn(sma, v)
    C = Ct + M @ Cc @ np.transpose(M, (0, 2, 1))

    vn = np.linalg.norm(v, axis=1)
    vh = np.where(vn[:, None] > 1e-9, v / np.where(vn > 1e-9, vn, 1)[:, None], np.array([0.0, 1.0, 0.0]))
    x = rp - (rp * vh).sum(1, keepdims=True) * vh
    d = np.linalg.norm(x, axis=1)
    # degenerate miss (0): any unit vector perpendicular to v
    alt = np.cross(vh, np.where(np.abs(vh[:, :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 0, 1.0])))
    alt /= np.linalg.norm(alt, axis=1, keepdims=True)
    x = np.where(d[:, None] > 1e-9, x / np.where(d > 1e-9, d, 1)[:, None], alt)
    y = np.cross(vh, x)
    B = np.stack([x, y], 1)
    C2 = B @ C @ np.transpose(B, (0, 2, 1))
    C2 = (C2 + np.transpose(C2, (0, 2, 1))) / 2
    C2, npd2 = clip_npd(C2)
    cxx, cxy, cyy = C2[:, 0, 0], C2[:, 0, 1], C2[:, 1, 1]

    lp = log10_pc_disc(d, np.zeros(n), cxx, cxy, cyy, hbr, chunk=chunk)

    w, V = np.linalg.eigh(C2)          # ascending
    sx, sy = np.sqrt(np.maximum(w[:, 1], 0)), np.sqrt(np.maximum(w[:, 0], 0))
    th = np.degrees(np.arctan2(V[:, 1, 1], V[:, 0, 1]))
    th = np.where(th > 90, th - 180, np.where(th <= -90, th + 180, th))
    det = cxx * cyy - cxy ** 2
    maha = d * np.sqrt(cyy / det)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_star = np.where(maha > 0, np.sqrt(2.0) / maha, np.inf)
        lpmax = np.where(maha > 0, np.log10(hbr ** 2 / (math.e * np.sqrt(det) * maha ** 2)), 0.0)
    lpmax = np.minimum(lpmax, 0.0)

    npd = npd_t | npd_c | npd2
    status = np.where(unknown, "COV_UNKNOWN", np.where(vn < LOW_VREL_MS, "LOW_VREL",
                                                       np.where(npd, "NPD_CLIPPED", "OK")))
    out = pd.DataFrame({
        "pc_status": status, "log10_pc": lp, "hbr_m": hbr, "sigma_x": sx, "sigma_y": sy, "theta_deg": th,
        "mahalanobis": maha, "k_star": k_star, "log10_pc_max": lpmax, "miss_x": d, "miss_y": np.zeros(n),
        "c2_xx": cxx, "c2_xy": cxy, "c2_yy": cyy, "npd_clipped": npd.astype("int8"), "v_rel_ms": vn,
    }, index=df.index)
    geo = ["log10_pc", "sigma_x", "sigma_y", "theta_deg", "mahalanobis", "k_star", "log10_pc_max",
           "c2_xx", "c2_xy", "c2_yy"]
    out.loc[unknown, geo] = np.nan
    return out


def _num(v):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return v


def pc_object(r: dict) -> dict:
    """Convert one pc_batch row (dict) to the design-6.3 `pc` object (JSON-safe)."""
    lp = _num(r.get("log10_pc"))
    ks = _num(r.get("k_star"))
    return {
        "status": r["pc_status"],
        "log10": lp,
        "method": METHOD,
        "hbr_m": _num(r.get("hbr_m")),
        "sigma_plane_m": [_num(r.get("sigma_x")), _num(r.get("sigma_y"))] if lp is not None else None,
        "theta_deg": _num(r.get("theta_deg")),
        "mahalanobis": _num(r.get("mahalanobis")),
        "k_star": ks,
        "log10_pc_max": _num(r.get("log10_pc_max")),
        "dilution": bool(ks is not None and ks > 1 and r["pc_status"] != "COV_UNKNOWN"),
        "miss_plane_m": [_num(r.get("miss_x")), _num(r.get("miss_y"))],
        "cov_plane_m2": ([[_num(r.get("c2_xx")), _num(r.get("c2_xy"))], [_num(r.get("c2_xy")), _num(r.get("c2_yy"))]]
                         if lp is not None else None),
        "npd_clipped": bool(r.get("npd_clipped", 0)),
        "v_rel_ms": _num(r.get("v_rel_ms")),
    }


def pc_for_row(row: dict, hbr_m: float | None = None) -> dict:
    """Pc for one CDM row (raw Kelvins columns or a cleaned row). Sentinel sigmas are detected here too."""
    df = pd.DataFrame([row])
    out = pc_batch(df, hbr_m=hbr_m).iloc[0].to_dict()
    return pc_object(out)
