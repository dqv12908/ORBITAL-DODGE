#!/usr/bin/env python3
"""ORBITAL DODGE live-sky screening: all-vs-all conjunction screening of the public
CelesTrak GP catalogue (LEO crossers) with SGP4, GPU spatial hashing, linear TCA
refinement, isotropic worst-case ("max") Pc, and validation against CelesTrak SOCRATES.

Run (on triologay):
  cd /root/orbital-dodge/research/livesky
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 .venv-sky/bin/python screen.py   # t0 = now (rounded down to 10 min)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 .venv-sky/bin/python screen.py --t0 2026-10-31T04:00:00Z --hours 72 --dt 10

Inputs (downloaded ONCE, cached under cache/, never re-fetched by this script):
  cache/gp/*.json   CelesTrak GP OMM JSON groups (active, *-debris)
  cache/gp/*.csv    CelesTrak GP OMM CSV name searches (NAME=DEB, R/B, COSMOS, OBJECT, OPS, METEOR)
  cache/satcat.csv  CelesTrak SATCAT (object type, RCS)
  cache/socrates_minrange.csv  CelesTrak SOCRATES (sort-minRange.csv)
Outputs:
  ../../data/livesky/latest.json      contract format (top conjunctions, summary, validation)
  out/events_all.csv                  every screened event < threshold in the window
  out/socrates_compare.csv            SOCRATES events in window with our match
  out/run_stats.json                  timings
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray, jday

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
DATA_OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "livesky"))
MU = 398600.4418
RE = 6378.137

# ----------------------------------------------------------------------------- catalogue

def _s(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def load_catalogue():
    recs = []
    for f in sorted(glob.glob(os.path.join(CACHE, "gp", "*.json"))):
        for d in json.load(open(f)):
            d = {k: _s(v) for k, v in d.items()}
            d["_src"] = os.path.basename(f)
            recs.append(d)
    for f in sorted(glob.glob(os.path.join(CACHE, "gp", "*.csv"))):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for d in df.to_dict("records"):
            d["_src"] = os.path.basename(f)
            recs.append(d)
    df = pd.DataFrame(recs)
    df["NORAD_CAT_ID"] = df["NORAD_CAT_ID"].astype(int)
    df["EPOCH"] = df["EPOCH"].map(lambda e: e if "." in e else e + ".000000")
    df["_ep"] = pd.to_datetime(df["EPOCH"], format="%Y-%m-%dT%H:%M:%S.%f")
    df = df.sort_values("_ep").drop_duplicates("NORAD_CAT_ID", keep="last").reset_index(drop=True)
    n = df["MEAN_MOTION"].astype(float).values * 2 * np.pi / 86400.0
    e = df["ECCENTRICITY"].astype(float).values
    a = (MU / n ** 2) ** (1 / 3)
    df["perigee_km"] = a * (1 - e) - RE
    df["apogee_km"] = a * (1 + e) - RE
    return df


def hbr_table(df):
    sat = pd.read_csv(os.path.join(CACHE, "satcat.csv"))
    sat = sat.set_index("NORAD_CAT_ID")
    typ = df["NORAD_CAT_ID"].map(sat["OBJECT_TYPE"]).fillna("UNK")
    rcs = pd.to_numeric(df["NORAD_CAT_ID"].map(sat["RCS"]), errors="coerce")
    default = typ.map({"PAY": 1.0, "R/B": 2.0, "DEB": 0.3}).fillna(0.5)
    # RCS (m^2) treated as a projected area -> equivalent-circle radius; clip to sane range.
    r = np.sqrt(rcs / np.pi).where(rcs > 0, default).clip(0.05, 15.0)
    return typ.values, rcs.values, r.values

# ----------------------------------------------------------------------------- propagation (CPU, forked pool)

G = {}


def _prop(args):
    i0, i1, jd0, fr = args
    sats = G["sats"]
    jd = np.full(i1 - i0, jd0)
    e, r, v = sats.sgp4(jd, fr)  # (N, T, 3)
    G["pos"][i0:i1] = np.transpose(r, (1, 0, 2))
    G["vel"][i0:i1] = np.transpose(v, (1, 0, 2))
    G["err"][i0:i1] = (e != 0).T
    return i1 - i0

# ----------------------------------------------------------------------------- GPU screening

HALF_OFFSETS = [(0, 0, 0)] + [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                              if (dx, dy, dz) > (0, 0, 0)]


def screen_block(torch, P, V, valid, dt, rs, dthr, dev, max_pairs=60_000_000):
    """P,V: (S,N,3) float64 torch on dev; valid (S,N) bool. Returns tensors of events
    (s_idx, a, b, tau, dmin, vrel, ra_tca(3))."""
    S, N, _ = P.shape
    flat = valid.reshape(-1).nonzero().squeeze(1)
    p = P.reshape(-1, 3)[flat]
    v = V.reshape(-1, 3)[flat]
    sidx = flat // N
    oidx = flat % N
    c = torch.floor(p / rs).long() + 32768
    c = c.clamp(1, 65534)
    key = (sidx << 48) | (c[:, 0] << 32) | (c[:, 1] << 16) | c[:, 2]
    skey, order = torch.sort(key)
    M = key.numel()
    out = []
    ar = torch.arange(M, device=dev)
    for (dx, dy, dz) in HALF_OFFSETS:
        nkey = key + ((dx << 32) + (dy << 16) + dz)
        lo = torch.searchsorted(skey, nkey, right=False)
        hi = torch.searchsorted(skey, nkey, right=True)
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            continue
        # chunk over source points if expansion is large
        csum = torch.cumsum(cnt, 0)
        nchunks = max(1, math.ceil(tot / max_pairs))
        bounds = torch.searchsorted(csum, torch.linspace(0, tot, nchunks + 1, device=dev).long()[1:-1])
        starts = [0] + bounds.tolist() + [M]
        for j in range(len(starts) - 1):
            s0, s1 = starts[j], starts[j + 1]
            if s1 <= s0:
                continue
            cc = cnt[s0:s1]
            t = int(cc.sum())
            if t == 0:
                continue
            a = torch.repeat_interleave(ar[s0:s1], cc)
            first = torch.cumsum(cc, 0) - cc
            k = torch.arange(t, device=dev) - torch.repeat_interleave(first, cc)
            b = order[lo[a] + k]
            if (dx, dy, dz) == (0, 0, 0):
                m = b > a
                a, b = a[m], b[m]
            dr = p[b] - p[a]
            d2 = (dr * dr).sum(1)
            m = d2 < rs * rs
            a, b, dr = a[m], b[m], dr[m]
            if a.numel() == 0:
                continue
            dv = v[b] - v[a]
            vv = (dv * dv).sum(1).clamp_min(1e-12)
            tau = -(dr * dv).sum(1) / vv
            m = (tau >= -dt / 2 - 0.25) & (tau < dt / 2 + 0.25)
            a, b, dr, dv, tau = a[m], b[m], dr[m], dv[m], tau[m]
            rm = dr + dv * tau[:, None]
            dmin = torch.sqrt((rm * rm).sum(1))
            m = dmin < dthr
            if m.sum() == 0:
                continue
            a, b, tau, dmin, dv = a[m], b[m], tau[m], dmin[m], dv[m]
            ra = p[a] + v[a] * tau[:, None]
            out.append(torch.cat([sidx[a][:, None].double(), oidx[a][:, None].double(), oidx[b][:, None].double(),
                                  tau[:, None], dmin[:, None], torch.sqrt((dv * dv).sum(1))[:, None], ra], 1).cpu())
    if not out:
        return np.zeros((0, 9))
    return torch.cat(out).numpy()

# ----------------------------------------------------------------------------- geometry / Pc

def gmst_rad(jd_ut1):
    t = (jd_ut1 - 2451545.0) / 36525.0
    g = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t ** 3
    return np.deg2rad((g % 86400.0) / 240.0)


def teme_to_geodetic(r, jd):
    th = gmst_rad(jd)
    x = r[:, 0] * np.cos(th) + r[:, 1] * np.sin(th)
    y = -r[:, 0] * np.sin(th) + r[:, 1] * np.cos(th)
    z = r[:, 2]
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h


def max_pc_isotropic(d_km, hbr_m):
    """Worst case over an isotropic 2-D Gaussian of unknown sigma in the encounter plane:
    Pc(sigma) = P[ncx2(df=2, nc=d^2/sigma^2) <= R^2/sigma^2]; return max over sigma."""
    from scipy.stats import ncx2
    d = np.maximum(np.asarray(d_km, float) * 1000.0, 1e-3)
    R = np.asarray(hbr_m, float)
    out = np.empty_like(d)
    ks = np.logspace(-3, 1.5, 300)
    for i in range(len(d)):
        sig = d[i] * ks
        pc = ncx2.cdf((R[i] / sig) ** 2, 2, (d[i] / sig) ** 2)
        out[i] = np.nanmax(pc)
    return np.clip(out, 1e-300, 1.0)

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default=None)
    ap.add_argument("--hours", type=float, default=72)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=5.0, help="km")
    ap.add_argument("--perigee-max", type=float, default=2000.0)
    ap.add_argument("--max-age-days", type=float, default=30.0)
    ap.add_argument("--chunk", type=int, default=1440)
    ap.add_argument("--gpu-block", type=int, default=120)
    ap.add_argument("--workers", type=int, default=52)
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()
    T0 = time.time()
    stats = {}
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)

    now = datetime.now(timezone.utc)
    if args.t0:
        t0 = datetime.fromisoformat(args.t0.replace("Z", "+00:00"))
    else:
        t0 = now.replace(minute=now.minute // 10 * 10, second=0, microsecond=0)
    nstep = int(round(args.hours * 3600 / args.dt)) + 1

    cat = load_catalogue()
    stats["n_catalogue"] = len(cat)
    age = (pd.Timestamp(t0.replace(tzinfo=None)) - cat["_ep"]).dt.total_seconds() / 86400
    keep = (cat["perigee_km"] < args.perigee_max) & (cat["perigee_km"] > 80) & (age < args.max_age_days)
    cat = cat[keep].reset_index(drop=True)
    sats = []
    good = []
    for i, d in enumerate(cat.to_dict("records")):
        s = Satrec()
        try:
            omm.initialize(s, d)
        except Exception:
            dd = dict(d)
            dd["NORAD_CAT_ID"] = "0"
            try:
                omm.initialize(s, dd)
            except Exception:
                continue
        sats.append(s)
        good.append(i)
    cat = cat.iloc[good].reset_index(drop=True)
    N = len(cat)
    typ, rcs, hbr = hbr_table(cat)
    stats["n_screened"] = N
    stats["gp_epoch_max"] = str(cat["_ep"].max())
    stats["gp_epoch_median"] = str(cat["_ep"].median())
    print(f"objects screened: {N} (catalogue {stats['n_catalogue']}), steps {nstep} dt={args.dt}s", flush=True)

    C = args.chunk
    shms = []
    def mk(shape, dtype):
        sh = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np.dtype(dtype).itemsize)
        shms.append(sh)
        return np.ndarray(shape, dtype=dtype, buffer=sh.buf)
    G["sats"] = SatrecArray(sats)
    G["pos"] = mk((C, N, 3), np.float64)
    G["vel"] = mk((C, N, 3), np.float64)
    G["err"] = mk((C, N), np.bool_)
    pool = mp.get_context("fork").Pool(args.workers)

    import torch  # CUDA initialised only after the fork
    dev = torch.device("cuda")
    stats["gpu"] = torch.cuda.get_device_name(0)
    # triologay: nvidia-smi index 3 (PCI order) = PG503 32 GB; V100s are reserved for the ML builders.
    want = os.environ.get("LIVESKY_GPU_NAME", "PG503")
    if want and want not in stats["gpu"]:
        raise SystemExit(f"refusing to run on {stats['gpu']!r}; set CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 "
                         f"(or LIVESKY_GPU_NAME='' to override)")
    jd0, fr0 = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second)
    rows = []
    tprop = tgpu = 0.0
    for c0 in range(0, nstep, C):
        c1 = min(nstep, c0 + C)
        L = c1 - c0
        t_a = time.time()
        per = max(1, math.ceil(L / args.workers))
        jobs = []
        for i0 in range(0, L, per):
            i1 = min(L, i0 + per)
            fr = fr0 + (c0 + np.arange(i0, i1)) * args.dt / 86400.0
            jobs.append((i0, i1, jd0, fr))
        pool.map(_prop, jobs)
        t_b = time.time()
        P = torch.from_numpy(G["pos"][:L]).to(dev)
        V = torch.from_numpy(G["vel"][:L]).to(dev)
        E = torch.from_numpy(G["err"][:L]).to(dev)
        valid = ~E & torch.isfinite(P).all(2)
        P = torch.nan_to_num(P)
        V = torch.nan_to_num(V)
        rs = math.sqrt(args.thresh ** 2 + (17.0 * args.dt / 2) ** 2) + 1.0  # 17 km/s max rel speed
        for b0 in range(0, L, args.gpu_block):
            b1 = min(L, b0 + args.gpu_block)
            ev = screen_block(torch, P[b0:b1], V[b0:b1], valid[b0:b1], args.dt, rs, args.thresh, dev)
            if len(ev):
                ev[:, 0] += c0 + b0
                rows.append(ev)
        torch.cuda.synchronize()
        del P, V, E, valid
        t_c = time.time()
        tprop += t_b - t_a
        tgpu += t_c - t_b
        print(f"  steps {c0}-{c1}: prop {t_b - t_a:.1f}s gpu {t_c - t_b:.1f}s events so far {sum(len(r) for r in rows)}", flush=True)
    pool.close()
    for sh in shms:
        sh.close(); sh.unlink()
    stats["t_propagate_s"] = round(tprop, 1)
    stats["t_gpu_screen_s"] = round(tgpu, 1)
    stats["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    ev = np.concatenate(rows) if rows else np.zeros((0, 9))
    df = pd.DataFrame(ev, columns=["step", "ia", "ib", "tau", "dmin", "vrel", "rx", "ry", "rz"])
    df[["ia", "ib"]] = df[["ia", "ib"]].astype(int)
    lo = np.minimum(df.ia, df.ib); hi = np.maximum(df.ia, df.ib)
    df["ia"], df["ib"] = lo, hi
    df["t_s"] = df.step * args.dt + df.tau
    # de-duplicate same encounter caught from adjacent steps
    df = df.sort_values(["ia", "ib", "t_s"]).reset_index(drop=True)
    grp = ((df.ia != df.ia.shift()) | (df.ib != df.ib.shift()) | (df.t_s - df.t_s.shift() > 2 * args.dt)).cumsum()
    df["absd"] = (df.tau.abs())
    df = df.loc[df.groupby(grp)["dmin"].idxmin()].reset_index(drop=True)
    df = df[(df.t_s >= 0) & (df.t_s <= args.hours * 3600)].reset_index(drop=True)

    nor = cat["NORAD_CAT_ID"].values
    names = cat["OBJECT_NAME"].values
    df["norad_1"] = nor[df.ia]; df["norad_2"] = nor[df.ib]
    df["name_1"] = names[df.ia]; df["name_2"] = names[df.ib]
    df["type_1"] = typ[df.ia]; df["type_2"] = typ[df.ib]
    df["hbr_m"] = hbr[df.ia] + hbr[df.ib]
    df["tca"] = [t0 + timedelta(seconds=float(s)) for s in df.t_s]
    jd_tca = jd0 + fr0 + df.t_s.values / 86400.0
    lat, lon, alt = teme_to_geodetic(df[["rx", "ry", "rz"]].values, jd_tca)
    df["lat"], df["lon"], df["alt_km"] = lat, lon, alt
    df["pc_max"] = max_pc_isotropic(df.dmin.values, df.hbr_m.values)
    stats["n_events"] = len(df)
    print(f"events < {args.thresh} km: {len(df)}", flush=True)

    # ------------------------------------------------------------- SOCRATES validation
    soc = pd.read_csv(os.path.join(CACHE, "socrates_minrange.csv"))
    soc["tca"] = pd.to_datetime(soc.TCA, utc=True)
    t_end = t0 + timedelta(hours=args.hours)
    soc_w = soc[(soc.tca >= pd.Timestamp(t0)) & (soc.tca <= pd.Timestamp(t_end))].copy()
    soc_w["lo"] = np.minimum(soc_w.NORAD_CAT_ID_1, soc_w.NORAD_CAT_ID_2)
    soc_w["hi"] = np.maximum(soc_w.NORAD_CAT_ID_1, soc_w.NORAD_CAT_ID_2)
    df["lo"] = np.minimum(df.norad_1, df.norad_2); df["hi"] = np.maximum(df.norad_1, df.norad_2)
    df["tca_ts"] = pd.to_datetime(df.tca, utc=True)
    m = soc_w.reset_index().merge(df.reset_index()[["index", "lo", "hi", "tca_ts", "dmin"]].rename(columns={"index": "eidx"}),
                                  on=["lo", "hi"], how="left")
    m["dt_s"] = (m.tca_ts - m.tca).dt.total_seconds()
    mm = m[m.dt_s.abs() <= 60].copy()
    mm["absdt"] = mm.dt_s.abs()
    mm = mm.sort_values("absdt").drop_duplicates("index").drop_duplicates("eidx")
    soc_w = soc_w.reset_index().merge(mm[["index", "eidx", "dt_s", "dmin"]], on="index", how="left")
    soc_w["matched"] = soc_w.eidx.notna()
    ids = set(nor.tolist())
    soc_w["both_in_cat"] = soc_w.NORAD_CAT_ID_1.isin(ids) & soc_w.NORAD_CAT_ID_2.isin(ids)
    # SOCRATES element epoch = TCA - DSE (DSE has 1e-3 d resolution). Same element set as ours -> pure screening check.
    ep = dict(zip(nor, cat["_ep"].dt.tz_localize("UTC")))
    tol = pd.Timedelta(seconds=150)
    def same(nid, dse):
        se = soc_w.tca - pd.to_timedelta(soc_w[dse], unit="D")
        oe = soc_w[nid].map(ep)
        return (oe - se).abs() <= tol
    soc_w["same_elset"] = same("NORAD_CAT_ID_1", "DSE_1") & same("NORAD_CAT_ID_2", "DSE_2")
    df["socrates_match"] = df.index.isin(set(mm.eidx.astype(int)))
    df["soc_range"] = df.index.map(dict(zip(mm.eidx.astype(int), mm.TCA_RANGE))) if len(mm) else np.nan
    df["soc_maxprob"] = df.index.map(dict(zip(mm.eidx.astype(int), mm.MAX_PROB))) if len(mm) else np.nan
    df["has_payload"] = (df.type_1 == "PAY") | (df.type_2 == "PAY")
    # per-object SOCRATES element epoch (TCA - DSE) -> events where we used the same element sets as SOCRATES
    se = {}
    for i in (1, 2):
        x = pd.DataFrame({"n": soc[f"NORAD_CAT_ID_{i}"], "e": soc.tca - pd.to_timedelta(soc[f"DSE_{i}"], unit="D")})
        se.update(x.groupby("n").e.median().to_dict())
    def same_ev(col):
        return (df[col].map(ep) - df[col].map(se)).abs() <= tol
    df["same_elset"] = same_ev("norad_1") & same_ev("norad_2")
    is_sl = lambda c: df[c].astype(str).str.startswith("STARLINK")
    df["starlink_pair"] = is_sl("name_1") & is_sl("name_2")

    def rec(sub):
        return float(sub.matched.mean()) if len(sub) else float("nan")
    s1 = soc_w[soc_w.TCA_RANGE < 1]
    val = {
        "socrates_run_window": f"{soc.tca.min():%Y-%m-%dT%H:%MZ} to {soc.tca.max():%Y-%m-%dT%H:%MZ}",
        "compare_window": f"{t0:%Y-%m-%dT%H:%MZ} to {t_end:%Y-%m-%dT%H:%MZ}",
        "match_rule": "same NORAD pair and |dTCA| <= 60 s",
        "n_socrates_events_in_window": int(len(soc_w)),
        "recall_all_lt5km": rec(soc_w),
        "recall_all_lt5km_both_in_catalogue": rec(soc_w[soc_w.both_in_cat]),
        "recall_lt1km": rec(s1),
        "recall_lt1km_both_in_catalogue": rec(s1[s1.both_in_cat]),
        "n_socrates_lt1km": int(len(s1)),
        "n_socrates_same_elset": int(soc_w.same_elset.sum()),
        "recall_lt5km_same_elset": rec(soc_w[soc_w.same_elset]),
        "recall_lt1km_same_elset": rec(s1[s1.same_elset]),
        "median_abs_drange_km_same_elset": float((soc_w[soc_w.same_elset & soc_w.matched].dmin - soc_w[soc_w.same_elset & soc_w.matched].TCA_RANGE).abs().median()) if (soc_w.same_elset & soc_w.matched).any() else None,
        "precision_payload_events": float(df[df.has_payload].socrates_match.mean()) if df.has_payload.any() else float("nan"),
        "n_payload_events_same_elset": int((df.has_payload & df.same_elset).sum()),
        "precision_payload_events_same_elset": float(df[df.has_payload & df.same_elset].socrates_match.mean()),
        "precision_same_elset_excl_starlink_pairs_and_formation": float(
            df[df.has_payload & df.same_elset & ~df.starlink_pair & (df.vrel >= 0.1)].socrates_match.mean()),
        "starlink_starlink_share_ours_payload": float(df[df.has_payload].starlink_pair.mean()),
        "starlink_starlink_share_socrates": float((soc.OBJECT_NAME_1.astype(str).str.startswith("STARLINK")
                                                   & soc.OBJECT_NAME_2.astype(str).str.startswith("STARLINK")).mean()),
        "median_abs_dtca_s": float(mm.dt_s.abs().median()) if len(mm) else None,
        "median_abs_drange_km": float((mm.dmin - mm.TCA_RANGE).abs().median()) if len(mm) else None,
        "p90_abs_drange_km": float((mm.dmin - mm.TCA_RANGE).abs().quantile(0.9)) if len(mm) else None,
    }
    val = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items()}

    # ------------------------------------------------------------- outputs
    df.drop(columns=["tca_ts"]).assign(tca=df.tca.map(lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z")) \
        .to_csv(os.path.join(OUT, "events_all.csv"), index=False)
    soc_w.drop(columns=["tca"]).to_csv(os.path.join(OUT, "socrates_compare.csv"), index=False)

    nowts = t0
    in24 = df[df.t_s <= 24 * 3600]
    # one row per pair (closest encounter) for the top list so a co-orbiting pair cannot fill it
    # formation / co-located pairs (rel speed < 0.1 km/s, e.g. TanDEM-X/TerraSAR-X, deployment clusters) are
    # counted in the summary but kept out of the globe's top list
    top = df[df.vrel >= 0.1].sort_values("dmin").drop_duplicates(["lo", "hi"]).head(args.top)
    conj = []
    for r in top.itertuples():
        tca = r.tca.strftime("%Y-%m-%dT%H:%M:%S.") + f"{r.tca.microsecond // 1000:03d}Z"
        conj.append({
            "id": f"{r.lo}-{r.hi}-{r.tca:%Y%m%dT%H%M%S}",
            "norad_1": int(r.norad_1), "name_1": r.name_1, "norad_2": int(r.norad_2), "name_2": r.name_2,
            "tca_utc": tca, "min_range_km": round(float(r.dmin), 4), "rel_speed_kms": round(float(r.vrel), 4),
            "log10_pc_max": round(float(np.log10(r.pc_max)), 3), "socrates_match": bool(r.socrates_match),
            "lat": round(float(r.lat), 3), "lon": round(float(r.lon), 3), "alt_km": round(float(r.alt_km), 2),
            "type_1": r.type_1, "type_2": r.type_2, "hbr_m": round(float(r.hbr_m), 2),
            "socrates_range_km": None if pd.isna(r.soc_range) else float(r.soc_range),
            "socrates_max_prob": None if pd.isna(r.soc_maxprob) else float(r.soc_maxprob),
            "low_rel_speed": bool(r.vrel < 0.1),
        })
    stats["t_total_s"] = round(time.time() - T0, 1)
    out = {
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"CelesTrak GP (OMM) snapshot, element epochs up to {cat['_ep'].max():%Y-%m-%d %H:%M} UTC; "
                  f"groups active, fengyun-1c/cosmos-2251/iridium-33 debris + name searches DEB, R/B, COSMOS, OBJECT, OPS, METEOR. "
                  f"Data: CelesTrak (celestrak.org)",
        "window_h": args.hours,
        "window_start_utc": f"{t0:%Y-%m-%dT%H:%M:%SZ}",
        "n_objects": int(N),
        "method": f"SGP4 (python-sgp4 SatrecArray, {args.workers} CPU procs) on a {args.dt:g} s grid; GPU sorted-grid "
                  f"spatial hash (cell {rs:.0f} km) per step on {stats['gpu']}; linear-relative-motion TCA refinement; "
                  f"threshold {args.thresh:g} km",
        "pc_note": "log10_pc_max = max Pc (no covariance in public TLEs): worst case over an isotropic Gaussian of unknown "
                   "size in the encounter plane, HBR from SATCAT RCS (equivalent-circle radius, both objects summed). "
                   "SGP4/TLE errors are ~1 km or more; this is a screening view, not an operational Pc.",
        "summary": {
            "n_events_window": int(len(df)),
            "n_pairs_window": int(df[["lo", "hi"]].drop_duplicates().shape[0]),
            "next24h_lt1km": int((in24.dmin < 1).sum()),
            "next24h_lt5km": int((in24.dmin < 5).sum()),
            "next24h_lt1km_excl_low_rel_speed": int(((in24.dmin < 1) & (in24.vrel >= 0.1)).sum()),
            "next24h_lt5km_excl_low_rel_speed": int(((in24.dmin < 5) & (in24.vrel >= 0.1)).sum()),
            "n_formation_pairs_window": int(df[df.vrel < 0.1][["lo", "hi"]].drop_duplicates().shape[0]),
            "window_lt1km": int((df.dmin < 1).sum()),
            "top_list": f"closest encounter per pair, {len(conj)} pairs with rel speed >= 0.1 km/s, sorted by min range",
        },
        "conjunctions": conj,
        "validation": {
            "socrates_overlap": round(float(top.socrates_match.mean()), 4) if len(top) else 0.0,
            **val,
            "notes": "socrates_overlap = fraction of the listed top pairs matched to a SOCRATES event (same pair, |dTCA|<=60 s). "
                     "SOCRATES lists only payload-involved pairs, so debris-on-debris events cannot match. "
                     "SOCRATES used older element sets (its run started at the window it reports) than our snapshot, and both use "
                     "the same public SGP4 data, so agreement checks the screening, not truth.",
        },
        "run_stats": stats,
    }
    with open(os.path.join(DATA_OUT, "latest.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    with open(os.path.join(OUT, "run_stats.json"), "w") as f:
        json.dump({**stats, "validation": val, "summary": out["summary"]}, f, indent=1, default=str)
    print(json.dumps({**stats, **val, **out["summary"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
