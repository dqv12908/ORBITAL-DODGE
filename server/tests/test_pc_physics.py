"""Physics property tests for the 2D Pc (design 6.9). Pure maths; no data files needed."""
import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from riskengine.pc import log10_pc_disc, pc_batch, pc_for_row

pos = st.floats(min_value=1.0, max_value=5000.0)
miss = st.floats(min_value=0.0, max_value=20000.0)
hbr = st.floats(min_value=0.5, max_value=30.0)
rho = st.floats(min_value=-0.95, max_value=0.95)
ang = st.floats(min_value=0.0, max_value=2 * math.pi)


def pc(mx, my, sx, sy, r, rh=0.0):
    return float(log10_pc_disc([mx], [my], [sx * sx], [rh * sx * sy], [sy * sy], [r])[0])


@settings(max_examples=60, deadline=None, derandomize=True)
@given(miss, miss, pos, pos, hbr, rho)
def test_bounds(mx, my, sx, sy, r, rh):
    lp = pc(mx, my, sx, sy, r, rh)
    assert lp <= 0.0 and math.isfinite(lp)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(miss, miss, pos, pos, hbr, rho)
def test_swap_objects_symmetry(mx, my, sx, sy, r, rh):
    # swapping the objects negates the miss vector; the combined covariance is unchanged
    assert pc(mx, my, sx, sy, r, rh) == pytest.approx(pc(-mx, -my, sx, sy, r, rh), abs=1e-9)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(st.floats(0, 3000), pos, pos, hbr, rho, ang)
def test_rotation_invariance(d, sx, sy, r, rh, a):
    # quadrature regime of the data: sigma >= HBR (HBR is metres, sigmas tens of metres to km)
    assume(min(sx, sy) * math.sqrt(1 - rh * rh) >= r)
    C = np.array([[sx * sx, rh * sx * sy], [rh * sx * sy, sy * sy]])
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    m = np.array([d, 0.3 * d])
    C2, m2 = Rm @ C @ Rm.T, Rm @ m
    a1 = float(log10_pc_disc([m[0]], [m[1]], [C[0, 0]], [C[0, 1]], [C[1, 1]], [r])[0])
    a2 = float(log10_pc_disc([m2[0]], [m2[1]], [C2[0, 0]], [C2[0, 1]], [C2[1, 1]], [r])[0])
    assert a1 == pytest.approx(a2, abs=1e-4 + 1e-5 * abs(a1))


@settings(max_examples=40, deadline=None, derandomize=True)
@given(pos, hbr, st.floats(0, 5000), st.floats(1, 5000), ang)
def test_monotone_in_miss_isotropic(s, r, d1, dd, a):
    c, si = math.cos(a), math.sin(a)
    lo = pc(d1 * c, d1 * si, s, s, r)
    hi = pc((d1 + dd) * c, (d1 + dd) * si, s, s, r)
    assert hi <= lo + 1e-9


@settings(max_examples=40, deadline=None, derandomize=True)
@given(miss, pos, pos, st.floats(0.5, 15), st.floats(0.01, 15))
def test_increasing_in_hbr(d, sx, sy, r, dr):
    assert pc(d, 0, sx, sy, r + dr) >= pc(d, 0, sx, sy, r) - 1e-9


def test_small_sigma_limits():
    r = 10.0
    assert 10 ** pc(0, 0, r / 6, r / 6, r) > 0.999        # miss inside HBR -> 1
    assert 10 ** pc(3 * r, 0, r / 6, r / 6, r) < 1e-10    # miss outside HBR -> 0


# 12 §3.1 cross-check table (x_m, y_m, sx, sy, R) -> Foster Pc, Pc_max
VECTORS = [((100, 50, 200, 50, 10), 2.676e-3, 2.94e-3), ((500, 0, 1000, 20, 15), 4.638e-3, 1.66e-2),
           ((20, 5, 30, 10, 5), 2.871e-2, 4.41e-2), ((1500, 200, 800, 100, 20), 5.922e-5, 2.45e-4)]


@pytest.mark.parametrize("case,expected,pcmax", VECTORS)
def test_reference_vectors(case, expected, pcmax):
    x, y, sx, sy, r = case
    assert 10 ** pc(x, y, sx, sy, r) == pytest.approx(expected, rel=2e-3)
    v = (x / sx) ** 2 + (y / sy) ** 2
    assert r * r / (math.e * sx * sy * v) == pytest.approx(pcmax, rel=1e-2)


def test_monte_carlo_agreement():
    rng = np.random.default_rng(0)
    n = 1_000_000
    for i in range(20):
        sx, sy = rng.uniform(5, 60, 2)
        rh = rng.uniform(-0.8, 0.8)
        r = rng.uniform(5, 20)
        mx, my = rng.uniform(-1.5, 1.5, 2) * np.array([sx, sy])
        C = np.array([[sx * sx, rh * sx * sy], [rh * sx * sy, sy * sy]])
        z = rng.multivariate_normal([mx, my], C, size=n)
        p_mc = np.mean((z ** 2).sum(1) <= r * r)
        p = 10 ** pc(mx, my, sx, sy, r, rh)
        se = math.sqrt(p * (1 - p) / n)
        assert abs(p - p_mc) <= max(0.01 * p, 4 * se), (i, p, p_mc)


def _row(**kw):
    base = dict(t_sigma_r=20.0, t_sigma_t=300.0, t_sigma_n=15.0, t_ct_r=0.1, t_cn_r=0.0, t_cn_t=0.0,
                c_sigma_r=60.0, c_sigma_t=3000.0, c_sigma_n=40.0, c_ct_r=-0.3, c_cn_r=0.1, c_cn_t=0.05,
                relative_position_r=50.0, relative_position_t=-200.0, relative_position_n=400.0,
                relative_velocity_r=3.0, relative_velocity_t=-9000.0, relative_velocity_n=-4500.0,
                t_j2k_sma=7000.0, t_span=5.0, c_span=2.0, miss_distance=450.0, relative_speed=10062.0)
    base.update(kw)
    return base


def test_row_api_statuses():
    p = pc_for_row(_row())
    assert p["status"] == "OK" and p["method"] == "2D-Foster-GL24x48" and p["hbr_m"] == 3.5
    assert p["log10"] <= 0 and p["sigma_plane_m"][0] >= p["sigma_plane_m"][1] > 0
    assert p["miss_plane_m"][0] == pytest.approx(450.0, rel=1e-2)
    assert pc_for_row(_row(c_sigma_r=63781363.0))["status"] == "COV_UNKNOWN"
    assert pc_for_row(_row(c_sigma_r=63781363.0))["log10"] is None
    slow = _row(relative_velocity_r=0.0, relative_velocity_t=-40.0, relative_velocity_n=-30.0, relative_speed=50.0)
    assert pc_for_row(slow)["status"] == "LOW_VREL"
    npd = _row(t_ct_r=0.99, t_cn_r=0.99, t_cn_t=-0.99)
    assert pc_for_row(npd)["status"] == "NPD_CLIPPED"


def test_batch_matches_row():
    import pandas as pd
    rows = [_row(), _row(relative_position_n=40.0), _row(c_sigma_t=50.0)]
    b = pc_batch(pd.DataFrame(rows))
    for i, r in enumerate(rows):
        assert pc_for_row(r)["log10"] == pytest.approx(b.log10_pc.iloc[i], abs=1e-9)
