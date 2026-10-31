"""Golden tests: physics Pc vs the dataset `risk` column, and the four golden events (needs data/processed)."""
import numpy as np
import pandas as pd
import pytest

from riskengine import paths, rules
from riskengine.cdm import brief_tier_codes
from riskengine.pc import pc_batch

pytestmark = pytest.mark.skipif(not paths.CDM_CLEAN.exists(), reason="run pipelines/ingest.py first")


@pytest.fixture(scope="module")
def train():
    df = pd.read_parquet(paths.CDM_CLEAN)
    return df[df.split == "train"]


@pytest.fixture(scope="module")
def clean_rows(train):
    return train[(train.cov_unknown_t == 0) & (train.cov_unknown_c == 0)]


def test_golden_sample_5000(clean_rows):
    s = clean_rows.sample(5000, random_state=1729)
    out = pc_batch(s)
    lp, y = out.log10_pc.to_numpy(), s.risk.to_numpy()
    m = y >= -10
    assert np.median(np.abs(lp[m] - y[m])) <= 0.07
    agree = np.mean(brief_tier_codes(lp, s.miss_distance) == brief_tier_codes(y, s.miss_distance))
    assert agree >= 0.985
    # never a safe<->critical swap
    tp, ty = brief_tier_codes(lp, s.miss_distance), brief_tier_codes(y, s.miss_distance)
    assert not np.any(np.abs(tp - ty) == 2)


def test_high_band_all_rows(clean_rows):
    hi = clean_rows[clean_rows.risk >= -4.5]
    lp = pc_batch(hi).log10_pc.to_numpy()
    y = hi.risk.to_numpy()
    band = y >= -4
    assert np.median(np.abs(lp[band] - y[band])) <= 0.01
    assert np.mean(lp[band] >= -4) >= 0.90          # Pc >= 1e-4 recall


def _assess_event(train, eid, mode="BRIEF"):
    ev = train[train.event_key == f"train:{eid}"].sort_values("cdm_index")
    out = pc_batch(ev)
    res = []
    for row, (_, p) in zip(ev.to_dict("records"), out.iterrows()):
        pc = {"status": p.pc_status, "log10": None if np.isnan(p.log10_pc) else p.log10_pc, "k_star": p.k_star}
        res.append(rules.assess(row, pc, mode=mode))
    return ev, out, res


def test_event_2563_late_escalation(train):
    ev, out, res = _assess_event(train, 2563)
    assert [r["tier"] for r in res] == ["SAFE", "ELEVATED", "CRITICAL"]
    assert [r["miss_tier"] for r in res] == ["SAFE", "ELEVATED", "CRITICAL"]
    assert "R-MISS-CRIT" in res[-1]["rule_ids"] and "T-MCP" in res[-1]["rule_ids"]


def test_event_2299_cov_unknown(train):
    ev, out, res = _assess_event(train, 2299)
    st = out.pc_status.tolist()
    assert st[:13] == ["COV_UNKNOWN"] * 13 and st[13] == "OK"
    for r in res[:13]:
        assert r["pc_tier"] == "UNKNOWN" and r["tier"] in ("ELEVATED", "CRITICAL") and r["automation"] == "BLOCKED"
        assert "Q-COV-DEFAULT" in r["rule_ids"]


def test_event_4200_phys_disagree(train):
    ev, out, res = _assess_event(train, 4200)
    assert out.log10_pc.iloc[0] < -300
    assert "Q-PHYS-DISAGREE" in res[0]["rule_ids"] and res[0]["escalate"]
    assert res[0]["pc_tier"] == "ELEVATED"            # tier from the higher of the two


def test_event_8756_rules_disagree(train):
    ev, out, res = _assess_event(train, 8756)
    last = res[-1]
    assert last["tier"] == "CRITICAL" and last["miss_tier"] == "CRITICAL" and last["pc_tier"] == "SAFE"
    _, _, res_cara = _assess_event(train, 8756, mode="CARA_PC")
    assert res_cara[-1]["tier"] == "SAFE" and "MISS-SCREEN" in res_cara[-1]["rule_ids"]


def test_pc_rows_artifact_consistent():
    if not paths.PC_ROWS.exists():
        pytest.skip("pc_rows not built")
    p = pd.read_parquet(paths.PC_ROWS, columns=["event_key", "cdm_index", "pc_status", "tier"])
    assert set(p.pc_status.unique()) <= {"OK", "COV_UNKNOWN", "LOW_VREL", "NPD_CLIPPED"}
    assert set(p.tier.unique()) <= {"SAFE", "ELEVATED", "CRITICAL"}
    assert len(p) == 187118
