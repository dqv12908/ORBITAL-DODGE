"""cdm_clean.parquet -> pc_rows.parquet: physics Pc + BRIEF-mode rules for every row (both splits)."""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import time

import numpy as np
import pandas as pd

from riskengine import cdm, paths, rules
from riskengine.pc import pc_batch

RULE_INPUT = ["miss_distance", "relative_speed", "time_to_tca", "risk", "cov_unknown_t", "cov_unknown_c"]


def compute(df: pd.DataFrame, mode: str = "BRIEF") -> pd.DataFrame:
    pcs = pc_batch(df)
    tiers, pct, mt, rids, reas, flags, esc, dis = [], [], [], [], [], [], [], []
    src = df[RULE_INPUT].to_dict("records")
    for row, st, lp, ks, vr in zip(src, pcs.pc_status.values, pcs.log10_pc.values, pcs.k_star.values,
                                   pcs.v_rel_ms.values):
        a = rules.assess(row, {"status": st, "log10": None if np.isnan(lp) else float(lp), "k_star": float(ks),
                               "v_rel_ms": float(vr)}, mode=mode)
        tiers.append(a["tier"]); pct.append(a["pc_tier"]); mt.append(a["miss_tier"])
        rids.append(a["rule_ids"]); reas.append(a["reasons"]); flags.append(a["flags"])
        esc.append(a["escalate"]); dis.append(a["disagree_dex"])
    out = pd.concat([df[["event_key", "cdm_index"]].reset_index(drop=True), pcs.reset_index(drop=True)], axis=1)
    out["tier"] = tiers
    out["pc_tier"] = pct
    out["miss_tier"] = mt
    out["rule_ids"] = rids
    out["reasons"] = reas
    out["flags"] = flags
    out["escalate"] = esc
    out["disagree_dex"] = pd.array(dis, dtype="Float64")
    return out


def main():
    t0 = time.time()
    df = pd.read_parquet(paths.CDM_CLEAN)
    out = compute(df)
    out.to_parquet(paths.PC_ROWS, index=False)
    tr = df.split.values == "train"
    ok = (out.pc_status.values != "COV_UNKNOWN") & tr
    y = df.risk.values
    m = ok & (y >= -10)
    e = np.abs(out.log10_pc.values[m] - y[m])
    ty = cdm.brief_tier_codes(y[ok], df.miss_distance.values[ok])
    tp = cdm.brief_tier_codes(out.log10_pc.values[ok], df.miss_distance.values[ok])
    print(f"[precompute_pc] {len(out)} rows; train clean rows {ok.sum()}: MAE(risk>=-10) {e.mean():.3f} dex, "
          f"median {np.median(e):.3f}, tier agreement {np.mean(ty == tp):.4f}; status {out.pc_status.value_counts().to_dict()} "
          f"-> {paths.PC_ROWS} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
