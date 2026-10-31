"""Write registry/futures/fans/<event_key>.json (CONTRACT.md format + extras).
Train events: out-of-fold fans from the by-mission models (the event's mission was held out).
Test events: model trained on all train events."""
import os, json, joblib, numpy as np, pandas as pd
from bm import HERE, ROOT, load_rows, events_table

OUT = os.path.join(HERE, "out")
FAN = os.path.join(ROOT, "registry", "futures", "fans"); os.makedirs(FAN, exist_ok=True)
d = load_rows(); ev = events_table(d).set_index("event_key")
QN = ["p05", "p25", "p50", "p75", "p95"]
VERSION = "futures-mdn-1.0"
n_written = 0
for src, npaths, note in [("futures_mission.joblib", 10000, "out-of-fold: model trained without this event's mission (GroupKFold by mission)"),
                          ("futures_full_test.joblib", 10000, "model trained on all train events; test truth hidden")]:
    p = os.path.join(OUT, src)
    if not os.path.exists(p):
        print("missing", src); continue
    F = joblib.load(p)
    grid = F.get("grid", np.round(np.arange(7.0, -0.001, -0.25), 3))
    for i, k in enumerate(F["event_key"]):
        e = ev.loc[k]; s = float(e.start_ttca); row = d.loc[int(e.start_idx)]
        keep = grid < s - 1e-9
        tg = [round(s, 4)] + [float(g) for g in grid[keep]]
        q_rk = F["q_rk"][i][:, keep]; q_lm = F["q_lm"][i][:, keep]
        r0 = float(np.clip(row.risk, -30, 0)); m0 = float(max(row.miss_distance, 1.0))
        out = dict(event_key=k, ttca_grid_d=tg,
                   quantiles={q: [round(r0, 3)] + [round(float(v), 3) for v in q_rk[j]] for j, q in enumerate(QN)},
                   miss_quantiles={q: [round(m0, 1)] + [round(float(10 ** v), 1) for v in q_lm[j]] for j, q in enumerate(QN)},
                   p_critical=round(float(F["p_tiers"][i][2]), 4),
                   p_tiers=dict(SAFE=round(float(F["p_tiers"][i][0]), 4), ELEVATED=round(float(F["p_tiers"][i][1]), 4),
                                CRITICAL=round(float(F["p_tiers"][i][2]), 4)),
                   n_paths=npaths, model_version=VERSION,
                   start_cdm_index=int(e.start_cdm), start_ttca_d=round(s, 4),
                   quantity="quantiles = log10 Pc on the CDM `risk` scale (floor -30); miss_quantiles in metres",
                   meaning="fan of the latest-known CDM value at each time-to-TCA; p_critical = share of simulated futures whose final CDM is CRITICAL under the brief's tiers",
                   provenance=note,
                   label="STATISTICS (simulated futures), not a collision probability")
        with open(os.path.join(FAN, f"{k}.json"), "w") as f:
            json.dump(out, f, separators=(",", ":"))
        n_written += 1
print("fans written", n_written)
