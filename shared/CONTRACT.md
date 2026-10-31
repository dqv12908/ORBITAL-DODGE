# ORBITAL DODGE — build contract (read before writing code)

Product name: **ORBITAL DODGE**. Log prompt: `dodge@ops:~$`. (The design doc calls it "CLOSE CALL"; that name is retired.)
Full design: `docs/design/40-final-design.md` (sections 3–7 are the spec). Style: `docs/design/10-reference-style.md`.
Build deadline: ~4 hours from 11:55 SGT, 31 Oct 2026. Ship MUSTs first; working > complete.

## Ownership (do not edit files outside your area; ask the lead via your final report)

| Area | Owner |
|---|---|
| `server/riskengine/`, `server/pipelines/`, `registry/`, `predictions/`, `shared/scenarios/`, `data/processed/`, `server/tests/test_pc*.py`, `test_rules.py`, `test_forecast.py` | engine |
| `server/app/`, `server/decisions/`, `server/llm/`, `server/ledger/`, `server/registry/`, `shared/schema/`, other `server/tests/*` | backend |
| `game/` | frontend |
| `research/bigml/`, `registry/futures/`, `registry/conformal/`, `registry/policy/` | bigml (Triologay) |
| `research/livesky/`, `data/livesky/` | livesky (Triologay) |
| `Makefile`, `README.md`, `JUDGING.md`, `shared/CONTRACT.md`, `server/pyproject.toml` | lead |

Python env: `server/.venv` (uv, Python 3.13, deps in `server/pyproject.toml` already installed). Run with `cd server && .venv/bin/python ...`.
Need a new dependency? Add it with `cd server && uv pip install X` and mention it in your report (lead updates pyproject).
Node 20.11.1 (Vite 6, not 7).

## Data
Raw: `data/raw/Challenge 1 Airbus /{train,test}_data.csv` (note the trailing space). Never modify.
Proven research code to port (do not reinvent): `research/experiments/{common.py, pc2d.py, e_orbits.py, b_baselines.py, f_stories*.py, d_*.py}`.
Event key everywhere: `"{split}:{event_id}"` e.g. `train:2563`, `test:891`.

## Artifacts produced by engine (consumed by backend + frontend)

| File | Content |
|---|---|
| `data/processed/cdm_clean.parquet` | all rows, both splits; cols: `event_key, split, event_id, mission_id, cdm_index` (0-based by descending ttca), all raw cols, `cov_unknown_t, cov_unknown_c` |
| `data/processed/pc_rows.parquet` | per row: `event_key, cdm_index, pc_status, log10_pc, hbr_m, sigma_x, sigma_y, theta_deg, mahalanobis, k_star, log10_pc_max, miss_x, miss_y, tier, pc_tier, miss_tier, rule_ids (list), reasons (list)` |
| `data/processed/events.parquet` | per event: `event_key, split, mission_id, n_cdm, eligible, final tier/log10/miss (train only), latest>=2d tier`, OOF `p_critical`, `p_tiers`, `persistence_tier` |
| `data/processed/geometry.parquet` | per event: reconstructed Keplerian elements for target and chaser (`a_km, e, i_deg, raan_deg, argp_deg, nu_deg_at_tca`), conjunction point lat/lon (lon synthetic), residuals `inc_res_deg, sma_res_km` |
| `registry/tier_forecaster/1.0.0/` | `model.joblib, calibrator.joblib, model_card.json, metrics.json, reliability.json, oof.parquet`; plus `registry/tier_forecaster/CURRENT` containing `1.0.0` |
| `predictions/test_events.csv`, `test_rows.csv` | brief deliverable for the 2,167 test events / 24,484 rows |
| `shared/scenarios/<id>.json` | see below |

### riskengine Python API (backend imports these; engine implements them)
```python
from riskengine.cdm import load_clean()                      # -> DataFrame (cdm_clean)
from riskengine.pc import pc_for_row(row: dict) -> dict      # pc object as in design §6.3 risk.assessed.data.pc
from riskengine.rules import assess(row: dict, pc: dict, mode="BRIEF") -> dict   # tier object (design §6.3)
from riskengine.forecast import Forecaster                    # Forecaster.load(path); .predict(cdm_rows_df) -> {valid, p_tiers, p_critical, calib_bin, baseline{persistence_tier, p_critical_empirical}, watch, model_version}
from riskengine.orbits import geometry(event_key) -> dict     # from geometry.parquet, plus encounter-plane per CDM
```

### Scenario JSON (`shared/scenarios/<id>.json`)
```json
{"id":"late-cdm","title":"The late CDM","tag":"LATE","teaches":"...","order":2,
 "d0_sim":"2026-10-31T06:00:00Z","seed":1729,
 "cast":[{"event_key":"train:2563","role":"focal","tca_sim":"2026-10-31T20:30:00Z"}],
 "preload_until_sim":"2026-10-31T06:00:00Z"}
```
CDM release time in SIM = `tca_sim - time_to_tca` (days). Outcome revealed at `tca_sim`.
Scenarios (design §7.6): `ops-day` (mission 15 + 8767, 2563, 2115), `late-cdm` (2563), `steady-climb` (8767),
`scary-then-nothing` (2427, 7465), `rules-disagree` (8756, 9361), `cov-unknown` (2299, 2759), `failure` (4200 + one missed critical), `blind-test` (test:891,1568,1448,2059,2091), `free-play`.

## API + SSE (backend implements, frontend consumes)
Exactly design §6.2 (REST) and §6.3 (SSE envelope + event types + `risk.assessed` schema). Backend on `:8000`, Vite on `:5173` proxying `/api`.
Additions:
- `GET /api/livesky` → contents of `data/livesky/latest.json` (or 404 JSON error if absent).
- `GET /api/futures/{event_key}` → contents of `registry/futures/fans/<event_key>.json` if present: `{event_key, ttca_grid_d[], quantiles{p05,p25,p50,p75,p95}[] of log10_pc, miss_quantiles{...}[], p_critical, n_paths, model_version}`; else 404.
- `GET /api/eval` merges `registry/*/metrics.json` incl. `research/bigml/results.json` if present.

### livesky JSON (`data/livesky/latest.json`)
```json
{"generated_utc":"...","source":"CelesTrak GP <date>","window_h":72,"n_objects":0,
 "conjunctions":[{"id":"...","norad_1":0,"name_1":"","norad_2":0,"name_2":"","tca_utc":"","min_range_km":0,"rel_speed_kms":0,
   "log10_pc_max":0,"socrates_match":true,"lat":0,"lon":0,"alt_km":0}],
 "validation":{"socrates_overlap":0.0,"notes":""}}
```

### Population for the globe
`game/public/population.f32.bin` + `population.meta.json` — built by frontend from CelesTrak GP (`https://celestrak.org/NORAD/elements/gp.php?GROUP=<g>&FORMAT=json`).

## Honesty rules (all owners)
- Physics computes Pc; rules compute tiers and allowed actions; ML forecasts and may only ADD attention. Decision model picks among allowed actions; LLM explains. Neither computes a Pc or a tier.
- Reconstructed orbits are always labelled "RECONSTRUCTED GEOMETRY". SIM clock always labelled SIM.
- Decision-model probabilities are never shown as collision probabilities.
- Never use the Kelvins Zenodo release with private test labels.
