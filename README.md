# ORBITAL DODGE

Challenge 1 (Airbus track, Autonomous SG 2026): AI-assisted satellite collision avoidance with a 3D whole-Earth simulator.

**Status:** research and design done; build not started.

## What's here

| Path | Contents |
|---|---|
| [`docs/design/40-final-design.md`](docs/design/40-final-design.md) | **The system design**: user, architecture, risk engine, decision layer, LLM layer, backend, 3D simulator, evaluation, build order, demo script |
| [`docs/design/11-dataset-experiments.md`](docs/design/11-dataset-experiments.md) | Experiments on the dataset: baselines, physics Pc check, live-feed replay, orbit reconstruction, demo events |
| [`docs/design/10-reference-style.md`](docs/design/10-reference-style.md) | Visual and engineering style taken from the reference project (agentic-factory) |
| [`docs/design/12-domain-research.md`](docs/design/12-domain-research.md) | NASA CARA / ESA practice, Pc methods, CDM standard, users, public orbit data |
| [`docs/design/13-tech-research.md`](docs/design/13-tech-research.md) | Stack research: three.js globe, orbit propagation, FastAPI backend, LLM and decision-model APIs |
| [`docs/design/00-context.md`](docs/design/00-context.md) | Brief summary, rubric and verified dataset facts |
| [`research/experiments/`](research/experiments/) | Python scripts behind the experiment numbers, with their outputs in `out/` |

## Key findings

- The dataset's `risk` (log10 Pc) can be recomputed from the covariance with a standard 2D Pc: median error 0.059 dex, tier agreement 98.97 %.
- The 13 suggested columns alone cannot compute Pc (only radial sigmas; the along-track sigma dominates).
- Tier persistence is a strong baseline (final tier unchanged in 92.9 % of events). The ML model ranks events better (AUC 0.980 vs 0.958) but does not beat persistence as an alarm at matched recall.
- Of 551 final-critical events, 549 are critical because miss distance < 1 km; only 6 because Pc ≥ 1e-4.

## Running the experiments

The dataset is not in this repo. Put `train_data.csv` and `test_data.csv` in `data/raw/Challenge 1 Airbus /` (or set `C1_DATA_DIR`), then:

```bash
pip install pandas pyarrow numpy scipy scikit-learn
cd research/experiments && python b_baselines.py
```
