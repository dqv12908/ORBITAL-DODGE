# Big ML on Triologay: risk futures, conformal alerts, decision policy

Source of every number: `research/bigml/results.json`, with per-split sections, CIs and tuned parameters. The code is the `*.py` files in this folder. Runs used Triologay's GPUs with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

Setup:
- Kelvins-eligible train events: 8,293, of which 551 end critical under the brief's tiers.
- Inputs are only CDMs with time to TCA ≥ 2 d.
- Splits:
  - by event: KFold 5
  - by mission: GroupKFold 5, the same folds as `research/bakeoff/folds.parquet`
  - leave-one-mission-out
- `mission_id` is never a feature.

## A. Risk futures
**Method:**
- A mixture-density network learns each CDM → next-CDM transition: time gap, miss distance, risk with its −30 floor, sigmas, and when the sequence ends. It uses 31 state features.
- From the latest CDM at ≥ 2 d before TCA it simulates 10,000 futures per event to TCA on the GPU.
- P(final tier) is the share of futures ending in each tier.

**Results by held-out mission, at persistence's critical recall of 0.820:**

| Method | AUC | Brier | False alarms |
|---|---|---|---|
| Tier persistence | 0.958 | 0.0223 | 113 |
| Risk futures | 0.982 | 0.0187 | 100 |
| LightGBM, Optuna-tuned (10 seeds) | 0.984–0.986 | 0.0174–0.0177 | 94 |
| Average of futures + LightGBM A13 | — | — | **91** |

**Reading the results:**
- Futures beat persistence. They don't beat tuned LightGBM; the confidence intervals overlap.
- By-mission and leave-one-mission-out metrics match the by-event numbers for every learned model. The pre-registered "does not generalise" trigger was not met.
- What futures add that a classifier can't:
  - per-event fan charts (`registry/futures/fans/<event_key>.json`; all 13,154 train events from models that never saw their mission, plus all 2,167 test events)
  - the full tier distribution
  - the simulator behind the decision policy

## B. Conformal alerts
Split conformal on futures P(critical), calibrated only on other missions' out-of-fold scores:

| Target | Realised critical recall [95% CI] | False alarms |
|---|---|---|
| 95 % | **0.949** [0.925, 0.967] | 613 |
| 90 % | 0.907 | 257 |
| Persistence (for reference) | 0.926 | 1,323 |

The guarantee is pooled, not per mission: mission 15 falls below target for every score. A per-tier (Mondrian) variant failed, flagging about 7,000 events; this is recorded as a negative result.

## C. Decision policy
Value iteration over 588 discretised CDM states, using real plus MDN-simulated transitions. The loss table is an extension of design §4.4 and includes a surcharge for late preparation. Mean loss per held-out event:

| Policy | Loss [95 % CI] | Missed criticals |
|---|---|---|
| Optimal policy | **0.859** [0.811, 0.913] | 0 |
| Prepare if critical | 1.039 | — |
| Brief rule (track at every ELEVATED CDM) | 3.896 | — |
| Perfect foresight (lower bound) | 0.332 | — |

Under these costs, requesting tracking is almost never optimal. The policy lookup is `registry/policy/policy.json`.

## Failure cases
- **train:9361** is critical by Pc 1.2e-4 at 3.93 km, but futures give it P(critical) 0.04. Pc-driven criticals are only 6 of 551, so the model barely learns them.
- **Mission 15 late escalations:** leave-one-mission-out critical recall is 0.685 for persistence and futures, 0.704 for LightGBM.
  - Partial win: train:2563 is SAFE at T−2.43 d but gets futures P(critical) 0.092. That is above the 95 % conformal threshold of 0.067, so it would be flagged two days early.
- **Mission 6:** critical recall is 0.50 for every method (8 criticals).
- **Fan scale:** fans use the dataset's `risk` scale (floor −30), not our physics Pc.
- **Calibration:** raw Monte Carlo P(critical) is slightly overconfident (ECE 0.008).
