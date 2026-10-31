# Contender `ebm`: Explainable Boosting Machine (interpret 0.7.8)

## Setup
- Features: A13 (all 84 columns of `data/features_A13.parquet`). No feature screening: one fit takes about 13 s on an idle box
  (25 to 55 s under the shared load during the run), so the full set fits easily in the budget.
  `last_c_object_type` is declared `nominal`; every other feature is `continuous`.
- Formulation: **ordinal Frank-Hall decomposition into two binary EBMs**, chosen before any scoring:
  - `crit`: P(tier == critical). PRIMARY depends only on this model, and its shapes are clean log-odds of "critical" for the UI.
  - `ge`: P(tier >= elevated).
  - `p_critical = pc`, `p_elevated = max(pge, pc) - pc`, `p_safe = 1 - max(pge, pc)`, then clipped and renormalised.
- Hyperparameters were fixed a priori, with no tuning: `interactions=10` (FAST-selected pairs per binary model),
  `outer_bags=14`, `inner_bags=0`, `learning_rate=0.015`, `max_bins=1024`, `max_interaction_bins=64`,
  `early_stopping_rounds=100`, `min_samples_leaf=4`, `smoothing_rounds=75`, `interaction_smoothing_rounds=75`, `random_state=0`.
- Leakage control: each outer fold fits only on its training positions. Early stopping uses **explicit EBM `bags`**
  built from the training fold only. On the by-mission split, each of the 14 bags holds out whole missions (about 15%
  of training events, so the inner validation is mission-grouped as PREREG 3 requires). On the by-event split, each bag
  holds out a stratified 15%. No scaler is needed because EBM bins features. No calibrator or target encoder is used,
  and `y_risk`/`y_miss` are not used.
- Parallelism: OOF runs 5 fits in parallel with `n_jobs=10` each. The final model runs 2 fits with `n_jobs=27`.

## Results (`results/ebm.json`, `results/ebm.md`; score.py run as is, B=1000, seed 20261031)
| split | PRIMARY [95% CI] | dPRIMARY vs persistence [CI] | AUC [CI] | Brier | ECE iso |
|---|---|---|---|---|---|
| by-mission | 0.831 [0.795, 0.868] | +0.011 [-0.019, 0.038] | 0.980 [0.973, 0.986] | 0.0188 | 0.004 |
| by-event | 0.824 [0.792, 0.856] | +0.004 [-0.013, 0.023] | 0.980 [0.974, 0.987] | 0.0190 | 0.006 |

- By-mission gate: (a) fails because the CI lower bound is below 0. (b) passes: PRIMARY >= persistence - 0.01 and the AUC CI
  lower bound 0.973 is above the persistence AUC 0.958. Calibration is ok.
- `results/ebm.md` shows a "serve ebm" verdict. That verdict compares ebm with persistence only, because score.py was run with
  `--models ebm`. It is **not** the bake-off decision. Against the reference rows, ebm ties lgbm_a13 on by-mission PRIMARY
  (0.831) and is below lgbm_all (0.842). Its AUC (0.980) is also below lgbm_a13 (0.983) and lgbm_all (0.985). Run
  `score.py` over all entrants for the decision.
- Its value is interpretability at roughly LightGBM-A13 accuracy. The top `crit` terms are last/min miss distance
  (both the raw and log forms), max relative_position_n, `last_relative_position_t x min_sig_r_comb`, d1_max_risk_estimate
  and std_log_miss.

## Runtime
About 20 min wall clock in total on a heavily shared box (load average up to 170): 1 timing fit, OOF by-event 367 s,
OOF by-mission 395 s, final 2 fits plus exports 321 s, and score.py about 1 min.

## Final model (`final/`)
- `ebm_crit.joblib` and `ebm_ge.joblib` are fit on all 8,293 eligible events with 14 mission-grouped early-stopping bags.
  `model_meta.json` holds the feature list, parameters and selected interactions.
- `predict.py <features_A13.parquet> <out.parquet>` writes event_key, p_safe, p_elevated and p_critical.
- `test_features_A13.parquet` is built from `test_data.csv` with the harness A13 recipe (`H.clean` +
  `H.event_features(..., ALLOWED[1:], TREND_COLS_A13)`). Every test CDM has ttca >= 2 d. There are 2,167 events, keyed `test:<id>`,
  and the c_object_type category sets are identical to train.
  -> `../test_predictions.parquet`.
- `shapes.json` (compact, about 4.5 MB): for each model (`crit`, `ge`), the intercept and every term sorted by importance. A main
  term has `bin_edges` (len = len(scores)+1; score[i] covers [edge i, edge i+1)), `scores` (additive log-odds),
  `lower`/`upper` (±1 sd across bags) and `missing_score`. An interaction has left/right edges and a score matrix.
  Consecutive bins with equal scores are merged and values are rounded to 5 dp. The full export is in `shapes_full.json` (15 MB).
- `local_examples.json`: train:2563, train:8767, train:2427 and train:8756. Each entry has feature values, p_tiers, and for each
  model the intercept, probability, logit and per-term contributions sorted by |score|. train:2563 is critical but its latest tier
  is safe. The EBM gives it p_critical 0.106, driven mainly by interaction terms. The other three get p_critical between 0.90 and 0.98,
  driven by miss distance.

## What was tried
One configuration only, per the pre-registration. There was no tuning, no feature screening (not needed for speed) and no
3-class variant.

## Code
`models/ebm/ebm.py` (`time` | `oof <split> [n_par]` | `final`). Logs are in `models/ebm/logs/` (`run.log` and per-fit `oof_<split>.json`).
