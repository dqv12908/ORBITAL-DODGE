# ORBITAL DODGE forecaster bake-off: pre-registration

Written 2026-10-31, before any contender result was scored or read. The reference models' OOF files had been
generated but not scored. `oof/futures_{event,mission}.parquet` were already in `oof/` (dropped there by the futures
builder) when this file was written. They were neither opened nor scored. After this file is
frozen, changes are allowed only as dated amendments at the bottom, and an amendment cannot change the decision rule.

## 1. Question

Which forecaster, if any, should the product serve to predict the brief tier of the final CDM (< 1 d before TCA)
from the CDMs available at the 2-day decision point? Should the product instead rank with tier persistence?

## 2. Data, task, label (frozen in `harness.py`)

- Data: Kelvins CDM `train_data.csv` from the hackathon (`data/raw/Challenge 1 Airbus /`). **The Kelvins Zenodo release is not used**: it
  contains the private test labels, which are the hackathon test set. `test_data.csv` is not used for scoring.
- Cleaning: any position sigma ≥ 1e7 m becomes NaN and sets `cov_unknown = 1`. `c_object_type` becomes a category code.
- Eligible events: at least one CDM with `time_to_tca ≥ 2` d and a final CDM with `time_to_tca < 1` d. That gives **8,293 events**.
  The key is `train:<event_id>`.
- Inputs: only CDMs with `time_to_tca ≥ 2` d. No feature can use the final CDM, any CDM < 2 d, or anything derived
  from them (final risk, final miss, final ttca, or the tier of CDMs < 2 d). `y_risk`/`y_miss` (in `data/labels.parquet`) may be
  used **only as auxiliary training targets inside the training folds**.
- Label: the brief tier of the FINAL CDM. CRITICAL if risk ≥ −4 or miss < 1,000 m. ELEVATED if risk ≥ −6 or miss ≤ 5,000 m.
  SAFE otherwise. Counts: **6,138 safe / 1,604 elevated / 551 critical**.
- Shared inputs: `data/features_A13.parquet` (84 features from the 13 allowed columns: latest values plus trends),
  `data/features_ALL.parquet` (233 features: all numeric columns plus risk, covariance and other histories) and
  `data/sequences.npz` (per event, the ≥2 d CDMs by decreasing ttca, padded to 17 steps, with a mask and 19 per-step features:
  the 12 allowed non-id columns, risk, log10 sigma t/n for both objects, mahalanobis and cov_unknown. NaN is kept. Each
  contender must fit standardisation **per fold, on training folds only**).

## 3. Splits (frozen in `data/folds.parquet`)

Events are sorted by event_id.
- **by-event**: `KFold(5, shuffle=True, random_state=0)`.
- **by-mission** (decision split): `GroupKFold(5)` on `mission_id`. Whole missions are held out.
- lomo: leave-one-mission-out (`fold_lomo = mission_id`). This one is diagnostic only and not part of the decision.

Every contender delivers out-of-fold predictions for **both** the by-event and by-mission splits. All fitting must happen inside the
training folds: preprocessing, standardisation, feature selection, hyperparameter choice, early stopping and calibration. Hyperparameters may
be tuned only on inner splits of the training folds, and on the by-mission split those inner splits must be grouped by mission. Tuning
on the outer test folds is not allowed.

Delivery: `oof/<model>_<split>.parquet` with columns `event_key, p_safe, p_elevated, p_critical`. Every one of the 8,293 events
must appear once. Each row must be finite and sum to 1. `harness.write_oof` enforces this.

## 4. Metrics (implemented in `score.py`, run as is)

**PRIMARY: critical recall at the persistence false-alarm budget.** Tier persistence predicts the tier of the latest ≥2 d CDM.
Its false-alarm count on the split (events it calls critical that are not critically final) is the budget. The model raises an alert when
`p_critical ≥ t`. We choose the lowest threshold t whose alert set has **≤ budget** false alarms. Tied scores enter or leave together, so
among feasible thresholds the best recall wins. PRIMARY is the recall at that threshold.

Secondary metrics for critical, on `p_critical`: ROC AUC, average precision, Brier, ECE (10 equal-width bins), and ECE/Brier after
**cross-fitted isotonic calibration** (isotonic fitted on the other folds' OOF predictions of the same split). Also the macro-F1 of the
argmax tier, and critical recall, precision and false alarms at p_critical ≥ 0.5 and ≥ 0.10.

Uncertainty: 1,000 bootstrap resamples of events, seed 20261031. Every model uses the **same** resamples, so the comparisons
are paired. Within each resample, the persistence budget is recomputed on that resample. We report percentile 95% CIs for PRIMARY,
AUC and Brier, and a paired-bootstrap 95% CI for ΔPRIMARY = PRIMARY(model) − PRIMARY(persistence).

## 5. Decision rule (by-mission split only)

1. **Winner:** the row with the highest by-mission PRIMARY among all rows except persistence. That means contenders plus the
   ML references `hgb_a13`, `lgbm_a13` and `lgbm_all`. Ties (at 1e-6) go to the higher by-mission AUC, then the lower Brier.
2. **Serving gate.** The winner becomes the served forecaster only if, on the by-mission split, **either**
   (a) the paired-bootstrap 95% CI lower bound of ΔPRIMARY vs persistence is **> 0**, **or**
   (b) PRIMARY is within 0.01 of persistence, i.e. PRIMARY(model) ≥ PRIMARY(persistence) − 0.01, **and** the
       model's AUC 95% CI lower bound is greater than the persistence AUC (point estimate).
   If the gate fails, **the product ranks with persistence and we say so**. We do not fall back to the runner-up.
   We publish the full table either way.
3. **Calibration requirement.** The served model's by-mission ECE after cross-fitted isotonic must be ≤ 0.03. If it is not, the model
   is served with a calibration warning shown in the UI and the write-up.
4. The by-event split is reported next to the decision split as a sanity check. It does not decide anything. If a model passes on by-mission but is
   clearly worse than persistence by-event, we note it in the write-up, and the decision stays as it is.

Interpretation note for (b): "within 0.01 of persistence" is read as non-inferiority, PRIMARY ≥ persistence − 0.01. Otherwise
a model better than persistence by more than 0.01, whose ΔPRIMARY CI still crosses 0, would fail (b) while a slightly worse one
passed. This is the only reading `score.py` implements.

## 6. Entrants

Contenders:

| name | what | owner/notes |
|---|---|---|
| `futures` | risk-futures Monte Carlo | delivered by another builder into `oof/futures_{event,mission}.parquet` |
| `tabpfn` | TabPFN classifier on A13 or ALL features | TabPFN **v2** weights (ungated, cached at `~/.cache/tabpfn/`, works offline). v2.5+ weights need a Prior Labs license token and are not used |
| `ebm` | Explainable Boosting Machine (`interpret`) | |
| `gru` | GRU over `sequences.npz` | standardisation fitted per fold |

References (produced by `refs.py`):

| name | what |
|---|---|
| `persistence` | tier of the latest ≥2 d CDM. p = empirical P(final tier \| latest tier), cross-fitted. argmax = latest tier |
| `hgb_a13` | sklearn HistGradientBoostingClassifier, A13, 300 iterations, lr 0.05, min_samples_leaf 20, seed 0 (current baseline) |
| `lgbm_a13` | LightGBM multiclass, A13, lr 0.03, 31 leaves, early stopping (100) on an inner 15% stratified split (by-event) or a 20% mission-grouped split (by-mission) |
| `lgbm_all` | same as `lgbm_a13`, on ALL features |

## 7. Budget and reproducibility

Each contender gets about 75 minutes of wall clock in total. There is no open-ended tuning. Software: the shared `.venv` (Python 3.12,
numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, lightgbm 4.7.0, interpret 0.7.8, tabpfn 9.2.0, torch 2.6.0+cu124).
Commands: `python harness.py build` → `python refs.py` → each contender → `python score.py`.

## Amendments

(none)
