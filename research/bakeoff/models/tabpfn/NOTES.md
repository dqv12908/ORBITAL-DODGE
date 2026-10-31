# tabpfn: TabPFN v2 on A13 features (bake-off contender)

## Setup
- Model: `TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=..., n_estimators=8, random_state=0)`,
  tabpfn 9.2.0, checkpoint `~/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (ungated v2), `HF_HUB_OFFLINE=1`.
  v2.5+ weights are licence-gated (Prior Labs token); no licence was accepted and they were not used.
- Features: `data/features_A13.parquet` (84 features), float32, non-finite -> NaN (TabPFN handles NaN natively).
  No feature selection needed (84 < 500). No external scaling: TabPFN's own preprocessing (quantile/power transforms,
  feature shuffling, ensembling over preprocessors) is fitted inside `.fit()` on the training folds only.
- Target: `y_tier` (0 safe / 1 elevated / 2 critical), 3-class. No auxiliary targets, no labels in features.
- Per outer fold: fit on the training folds, `predict_proba` on the held-out fold, both splits (`fold_event`, `fold_mission`).
- Hyperparameters: all TabPFN v2 defaults except `n_estimators=8` (fixed a priori for the time budget). No tuning at all,
  so nothing was chosen on outer folds. The A13 feature set was declared as the `tabpfn` entry before any scoring.

## Runtime (wall clock, 56-core box shared with other builders; load average up to ~150)
- by-event OOF: 611 s on a V100 (torch `cuda:1`), folds 45-247 s.
- by-mission OOF: 727 s on the RTX 2060 (torch `cuda:3`), folds 105-260 s (run in parallel with by-event).
- Final fit on 8,293 events + predict 2,167 test events: 3 min 44 s (PG503).
- Most per-fold time is TabPFN's CPU preprocessing (processes sat at ~800-1000% CPU), not the GPU.
- Note: torch's default CUDA order is FASTEST_FIRST, so `cuda:k` != nvidia-smi index k on this server. The runlogs record the
  actual device name. Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` to address cards by nvidia-smi index (done for the final predict).

## Results (score.py, pre-registered, B=1000 paired bootstrap) -> results/tabpfn.json, results/tabpfn.md
| split | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | Brier | ECE iso |
|---|---|---|---|---|---|
| by-mission | 0.838 [0.807, 0.874] | +0.018 [+0.002, +0.042] | 0.984 [0.978, 0.989] | 0.018 | 0.004 |
| by-event   | 0.833 [0.798, 0.866] | +0.013 [-0.007, +0.032] | 0.984 [0.978, 0.989] | 0.018 | 0.003 |

- Gate (by-mission): passes via (a), dPRIMARY CI lower bound > 0. It is the only row so far (refs included) that passes via (a).
- Against the references (results/tabpfn_vs_refs.md, same scorer): by-mission PRIMARY 0.838 is second to lgbm_all (0.842,
  i.e. 2 critical events of 551 more). Under the pre-registered rule (highest by-mission PRIMARY wins) lgbm_all still wins
  among refs + tabpfn. tabpfn has the best by-event PRIMARY (0.833 vs 0.822 lgbm_a13, 0.820 lgbm_all).
- Calibration fine (ECE 0.004 raw, 0.004 after isotonic).

## Diagnostic only (NOT an entrant): TabPFN v2 on ALL features (233), by-mission split
- Run in parallel, out of curiosity / per the brief's "try ALL too if time". Stored in `models/tabpfn/diag/` (not in `oof/`)
  and scored with the unmodified scorer via `score_diag.py` -> `diag/leaderboard_diag.md`.
- by-mission PRIMARY 0.844 [0.812, 0.878], dPRIMARY +0.024 [0.000, 0.046], AUC 0.984, ECE iso 0.005. Runtime 1,049 s.
- Picking ALL over A13 because of this number would be selection on the outer folds, which PREREG forbids, so the entry
  stays A13. The ALL vs A13 gap is 3 critical events; the ALL vs lgbm_all gap is 1 event: noise.
- No by-event run for ALL (time).

## Final model and test predictions
- `final/fit_final.py`: TabPFN is in-context, so the final model = training table (`final/train_X.parquet`, A13, all 8,293
  eligible events; `final/train_y.parquet`) + `final/config.json`. `final/predict.py FEATURES.parquet OUT.parquet [--device]`
  rebuilds the fitted estimator (random_state=0) and writes event_key, p_safe, p_elevated, p_critical.
- `final/build_test_features.py`: same harness recipe on `test_data.csv` (all 24,484 test CDMs have ttca >= 2 d; 2,167
  events), c_object_type coded with the train category order (identical set) -> `final/test_features_{A13,ALL}.parquet`.
- `test_predictions.parquet`: 2,167 rows, keys `test:<id>`, mean p_critical 0.081 (train critical rate 0.066).

## Files
`run_oof.py` (OOF, both splits), `score_diag.py`, `runlog_*.json`, `final/`, `diag/`, logs in `../../logs/tabpfn_*.log`.
