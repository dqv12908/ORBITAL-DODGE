# gru: sequence model over the >= 2 d CDM history (bake-off contender)

## Setup
- Input: `data/sequences.npz` (8293 x 17 x 19). It is right-padded, its steps run by decreasing ttca, and it holds only CDMs with ttca >= 2 d. No label
  information is used as a feature, and `y_risk`/`y_miss` are not used at all. The target is `y_tier` (3 tiers).
- Per-step engineering (`final/gru_lib.py: engineer`, 35 channels):
  - time: ttca and the gap since the previous CDM
  - distance and position: log10 miss and its step change, relative speed, signed-log relative position r/t/n
  - uncertainty: log10 sigma_r (c, t, combined), log |r|/sigma_r, the four log sigma t/n, log1p mahalanobis, cov_unknown
  - risk: max_risk_estimate and its change, risk and its change, a risk-floor flag (risk <= -29.9)
  - space weather: F10, AP, an F10-missing flag
  - the four brief-tier threshold indicators of each input CDM (risk >= -4, miss < 1 km, risk >= -6, miss <= 5 km)
  - c_object_type one-hot (5)
- Standardisation (mean/std over valid steps) is fitted **on the training folds only** for each outer fold, and on all events for
  the final fit. Values are clipped to +-6 sd, and NaN and padding become 0 after standardising.
- Model: Linear(35->64)+GELU+Dropout(0.1), then a 1-layer GRU(64). The read-out is at the last valid step. With right padding,
  padded steps never reach the read-out, so this is the masking. The read-out is concatenated with the last step's inputs and
  length/17, then passed to an MLP(64, dropout 0.2) that gives 3 logits.
- Loss: class-weighted cross-entropy with weights (n / 3 n_k)^0.5 from the training folds. Output probabilities are
  prior-corrected (p_k proportional to p_k / w_k) to undo the shift the weighting causes.
- Optimiser: AdamW, lr 2e-3, weight decay 1e-2, batch 256, gradient clip 1.0, up to 200 epochs. Early stopping (patience 20) uses
  the inner-validation weighted CE.
- Inner split: by-event uses StratifiedShuffleSplit at 15 %. By-mission and the final fit use GroupShuffleSplit at 20 % by mission.
- Ensemble: the mean of the probabilities from seeds 0-4. Each seed uses its own inner split (random_state = seed), so the ensemble
  is also bagged over early-stopping splits.
- Best epochs were small (3-14): the model fits fast, so early stopping matters.

## Inner-split tuning
Only inner splits were used and no outer fold was scored. The selection criterion was fixed beforehand: mean inner-val weighted CE
on by-mission. See `tune.sh` and `tune/*.json`. Every config ran all 5 folds x 5 seeds on the same inner splits.

| config | mission inner wCE | mission inner AUC | event inner wCE |
|---|---|---|---|
| **GRU h64 lr 2e-3 (chosen; the default)** | **0.3060** | 0.9772 | **0.2776** |
| GRU h64 lr 5e-4 | 0.3117 | 0.9748 | 0.2788 |
| GRU h32 lr 5e-4 | 0.3143 | 0.9763 | 0.2783 |
| GRU h64 x2 layers lr 5e-4 | 0.3120 | 0.9752 | 0.2801 |
| GRU h64 lr 5e-4 wd 0.1 dropout 0.3 | 0.3131 | 0.9750 | 0.2789 |
| 2-layer transformer encoder h64 lr 5e-4 | 0.3115 | 0.9758 | 0.2892 |

The transformer did no better, so the GRU is the contender. The default config won, so the OOF files it produced first were kept.
The tuning runs never predict the outer folds.

## Results
From `results/gru.json` (`score.py --models gru`, B=1000, seed 20261031):

| split | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier | ECE iso | macro-F1 |
|---|---|---|---|---|---|---|---|
| by-mission | 0.831 [0.796, 0.865] | +0.011 [-0.012, 0.032] | 0.983 [0.977, 0.988] | 0.877 | 0.0187 | 0.003 | 0.873 |
| by-event | 0.824 [0.794, 0.863] | +0.004 [-0.014, 0.030] | 0.982 [0.975, 0.987] | 0.871 | 0.0188 | 0.004 | 0.872 |

- Against persistence alone, gru passes the gate via (b).
- Among the pre-registered rows, its by-mission PRIMARY equals lgbm_a13 (0.831) and is below lgbm_all (0.842), so on its own it would
  not be the winner.
- The full leaderboard (`score.py` without `--models`) decides.

## Final model and test predictions
- `final/` holds the 5 seed state dicts, `norm.npz` (mu, sd and class weights fitted on all 8293 events), `meta.json` (config),
  `gru_lib.py` and `predict.py`. The final model was fit on all eligible events, with a mission-grouped 20 % inner early-stopping split per seed.
- `make_test_sequences.py` builds `test_sequences.npz` from `test_data.csv` with the harness recipe (`H.clean`, `H.build_sequences`).
  It has 2167 events, all CDMs are >= 2 d, and the c_object_type categories match train.
- `predict.py --input test_sequences.npz --out test_predictions.parquet` writes `test_predictions.parquet` (2167 rows,
  event_key `test:<id>`).
- `predict.py` also accepts a raw CDM csv/parquet (`--prefix test`). That path gave identical predictions (max diff 0).

## Runtime
| step | wall clock |
|---|---|
| CV, both splits in parallel on 2 GPUs (V100 idx 1, PG503 idx 3) | about 1.7 min |
| Final fit | 0.5 min |
| Inner tuning, 6 configs x 2 splits | about 5 min |
| Scoring | about 1 min |
| End to end, including setup | about 15 min |
