# Bake-off leaderboard (leaderboard_diag.md)

8293 eligible events, 551 critical. Bootstrap B=1000 over events (seed 20261031, same resamples for every model, so deltas are paired). PRIMARY = critical recall at <= the tier-persistence false-alarm count on the split (recomputed inside each bootstrap resample).

## Split: by-mission  (false-alarm budget = 113)

| model | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE | ECE iso | macro-F1 | @0.5 rec/prec/FA | @0.10 rec/prec/FA | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn_all | 0.844 [0.812, 0.878] | 0.024 [0.000, 0.046] | 0.984 [0.978, 0.989] | 0.886 | 0.018 [0.016, 0.020] | 0.004 | 0.005 | 0.881 | 0.802/0.844/82 | 0.920/0.602/335 | PRIMARY within 0.01 & AUC CI lo > persistence AUC |
| persistence | 0.820 [0.789, 0.852] | 0.000 [0.000, 0.000] | 0.958 [0.947, 0.967] | 0.699 | 0.022 [0.020, 0.025] | 0.001 | 0.002 | 0.868 | 0.820/0.800/113 | 0.820/0.800/113 | n/a (reference) |

## Pre-registered decision (by-mission split)

- Highest by-mission PRIMARY (excluding persistence): **tabpfn_all**
- Gate: PASS (PRIMARY within 0.01 & AUC CI lo > persistence AUC)
- Calibration (ECE after cross-fitted isotonic <= 0.03): ok
- Verdict: serve tabpfn_all
