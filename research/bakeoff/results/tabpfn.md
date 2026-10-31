# Bake-off leaderboard (tabpfn.md)

8293 eligible events, 551 critical. Bootstrap B=1000 over events (seed 20261031, same resamples for every model, so deltas are paired). PRIMARY = critical recall at <= the tier-persistence false-alarm count on the split (recomputed inside each bootstrap resample).

## Split: by-event  (false-alarm budget = 113)

| model | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE | ECE iso | macro-F1 | @0.5 rec/prec/FA | @0.10 rec/prec/FA | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn | 0.833 [0.798, 0.866] | 0.013 [-0.007, 0.032] | 0.984 [0.978, 0.989] | 0.886 | 0.018 [0.016, 0.020] | 0.004 | 0.003 | 0.878 | 0.789/0.845/80 | 0.913/0.633/292 |  |
| persistence | 0.820 [0.789, 0.852] | 0.000 [0.000, 0.000] | 0.958 [0.948, 0.967] | 0.674 | 0.022 [0.020, 0.025] | 0.005 | 0.005 | 0.868 | 0.820/0.800/113 | 0.820/0.800/113 |  |

## Split: by-mission  (false-alarm budget = 113)

| model | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE | ECE iso | macro-F1 | @0.5 rec/prec/FA | @0.10 rec/prec/FA | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn | 0.838 [0.807, 0.874] | 0.018 [0.002, 0.042] | 0.984 [0.978, 0.989] | 0.874 | 0.018 [0.016, 0.020] | 0.004 | 0.004 | 0.879 | 0.799/0.848/79 | 0.909/0.641/281 | dPRIMARY CI lo > 0 |
| persistence | 0.820 [0.789, 0.852] | 0.000 [0.000, 0.000] | 0.958 [0.947, 0.967] | 0.699 | 0.022 [0.020, 0.025] | 0.001 | 0.002 | 0.868 | 0.820/0.800/113 | 0.820/0.800/113 | n/a (reference) |

## Pre-registered decision (by-mission split)

- Highest by-mission PRIMARY (excluding persistence): **tabpfn**
- Gate: PASS (dPRIMARY CI lo > 0)
- Calibration (ECE after cross-fitted isotonic <= 0.03): ok
- Verdict: serve tabpfn
