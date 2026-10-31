# Bake-off leaderboard (gru.md)

8293 eligible events, 551 critical. Bootstrap B=1000 over events (seed 20261031, same resamples for every model, so deltas are paired). PRIMARY = critical recall at <= the tier-persistence false-alarm count on the split (recomputed inside each bootstrap resample).

## Split: by-event  (false-alarm budget = 113)

| model | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE | ECE iso | macro-F1 | @0.5 rec/prec/FA | @0.10 rec/prec/FA | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gru | 0.824 [0.794, 0.863] | 0.004 [-0.014, 0.030] | 0.982 [0.975, 0.987] | 0.871 | 0.019 [0.017, 0.021] | 0.003 | 0.004 | 0.872 | 0.789/0.838/84 | 0.906/0.629/294 |  |
| persistence | 0.820 [0.789, 0.852] | 0.000 [0.000, 0.000] | 0.958 [0.948, 0.967] | 0.674 | 0.022 [0.020, 0.025] | 0.005 | 0.005 | 0.868 | 0.820/0.800/113 | 0.820/0.800/113 |  |

## Split: by-mission  (false-alarm budget = 113)

| model | PRIMARY [95% CI] | dPRIMARY vs persistence [95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE | ECE iso | macro-F1 | @0.5 rec/prec/FA | @0.10 rec/prec/FA | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gru | 0.831 [0.796, 0.865] | 0.011 [-0.012, 0.032] | 0.983 [0.977, 0.988] | 0.877 | 0.019 [0.017, 0.021] | 0.003 | 0.003 | 0.873 | 0.804/0.830/91 | 0.909/0.616/312 | PRIMARY within 0.01 & AUC CI lo > persistence AUC |
| persistence | 0.820 [0.789, 0.852] | 0.000 [0.000, 0.000] | 0.958 [0.947, 0.967] | 0.699 | 0.022 [0.020, 0.025] | 0.001 | 0.002 | 0.868 | 0.820/0.800/113 | 0.820/0.800/113 | n/a (reference) |

## Pre-registered decision (by-mission split)

- Highest by-mission PRIMARY (excluding persistence): **gru**
- Gate: PASS (PRIMARY within 0.01 & AUC CI lo > persistence AUC)
- Calibration (ECE after cross-fitted isotonic <= 0.03): ok
- Verdict: serve gru
