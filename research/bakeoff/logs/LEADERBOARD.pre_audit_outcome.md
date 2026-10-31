# ORBITAL DODGE forecaster bake-off: LEADERBOARD

Generated 2026-10-31 by `make_leaderboard.py` from `leaderboard.json`, which `score.py` wrote when run unmodified over every file in `oof/`. The decision rule is `PREREG.md` section 5 (its sha256 matches `PREREG.sha256`; no amendments), applied mechanically.

8293 eligible events, 551 critical. Bootstrap B=1000 over events, seed 20261031, with the same resamples for every model (paired). PRIMARY = critical recall at no more than the tier-persistence false-alarm count (113 on both splits, recomputed inside each resample). ECE uses 10 equal-width bins on p_critical. 'ECE after iso' is after cross-fitted isotonic calibration (fitted on the other folds' OOF predictions of the same split).

## Decision (by-mission split)

- **Winner: `lgbm_all`.** It has the highest by-mission PRIMARY, 0.842 (464/551 critical events caught at 113 false alarms). The runner-up is `tabpfn` at 0.838 (462/551). There is no tie at 1e-6, so the AUC and Brier tie-breaks were not used.
- **Serving gate: PASS via (b).** (a) fails: the paired dPRIMARY CI is +0.022 [+0.000, +0.041], and its lower bound 0.000 is not > 0 (P(d <= 0) = 0.037). (b) passes: PRIMARY 0.842 >= persistence 0.820 - 0.01, and the AUC CI lower bound 0.980 > persistence AUC 0.958.
- **Calibration requirement: met.** ECE after cross-fitted isotonic is 0.005 (limit 0.03; raw ECE 0.005).
- **Verdict: serve lgbm_all.**
- By-event sanity check (section 5.4): `lgbm_all` scores by-event PRIMARY 0.820, dPRIMARY +0.000 [-0.017, +0.029]. That ties persistence and is not clearly worse, so the rule has nothing to flag. Its by-mission lead does not reproduce on the by-event split. The decision stands.
- Context, not part of the rule: `tabpfn` is the only row that passes the gate via (a), and it has the best by-event PRIMARY (0.833). The rule ranks by by-mission PRIMARY, not by gate strength. The gap to the winner is 2 of 551 critical events, well inside both CIs.
- Risk-futures (`futures`) was delivered on time (before the pre-registration was frozen) and passed validation. It was scored as a contender.

## Split: by-mission (decision split), false-alarm budget 113

| model | PRIMARY [95% CI] | crit caught | dPRIMARY vs persistence [paired 95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE before | ECE after iso | macro-F1 | @0.50 recall / FA | @0.10 recall / FA | OOF runtime | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lgbm_all** | 0.842 [0.808, 0.876] | 464/551 | +0.022 [+0.000, +0.041] | 0.985 [0.980, 0.990] | 0.889 | 0.018 [0.016, 0.020] | 0.005 | 0.005 | 0.882 | 0.791 / 74 | 0.898 / 247 | 56.8 s* | PASS (b) |
| tabpfn | 0.838 [0.807, 0.874] | 462/551 | +0.018 [+0.002, +0.042] | 0.984 [0.978, 0.989] | 0.874 | 0.018 [0.016, 0.020] | 0.004 | 0.004 | 0.879 | 0.799 / 79 | 0.909 / 281 | 727 s | PASS (a) |
| gru | 0.831 [0.796, 0.865] | 458/551 | +0.011 [-0.012, +0.032] | 0.983 [0.977, 0.988] | 0.877 | 0.019 [0.017, 0.021] | 0.003 | 0.003 | 0.873 | 0.804 / 91 | 0.909 / 312 | ~100 s | PASS (b) |
| lgbm_a13 | 0.831 [0.795, 0.863] | 458/551 | +0.011 [-0.017, +0.030] | 0.983 [0.977, 0.988] | 0.883 | 0.018 [0.016, 0.021] | 0.004 | 0.004 | 0.876 | 0.795 / 79 | 0.898 / 256 | 56.8 s* | PASS (b) |
| futures | 0.831 [0.798, 0.861] | 458/551 | +0.011 [-0.007, +0.026] | 0.982 [0.976, 0.988] | 0.883 | 0.019 [0.017, 0.021] | 0.008 | 0.004 | 0.875 | 0.795 / 86 | 0.940 / 481 | 372 s | PASS (b) |
| ebm | 0.831 [0.795, 0.868] | 458/551 | +0.011 [-0.019, +0.038] | 0.980 [0.973, 0.986] | 0.868 | 0.019 [0.017, 0.021] | 0.005 | 0.004 | 0.872 | 0.789 / 87 | 0.907 / 349 | 395 s | PASS (b) |
| hgb_a13 | 0.824 [0.786, 0.856] | 454/551 | +0.004 [-0.027, +0.025] | 0.983 [0.977, 0.988] | 0.876 | 0.022 [0.019, 0.025] | 0.020 | 0.004 | 0.872 | 0.775 / 78 | 0.824 / 119 | 56.8 s* | PASS (b) |
| persistence | 0.820 [0.789, 0.852] | 452/551 | +0.000 [+0.000, +0.000] | 0.958 [0.947, 0.967] | 0.699 | 0.022 [0.020, 0.025] | 0.001 | 0.002 | 0.868 | 0.820 / 113 | 0.820 / 113 | <1 s | reference |

## Split: by-event (sanity check, does not decide), false-alarm budget 113

| model | PRIMARY [95% CI] | crit caught | dPRIMARY vs persistence [paired 95% CI] | AUC [95% CI] | AP | Brier [95% CI] | ECE before | ECE after iso | macro-F1 | @0.50 recall / FA | @0.10 recall / FA | OOF runtime | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn | 0.833 [0.798, 0.866] | 459/551 | +0.013 [-0.007, +0.032] | 0.984 [0.978, 0.989] | 0.886 | 0.018 [0.016, 0.020] | 0.004 | 0.003 | 0.878 | 0.789 / 80 | 0.913 / 292 | 611 s |  |
| futures | 0.831 [0.797, 0.861] | 458/551 | +0.011 [-0.010, +0.026] | 0.983 [0.978, 0.988] | 0.883 | 0.019 [0.017, 0.021] | 0.007 | 0.003 | 0.875 | 0.786 / 85 | 0.938 / 471 | 206 s |  |
| gru | 0.824 [0.794, 0.863] | 454/551 | +0.004 [-0.014, +0.030] | 0.982 [0.975, 0.987] | 0.871 | 0.019 [0.017, 0.021] | 0.003 | 0.004 | 0.872 | 0.789 / 84 | 0.906 / 294 | ~100 s |  |
| ebm | 0.824 [0.792, 0.856] | 454/551 | +0.004 [-0.013, +0.023] | 0.980 [0.974, 0.987] | 0.859 | 0.019 [0.017, 0.021] | 0.004 | 0.006 | 0.877 | 0.795 / 74 | 0.913 / 350 | 367 s |  |
| lgbm_a13 | 0.822 [0.788, 0.857] | 453/551 | +0.002 [-0.019, +0.025] | 0.985 [0.979, 0.989] | 0.875 | 0.018 [0.016, 0.020] | 0.004 | 0.004 | 0.875 | 0.773 / 65 | 0.897 / 252 | 58.6 s* |  |
| lgbm_all | 0.820 [0.790, 0.861] | 452/551 | +0.000 [-0.017, +0.029] | 0.984 [0.979, 0.989] | 0.875 | 0.018 [0.016, 0.020] | 0.004 | 0.003 | 0.881 | 0.782 / 65 | 0.895 / 237 | 58.6 s* |  |
| persistence | 0.820 [0.789, 0.852] | 452/551 | +0.000 [+0.000, +0.000] | 0.958 [0.948, 0.967] | 0.674 | 0.022 [0.020, 0.025] | 0.005 | 0.005 | 0.868 | 0.820 / 113 | 0.820 / 113 | <1 s |  |
| hgb_a13 | 0.813 [0.779, 0.847] | 448/551 | -0.007 [-0.026, +0.017] | 0.981 [0.975, 0.987] | 0.868 | 0.022 [0.019, 0.025] | 0.020 | 0.004 | 0.872 | 0.779 / 81 | 0.817 / 116 | 58.6 s* |  |

Rows are sorted by the decision key: PRIMARY, then AUC, then Brier. Brier and ECE are on p_critical. The @t columns give critical recall and false alarms when alerting at p_critical >= t. Precision is in `leaderboard.md` and `leaderboard.json`.

## OOF file validation

Checks: all 8,293 eligible event_keys present exactly once, no extra keys, no NaN, probabilities in [0, 1], and each row sums to 1 within 1e-6. The futures builder dropped its files into `oof/` at 04:03, before PREREG was frozen at 04:05. PREREG records that they were neither opened nor scored at that point. Their `fold` column matches `folds.parquet` exactly on both splits (model_version futures-mdn-1.0). No file was rejected.

| file | rows | max row-sum deviation | result |
|---|---|---|---|
| ebm_event.parquet | 8293 | 2.2e-16 | valid |
| ebm_mission.parquet | 8293 | 2.2e-16 | valid |
| futures_event.parquet | 8293 | 8.9e-08 | valid |
| futures_mission.parquet | 8293 | 6.9e-08 | valid |
| gru_event.parquet | 8293 | 2.2e-16 | valid |
| gru_mission.parquet | 8293 | 2.2e-16 | valid |
| hgb_a13_event.parquet | 8293 | 3.3e-16 | valid |
| hgb_a13_mission.parquet | 8293 | 3.3e-16 | valid |
| lgbm_a13_event.parquet | 8293 | 3.3e-16 | valid |
| lgbm_a13_mission.parquet | 8293 | 3.3e-16 | valid |
| lgbm_all_event.parquet | 8293 | 3.3e-16 | valid |
| lgbm_all_mission.parquet | 8293 | 3.3e-16 | valid |
| persistence_event.parquet | 8293 | 0.0e+00 | valid |
| persistence_mission.parquet | 8293 | 1.1e-16 | valid |
| tabpfn_event.parquet | 8293 | 2.2e-16 | valid |
| tabpfn_mission.parquet | 8293 | 2.2e-16 | valid |

## Runtime notes

- `persistence`: by-event <1 s, by-mission <1 s. refs.py, cross-fitted lookup table.
- `hgb_a13`: by-event 58.6 s*, by-mission 56.8 s*. *refs.py total for hgb_a13 + lgbm_a13 + lgbm_all per split (logs/refs.log).
- `lgbm_a13`: by-event 58.6 s*, by-mission 56.8 s*. *shared refs.py total.
- `lgbm_all`: by-event 58.6 s*, by-mission 56.8 s*. *shared refs.py total.
- `tabpfn`: by-event 611 s, by-mission 727 s. GPU (V100 / RTX 2060); final fit + test prediction 3.7 min.
- `ebm`: by-event 367 s, by-mission 395 s. CPU; final 2 fits + exports 321 s.
- `gru`: by-event ~100 s, by-mission ~100 s. both splits in parallel on 2 GPUs (~1.7 min total); inner tuning ~5 min; final fit 0.5 min.
- `futures`: by-event 206 s, by-mission 372 s. MDN + Monte Carlo rollouts (4,000 / 10,000 paths per event); research/bigml/out/futures_*.log.

Runtimes are wall clock on a shared 56-core, 4-GPU server under heavy load (load average up to about 170), so treat them as indicative only.

## Provenance

- `score.py` was run unmodified. Its full output is in `leaderboard.md` and `leaderboard.json` (precision, thresholds, P(d <= 0)).
- Contender details are in `results/{tabpfn,ebm,gru}.md` and `models/*/NOTES.md`. The futures source is in `research/bigml/` (`export_bakeoff.py`).
- The TabPFN-ALL diagnostic in `models/tabpfn/diag/` is not entered. Choosing it would mean selecting on the outer folds, which PREREG forbids.
