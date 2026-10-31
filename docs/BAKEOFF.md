# Forecaster bake-off: what we serve and why

ORBITAL DODGE, Airbus hackathon Challenge 1 (Kelvins CDM training set), 2026-10-31. The full record is in `research/bakeoff/` (PREREG.md, LEADERBOARD.md, audit/). The served artifact is in `registry/forecaster_bakeoff/`.

## Question
At the 2-day decision point, can any model beat a simple rule at predicting the tier of the **final** CDM (< 1 d before TCA)? The rule is tier persistence: "the latest CDM's tier is the final tier". Tiers: critical if risk ≥ −4 or miss < 1 km, elevated if risk ≥ −6 or miss ≤ 5 km, safe otherwise. The data is 8,293 eligible events, of which 551 are critical. Inputs use only CDMs with time_to_tca ≥ 2 d. The Kelvins Zenodo release is never used, because it holds the hackathon test labels.

## Pre-registered rule (PREREG.md, frozen before any score was read, sha256-pinned, no amendments)
- **PRIMARY:** critical recall at persistence's own false-alarm count (113). Bootstrap: 1,000 resamples of events, paired across models.
- **Decision split:** whole missions held out (GroupKFold 5). The by-event split is reported as a sanity check only.
- **Winner:** the highest by-mission PRIMARY. It is served only if one of these holds:
  - (a) the lower bound of the paired ΔPRIMARY CI vs persistence is > 0, or
  - (b) PRIMARY ≥ persistence − 0.01 and the AUC CI lower bound is > persistence's AUC.

  Otherwise we rank with persistence. If ECE after isotonic is > 0.03, the model gets a calibration warning.

## Contenders
- **Entrants:** `futures` (risk-futures Monte Carlo MDN), `tabpfn` (TabPFN v2), `ebm` (Explainable Boosting Machine) and `gru` (sequence model).
- **References:** `persistence`, `hgb_a13` (the old baseline), `lgbm_a13` and `lgbm_all` (LightGBM on 84 or 233 hand-built features).

## Leaderboard, by mission (decision split, false-alarm budget 113)
| model | PRIMARY [95% CI] | crit caught | ΔPRIMARY vs persistence [paired CI] | AUC [95% CI] | Brier | ECE after iso | gate |
|---|---|---|---|---|---|---|---|
| **lgbm_all** | **0.842** [0.808, 0.876] | 464/551 | +0.022 [+0.000, +0.041] | 0.985 [0.980, 0.990] | 0.018 | 0.005 | PASS (b) |
| tabpfn | 0.838 [0.807, 0.874] | 462/551 | +0.018 [+0.002, +0.042] | 0.984 [0.978, 0.989] | 0.018 | 0.004 | PASS (a) |
| gru | 0.831 [0.796, 0.865] | 458/551 | +0.011 [−0.012, +0.032] | 0.983 [0.977, 0.988] | 0.019 | 0.003 | PASS (b) |
| lgbm_a13 | 0.831 [0.795, 0.863] | 458/551 | +0.011 [−0.017, +0.030] | 0.983 [0.977, 0.988] | 0.018 | 0.004 | PASS (b) |
| futures | 0.831 [0.798, 0.861] | 458/551 | +0.011 [−0.007, +0.026] | 0.982 [0.976, 0.988] | 0.019 | 0.004 | PASS (b) |
| ebm | 0.831 [0.795, 0.868] | 458/551 | +0.011 [−0.019, +0.038] | 0.980 [0.973, 0.986] | 0.019 | 0.004 | PASS (b) |
| hgb_a13 | 0.824 [0.786, 0.856] | 454/551 | +0.004 [−0.027, +0.025] | 0.983 [0.977, 0.988] | 0.022 | 0.004 | PASS (b) |
| persistence | 0.820 [0.789, 0.852] | 452/551 | — | 0.958 [0.947, 0.967] | 0.022 | 0.002 | reference |

**By event (sanity check only):** tabpfn 0.833, futures 0.831, gru, ebm 0.824, lgbm_a13 0.822, **lgbm_all 0.820**, persistence 0.820, hgb_a13 0.813.

## Winner and why
**We serve `lgbm_all` 1.0.0.** It has the highest by-mission PRIMARY (0.842), and the tie-breaks were not needed.
- **Gate:** it passes through (b), not (a). Its paired ΔPRIMARY CI touches zero: [+0.000, +0.041], P(Δ ≤ 0) = 0.037. Its AUC lower bound of 0.980 is well above persistence's 0.958.
- **Calibration:** ECE after isotonic is 0.005, so there is no warning.

**How to read this honestly:**
- Every ML model beats persistence clearly on **ranking**: AUC 0.98 vs 0.96, and AP 0.87–0.89 vs 0.70.
- On the **operating point that matters**, critical events caught at the same false-alarm count, the gain is small. lgbm_all catches 12 more of 551. Its by-mission lead disappears by event, where it ties persistence.
- tabpfn is 2 events behind and is the only model whose gain is significant (gate (a)).
- An audit re-ran lgbm_all with other seeds and got PRIMARY 0.828–0.842. Seed 0, the pre-registered seed, is the top of that range.
- The rule picks lgbm_all mechanically. The real-world claim is narrower: "≥ persistence on recall, much better ranking and calibrated probabilities."

## Audit outcome
- **Blocker found:** the winner had no final model, and the final-fit recipe had never been declared.
- **Fix:** the recipe is now declared, and it is exactly the procedure the OOF evaluated, applied to all events. It trains on 80% of missions, early-stops on the other 20% (best iteration 161) and does **not** refit.
- **Diagnostic:** the audit's refit-on-all variant agrees with the served model at r = 0.995 on test p_critical.
- **Calibration:** isotonic calibration of p_critical was added.
- **Result:** the model was built, exported and checked; the registry `predict.py` reproduces the exported predictions exactly. No OOF file changed, the rule was re-applied and the verdict is unchanged.
- **Other checks:** label-shuffle checks gave a null AUC of about 0.5, and two independent test-feature builds match.

Details are in LEADERBOARD.md, under "Audit outcome".

## Failure case: train:4973 (mission 2), missed by every model
68 critical events are missed by **all eight** rows at their budget thresholds. This is the one with the longest history:

| time_to_tca (d) | 6.99 → 2.11 (16 CDMs, model input) | 1.68 | 1.51 | 1.08 | **0.81** | 0.39 | 0.13 (final) |
|---|---|---|---|---|---|---|---|
| miss distance (m) | 1,715 → 1,690, flat | 1,688 | 1,689 | 1,691 | **814** | 815 | **815** |
| chaser sigma_r (m) | 32.7, constant | 32.7 | 32.7 | 32.7 | **53.4** | 53.4 | 53.4 |
| risk (log10) | −18.3 → −17.9 | −17.9 | −17.9 | −18.0 | −13.9 | −13.9 | −13.9 |
| tier | elevated | elevated | elevated | elevated | **critical** | critical | critical |

- **What the inputs showed:** for six days the debris conjunction sat stably at about 1.7 km. Collision probability was negligible (about 1e-18), the covariance was frozen, and the history was quiet.
- **What changed:** at 0.8 d a new orbit determination for the debris object (sigma_r 32.7 → 53.4 m) moved the predicted miss distance to 814 m. That flips the tier to critical through the **miss < 1 km** clause, even though the risk is still about 1e-14.
- **Model scores:** p_critical was 0.015 (lgbm_all), 0.024 (tabpfn), 0.054 (persistence) and < 0.04 for all other models.
- **Lesson:** nothing in the ≥ 2 d data predicts this. It is a late, new measurement. Misses like this put a hard ceiling on PRIMARY: 68 of 551 criticals are missed by all eight models, so the ceiling is about 88% (483/551), and they are an argument for showing the *uncertainty* ("could still move by about 1 km") as well as the tier.

## What we'd do next
1. **Re-run with several seeds.** Re-run the bake-off with averages over 5 seeds, or seed-bagged models, as a pre-registered amendment, so that a 2-event gap stops deciding the winner. Consider a tabpfn + lgbm_all average.
2. **Model the miss-distance clause directly.** Use a regression on the final miss with a predictive interval, for example by feeding the futures MDN into the tier rule, because "miss < 1 km" drives many late flips.
3. **Re-check by mission.** Only 19 missions exist, so check per-mission recall, and re-validate the alert threshold (p_critical ≥ 0.30) once labelled hold-out data is available. The test set (2,167 events) is unscored here. 191 of its events (8.8%) alert, against 7.0% on train.
