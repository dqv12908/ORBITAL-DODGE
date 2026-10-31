# ORBITAL DODGE: the machine-learning deep dive

*For the ones that want to read more. The [README](README.md) has the short version and the [full report](docs/REPORT.md) covers the whole system. This page is only about the models: what we asked, what we tried, what worked, what did not, and why we made each decision.*

**How to read the numbers.** Every number on this page comes from a file in this repository, and that file is named next to the number, either in a **Source** column or in a line under the table. [`docs/FACTS.json`](docs/FACTS.json) is the checked index of the headline values. Where we got a number by arithmetic, we show the inputs. A few facts about the compute server are not recorded in any repo file; we mark those as *team-reported*.

**One naming note.** The reference model is called **the baseline** everywhere on this page: *the tier of the latest message at least 2 days before closest approach is the final tier.* In code and result files it is called `persistence`, so you will see that name in backticks when we quote a file.

---

## Contents

0. [TL;DR](#0-tldr)
1. [The question we are answering](#1-the-question-we-are-answering)
2. [The baseline, and why it is hard to beat](#2-the-baseline-and-why-it-is-hard-to-beat)
3. [Data preparation and features](#3-data-preparation-and-features)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [The cost ratio R = 10](#5-the-cost-ratio-r--10)
6. [Experiment log](#6-experiment-log)
7. [The served model](#7-the-served-model)
8. [Leakage and honesty audits](#8-leakage-and-honesty-audits)
9. [Failure analysis](#9-failure-analysis)
10. [Compute and engineering notes](#10-compute-and-engineering-notes)
11. [What we would do next](#11-what-we-would-do-next)
12. [File index](#12-file-index)

---

## 0. TL;DR

**The question.** Two days before closest approach (TCA), using only the conjunction data messages (CDMs) that exist at that moment, will this encounter end **CRITICAL**?

**The baseline.** Keep the tier of the latest message that is at least 2 days out. It is right for 92.9 % of events ([doc 11 §A](docs/design/11-dataset-experiments.md)), which makes it hard to beat.

**The winner.** A LightGBM multiclass model on 233 event features, trained on the organizer's 11,942 usable events, with its CRITICAL alert threshold set by cost (R = 10, recall floor 90 %). It won a pre-registered bake-off of 8 contenders and survived a 13-candidate attempt to dethrone it.

**Headline, on the organizer's official labels:**

| | Baseline | ORBITAL DODGE | Source |
|---|---|---|---|
| Critical recall, 80/20 split (5 seeds) | 85.1 % [83.0, 87.8] | **93.7 %** [91.9, 95.5] | [`research/organizer/results.json`](research/organizer/results.json) `metrics.event80_20` |
| Critical recall, held-out missions | 86.6 % [84.1, 89.0] | **94.8 %** [93.2, 96.4] | same, `metrics.mission` |
| Expected cost per 1,000 events, 80/20 | 110.8 | **80.1** (−28 %) | same |
| Expected cost per 1,000 events, held-out missions | 104.3 | **72.8** (−30 %) | same |
| AUC (critical), 80/20 | 0.970 | **0.988** | same |
| False alarms per 80/20 test set | **24.8** | 89.4 | same |
| Macro-F1 (3 tiers), 80/20 | 0.895 | 0.895 (tie) | same |

Brackets are 95 % bootstrap intervals. The cost reductions are derived: (110.84 − 80.12) / 110.84 = 27.7 % and (104.34 − 72.77) / 104.34 = 30.3 %.

**The trade-off, said plainly.** We raise more false alarms on purpose: about 3.5 times as many as the baseline. One CRITICAL alert in three is false (precision 0.64 by mission against 0.85 for the baseline). The cost ratio R = 10, which says one missed critical costs as much as ten false alarms, was derived from operator cost evidence before we read the sweep scores ([`docs/COST_RATIO.md`](docs/COST_RATIO.md)). Macro-F1 does not improve, and on the slice where the latest message is already the final one the baseline is exact and we lose. We report all of that below.

**Reproduce the held-out-mission row in one command, without retraining:**

```bash
server/.venv/bin/python research/organizer/recompute_table.py
# baseline (tier at T-2 d)                     722 caught, 112 missed, 126 FA, cost 1246 = 104.3 per 1,000, AUC 0.969, macro-F1 0.898
# ORBITAL DODGE served (per-fold thresholds)   791 caught,  43 missed, 439 FA, cost  869 =  72.8 per 1,000, AUC 0.988, macro-F1 0.893
```

---

## 1. The question we are answering

### 1.1 Why a forecast, not a score

The physics layer already recomputes the collision probability (Pc) of every CDM, and the rules layer already turns each CDM into a tier ([REPORT §4](docs/REPORT.md#4-physics-pc-from-the-covariance)). So we do not need ML to *score* a message. What the analyst cannot get from one message is **where the event is going**.

That matters because the decision has a deadline. Manoeuvres are planned at least 2 days before TCA (Uriot et al. 2022 §4.2, via [REPORT §2](docs/REPORT.md#the-forecasting-question-kelvins-style)), and the hidden test set's inputs stop at `time_to_tca ≥ 2 d`: its latest message is at 2.0–6.9 d, median 2.19 d ([`research/organizer/out/build_summary.json`](research/organizer/out/build_summary.json) `test_latest_ttca_quantiles`). So the product question is the Kelvins question:

> Using only CDMs at T−2 d or earlier, will this event's **final** tier be CRITICAL?

### 1.2 The label

| Tier | Rule (organizer `tier_rules.csv`) |
|---|---|
| CRITICAL | risk ≥ −4 **or** miss distance < 1,000 m |
| ELEVATED | −6 ≤ risk < −4 **or** miss 1–5 km |
| SAFE | otherwise |

Precedence is CRITICAL > ELEVATED > SAFE. `risk` is log10 of the collision probability and −30 is a placeholder for "too small to compute". Source: `data/raw/organizer/tier_rules.csv`, quoted in [`docs/FACTS.json`](docs/FACTS.json) `dataset.tier_rule`.

**Our labels match the organizer's exactly.** We recomputed every training event's final tier from the raw CDMs and it agrees with the organizer's `final_tier` on 13,154 of 13,154 events (`label_mismatches_vs_organizer = 0` in [`build_summary.json`](research/organizer/out/build_summary.json)).

**One thing worth knowing about this label:** it is mostly a miss-distance label. In the Kelvins-eligible set, 549 of 551 critical events are critical because the miss is under 1 km, and only 6 because risk ≥ −4 (4 meet both) ([doc 11 §A](docs/design/11-dataset-experiments.md)). The mean final Pc of events that end CRITICAL is only 6.1e-6 ([COST_RATIO §2](docs/COST_RATIO.md)). This shapes everything later: features about miss distance dominate, and Pc-driven criticals are too rare to learn.

### 1.3 Two populations, three sub-populations

We worked on two populations, in this order:

| Population | Events | Critical | Tiers S / E / C | Used for | Source |
|---|---|---|---|---|---|
| **Kelvins-eligible**: ≥ 1 CDM at ≥ 2 d **and** final CDM < 1 d | 8,293 | 551 (6.6 %) | 6,138 / 1,604 / 551 | v1 model, bake-off, risk futures, conformal, policy, cost sweep | [doc 11 §A](docs/design/11-dataset-experiments.md); [`research/bakeoff/PREREG.md`](research/bakeoff/PREREG.md) §2 |
| **Organizer "usable for training"**: ≥ 1 message at ≥ 2 d | 11,942 | 834 (7.0 %) | 8,946 / 2,162 / 834 | headline results, served model, false-alarm push | [`build_summary.json`](research/organizer/out/build_summary.json) |

The full training file has 13,154 events (1,212 not usable) and the test file 2,167 events whose labels are hidden ([`docs/FACTS.json`](docs/FACTS.json) `dataset`, `organizer_population`).

The organizer population splits cleanly by **when the final message arrives**:

| Sub-population | Events | Critical | What it means | Source |
|---|---|---|---|---|
| `final_lt1d`: final CDM < 1 d before TCA | 8,293 | 551 | Exactly the Kelvins-eligible set. **The forecasting problem lives here** | `build_summary.json` `subpop_counts`, `subpop_crit` |
| `final_1to2d`: final CDM 1–2 d | 599 | 39 | Small and hard | same |
| `final_ge2d`: final CDM ≥ 2 d | 3,050 | 244 | The final message **is** the latest input, so the baseline is exact by construction (recall 1, 0 false alarms) | same; [`model_card.json`](registry/forecaster_organizer/1.0.0/model_card.json) `caveats[0]` |

The third slice is 26 % of the organizer population. Any skill number on the full population is diluted by it, which is why every headline below also comes split by sub-population (§6h).

---

## 2. The baseline, and why it is hard to beat

**The rule:** take the latest CDM with `time_to_tca ≥ 2 d`, compute its tier with the rules above, and predict that the event ends in that tier. For probabilities (needed for AUC and Brier), the baseline uses the in-fold empirical rate P(final CRITICAL | latest tier) ([`research/bakeoff/PREREG.md`](research/bakeoff/PREREG.md) §6; [`research/organizer/organizer.py`](research/organizer/organizer.py) `persistence_P`).

**Why it is strong.** Most encounters do not change tier in their last two days. On the Kelvins-eligible set, from latest ≥ 2 d CDM to final CDM ([doc 11 §A](docs/design/11-dataset-experiments.md)):

| from \ to | SAFE | ELEVATED | CRITICAL |
|---|---|---|---|
| SAFE (6,040) | 5,885 (97.4 %) | 145 (2.4 %) | **10 (0.2 %)** |
| ELEVATED (1,688) | 235 (13.9 %) | 1,364 (80.8 %) | **89 (5.3 %)** |
| CRITICAL (565) | 18 (3.2 %) | 95 (16.8 %) | 452 (80.0 %) |

- The tier changed in only 592 events (7.1 %): 244 went up and 348 went down. That is the 92.9 % the baseline gets right.
- The miss distance barely moves: the median change from latest to final is +8 m, and the median |log10 ratio| is 0.015.
- On the Kelvins set the baseline catches 452 of 551 criticals (82.0 %) with 113 false alarms. That **113** becomes the false-alarm budget of the bake-off (§6b).

**The history agrees.** In the original Kelvins competition, only 12 of 97 teams beat the related latest-risk baseline (Uriot et al. 2022, via [REPORT §5.1](docs/REPORT.md#51-the-baseline-we-had-to-beat) and [doc 12 §0](docs/design/12-domain-research.md)). We replicated that metric: the latest-risk baseline scores Kelvins L = 0.804 (F2 0.411, MSE_HR 0.330), while our first HGB on the 13 brief columns scored L = 11.465 by event ([`research/experiments/out/b_baselines.txt`](research/experiments/out/b_baselines.txt)). That early result is why we stopped chasing the Kelvins regression metric and moved to the tier question with an explicit cost.

**So where can a model win?** Only on the 7 % of events that change, and especially on the 99 criticals the baseline misses (10 that were SAFE and 89 that were ELEVATED at T−2 d). A model that is equally good at "nothing changes" and slightly better at "this ELEVATED event is closing in" is the whole game.

---

## 3. Data preparation and features

### 3.1 Cleaning the sentinel covariance

About 3.2 % of training CDM rows carry a placeholder covariance: **63,781,363 m** (ten Earth radii) in all three position sigmas of one object. That is 5,129 train rows across 1,313 events in doc 11's count, and 659 test rows. On those rows the dataset's `risk` is pinned to a meaningless band (mean −14.31, sd 0.34) ([doc 11 §C.1](docs/design/11-dataset-experiments.md)).

**Rule:** any position sigma ≥ 1e7 m becomes NaN and sets `cov_unknown = 1` ([`PREREG.md`](research/bakeoff/PREREG.md) §2; [doc 11 §C.1](docs/design/11-dataset-experiments.md)). The pipeline also catches a few other meaningless sigmas (for example `c_sigma_t` between 1e7 and 6e7 m), so it flags slightly more: 5,166 train rows in 1,325 events and 665 test rows ([`docs/FACTS.json`](docs/FACTS.json) `dataset.sentinel_covariance`).

**Why NaN and a flag, not a drop:** a default covariance is itself information (the operator had no usable orbit), and dropping those events would bias the population. The tree models handle NaN natively, TabPFN does too, and the GRU sees the flag as a channel.

### 3.2 Two feature sets: A13 (84) and ALL (233)

Each event becomes one row, built **only** from its CDMs with `time_to_tca ≥ 2 d` ([`research/bakeoff/harness.py`](research/bakeoff/harness.py); the 84-feature recipe is ported to the backend in [`server/riskengine/features.py`](server/riskengine/features.py)).

| Set | Count | What is in it | Source |
|---|---|---|---|
| **A13** | 84 | Built only from the brief's 13 columns (`event_id`, `time_to_tca`, `miss_distance`, `relative_speed`, `relative_position_r/t/n`, `c_object_type`, `c_sigma_r`, `t_sigma_r`, `max_risk_estimate`, `F10`, `AP`). Latest values of the 12 non-id columns plus 4 derived ones (`log_miss`, `sig_r_comb` = combined radial sigma, `r_over_sig` = \|r\|/σ_r, `cov_unknown`), plus `n_cdm` and `first_ttca` (18), plus 6 trend statistics (`d1` last−previous, `dall` last−first, `min`, `max`, `std`, `slope` vs time) for 11 columns (66). **No `risk`** | [`server/riskengine/features.py`](server/riskengine/features.py) (asserts 84); [doc 11 §B](docs/design/11-dataset-experiments.md) |
| **ALL** | 233 | Latest values of all numeric columns plus the derived ones, `n_cdm`, `first_ttca`, and the same 6 trends for 21 columns: miss, speed, relative position, sigmas, risk, `max_risk_estimate`, Mahalanobis distance, covariance determinants, `c_obs_used`, `c_weighted_rms` | [`registry/forecaster_bakeoff/model_card.json`](registry/forecaster_bakeoff/model_card.json) `features.description` |
| **Sequences** (GRU) | 17 steps × 19 per-step features | The ≥ 2 d CDMs in decreasing `ttca`, padded to 17 steps, with a mask | [`PREREG.md`](research/bakeoff/PREREG.md) §2 |

**Why trends:** the information that the baseline throws away is *direction*. An ELEVATED event whose miss distance has fallen at every message is not the same as one that has been flat for a week. The `d1`, `dall` and `slope` features carry exactly that.

**What the model ended up using.** The served model's top features by gain are all about miss distance: `last_miss_distance` (104,768), `last_log_miss` (50,650), `min_miss_distance` (28,363), `min_log_miss`, `max_miss_distance`, `max_log_miss`, then `last_risk` (8,554), `dall_log_miss`, `dall_miss_distance`, `min_risk` ([`model_card.json`](registry/forecaster_organizer/1.0.0/model_card.json) `top_features_gain`). That is what you would expect from a label that is 549/551 miss-driven.

### 3.3 What is excluded, and why

| Excluded | Why | Where enforced |
|---|---|---|
| Any CDM with `time_to_tca < 2 d`, and anything derived from it (final risk, final miss, the tier of late messages) | That is the future we are forecasting | [`PREREG.md`](research/bakeoff/PREREG.md) §2; tested by noise substitution (§8) |
| `mission_id` | We want a model that works for a satellite it has never seen. A mission id would let it memorise per-satellite base rates | [`research/bigml/REPORT.md`](research/bigml/REPORT.md) setup; [REPORT §5.2](docs/REPORT.md#52-target-and-splits) |
| `risk` in A13 | It is not among the brief's 13 columns. A13 exists to answer "what can the 13 columns alone do?" | [`server/riskengine/features.py`](server/riskengine/features.py) docstring |
| An "is this the final message?" feature | Unknowable at prediction time. The model does see `last_time_to_tca`, which is legitimate | [`model_card.json`](registry/forecaster_organizer/1.0.0/model_card.json) `caveats[1]` |
| The Kelvins Zenodo release | It contains the private test labels, which are the hackathon's test set | [`PREREG.md`](research/bakeoff/PREREG.md) §2 |
| Velocity sigmas | Values up to 79,054 m/s are unphysical | [doc 11 §C.1](docs/design/11-dataset-experiments.md) |

### 3.4 The 13-column question

The brief lists 13 columns that can be used "for simplicity purposes". We tested what they can and cannot do:

- **They cannot compute Pc.** With only the radial sigmas, an isotropic 2D Pc reaches Spearman 0.34 against `risk`, against 0.94 with the full covariance, because the dominant uncertainty is along-track (`c_sigma_t` median about 4 km, `c_sigma_r` median 68 m). A gradient-boosted model on the 13 columns still has a 3.7 dex error ([doc 11 headline 2, §C.4](docs/design/11-dataset-experiments.md)). So the physics layer reads the full covariance, and says so.
- **They are enough to beat the baseline as a forecaster.** The same LightGBM on A13 gets 93.4 % recall with 78.4 false alarms and cost 77.2 on 80/20, and 94.6 % with 435 false alarms and cost 74.1 on held-out missions ([`research/organizer/results.json`](research/organizer/results.json) `lgbm_a13__train_organizer`). That is within the confidence intervals of the served ALL model. **A13 is a drop-in alternative** for anyone who reads the brief strictly.

We serve ALL because it won the pre-registered bake-off (§6b), not because A13 failed.

---

## 4. Evaluation protocol

### 4.1 Splits

| Split | Definition | Why | Source |
|---|---|---|---|
| **By event** (Kelvins set) | `KFold(5, shuffle=True, random_state=0)`; one row per event | Sanity check | [`PREREG.md`](research/bakeoff/PREREG.md) §3 |
| **By mission** (decision split) | `GroupKFold(5)` on `mission_id`: whole satellites held out | The honest test for a new operator | same |
| **Leave-one-mission-out (LOMO)** | 19 folds, one per mission | Diagnostic: per-satellite weaknesses | [`research/bigml/results.json`](research/bigml/results.json) `setup.splits` |
| **Organizer 80/20** | `GroupShuffleSplit(test_size=0.2, groups=event_id)`, `random_state` 0–4; about 2,389 test events and 159.2 criticals per seed; point = mean over the 5 seeds | Mirrors the organizer's own split style; the test missions are all seen in training (`test_missions_seen_in_train = true`) | [`research/organizer/results.json`](research/organizer/results.json) `setup` |
| **Organizer by mission** | `GroupKFold(5)` over the 19 missions; totals pooled over all 11,942 events | The stricter check | same |

**Why held-out missions matter.** Missions differ a lot: 2,322 events for mission 1 against 8 for mission 13 ([doc 11 §D](docs/design/11-dataset-experiments.md)), different object types, different covariance habits. Two events from the same satellite, days apart, can look alike. A by-event split can reward a model for recognising "this kind of satellite" instead of "this kind of encounter". By-mission is the number we would quote to a new operator, and the bake-off was decided on it.

The by-mission fold assignment of the organizer evaluation, with the threshold each fold chose (from `recompute_table.py`):

| Fold | Held-out missions | Threshold chosen inside the training missions |
|---|---|---|
| 0 | 4, 5, 13 | 0.0787 |
| 1 | 1, 9, 14, 18, 20 | 0.1035 |
| 2 | 3, 10, 15, 16, 23 | 0.0502 |
| 3 | 7, 19, 22, 24 | 0.0947 |
| 4 | 2, 6 | 0.0907 |

Source: [`research/organizer/recompute_table.py`](research/organizer/recompute_table.py) on [`oof_organizer.parquet`](registry/forecaster_organizer/1.0.0/oof_organizer.parquet) (`fold_mission`, `thr_mission`).

### 4.2 Nested choice of every threshold

Every number that is "tuned" is tuned **inside the training part** of each outer split. The rule is in [`organizer.py`](research/organizer/organizer.py) `pick_thr` and `run_lgbm`:

1. Fit the model on the outer training part and predict the outer test part.
2. Inside the outer training part, run inner folds (`GroupKFold(5)` by mission for the mission split; `KFold(5, random_state=100+seed)` for 80/20) to get inner out-of-fold scores.
3. On those inner scores, pick the threshold `t` that minimises `10·FN + FP` subject to recall ≥ 0.90.
4. Apply that `t` to the outer test part. Report.

So the held-out-mission numbers use each fold's own threshold (0.050–0.103 above), never one chosen on the outer labels. If you instead apply the final model's global 0.0634 to all OOF scores, you get 796 caught, 511 false alarms, cost 74.6 per 1,000: also below the baseline's 104.3, but not what we report ([`recompute_table.py`](research/organizer/recompute_table.py) output).

### 4.3 Metrics

| Metric | Definition | Why |
|---|---|---|
| **Critical recall** | Criticals alerted at T−2 d / all criticals | What the analyst fears missing |
| **False alarms (FA)** | Alerts on events that do not end CRITICAL | What the analyst pays for |
| **Expected cost per 1,000 events** | (10 × missed + FA) / n × 1,000 | The one number that trades the two at R = 10 |
| **AUC (critical)** | Rank quality of P(critical) | Threshold-free |
| **Macro-F1 (3 tiers)** | Argmax tier, averaged over SAFE, ELEVATED, CRITICAL | The task's classic score; we report it even where we do not win |
| **PRIMARY** (bake-off only) | Critical recall at the baseline's own false-alarm budget (113) | Compares models at one fair operating point, before any cost choice |
| Brier, ECE, AP | Probability quality | Needed because the app shows probabilities |

### 4.4 Confidence intervals, and one honest limitation

| Where | Bootstrap | Source |
|---|---|---|
| Bake-off | B = 1,000 resamples of **events**, seed 20261031, the same resamples for every model (paired); the baseline's budget is recomputed inside each resample | [`PREREG.md`](research/bakeoff/PREREG.md) §4 |
| Risk futures, conformal, policy | Event bootstrap for by-event; **mission-cluster** bootstrap for by-mission and LOMO | [`research/bigml/results.json`](research/bigml/results.json) `setup.ci` |
| Organizer headline | B = 500 **event** bootstrap; for 80/20 each seed is resampled and the metric averaged per replicate | [`organizer.py`](research/organizer/organizer.py) `summarise`; `results.json` `setup.bootstrap_B` |
| 1.1.0 model card | Mission bootstrap | [`1.1.0/model_card.json`](registry/forecaster_organizer/1.1.0/model_card.json) `audit.open_items` |

The limitation: the organizer headline's by-mission intervals resample events, not missions. With only 19 missions, a mission-level bootstrap would give **wider** intervals. The 1.1.0 audit already flagged the same issue for the false-alarm study ("Earlier select.py CIs used an event bootstrap for the by-mission split (too narrow)"). Read the by-mission brackets in §0 as optimistic. The point estimates are not affected.

---

## 5. The cost ratio R = 10

Full derivation: [`docs/COST_RATIO.md`](docs/COST_RATIO.md). Here is the short version and how it drives the threshold.

**What the two errors cost.** A false alarm at T−2 d costs analyst review, perhaps a tracking request, sometimes manoeuvre planning, and rarely an unnecessary burn with an instrument outage. A missed critical at T−2 d is **late discovery, not a collision**: the operator's own screening still sees the event inside the last 2 days, so the cost is a rushed response (compressed analysis, about 2× the Δv per CARA App. Q.8, an outage that cannot be scheduled) plus a small chance that the late response cannot be finished.

**The model.** `C_fa = review + tracking + share planned × planning + share executed × (planning + outage)` and `C_miss = rush hours × overtime + share needing a CAM × late extra + p_late × Pc × L`, with inputs from NASA OTPS 2023, ESA alert volumes, the Swarm-B timeline and CARA's 2× late Δv ([COST_RATIO §3](docs/COST_RATIO.md)).

| | Low | Central | High | Monte Carlo (200,000 draws) | Source |
|---|---|---|---|---|---|
| C_fa | €80 | **€1,120** | €9,000 | | [COST_RATIO §3](docs/COST_RATIO.md) |
| C_miss | €830 | **€6,750** | €73,000 | | same |
| R | 10.3 | **6.0** | 8.1 | P10 **2.4** · P50 **5.1** · P90 **10.3** | same |

**Why 10 and not 6.** R = 10 sits at the 90th percentile. The user puts catching criticals first, and R = 10 also covers costs nobody priced (debris externality, duty of care, Pc under-estimation). Higher is not supported: R ≥ 20 appears in under 3 % of draws, and at R ≥ 14 "flag every event" already beats "flag none" on this label, so the threshold would stop discriminating ([COST_RATIO §4](docs/COST_RATIO.md)).

**Why it was fixed first, and what that claim is worth.** `COST_RATIO.md` declares itself written before any sweep result was produced or read: only the label file was opened. That is our attestation, not something git can prove. The file is not hash-pinned, and it was first committed at 14:02 (`1ddfa4e`), after the v1 metrics (12:21, `e9d9449`) and after the organizer results that already use R = 10 (13:16, `2044eb8`) ([REPORT §1](docs/REPORT.md#the-cost-ratio-we-use-r--10-and-how-we-derived-it); [`docs/FACTS.json`](docs/FACTS.json) `cost_ratio`, `git_history`).

**How the threshold follows from R.**
- On a perfectly calibrated P(critical), the cost-optimal rule is "alert when p ≥ 1/(1+R) = 0.091" ([COST_RATIO §4](docs/COST_RATIO.md)).
- We do not assume calibration. We pick the threshold empirically: `argmin 10·FN + FP` subject to recall ≥ 0.90, on out-of-fold scores inside the training data (§4.2). The recall floor is a constraint, not a second cost; the target is 95 %.
- For the final model, trained on all 11,942 events, the inner recall at that threshold was 95.4 % and the threshold is **p_critical ≥ 0.0634** ([`results.json`](research/organizer/results.json) `final.threshold`, `threshold_info`).
- A side observation: the pooled cost-optimal cut on the held-out-mission OOF scores lands at 0.090 ([`recompute_table.py`](research/organizer/recompute_table.py) output), right next to the 0.091 a calibrated model would use. That is a hint, not a proof, that the raw LightGBM probabilities are reasonably calibrated in that range.

**Is the trade robust to R?** Against the baseline on held-out missions we pay 313 extra false alarms (439 − 126) for 69 extra criticals caught (791 − 722). So the model has the lower expected cost whenever 69·R > 313, that is for **any R above about 4.5** (derived; [REPORT §12](docs/REPORT.md#12-impact-before-and-after-for-the-analyst)). That covers the central 6.0 and the median 5.1, but not the P10 of 2.4. An operator at the low tail should re-derive the threshold.

---

## 6. Experiment log

Roughly chronological. All of it happened on 31 October 2026. Stages b–g overlapped on the GPU server; the commit times are from [`docs/FACTS.json`](docs/FACTS.json) `git_history`.

**At a glance:**

| Stage | What | Configs | Population | Verdict |
|---|---|---|---|---|
| a | v1 HGB, fixed defaults | 1 | Kelvins 8,293 | Better ranking, worse alarm at argmax |
| b | Pre-registered bake-off | 8 contenders | Kelvins 8,293 | LightGBM ALL wins via gate (b) |
| c | Risk futures (MDN + Monte Carlo) | 1 model, 3 splits | Kelvins 8,293 | Beats the baseline, ties tuned LightGBM |
| d | Conformal alerts | 5 scores × 2 targets × 2 variants | Kelvins 8,293 | 94.9 % realised at a 95 % target; Mondrian fails |
| e | Optimal decision policy | 1 | Kelvins 8,293 | Loss 0.859 vs 1.039 for the best simple rule |
| f | Optuna LightGBM | 60 trials × 2 feature sets | Kelvins 8,293 | The parameters we still serve |
| g | Nested cost sweep | 403 completed | Kelvins 8,293 | Stopped before final selection |
| h | Organizer-label realignment | 4 LightGBM variants + 2 baseline variants | Organizer 11,942 | The headline numbers |
| i | False-alarm push | 13 candidates | Organizer 11,942 | A tie; 1.0.0 stays served |
| j | Operating curve | — | Organizer 11,942 | The knee is at about 95 % recall |

That is 8 + 120 + 403 + 13 = 544 configurations in the four search stages, which the README rounds to "about 540" ([README](README.md#-the-ml-grind)).

### 6a. v1: HistGradientBoosting with fixed defaults, and the "argmax lesson"

**What.** sklearn `HistGradientBoostingClassifier`, 3 classes, 300 iterations, learning rate 0.05, 31 leaves, min leaf 20, seed 0, on the 84 A13 features, with isotonic calibration of P(critical). Kelvins-eligible 8,293 events ([`registry/tier_forecaster/1.0.0/README.md`](registry/tier_forecaster/1.0.0/README.md)). Committed at 12:21 (`e9d9449`).

**Why these settings.** LightGBM would not load on the development Mac (`libomp.dylib` missing), so the first experiments used sklearn HGB with plain defaults and no tuning ([doc 11](docs/design/11-dataset-experiments.md) header). We wanted a first honest number before any search.

**Result** (out-of-fold; [`registry/tier_forecaster/1.0.0/metrics.json`](registry/tier_forecaster/1.0.0/metrics.json), README table):

| | By event | By mission |
|---|---|---|
| Baseline: crit recall / FA / missed | 0.820 / 113 / **99** | 0.820 / 113 / **99** |
| HGB argmax (raw): crit recall / FA / missed | 0.779 / 80 / 122 | 0.775 / 78 / 124 |
| HGB argmax after isotonic | 0.739 / 59 / 144 | 0.764 / 85 / 130 |
| P(crit) AUC, HGB calibrated vs baseline | **0.978** vs 0.958 | **0.980** vs 0.958 |
| P(crit) Brier, HGB calibrated vs baseline vs climatology | **0.0193** vs 0.0223 vs 0.0620 | **0.0192** vs 0.0223 vs 0.0620 |

**The lesson.** The model ranks clearly better than the baseline (AUC 0.98 vs 0.96) and gives better probabilities, yet **as an alarm at argmax it is worse**: it misses 122 criticals where the baseline misses 99. Calibration makes the probabilities better and the argmax alarm worse still (144 missed). Argmax quietly picks a precision-heavy operating point that nobody chose.

So we stopped asking "which tier is most likely?" and started asking "**at what probability should we alert, given what each error costs?**" Everything after this point compares models at an explicit operating point: first a fixed false-alarm budget (the bake-off), then the cost-optimal threshold.

v1 lives on as the 13-column forecaster for `make precompute` and the CSV predictions ([`server/pipelines/train.py`](server/pipelines/train.py)); at τ = 0.20 it reaches recall 0.853 with 186 false alarms ([`docs/FACTS.json`](docs/FACTS.json) `tier_forecaster_v1.desk_tau_0.20`). It is used only to rank and add WATCH, never to gate.

### 6b. The pre-registered bake-off (8 contenders)

**Why pre-register.** With a dozen model families and a strong baseline, it is easy to fool yourself by picking whichever number looks best. So we wrote the rule down first, in [`research/bakeoff/PREREG.md`](research/bakeoff/PREREG.md), pinned it with sha256 ([`PREREG.sha256`](research/bakeoff/PREREG.sha256)), and made no amendments.

**The rule** ([`PREREG.md`](research/bakeoff/PREREG.md) §4–5):
- **PRIMARY:** critical recall at the baseline's own false-alarm count (113), by held-out mission. The model alerts at the lowest threshold whose alert set has ≤ 113 false alarms.
- **Winner:** the highest by-mission PRIMARY (ties: higher AUC, then lower Brier).
- **Serving gate**, either: (a) the paired ΔPRIMARY CI lower bound is above 0; or (b) PRIMARY ≥ baseline − 0.01 **and** the AUC CI lower bound is above the baseline's AUC. If the gate fails, the product ranks with the baseline and says so; no falling back to the runner-up.
- **Calibration:** ECE after cross-fitted isotonic must be ≤ 0.03, or the model gets a warning in the UI.
- **Budget:** about 75 minutes of wall clock per contender, no open-ended tuning; hyperparameters may only be tuned on inner splits that are themselves mission-grouped on the by-mission split.

**The contenders** ([`docs/BAKEOFF.md`](docs/BAKEOFF.md); per-model notes in [`research/bakeoff/models/`](research/bakeoff/models/)):

| Name | What | Why it was in | Notes |
|---|---|---|---|
| `futures` | Risk-futures MDN + Monte Carlo (§6c) | Models how an event evolves, not just where it is | Delivered before the freeze |
| `tabpfn` | TabPFN v2, A13, `n_estimators=8`, otherwise defaults | A foundation model for small tables; no tuning needed | v2.5+ weights need a licence and were not used ([NOTES](research/bakeoff/models/tabpfn/NOTES.md)) |
| `ebm` | Explainable Boosting Machine, A13, ordinal Frank-Hall (two binary EBMs) | Interpretable shapes at near-GBM accuracy | Hyperparameters fixed a priori ([NOTES](research/bakeoff/models/ebm/NOTES.md)) |
| `gru` | GRU(64) over the ≥ 2 d sequence, 35 engineered channels, 5-seed ensemble | Sequence model; sees order and timing directly | Inner tuning of 6 configs; a 2-layer transformer did no better ([NOTES](research/bakeoff/models/gru/NOTES.md)) |
| `persistence` | The baseline | Reference | |
| `hgb_a13` | The v1 model's recipe | Reference | |
| `lgbm_a13`, `lgbm_all` | LightGBM, lr 0.03, 31 leaves, early stopping on an inner 20 % mission-grouped split | Reference | |

**The leaderboard, by held-out mission** ([`research/bakeoff/LEADERBOARD.md`](research/bakeoff/LEADERBOARD.md), [`leaderboard.json`](research/bakeoff/leaderboard.json)). *These are before cost-threshold optimisation*: every row is read at the baseline's 113-false-alarm budget.

| Model | PRIMARY [95 % CI] | Caught | ΔPRIMARY vs baseline [paired CI] | AUC [95 % CI] | Brier | ECE after iso | Gate |
|---|---|---|---|---|---|---|---|
| **lgbm_all** | **0.842** [0.808, 0.876] | 464/551 | +0.022 [+0.000, +0.041] | 0.985 [0.980, 0.990] | 0.018 | 0.005 | PASS (b) |
| tabpfn | 0.838 [0.807, 0.874] | 462/551 | +0.018 [+0.002, +0.042] | 0.984 [0.978, 0.989] | 0.018 | 0.004 | PASS (a) |
| gru | 0.831 [0.796, 0.865] | 458/551 | +0.011 [−0.012, +0.032] | 0.983 [0.977, 0.988] | 0.019 | 0.003 | PASS (b) |
| lgbm_a13 | 0.831 [0.795, 0.863] | 458/551 | +0.011 [−0.017, +0.030] | 0.983 [0.977, 0.988] | 0.018 | 0.004 | PASS (b) |
| futures | 0.831 [0.798, 0.861] | 458/551 | +0.011 [−0.007, +0.026] | 0.982 [0.976, 0.988] | 0.019 | 0.004 | PASS (b) |
| ebm | 0.831 [0.795, 0.868] | 458/551 | +0.011 [−0.019, +0.038] | 0.980 [0.973, 0.986] | 0.019 | 0.004 | PASS (b) |
| hgb_a13 | 0.824 [0.786, 0.856] | 454/551 | +0.004 [−0.027, +0.025] | 0.983 [0.977, 0.988] | 0.022 | 0.004 | PASS (b) |
| baseline | 0.820 [0.789, 0.852] | 452/551 | — | 0.958 [0.947, 0.967] | 0.022 | 0.002 | reference |

By event (sanity check only): tabpfn 0.833, futures 0.831, gru and ebm 0.824, lgbm_a13 0.822, **lgbm_all 0.820**, baseline 0.820, hgb_a13 0.813 ([`LEADERBOARD.md`](research/bakeoff/LEADERBOARD.md)).

**Verdict: serve lgbm_all** through gate (b). Its paired gain CI touches zero, [+0.000, +0.041], with P(Δ ≤ 0) = 0.037. Its AUC lower bound (0.980) is well above the baseline's 0.958, and its ECE after isotonic is 0.005, so no calibration warning ([`docs/BAKEOFF.md`](docs/BAKEOFF.md)).

**How we read it honestly:**
- Every ML model clearly beats the baseline on **ranking**: AUC 0.98 against 0.96, AP 0.87–0.89 against 0.70.
- At the baseline's own false-alarm budget the gain is **small**: 12 more criticals out of 551.
- lgbm_all's by-mission lead disappears by event, where it ties the baseline at 0.820.
- tabpfn is 2 events behind and is the **only** contender whose gain is significant (gate a).
- An audit re-ran lgbm_all on other seeds and got PRIMARY 0.828–0.842. Seed 0, the pre-registered seed, is the top of that range ([`research/bakeoff/audit/leaderboard_audit.md`](research/bakeoff/audit/leaderboard_audit.md)).
- A diagnostic TabPFN on ALL features got 0.844, but it was not the declared entry, and picking it would have been selection on the outer folds, which the pre-registration forbids ([tabpfn NOTES](research/bakeoff/models/tabpfn/NOTES.md)).

The rule picked lgbm_all mechanically. The real-world claim is narrower: "at least as good as the baseline on recall at equal false alarms, much better ranking, calibrated probabilities." The big win only appears once the threshold is set by cost (§6h), because then the model can **buy recall above the baseline cheaply** (§6j).

**The bake-off audit.** It found one real blocker: the winner had no final model and no declared final-fit recipe. The fix declared the recipe as exactly the procedure the OOF evaluated (train on 80 % of missions, early-stop on the other 20 %, best iteration 161, no refit). A refit-on-all variant agrees with the served bake-off model at r = 0.995 on test p_critical, and the registry's `predict.py` reproduces the exported predictions exactly. No OOF file changed and the verdict stood ([`docs/BAKEOFF.md`](docs/BAKEOFF.md) "Audit outcome"; [`registry/forecaster_bakeoff/model_card.json`](registry/forecaster_bakeoff/model_card.json) `final_fit`, `final_fit_refit_diagnostic`).

**The timing caveat.** The hash proves `PREREG.md` has not changed since it was committed. It does **not** prove it was written before scoring, because it was first committed together with the results (`1ddfa4e`, 14:02). The only ordering evidence is local file times: `PREREG.md` 12:05:43, `PREREG.sha256` 12:05:44, then `results/gru.json` 12:18, `tabpfn.json` 12:22, `ebm.json` 12:26, `logs/score_all.log` 12:34 and `leaderboard.json` 12:37. The one earlier log, `logs/refs.log` (12:04), holds only early-stopping iteration counts, no scores ([`docs/FACTS.json`](docs/FACTS.json) `bakeoff.prereg_timing`). The order is our attestation.

### 6c. Risk futures: a mixture-density network and 10,000 Monte Carlo paths per event

**What.** Instead of classifying the event directly, learn **how one CDM turns into the next**, then simulate. A mixture-density network (10 Gaussian components, a 3 × 256 MLP) takes 31 state features of a CDM and predicts the distribution of the next one: the time gap, miss distance, risk (with its −30 floor), sigmas, and whether the sequence ends. From the latest CDM at ≥ 2 d, we roll **10,000 futures per event** forward to TCA on the GPU (4,000 for the by-event and LOMO runs). P(final tier) is the share of futures that end in each tier ([`research/bigml/REPORT.md`](research/bigml/REPORT.md) §A; [`results.json`](research/bigml/results.json) `methods.futures_mc`; code [`mdn.py`](research/bigml/mdn.py), [`run_futures.py`](research/bigml/run_futures.py)).

**Why.** A classifier gives one number. The analyst also wants to know *how* the event can still evolve: "could the miss still drop below 1 km?" A generative model answers that, and the same simulator can drive a decision policy (§6e).

**Result, by held-out mission, at the baseline's recall of 0.820** ([`research/bigml/results.json`](research/bigml/results.json) `splits.mission.*.op_points.at_persistence_recall`):

| Method | AUC | Brier | False alarms at recall 0.820 |
|---|---|---|---|
| Baseline | 0.958 | 0.0223 | 113 |
| Risk futures (10,000 paths) | 0.982 | 0.0187 | 100 |
| LightGBM A13, Optuna-tuned (10 seeds) | 0.984 | 0.0177 | 94 |
| LightGBM ALL, Optuna-tuned (10 seeds) | 0.986 | 0.0174 | 94 |
| Average of calibrated futures + LightGBM A13 | — | 0.0176 | **91** |

The same pattern holds on leave-one-mission-out: AUC 0.982 (futures) and 0.986 (LightGBM ALL) against 0.955 for the baseline (`splits.lomo`). The pre-registered "does not generalise" trigger was not met for any learned model.

**Verdict.** Futures **beat the baseline** and **tie tuned LightGBM**: the confidence intervals overlap. Raw Monte Carlo P(critical) is slightly over-confident (ECE 0.008). What futures give that a classifier cannot:
- **Fan charts** for every event: [`registry/futures/fans/`](registry/futures/fans/) holds 15,321 files, one per train event (13,154, each from a model that never saw that event's mission) and per test event (2,167). The inspector draws them.
- The **full tier distribution**, not just P(critical).
- The **simulator** behind the decision policy.
- One case where it sees what LightGBM does not: `train:2563` (§9).

Fans are on the dataset's `risk` scale (floor −30), not our physics Pc.

### 6d. Conformal alerts: a recall target you can state in advance

**What.** Split conformal prediction on the futures P(critical), calibrated **only on other missions'** out-of-fold scores. Pick the score cut-off so that a target share of criticals in the calibration missions are caught, then apply it to the held-out mission ([`research/bigml/REPORT.md`](research/bigml/REPORT.md) §B).

**Why.** A threshold chosen by cost says nothing in advance about recall. A conformal threshold comes with a stated recall target, which is how an operator would write a requirement ("catch 95 % of criticals").

| Score, target (held-out missions) | Realised recall [95 % CI] | False alarms | Alerts | Source |
|---|---|---|---|---|
| **Futures, 95 %** | **0.949** [0.925, 0.967] | 613 | 1,136 | [`results.json`](research/bigml/results.json) `conformal.mission['futures_mc\|target_recall_0.95\|plain']` |
| Futures, 90 % | 0.907 [0.868, 0.939] | 257 | 757 | same, `0.90` |
| LightGBM A13, 95 % | 0.955 [0.917, 0.977] | 552 | 1,078 | same, `lgbm_A13` |
| Futures + LightGBM A13 average, 95 % | 0.951 [0.914, 0.974] | 541 | 1,065 | same, `ens_futures_lgbmA13` |
| Baseline score as the conformal score, 95 % | 0.926 [0.867, 0.986] | 1,323 | 1,833 | same, `persistence` |

The baseline row is not the baseline alert used elsewhere. Its score has only three levels (one per latest tier), and the 95 % cut-off falls below the ELEVATED level, so it alerts on many ELEVATED events too.

**What it does not promise.** The guarantee is **pooled, not per mission**. At the 95 % target, missions 1, 2, 3, 4, 7, 9 and 15 fall below target for the futures score, and mission 15 falls below target for every score (`missions_below_target`). The median futures cut-off is τ95 = 0.067.

**A negative result we keep.** A Mondrian variant (a separate cut-off per latest tier) should give per-tier coverage. It flagged about 7,000 events out of 8,293 (7,037 for futures at 95 %; 7,671 for the baseline score) and is useless as an alert. We report it rather than hide it ([`results.json`](research/bigml/results.json) `mondrian_latest_tier` rows).

### 6e. An optimal decision policy by backward induction

**What.** Value iteration over 588 discretised CDM states. At each state the operator can monitor, request tracking, or prepare a manoeuvre; transitions come from real CDM sequences plus MDN-simulated ones. The loss table extends design §4.4: prepare 5 if the event ends critical and 8 if not, a +10 surcharge for late preparation, tracking 1, a missed critical 100, manoeuvre commitment point 1 d ([`registry/policy/metrics.json`](registry/policy/metrics.json); [`research/bigml/policy.py`](research/bigml/policy.py)).

**Why.** The forecaster says *how likely*; the analyst needs *what to do now*, and the right action depends on time left. Backward induction is the textbook way to get a policy that is optimal for a stated loss.

| Policy (8,293 held-out events) | Mean loss per event [95 % CI] | Missed criticals | Tracking requests | Source |
|---|---|---|---|---|
| **Optimal, real + simulated transitions** | **0.859** [0.811, 0.913] | 0 | 0 | [`results.json`](research/bigml/results.json) `policy.policies.dp_real_sim` |
| Optimal, real transitions only | 0.993 [0.929, 1.065] | 3 | 9 | `dp_real` |
| Prepare if critical | 1.039 [0.988, 1.096] | 0 | 0 | `prepare_if_critical` |
| Brief rule: track at every ELEVATED CDM | 3.896 [3.766, 4.032] | 0 | 24,165 | `brief_rule` |
| Never act | 6.644 | — | — | `never` |
| Perfect foresight (lower bound) | 0.332 | — | — | `oracle` |

**What we learned.**
- Adding MDN-simulated transitions lowered the loss from 0.993 to 0.859 and removed the 3 missed criticals: the simulator fills sparse states.
- Under these costs, **requesting tracking is almost never optimal**. The brief's "track every ELEVATED" rule makes 24,165 requests and has 4.5× the loss.
- The loss table is ours, not an operator's, so this is a demonstration of method, not a recommendation. The served app uses a simpler time-aware expected-cost rule (§6j, [REPORT §6.6](docs/REPORT.md#66-fewer-hand-offs-to-the-analyst-live-typesafe-jev-ops-day-scenario)).

### 6f. Optuna LightGBM: 60 trials × 2 feature sets

**What.** Optuna TPE (seed 0), **60 trials** each for A13 and ALL, minimising multiclass log-loss on a **separate by-event split** (seed 99, 4 folds). The search space: learning rate 0.01–0.2 (log), leaves 7–127, min data in leaf 5–200, feature fraction 0.3–1, bagging fraction 0.5–1, L2 1e-3–30, trees 100–1,200. Each tuned model was then evaluated as the mean of 10 seeds ([`research/bigml/run_gbm.py`](research/bigml/run_gbm.py); [`results.json`](research/bigml/results.json) `methods.lgbm_A13`, `best_lgbm_params`).

**Why a separate split.** Tuning on the folds you report is the most common way to inflate a result. The seed-99 split is not any of the reported fold assignments.

| Feature set | Best log-loss | learning rate | leaves | min leaf | feature frac. | bagging frac. | L2 | trees | Source |
|---|---|---|---|---|---|---|---|---|---|
| A13 | 0.1899 | 0.0102 | 19 | 82 | 0.390 | 0.864 | 2.918 | 436 | [`research/bigml/out_gbm.txt`](research/bigml/out_gbm.txt) |
| **ALL** | **0.1814** | **0.0223** | **46** | **113** | **0.366** | **0.997** | **0.051** | **175** | same; these are the served parameters |

Seed stability over 10 seeds (by event): LightGBM ALL macro-F1 0.882 ± 0.002, AUC 0.985 ± 0.0004 ([`results.json`](research/bigml/results.json) `splits.event.lgbm_ALL.seeds10`).

**Caveat.** The tuning split is different, but the **events** are the same 8,293 Kelvins events that later appear in the evaluation folds, and they are inside the organizer population too. Hyperparameters tuned on the same events can be mildly optimistic. With only 8 hyperparameters and a log-loss objective (not the reported recall or cost), we expect the effect to be small, but we have not measured it.

### 6g. The big nested cost sweep (403 configurations), and why we stopped it

**What.** A sweep over every model family we could think of, scored the way the product is scored: expected cost at R = 10 with a 90 % recall floor, with the threshold chosen on inner folds. Nested by-mission cross-validation on the Kelvins 8,293 events: **5 outer × 4 inner folds** (team-reported; the sweep code lived on the server under `research/sweep/` and is not in this repository).

**Configurations that completed on the GPU server** (team-reported; the total 403 is also stated in the [README](README.md#-the-ml-grind)):

| Family | Completed | Examples |
|---|---|---|
| CPU boosters | 223 | LightGBM, XGBoost, sklearn HGB; tier and binary-critical objectives, class weights 3–100 |
| GPU boosters | 17 | CatBoost (GPU), XGBoost (CUDA) |
| Neural | 81 | GRU, LSTM, transformer encoder, TCN, MLP over the sequence and/or A13 |
| Tabular, other | 82 | TabPFN v2, EBM, random forest, extra trees, logistic regression, spline GAM |
| **Total** | **403** | |

**What the inner screen showed.** The repository holds a ranking snapshot taken while the sweep was still running: [`research/fa/ens/config_rank.csv`](research/fa/ens/config_rank.csv), written by [`rank_configs.py`](research/fa/ens/rank_configs.py). It covers the 315 configurations whose inner out-of-fold files were readable at that moment (164 CPU boosters, 17 GPU boosters, 61 neural, 72 tabular-other, 1 reference) and 17 model classes. Its `inner_cost` is the mean over the 5 outer folds of the inner-OOF cost per 1,000 events at R = 10, recall ≥ 0.90 (lower is better):

| Model class (snapshot) | Configs | Best inner cost | Median inner cost |
|---|---|---|---|
| LightGBM | 98 | **85.4** | 91.1 |
| XGBoost (CPU) | 34 | 85.8 | 90.8 |
| Random forest | 8 | 85.8 | 88.0 |
| sklearn HGB | 32 | 86.3 | 94.4 |
| XGBoost (CUDA) | 11 | 86.6 | 91.6 |
| TabPFN v2 | 10 | 87.2 | 89.8 |
| Extra trees | 4 | 89.1 | 95.8 |
| Logistic regression | 19 | 90.5 | 124.5 |
| CatBoost (GPU) | 6 | 90.7 | 99.8 |
| EBM | 6 | 91.9 | 94.0 |
| Spline GAM | 25 | 96.0 | 140.2 |
| GRU | 15 | 96.1 | 102.2 |
| Transformer | 17 | 99.1 | 106.6 |
| TCN | 8 | 99.1 | 105.2 |
| LSTM | 10 | 100.3 | 108.1 |
| MLP | 11 | 126.5 | 137.6 |
| Baseline (`TierPersistence`) | 1 | 206.4 | — |

Derived from [`config_rank.csv`](research/fa/ens/config_rank.csv) (group by `model`, min and median of `inner_cost`).

How we read it:
- **Gradient boosting on hand-built features wins**, and the best configuration is a LightGBM on ALL (`lgbm-ALL-crit-w30-0044`, 85.4). The tuned family we already had was not beaten.
- **Random forest, TabPFN and extra trees are roughly level** with the boosters (85.8–89.1 best).
- **Logistic regression and the GAM are clearly worse** on median (124.5 and 140.2): the signal is non-linear (thresholds on miss distance and its trend).
- **The sequence models trail** the tabular ones by about 10 cost points. With a median of 13 CDMs per event ([doc 11 §D](docs/design/11-dataset-experiments.md)), hand-built trends already capture what the sequence has to say.
- The baseline scores 206.4 here because the 90 % recall floor forces it to alert on ELEVATED events as well, its only way to reach that recall. The same thing shows up on the organizer data as `persistence_costcut` (§6h).

**Why we stopped it.** The sweep was built for the Kelvins 8,293 population. When the organizer published its official label file, the target population changed to 11,942 events (§6h), and a final outer-fold selection on the old population would no longer have answered the question. We stopped the sweep before its final selection. The serving slot it was meant to fill (`forecaster_cost`, still listed as a fallback in [`server/app/engine_adapter.py`](server/app/engine_adapter.py)) was never created: there is no `registry/forecaster_cost/`. The sweep was not wasted. Its best members fed the false-alarm push as ensemble members (`ens_member_rf_all_crit_bsub`, `ens_member_hgb_a13_tier_w100`; §6i), and it told us that no family was hiding a large win.

### 6h. Realigning with the organizer's labels (11,942 events)

**What.** When the organizer published `train_event_labels.csv`, we re-ran the evaluation on its population ([`research/organizer/organizer.py`](research/organizer/organizer.py); commit `2044eb8`, 13:16):
- Population: the 11,942 usable events (834 critical). Target: the tier of each event's final message, at any `ttca`.
- Inputs: the same harness recipe, messages with `ttca ≥ 2 d` only.
- Models: LightGBM with the Optuna parameters, on A13 and ALL, trained on the organizer population and on the Kelvins subset only (4 variants), plus the baseline and `persistence_costcut` (the baseline with a cost-chosen cut that may also alert on ELEVATED).
- Splits: 80/20 by event (5 seeds) and `GroupKFold(5)` by mission; nested thresholds (§4.2); B = 500 bootstrap.

**Why.** Judges score on the organizer's labels and population, so that is where our claims must hold. The organizer population is also 44 % larger (11,942 vs 8,293), with the 1–2 d and ≥ 2 d slices the Kelvins set excludes.

**All rows** ([`research/organizer/results.json`](research/organizer/results.json) `summary.headline`, `metrics`):

| Model | Split | Crit recall [CI] | FA | Cost / 1,000 [CI] | Macro-F1 | AUC |
|---|---|---|---|---|---|---|
| Baseline | 80/20 | 0.851 [0.830, 0.878] | 24.8 | 110.8 [93.3, 127.1] | 0.895 | 0.970 |
| **LightGBM ALL, organizer-trained (served)** | 80/20 | **0.937** [0.919, 0.955] | 89.4 | **80.1** [67.9, 92.5] | 0.895 | **0.988** |
| LightGBM A13, organizer-trained | 80/20 | 0.934 [0.917, 0.951] | 78.4 | 77.2 [65.3, 89.1] | 0.888 | 0.988 |
| LightGBM ALL, Kelvins-trained | 80/20 | 0.933 [0.917, 0.951] | 87.8 | 82.0 [69.8, 94.4] | 0.874 | 0.987 |
| `persistence_costcut` | 80/20 | 0.981 | 447.8 | 200.0 | — | — |
| Baseline | mission | 0.866 [0.841, 0.890] | 126 | 104.3 [86.2, 123.8] | **0.898** | 0.969 |
| **LightGBM ALL, organizer-trained (served)** | mission | **0.948** [0.932, 0.964] | 439 | **72.8** [61.5, 85.3] | 0.893 | **0.988** |
| LightGBM A13, organizer-trained | mission | 0.946 [0.931, 0.962] | 435 | 74.1 [62.4, 86.3] | 0.888 | 0.987 |
| LightGBM ALL, Kelvins-trained | mission | 0.935 [0.917, 0.952] | 405 | 79.1 [67.6, 92.6] | 0.873 | 0.987 |
| `persistence_costcut` | mission | 0.984 | 2,276 | 201.5 | — | — |

**What this shows:**
- **The win holds on both splits:** more criticals caught, lower expected cost, better AUC.
- **Macro-F1 does not improve:** a tie on 80/20 and the baseline slightly ahead by mission. Per-tier recall at argmax (80/20) is 0.977 / 0.866 / 0.839 for the model against 0.971 / 0.884 / 0.851 for the baseline (`summary.headline`): the argmax lesson again. The recall gain comes entirely from the cost-set threshold.
- **The naive way to buy recall is far worse.** Lowering the baseline to alert on ELEVATED reaches 98 % recall, but at 2.5× (80/20) to 2.8× (mission) our cost.
- **Training on 11,942 beats training on 8,293** on every point estimate: by mission, cost 72.8 vs 79.1, recall 0.948 vs 0.935, macro-F1 0.893 vs 0.873. The cost intervals overlap, so the honest word is "better on every point estimate", not "significantly better". It also tells us the win does not depend on which population we trained on.

**By sub-population, held-out missions** ([`results.json`](research/organizer/results.json) `metrics.mission.*.{final_lt1d,final_1to2d,final_ge2d}`):

| Slice (events / criticals) | Baseline recall · FA · cost | Served model recall · FA · cost | Kelvins-trained recall · FA · cost |
|---|---|---|---|
| final < 1 d (8,293 / 551) | 0.820 · 113 · 133.0 | **0.931 · 335 · 86.2** | 0.913 · 300 · 94.1 |
| final 1–2 d (599 / 39) | 0.667 · 13 · 238.7 | **0.897 · 23 · 105.2** | 0.897 · 20 · 100.2 |
| final ≥ 2 d (3,050 / 244) | **1.000 · 0 · 0.0** (exact by construction) | 0.996 · 81 · 29.8 | — |

On 80/20 the pattern is the same: 0.917 vs 0.804 on < 1 d and 0.868 vs 0.698 on 1–2 d ([`docs/FACTS.json`](docs/FACTS.json) `subpopulation_breakdown`). **The forecasting gain is where forecasting is needed.** On the ≥ 2 d slice the baseline is exact and we lose 81 false alarms and 1 miss; we say so instead of letting it hide in the total.

### 6i. The false-alarm push: 13 candidates, and an honest tie

**What.** We tried to cut false alarms without losing recall, with 13 new candidates scored on the organizer population ([`research/fa/select/results.json`](research/fa/select/results.json); code [`select.py`](research/fa/select/select.py), [`physics/phys.py`](research/fa/physics/phys.py), [`ens/run.py`](research/fa/ens/run.py), [`thresholds/thr.py`](research/fa/thresholds/thr.py), [`ref/ref.py`](research/fa/ref/ref.py)):
- **Physics features:** 49 features from our own Pc pipeline (`ph_*`: last, delta, min, max, mean and slope of log10 Pc, Pc_max, Mahalanobis distance, principal sigmas, physics tier, escalation flags), added to ALL (`lgbm_all_phys`, 282 features). Also with futures features on top (`lgbm_all_phys_fut`).
- **Ensembles:** mean and rank averages of sweep members, and the members alone (random forest, HGB).
- **Bagging:** a seed-bagged LightGBM ALL.
- **Smarter thresholds:** a Mondrian threshold per latest tier, the same with a cost objective, and a two-signal rule, each on the current model.
- **Three baseline variants** with the same threshold tricks.

**The rule, fixed before scoring** (`rule` in [`results.json`](research/fa/select/results.json)): the lowest by-mission cost at R = 10 with recall ≥ 0.90. If the paired cost-difference CI includes 0, the candidate is preferred only if it also has fewer false alarms and recall within 0.01 of the current model; otherwise keep the current one.

**The table** (15 rows = the current model, the plain baseline with a cost-chosen cut, and the 13 candidates; [`results.json`](research/fa/select/results.json) `table_markdown`):

| Candidate | 80/20 recall | 80/20 FA | 80/20 cost | ΔFA vs current [CI] | ΔCost vs current [CI] | Mission recall | Mission FA | Mission cost | Mission ΔCost [CI] | FA at matched recall (80/20 · mission) |
|---|---|---|---|---|---|---|---|---|---|---|
| **lgbm_all_phys** (rule winner) | 0.935 | 80.6 | 77.3 | −8.8 [−12.8, −5.1] | −2.7 [−8.4, +3.2] | 0.951 | 438 | 71.0 | −1.8 [−7.0, +3.0] | 82.6 · 442 |
| current + two-signal | 0.937 | 90.4 | 80.5 | +1.0 [+0.3, +1.8] | +0.4 [+0.1, +0.8] | 0.948 | 438 | 72.7 | −0.1 [−0.3, +0.0] | — |
| **current lgbm_all (1.0.0)** | 0.937 | 89.4 | 80.1 | 0 | 0 | 0.948 | 439 | 72.8 | 0 | 81.0 · 423 |
| ens_mean | 0.934 | 91.4 | 82.6 | +2.0 [−1.4, +5.9] | +2.8 [−2.9, +9.0] | 0.944 | 407 | 73.4 | +0.5 [−4.1, +5.2] | 83.4 · 426 |
| current + Mondrian tier | 0.937 | 91.4 | 81.0 | +2.0 [+0.6, +3.2] | +0.8 [+0.3, +1.3] | 0.948 | 448 | 73.5 | +0.7 [+0.3, +1.3] | — |
| RF member (ALL, crit, balanced subsample) | 0.928 | 77.0 | 80.8 | −12.3 [−16.2, −8.3] | +0.8 [−5.1, +6.8] | 0.941 | 388 | 73.5 | +0.6 [−4.2, +5.6] | 87.8 · 420 |
| lgbm_all_phys_fut | 0.939 | 95.8 | 81.1 | +6.5 [+2.4, +10.7] | +1.2 [−4.4, +6.9] | 0.942 | 401 | 73.8 | +0.9 [−5.7, +6.7] | 93.6 · 415 |
| ens_rank | 0.932 | 87.6 | 82.7 | −1.7 [−5.0, +2.0] | +2.9 [−2.2, +8.4] | 0.942 | 416 | 75.0 | +2.1 [−2.6, +8.0] | 82.4 · 412 |
| HGB member (A13, tier, w100) | 0.926 | 81.0 | 84.1 | −8.3 [−13.2, −2.9] | +4.2 [−3.1, +11.3] | 0.942 | 446 | 77.5 | +4.5 [−2.3, +10.5] | 94.6 · 445 |
| lgbm_all_bag | 0.934 | 87.0 | 80.8 | −2.3 [−5.5, +0.8] | +0.8 [−3.9, +5.7] | 0.939 | 421 | 78.0 | +5.1 [+0.3, +10.2] | 81.2 · 420 |
| current + Mondrian (cost objective) | 0.931 | 86.2 | 82.1 | −3.2 [−7.2, +0.6] | +2.3 [−3.6, +8.8] | 0.938 | 457 | 81.8 | +9.1 [+1.8, +16.1] | — |
| baseline + Mondrian (cost objective) | 0.907 | 187.2 | 141.1 | +97.8 | +60.8 | 0.892 | 413 | 109.9 | +37.0 | — |
| baseline, cost-chosen cut | 0.889 | 197.4 | 158.8 | +107.7 | +78.2 | 0.902 | 586 | 117.7 | +44.9 | — |
| baseline + Mondrian tier | 0.889 | 197.4 | 158.8 | +107.7 | +78.2 | 0.902 | 586 | 117.7 | +44.9 | — |
| baseline + two-signal | 0.889 | 197.4 | 158.8 | +107.7 | +78.2 | 0.902 | 586 | 117.7 | +44.9 | — |

**The verdict: a tie.** The rule picked `lgbm_all_phys` because its cost CI includes 0 but it has fewer false alarms with recall within 0.01 (`winner_reason`). Then the audit looked harder ([`registry/forecaster_organizer/1.1.0/model_card.json`](registry/forecaster_organizer/1.1.0/model_card.json) `audit`):
- **No leakage found.**
- **Not statistically significant.** By mission, ΔCost −1.8 [−7.0, +3.0] and ΔFA −1 [−19, +16]. On 80/20 it cuts false alarms (−8.8 [−12.8, −5.1]) but the cost CI includes 0.
- **Ranking is not better.** At the current model's recall, it needs about the same false alarms (82.6 vs 81.0 on 80/20; 442 vs 423 by mission). The 80/20 FA reduction mostly comes from where the inner-chosen cut lands, and it flips with the seed (seed 0: 71 vs 113 false alarms; seed 3: 92 vs 79).
- **Selection bias.** It was picked out of 15 on the same by-mission data it is reported on, so the gain is optimistically biased.
- **An extra dependency.** It needs the physics Pc pipeline (`pc_rows.parquet`) for every event at inference time, and that file's provenance was not fully confirmed in the audit.

**Decision.** **1.0.0 stays served** ([`registry/forecaster_organizer/CURRENT`](registry/forecaster_organizer/CURRENT) = `1.0.0`). 1.1.0 is kept in the registry as an on-par alternative, with its threshold (0.0610) and test predictions; the two agree on 99.2 % of test alerts (286 vs 287 alerts; `test.alert_agreement_with_1.0.0`).

**The lesson.** Extra compute did not buy fewer false alarms. Every candidate lands near the same curve; what moves false alarms is **where on that curve we stand**, which is the cost and recall choice of §5 and §6j, not the model.

### 6j. The operating curve, and why about 95 % recall

**What.** Sort the 11,942 held-out-mission OOF scores, and read off the fewest false alarms needed to catch k criticals ([`docs/FACTS.json`](docs/FACTS.json) `operating_curve`, recomputed from [`oof_organizer.parquet`](registry/forecaster_organizer/1.0.0/oof_organizer.parquet); also [REPORT §5.7](docs/REPORT.md#57-the-operating-curve-and-why-we-chose-about-95--recall)):

| Recall | Caught (of 834) | False alarms | Extra FA per extra critical (vs previous row) | Cost per 1,000 (derived) |
|---|---|---|---|---|
| **86.6 % (the baseline's recall)** | 722 | 123 (baseline: 126) | — | 104.1 |
| 90.0 % | 751 | 179 | 1.9 | 84.5 |
| 92.1 % | 768 | 269 | 5.3 | 77.8 |
| **95.1 %** | 793 | 475 | **8.2** (< R = 10, worth it) | 74.1 |
| 97.0 % | 809 | 862 | **24.2** (> R = 10, not worth it) | 93.1 |
| *served, per-fold thresholds* | *791 (94.8 %)* | *439* | | *72.8* |

The last two columns are derived: ΔFA / Δcaught between rows, and (10 × (834 − caught) + FA) / 11,942 × 1,000.

**How to read it.**
- **At the baseline's own recall the model is no better:** 123 against 126 false alarms. That is the bake-off result again (§6b).
- **The gain is that the model can buy recall above the baseline cheaply.** Up to about 95 %, each extra critical costs fewer than 10 extra false alarms, which is worth it at R = 10. Beyond that, the price jumps to 24.
- **So the cost-optimal point sits at the knee, about 95 % recall**, which is also the recall target written in [COST_RATIO §4](docs/COST_RATIO.md).

**Other τ choices on the same OOF scores** ([`recompute_table.py`](research/organizer/recompute_table.py) output):

| Threshold rule | Caught | Missed | Recall | FA | Cost per 1,000 |
|---|---|---|---|---|---|
| Baseline | 722 | 112 | 86.6 % | 126 | 104.3 |
| **Per-fold thresholds (reported)** | **791** | **43** | **94.8 %** | **439** | **72.8** |
| Global 0.0634 (the final model's threshold) | 796 | 38 | 95.4 % | 511 | 74.6 |
| Pooled optimum 0.090 | 786 | 48 | 94.2 % | 397 | 73.4 |
| Watch τ = 0.20 | 758 | 76 | 90.9 % | 214 | 81.6 |

**The app follows the served threshold.** In the running app, WATCH (the flag that turns a SAFE event into "SAFE on watch" and sends it to the decision model) follows the served model's own alert flag, p_critical ≥ 0.0634; the desk τ defaults to it and can override it, and each change is written to the ledger ([REPORT §5.5](docs/REPORT.md#55-aligning-with-the-organizers-labels)). So the system acts at the operating point this page evaluates. An operator who wants fewer alerts can move τ to 0.20: 90.9 % recall and 214 false alarms.

**Downstream: fewer hand-offs.** The forecaster's alerts feed a decision step. Replaying the 18 flagged decisions of the *Ops day* scenario through the live TypeSafe Jev (`jev-1.14.0`) before and after two policy fixes ([REPORT §6.6](docs/REPORT.md#66-fewer-hand-offs-to-the-analyst-live-typesafe-jev-ops-day-scenario)):

| Policy | AUTO | ESCALATE | BLOCK |
|---|---|---|---|
| Before: 75 % bar on TypeSafe's `confidence` field, cost rule not time-aware | 0 | 18 | 0 |
| After: **70 % bar on p(chosen action)**, time-aware cost rule | **12** | **6** | 0 |

The 12 now handled automatically are ELEVATED events with a tiny Pc, 1 to 19 h before TCA, where Jev chose *Keep monitoring* at 72–91 % and the cost rule agrees. The 6 still escalated are the two CRITICAL events (critical always goes to a human) and four genuinely split decisions on `train:2115`. This is one replayed shift, not an offline validation of Jev.

---

## 7. The served model

`registry/forecaster_organizer/1.0.0` ([`model_card.json`](registry/forecaster_organizer/1.0.0/model_card.json); `CURRENT` = `1.0.0`).

| Field | Value |
|---|---|
| Model | LightGBM multiclass (SAFE / ELEVATED / CRITICAL), ALL feature set, 233 features |
| Training population | Organizer usable-for-training, 11,942 events |
| Parameters | learning rate 0.0223, 46 leaves, min data in leaf 113, feature fraction 0.366, bagging fraction 0.997 (freq 1), L2 0.051, 175 trees, seed 0 (Optuna, §6f) |
| Alert rule | **p_critical ≥ 0.0634** |
| Threshold source | By-event 5-fold OOF (`KFold`, shuffle, random_state 7) over all 11,942 training events; argmin `10·FN + FP` s.t. recall ≥ 0.90. Inner recall 95.4 %, inner cost 71.7 per 1,000 ([`results.json`](research/organizer/results.json) `final.threshold_info`) |
| Per-fold thresholds (for training events) | 0.0787 / 0.1035 / 0.0502 / 0.0947 / 0.0907 (§4.1) |
| Cost ratio | R = 10 |
| Calibration | No post-hoc calibrator on this model; the threshold is chosen on raw scores. Its bake-off sibling had ECE 0.005 by mission ([`docs/BAKEOFF.md`](docs/BAKEOFF.md)), and the pooled cost-optimal cut (0.090) sits next to the calibrated 1/(1+R) = 0.091 (§5) |
| Test set (2,167 events, labels hidden) | 287 CRITICAL alerts; argmax tiers 1,512 / 480 / 175 (the baseline's latest tiers: 1,481 / 500 / 186) ([`results.json`](research/organizer/results.json) `final`). **No accuracy claim** |
| Code | [`research/organizer/organizer.py`](research/organizer/organizer.py) `final` |

**Where its predictions live:**

| File | Rows | What |
|---|---|---|
| [`oof_organizer.parquet`](registry/forecaster_organizer/1.0.0/oof_organizer.parquet) | 11,942 | By-mission outer OOF (`p_*_mission`, `fold_mission`, `thr_mission`, `alert_mission`) and the by-event 5-fold OOF used for the threshold (`p_*_event5`) |
| [`predictions.parquet`](registry/forecaster_organizer/1.0.0/predictions.parquet) | 14,109 | What the backend reads: 11,942 training events (held-out-mission OOF with their fold threshold) + 2,167 test events (final model, 0.0634) |
| [`predictions_test.parquet`](registry/forecaster_organizer/1.0.0/predictions_test.parquet) | 2,167 | Test predictions with `alert_critical` and `persistence_tier` |
| [`model.txt`](registry/forecaster_organizer/1.0.0/model.txt) | — | The LightGBM booster |

**How the backend uses it** ([`server/app/engine_adapter.py`](server/app/engine_adapter.py) forecast sources):
- The forecast for an event is looked up in a fixed order: `forecaster_organizer` first, then the older registries, then v1 out-of-fold, then the baseline's empirical rate.
- **For a training event the app only ever shows a held-out-mission out-of-fold forecast**, with that fold's own threshold. The live model is never run on a `train:*` event. Example: `train:2563` is shown at 1.1 % (OOF, fold threshold 0.050); the in-sample value would have been 0.88 and is never shown ([`docs/FACTS.json`](docs/FACTS.json) `failure_cases.train:2563`; tested in [`server/tests/test_perf_honesty.py`](server/tests/test_perf_honesty.py)).
- The forecast **can only add attention**: it can set WATCH or raise the queue rank, never lower a rule tier or drop an escalation ([`server/riskengine/forecast.py`](server/riskengine/forecast.py) `attention`).
- Every forecast carries its source label, threshold and R in the `forecast.updated` event, so the analyst can see what produced it ([REPORT §7](docs/REPORT.md#7-trust-and-traceability)).

---

## 8. Leakage and honesty audits

All in [`research/bakeoff/audit/`](research/bakeoff/audit/), run on the Kelvins population and pipeline that the organizer evaluation reuses.

| Audit | What we did | Result | Source |
|---|---|---|---|
| **Noise substitution** | Replaced all **50,695** CDM rows with `ttca < 2 d` (the future) with noise and rebuilt every feature | A13, ALL and the sequences are **bit-identical**. No future data reaches the inputs | [`a2_out.json`](research/bakeoff/audit/a2_out.json), [`a2_perturb_late_cdms.py`](research/bakeoff/audit/a2_perturb_late_cdms.py) |
| **Shuffled labels, LightGBM** | Retrained with permuted labels | AUC 0.453 / 0.469 / 0.506 by mission and 0.521 / 0.552 / 0.522 by event: chance | [`a3_out.json`](research/bakeoff/audit/a3_out.json) |
| Shuffled labels, per fold | Same, fold by fold | Mean AUC 0.536 (event), 0.513 (mission) | [`a5_out.json`](research/bakeoff/audit/a5_out.json) |
| Shuffled labels, TabPFN | Two shuffle seeds | AUC 0.378 and 0.671, Spearman 0.014 between them. A noisy null; the 0.671 does not track outlier scores, `max_risk_estimate`, miss or `n_cdm` | [`a6_out.json`](research/bakeoff/audit/a6_out.json) |
| Shuffle **within** mission | Labels permuted only inside each mission, which keeps each mission's critical rate | AUC 0.671 on the mission split: how much is "which kind of mission is this". A full shuffle collapses to chance, so there is no hidden label signal in the pipeline | [`a3_out.json`](research/bakeoff/audit/a3_out.json) |
| **Exact reproduction** | Re-ran lgbm_all by mission and by event | Max absolute difference **0.0** (13.8 s and 18.3 s) | [`a3_out.json`](research/bakeoff/audit/a3_out.json) |
| Re-scoring | Re-scored every contender independently | Same PRIMARY for all 16 rows, e.g. lgbm_all 464 caught at 113 FA by mission | [`a3_out.json`](research/bakeoff/audit/a3_out.json) |
| **Data, labels, folds** | Independent rebuild from the raw CSV | 162,634 rows; 8,293 eligible events; **0 label mismatches**; tiers 6,138 / 1,604 / 551; max final ttca 0.997 d; min latest input ttca 2.00001 d; no mission in more than one fold | [`a1_out.json`](research/bakeoff/audit/a1_out.json) |
| Feature recompute | Independent feature build | 11 cell mismatches (10 on sigma-delta features of event 5826, where the recompute gives NaN; 1 on a covariance-determinant slope of event 8154), 1 missing column (`last_c_object_type`). Every `last_*` feature equals the latest ≥ 2 d CDM (fraction 1.0), and `last_miss_distance` equals the final CDM in only 0.18 % of events | [`a1_out.json`](research/bakeoff/audit/a1_out.json) |
| **Seed sensitivity** | lgbm_all with seeds 0–4 and 7 | PRIMARY 0.828–0.842; the pre-registered seed 0 is the top | [`leaderboard_audit.md`](research/bakeoff/audit/leaderboard_audit.md) |
| **Headline recompute** | One script, from the committed OOF file | Reproduces the held-out-mission row exactly | [`recompute_table.py`](research/organizer/recompute_table.py) |
| 1.1.0 audit | Leakage and significance check of the false-alarm winner | No leakage; not significant (§6i) | [`1.1.0/model_card.json`](registry/forecaster_organizer/1.1.0/model_card.json) |

**What we cannot prove, stated once more:**
- **Pre-registration timing.** The hash pins `PREREG.md` from its first commit, which also contains the results. The "written before scoring" order rests on local file times and our word (§6b).
- **The cost ratio's timing.** `COST_RATIO.md` is not hash-pinned and was committed after results that already used R = 10 (§5).
- **Optuna on the same events** (§6f) and **selection on the reported data** in the false-alarm push (§6i) both bias toward optimism; the first is probably small, the second is why we call 1.1.0 a tie.
- **Event bootstrap on the mission split** (§4.4) makes the organizer by-mission CIs too narrow.
- **The test set is unscored.** Every accuracy number is on training events, out-of-fold.

---

## 9. Failure analysis

### 9.1 Late escalations nobody could see

**`train:2563` (mission 15, scenario *The late CDM*).** SAFE at T−2.43 d with an 8.05 km miss; CRITICAL at T−0.44 d with a **136 m** miss (risk −4.19). The served model's held-out-mission forecast at T−2 d is **1.1 %**, below its fold's 0.050 threshold, so there is no early alert. The v1 forecaster gave 0.5 % ([`docs/FACTS.json`](docs/FACTS.json) `failure_cases.train:2563`).

*The partial win:* the risk-futures model gives it P(critical) **0.092**, above the 95 % conformal cut-off of 0.067, so **futures would have flagged it two days early** ([`research/bigml/REPORT.md`](research/bigml/REPORT.md) failure cases). That is the strongest argument we have for a futures + LightGBM combination (§11).

**`train:4973` (mission 2): missed by all 8 bake-off models** ([`docs/BAKEOFF.md`](docs/BAKEOFF.md); [`research/bakeoff/failure_case.json`](research/bakeoff/failure_case.json)):

| time to TCA (d) | 6.99 → 2.11 (16 CDMs, model input) | 1.68 | 1.51 | 1.08 | **0.81** | 0.39 | 0.13 (final) |
|---|---|---|---|---|---|---|---|
| Miss (m) | 1,715 → 1,690, flat | 1,688 | 1,689 | 1,691 | **814** | 815 | 815 |
| Chaser σ_r (m) | 32.7, constant | 32.7 | 32.7 | 32.7 | **53.4** | 53.4 | 53.4 |
| Tier | ELEVATED | ELEVATED | ELEVATED | ELEVATED | **CRITICAL** | CRITICAL | CRITICAL |

For six days the encounter sat quietly at about 1.7 km. At 0.81 d a new orbit determination for the debris moved the predicted miss to 814 m, which is CRITICAL through the miss clause while the risk stays about 1e-14. p_critical was 0.015 (lgbm_all), 0.024 (tabpfn), 0.054 (baseline). Nothing in the ≥ 2 d data predicts a new measurement.

**The ceiling this implies.** 68 of 551 criticals are missed by **all eight** bake-off models at their budget thresholds, so even a perfect combination of those eight would reach a PRIMARY of at most 483/551 ≈ 88 % ([`docs/BAKEOFF.md`](docs/BAKEOFF.md); arithmetic in [`docs/FACTS.json`](docs/FACTS.json) `bakeoff.missed_by_all_eight`).

### 9.2 Pc-driven criticals are too rare to learn

**`train:9361` (mission 18, scenario *Rules disagree*)** is CRITICAL by Pc 1.2e-4 at a 3.93 km miss. The served forecaster gives 1.3 %, futures 0.04. Only **6 of 551** Kelvins-eligible criticals are Pc-driven; 549 are miss-driven ([`docs/FACTS.json`](docs/FACTS.json) `failure_cases.train:9361`; [doc 11 §A](docs/design/11-dataset-experiments.md)). The model barely sees the Pc path. The rules layer catches the event when the Pc crosses 1e-4, regardless of the forecast.

### 9.3 Hard missions

| Mission | What we see | Source |
|---|---|---|
| 15 | Leave-one-mission-out critical recall 0.685 for the baseline and futures, 0.704 for LightGBM. Its by-mission fold chose the lowest threshold of all (0.050). Below the conformal target for every score | [`research/bigml/REPORT.md`](research/bigml/REPORT.md); `recompute_table.py`; [`results.json`](research/bigml/results.json) `conformal` |
| 6 | Critical recall 0.50 for every method (8 criticals) | [`research/bigml/REPORT.md`](research/bigml/REPORT.md) |
| 6, 7 | Hold 28.5 % of the rows where our physics Pc and ESA's `risk` disagree by more than 1 dex, against 10.7 % of rows | [doc 11 §C.3](docs/design/11-dataset-experiments.md) |

Satellite-specific behaviour that other missions do not teach is the main reason by-mission numbers are the ones to quote.

### 9.4 The ≥ 2 d slice, where the baseline is exact

On `final_ge2d` (3,050 events, 244 critical) the latest input **is** the final message, so the baseline scores recall 1.0 with 0 false alarms. Our model scores 0.996 with 81 false alarms and 1 miss by mission (cost 29.8 per 1,000) ([`results.json`](research/organizer/results.json) `metrics.mission.*.final_ge2d`). A product could route these events to the baseline whenever it knows no further message will come, but at prediction time that is unknowable, so we do not add such a feature (§3.3) and accept the loss.

### 9.5 The cost of the win

- About **one alert in three is false**: precision 0.626 (80/20) and 0.643 (mission), against 0.845 and 0.851 for the baseline ([`model_card.json`](registry/forecaster_organizer/1.0.0/model_card.json)).
- **36.8 false alarms per 1,000 events**, against 10.6 ([`model_card.json`](registry/forecaster_organizer/1.0.0/model_card.json) `fa_per_1000`).
- **Macro-F1 is not improved** (§6h).

---

## 10. Compute and engineering notes

### 10.1 The server

**Triologay**, our own open-frame GPU server, reached from a MacBook M2 (8 GB) over Tailscale. Every heavy job ran on it; the laptop ran the app, the tests and the end-to-end runs ([REPORT §5.4](docs/REPORT.md#54-the-big-compute-run-on-our-own-gpu-server-triologay)).

| Component | Value | Source |
|---|---|---|
| CPU | 56 cores | [`registry/forecaster_bakeoff/model_card.json`](registry/forecaster_bakeoff/model_card.json) `runtime.server` ("shared 56-core / 4-GPU server") |
| GPUs | 2× Tesla V100-SXM2 16 GB, Tesla PG503-216 32 GB, GeForce RTX 2060 | [README](README.md#-the-ml-grind); GPU names in [tabpfn NOTES](research/bakeoff/models/tabpfn/NOTES.md) and [`research/livesky/REPORT.md`](research/livesky/REPORT.md) |
| RTX 2060 memory, RAM | 12 GB, 247 GB | *team-reported*; no repo file records them ([`docs/FACTS.json`](docs/FACTS.json) `compute.ram_247GB`) |

### 10.2 The GPU-ordering gotcha

PyTorch's default CUDA device order is `FASTEST_FIRST`, so `cuda:k` is **not** `nvidia-smi` index k on a mixed-GPU box. The TabPFN run logs show it: they record the real device name, and the by-event OOF ran on a V100 as `cuda:1` while the by-mission OOF ran on the RTX 2060 as `cuda:3`. The fix is `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which makes indices match `nvidia-smi`; every big-ML run used it ([tabpfn NOTES](research/bakeoff/models/tabpfn/NOTES.md); [`research/bigml/REPORT.md`](research/bigml/REPORT.md) header).

### 10.3 Concurrency

The box was shared by several jobs at once: the bake-off contenders, the big-ML runs and the sweep. The EBM notes report load averages up to 170 and the TabPFN notes up to about 150 ([ebm NOTES](research/bakeoff/models/ebm/NOTES.md), [tabpfn NOTES](research/bakeoff/models/tabpfn/NOTES.md)). How we kept it workable:
- **Pin each GPU job to one card** by `nvidia-smi` index. The GRU ran both splits in parallel on two GPUs (V100 index 1, PG503 index 3) ([gru NOTES](research/bakeoff/models/gru/NOTES.md)).
- **Bound CPU parallelism per job** (`n_jobs` 10–40 for joblib, `num_threads` for LightGBM) so jobs did not starve each other.
- **Expect CPU to be the bottleneck for TabPFN**: most of its per-fold time was CPU preprocessing at 800–1,000 % CPU, not the GPU.
- The sweep families ran as parallel workers: CPU boosters and tabular models on the cores, CatBoost/XGBoost CUDA and the neural nets on the GPUs (team-reported).

### 10.4 Runtimes we recorded

| Job | Wall clock | Source |
|---|---|---|
| Optuna (60 × 2) + 10-seed LightGBM evaluation, all splits | 1,795 s | [`research/bigml/out_gbm.txt`](research/bigml/out_gbm.txt) |
| Risk futures, by mission (10,000 paths/event) | 372 s | [`research/bigml/out_mission.txt`](research/bigml/out_mission.txt) |
| Risk futures, by event / LOMO (4,000 paths/event) | 206 s / 725 s | [`out_event.txt`](research/bigml/out_event.txt), [`out_lomo.txt`](research/bigml/out_lomo.txt) |
| TabPFN OOF by event / by mission / final fit + test | 611 s (V100) / 727 s (RTX 2060) / 3 min 44 s (PG503) | [tabpfn NOTES](research/bakeoff/models/tabpfn/NOTES.md) |
| EBM, everything | about 20 min | [ebm NOTES](research/bakeoff/models/ebm/NOTES.md) |
| GRU, end to end incl. tuning | about 15 min | [gru NOTES](research/bakeoff/models/gru/NOTES.md) |
| Bake-off lgbm_all final fit | 162.1 s | [`registry/forecaster_bakeoff/model_card.json`](registry/forecaster_bakeoff/model_card.json) `runtime` |
| Organizer evaluation (all 4 LightGBM variants × 10 splits + bootstrap) | about 533 s | [`research/organizer/out/eval.log`](research/organizer/out/eval.log) |
| False-alarm ensemble run | 521 s | [`research/fa/ens/run.log`](research/fa/ens/run.log) |

### 10.5 How to reproduce

| Stage | Command / script |
|---|---|
| **Headline table, no retraining** | `server/.venv/bin/python research/organizer/recompute_table.py` |
| Dataset experiments (doc 11) | [`research/experiments/`](research/experiments/) (`a_target.py`, `b_baselines.py`, `b2_kelvins.py`, `c_*.py`, `pc2d.py`); outputs in `research/experiments/out/` |
| v1 tier forecaster | `make precompute` → [`server/pipelines/train.py`](server/pipelines/train.py) (needs `data/raw/`, git-ignored) |
| Bake-off | `python harness.py build` → `python refs.py` → each contender ([`models/`](research/bakeoff/models/)) → `python score.py`, in [`research/bakeoff/`](research/bakeoff/) ([`PREREG.md`](research/bakeoff/PREREG.md) §7) |
| Big ML (Optuna, futures, conformal, policy) | [`research/bigml/run_gbm.py`](research/bigml/run_gbm.py), [`run_futures.py`](research/bigml/run_futures.py) `mission\|event\|lomo\|full`, [`evaluate.py`](research/bigml/evaluate.py), [`policy.py`](research/bigml/policy.py), [`write_fans.py`](research/bigml/write_fans.py) |
| Organizer evaluation and served model | `python organizer.py build` → `eval` → `final`, in [`research/organizer/`](research/organizer/organizer.py) |
| False-alarm push | [`research/fa/`](research/fa/) `physics/phys.py`, `ens/run.py`, `thresholds/thr.py`, `ref/ref.py`, `final/final_phys.py`, then `select/select.py` |
| Audits | [`research/bakeoff/audit/`](research/bakeoff/audit/) `a1`–`a6`, `score_audit.py`, `rerun_lgbm_all.py` |

Software for the bake-off: Python 3.12, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, lightgbm 4.7.0, interpret 0.7.8, tabpfn 9.2.0, torch 2.6.0+cu124 ([`PREREG.md`](research/bakeoff/PREREG.md) §7). Several research scripts have absolute paths from the server (`/root/orbital-dodge/...`); adjust them before running elsewhere.

---

## 11. What we would do next

In priority order, each tied to something above:

1. **A futures + LightGBM combination for late escalations.** Futures would have flagged `train:2563` at the 95 % conformal cut-off (§9.1), and the futures + LightGBM average already had the fewest false alarms at matched recall on the Kelvins set (91, §6c). Pre-register it as an amendment and test it on the organizer population.
2. **Model the miss-distance clause directly.** 549 of 551 criticals are miss-driven, and late flips like `train:4973` are jumps in the predicted miss. A regression on the final miss with a predictive interval would let the app say "could still move by about 1 km", not only a tier ([`docs/BAKEOFF.md`](docs/BAKEOFF.md) next steps).
3. **Mission-level bootstrap for the organizer headline**, so the by-mission CIs are honest about having only 19 missions (§4.4).
4. **Multi-seed decisions.** Re-run the bake-off rule on 5-seed averages or seed-bagged models, so a 2-event gap stops deciding the winner (§6b).
5. **Finish the cost sweep on the organizer population**, with its outer-fold selection, and fill the `forecaster_cost` slot only if it beats 1.0.0 under the same pre-registered rule (§6g).
6. **Per-operator calibration and conformal coverage** once an operator's labelled history exists, because pooled coverage hides mission 15 (§6d, §9.3).
7. **A Pc-only label with its own R.** If the product moves to a Pc ≥ 1e-4 "red" label, re-derive R in the 20–50 range first, as [COST_RATIO §5](docs/COST_RATIO.md) asks, then re-tune.
8. **Offline evaluation of the decision model** against the expected-cost rule on held-out events, pre-registered (§6j).

---

## 12. File index

Every file this page cites, by stage.

**Facts and write-ups**
- [`docs/FACTS.json`](docs/FACTS.json): verified fact sheet, every value with its file and key
- [`docs/REPORT.md`](docs/REPORT.md): full technical report (§2 data, §5 ML process, §6 results, §8 failures)
- [`docs/BAKEOFF.md`](docs/BAKEOFF.md): bake-off summary, audit outcome, failure case
- [`docs/COST_RATIO.md`](docs/COST_RATIO.md): derivation of R = 10
- [`docs/design/11-dataset-experiments.md`](docs/design/11-dataset-experiments.md): target, transition matrix, physics agreement, cleaning
- [`docs/design/12-domain-research.md`](docs/design/12-domain-research.md): domain sources (Kelvins history, CARA, ESA)

**Dataset experiments**
- [`research/experiments/`](research/experiments/): `a_target.py`, `b_baselines.py`, `b2_kelvins.py`, `c_physics.py`, `c_physics2.py`, `c_span.py`, `c_final.py`, `pc2d.py`, `common.py`, and [`out/`](research/experiments/out/) (e.g. [`b_baselines.txt`](research/experiments/out/b_baselines.txt), [`b2_kelvins.txt`](research/experiments/out/b2_kelvins.txt))

**v1 forecaster**
- [`registry/tier_forecaster/1.0.0/README.md`](registry/tier_forecaster/1.0.0/README.md), [`metrics.json`](registry/tier_forecaster/1.0.0/metrics.json), [`model_card.json`](registry/tier_forecaster/1.0.0/model_card.json)
- [`server/pipelines/train.py`](server/pipelines/train.py), [`server/riskengine/features.py`](server/riskengine/features.py), [`server/riskengine/forecast.py`](server/riskengine/forecast.py)

**Bake-off**
- [`research/bakeoff/PREREG.md`](research/bakeoff/PREREG.md), [`PREREG.sha256`](research/bakeoff/PREREG.sha256)
- [`research/bakeoff/LEADERBOARD.md`](research/bakeoff/LEADERBOARD.md), [`leaderboard.json`](research/bakeoff/leaderboard.json), [`leaderboard_refs.md`](research/bakeoff/leaderboard_refs.md)
- [`research/bakeoff/harness.py`](research/bakeoff/harness.py), [`refs.py`](research/bakeoff/refs.py), [`score.py`](research/bakeoff/score.py), [`make_leaderboard.py`](research/bakeoff/make_leaderboard.py), [`export_registry.py`](research/bakeoff/export_registry.py)
- [`research/bakeoff/results/`](research/bakeoff/results/): `gru.json|md`, `tabpfn.json|md`, `ebm.json|md`, `tabpfn_vs_refs.json|md`
- [`research/bakeoff/models/`](research/bakeoff/models/): [`ebm/NOTES.md`](research/bakeoff/models/ebm/NOTES.md), [`gru/NOTES.md`](research/bakeoff/models/gru/NOTES.md), [`tabpfn/NOTES.md`](research/bakeoff/models/tabpfn/NOTES.md)
- [`research/bakeoff/failure_case.json`](research/bakeoff/failure_case.json)
- [`registry/forecaster_bakeoff/model_card.json`](registry/forecaster_bakeoff/model_card.json)

**Audits**
- [`research/bakeoff/audit/`](research/bakeoff/audit/): [`a1_out.json`](research/bakeoff/audit/a1_out.json), [`a2_out.json`](research/bakeoff/audit/a2_out.json), [`a3_out.json`](research/bakeoff/audit/a3_out.json), [`a5_out.json`](research/bakeoff/audit/a5_out.json), [`a6_out.json`](research/bakeoff/audit/a6_out.json), [`leaderboard_audit.md`](research/bakeoff/audit/leaderboard_audit.md), and their scripts

**Big ML (futures, conformal, policy, Optuna)**
- [`research/bigml/REPORT.md`](research/bigml/REPORT.md), [`results.json`](research/bigml/results.json)
- [`research/bigml/mdn.py`](research/bigml/mdn.py), [`run_futures.py`](research/bigml/run_futures.py), [`run_gbm.py`](research/bigml/run_gbm.py), [`evaluate.py`](research/bigml/evaluate.py), [`policy.py`](research/bigml/policy.py), [`write_fans.py`](research/bigml/write_fans.py)
- Logs: [`out_gbm.txt`](research/bigml/out_gbm.txt), [`out_mission.txt`](research/bigml/out_mission.txt), [`out_event.txt`](research/bigml/out_event.txt), [`out_lomo.txt`](research/bigml/out_lomo.txt)
- [`registry/futures/`](registry/futures/) (`metrics.json`, `fans/`), [`registry/conformal/`](registry/conformal/), [`registry/policy/`](registry/policy/)

**Cost sweep**
- [`research/fa/ens/config_rank.csv`](research/fa/ens/config_rank.csv) (mid-run snapshot, 315 configs), [`rank_configs.py`](research/fa/ens/rank_configs.py)

**Organizer evaluation and served model**
- [`research/organizer/organizer.py`](research/organizer/organizer.py), [`results.json`](research/organizer/results.json), [`recompute_table.py`](research/organizer/recompute_table.py)
- [`research/organizer/out/`](research/organizer/out/): [`build_summary.json`](research/organizer/out/build_summary.json), [`eval_metrics.json`](research/organizer/out/eval_metrics.json), [`final_summary.json`](research/organizer/out/final_summary.json), [`eval.log`](research/organizer/out/eval.log)
- [`registry/forecaster_organizer/CURRENT`](registry/forecaster_organizer/CURRENT)
- [`registry/forecaster_organizer/1.0.0/`](registry/forecaster_organizer/1.0.0/): `model_card.json`, `model.txt`, `oof_organizer.parquet`, `predictions.parquet`, `predictions_test.parquet`

**False-alarm push**
- [`research/fa/select/results.json`](research/fa/select/results.json), [`select.py`](research/fa/select/select.py)
- [`research/fa/physics/`](research/fa/physics/), [`research/fa/ens/`](research/fa/ens/), [`research/fa/thresholds/`](research/fa/thresholds/), [`research/fa/ref/`](research/fa/ref/), [`research/fa/final/`](research/fa/final/)
- [`registry/forecaster_organizer/1.1.0/model_card.json`](registry/forecaster_organizer/1.1.0/model_card.json)

**Backend use**
- [`server/app/engine_adapter.py`](server/app/engine_adapter.py) (forecast source chain), [`server/riskengine/pc.py`](server/riskengine/pc.py) (physics Pc), [`server/tests/test_perf_honesty.py`](server/tests/test_perf_honesty.py), [`server/tests/test_policy.py`](server/tests/test_policy.py)
