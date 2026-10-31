# 12 — Domain research: conjunction assessment, Pc, CDMs, users, visuals, public data

Status: research and design input only. Nothing here is built. Written 31 Oct 2026.
Every external fact has a URL. Where I could not check a fact, I say so (search for **UNVERIFIED**).
Dataset facts come from `00-context.md` and are cross-checked against the Kelvins paper where possible.

Primary sources I read in full text (downloaded and text-extracted, not just summarised by a search engine):
- NASA CA Handbook **Rev 2, Vol 2 (Technical Appendices), Aug 2026**, NASA-SP-20205011318/REV2-VOL2 —
  https://www.nasa.gov/wp-content/uploads/2026/08/ca-handbook-volume2-nasa-sp-20205011318-rev2-vol2.pdf
  (Vol 1, Best Practices, is at https://ntrs.nasa.gov/citations/20260000453 but returned HTTP 403 to me. I did **not** read Vol 1.)
- Uriot et al., "Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition",
  arXiv:2008.03069v2 (https://arxiv.org/abs/2008.03069), journal version Astrodynamics 6(2):121–140, 2022,
  https://doi.org/10.1007/s42064-021-0101-5
- CCSDS 508.0-B-1 Conjunction Data Message Blue Book (with Cor. 1 2018 and Cor. 2) — https://ccsds.org/Pubs/508x0b1e2c2.pdf
- 18/19 SDS "Spaceflight Safety Handbook for Satellite Operators" v1.7, Apr 2023 —
  https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf

---

## 0. TL;DR: what this means for our design

1. **The thresholds are real but they are not the same everywhere.** NASA CARA uses red Pc > 1e-4 and yellow Pc > 1e-7 (Vol 2 App. F).
   The 18/19 SDS emergency criteria for LEO are TCA ≤ 3 d, miss ≤ 1 km and Pc ≥ 1e-4 (SFS Handbook Table 5).
   ESA LEO missions react at 1e-5 to 1e-4 and start monitoring one order of magnitude lower (Uriot §4.1).
   Starlink manoeuvres at 3e-7. The brief's tiers (critical ≥ 1e-4, elevated ≥ 1e-6) fit inside this range, so we can defend them.
2. **CARA warns against tiers based only on miss distance.** Vol 2 App. M.2 says nearly all conjunctions under 5 km are perfectly safe.
   App. E says a miss-only rule that catches 95 % of Pc > 1e-4 events needs a 10 km threshold, and that threshold also sweeps in more than 50 % of the harmless events.
   The brief's "miss < 1 km ⇒ Critical" rule produces most of our 908 "critical" train events, against only 13 with final risk ≥ −4.
   The UI should show the miss rule as a separate, labelled flag (for example "miss-distance screen (brief rule)") next to the Pc tier.
   The deck should tell the judges about this disagreement openly. It is a sound-restraint point.
3. **The decision point is about 1 day before TCA, and the data stops at 2 days.** CARA puts the maneuver commitment point (MCP) at 0.5–1 d before TCA (App. F).
   ESA alerts control teams about 3 d before TCA and makes the final decision about 1 d before (ESA Kelvins announcement).
   Kelvins cuts its inputs at 2 d because manoeuvres are planned at least 2 d before TCA (Uriot §4.2).
   Our test set (time_to_tca between 2.0 and 6.99 d) is exactly this cut. So our product question is: "at T−2 d, will this event still be red at the MCP?"
4. **The latest-risk baseline (LRP) is very hard to beat.** It scores L = 0.694 on the official test set. Only 12 of 97 teams beat it, and the winner reached 0.556.
   In a post-hoc experiment, only 1 of 5,000 LightGBM models beat LRP on both train and test (Uriot Table 3 and §6.1).
   Our honest evaluation must report LRP. The winner's rule cascade (sesc) is a second, stronger rule baseline.
5. **Our numbers match the paper exactly, so we have the true Kelvins split.** The paper's post-hoc train table has **8,293** eligible train events and **66** train high-risk events. That is exactly our context-pack numbers.
   The official test set had 150 high-risk events and 2,017 low-risk events, 2,167 in total. Our test file also has 2,167 events.
   **Metric-code check:** on the train-eligible events, LRP should give MSE_HR ≈ 0.330, F2 ≈ 0.411 and L ≈ 0.804 (Uriot Table 4, "Train set" column).
6. **The sigma sentinel 63,781,363 m is a documented DoD convention.** It is the "default covariance": a general-perturbations fallback whose position variances are each "ten Earth radii squared". Vol 2 App. P.7 says that **state is not suitable for conjunction risk assessment** (expected error 1–2 km).
   So the sentinel is not just a missing value. It is a rule-based **refusal/"not actionable"** trigger we can cite.
7. **Pc is implementable in about 40 lines** with the Foster/Alfano erfc 1-D integral or Chan's series. Section 3 gives test vectors and the encounter-plane construction.
   The dataset has everything except absolute frames. A documented approximation that reconstructs the chaser frame lets us recompute Pc and check it against the `risk` column.
8. **An expert expects a specific set of displays.** CARA App. Q lists them: object vital statistics, ground trace, conjunction-plane plot, Pc time history, miss-distance history with RTN components and 1σ, a normalised-consistency plot, space weather, a drag-sensitivity heatmap and a maneuver trade-space heatmap.
   We can build about 6 of these 9 honestly from the Kelvins columns. See §6.
9. **CelesTrak OMM JSON makes the globe live.** I pulled the groups on 30–31 Oct 2026: 16,846 active, 11,157 Starlink, 1,980 FY‑1C debris, 584 COSMOS‑2251 debris, 109 IRIDIUM‑33 debris and 20 station objects.
   We must use OMM, because TLE ran out of 5-digit catalogue numbers on 2026‑07‑11 and the highest active NORAD ID is now 101227. Use satellite.js 7.1.0 `json2satrec`.
   CelesTrak allows one download per 2-hour update, so ship a snapshot.

---

## 1. NASA CARA (NASA-SP-20205011318)

### 1.1 Which document
- The original handbook came out in Dec 2020 as NASA/SP-20205011318. Rev 1 (NASA/SP-20230002470) aligned it with NPR 8079.1 (https://ntrs.nasa.gov/citations/20230002470).
  The current version is **Rev 2, Aug 2026**, split into Vol 1 "Best Practices" (NTRS 20260000453) and Vol 2 "Technical Appendices" (NTRS 20260000457), https://ntrs.nasa.gov/citations/20260000453.
- In Vol 2, the appendices that matter for us are E (TLE use), F (event rates), I (covariance realism), M (Pc as metric), N (Pc calculation), P (actionability) and Q (display flow).

### 1.2 Pc thresholds (with exact provenance)
| Threshold | Value | Source |
|---|---|---|
| CARA red (mitigation planning; maneuver if still red at MCP) | Pc > 1e-4 | Vol 2 App. F ("red events ... Pc value > 1E-04") |
| CARA yellow (heightened monitoring, OD/tracking review) | Pc > 1e-7 | Vol 2 App. F |
| NASA red as a requirement | 1e-4 requires a risk-mitigation maneuver under NPR 8079.1. CARA's default yellow is 7e-5. | NASA NTRS extended abstract, https://ntrs.nasa.gov/citations/20240011653 (search-engine summary, not read in full) |
| Post-mitigation target | Fully mitigated at 1e-7. The CARA-recommended mitigation level is 3e-6. | Vol 2 App. Q.8 |
| Most operators' action threshold | Pc > 1e-4 | NASA SmallSat 2024 paper (search summary), https://ntrs.nasa.gov/api/citations/20240006391/downloads/CA%20and%20automated%20maneuvers%20Smallsat2024%20final.docx |
| Historic ISS | red 1e-4, yellow 1e-5 | search summary of NTRS ISS documents (**UNVERIFIED** in primary text) |

### 1.3 Operational workflow (screening → CDM → assessment → maneuver planning)
1. **Screening.** NASA screens its assets against the DoD catalogue **every 8 hours**. The example LEO asset in App. Q used a screening volume of **0.5 km × 17 km × 20 km** (Vol 2 App. Q.1).
   19 SDS runs one full HAC‑vs‑HAC screening every 24 h, plus Concern-List screenings every 8 h (SFS Handbook p.9).
   In LEO, events are first discovered about **7 days** before TCA, and about 10 days before in other regimes (App. F).
2. **CDMs.** A new CDM arrives after every screening that still contains the event. ESA sees about **3 CDMs/day over a week** per event (Uriot §2), and our data has a median of 13 CDMs per event.
3. **Assessment.** For each update the analyst looks at:
   - Pc and its trend.
   - OD quality ("actionability", App. P): WRMS limits are payload 1.5, rocket body 2.0, debris 5.0. Residual acceptance below 80 % triggers review. A default covariance makes the event non-actionable. LUPI (fit span) is checked.
   - Space weather. CARA treats Dst < −75 nT as disturbed (App. Q.7.1).
   - Drag sensitivity (App. Q.7.2).
4. **Maneuver planning.** A maneuver trade-space is computed over burn size × burn time, with the remediated Pc colour-coded (App. Q.8).
   The candidate maneuver ephemeris is sent to 19 SDS for screening, so the new trajectory is checked against the whole catalogue.
5. **Decision at the MCP.** The decision is made at the maneuver commitment point, **typically 12–24 h (0.5–1 d) before TCA** (App. F). In the App. Q worked example the MCP was 1 d before TCA.
   App. Q.8 states the trade-off directly: commit early to a smaller burn, or wait for tracking that may make the burn unnecessary. **Our decision-support layer should sit in exactly this tension.**

### 1.4 What a conjunction analyst actually does each day (from App. P/Q, ESA, and the Swarm incident)
- Triage the new CDMs from every 8-hour screening. Most stay green. Yellow ones (Pc > 1e-7) get OD-quality checks and may trigger a request for extra tracking (App. F).
- Read the secondary object's "vital statistics": tracks per day, observations used, time since last observation, WRMS, ballistic coefficient, EDR and RCS (App. Q.2). Decide whether the data can be trusted.
- Watch the Pc history and the RTN miss components. The radial miss matters most and "needs to be watched carefully" (App. Q.6.1).
- Check the space-weather forecast and the drag-sensitivity heatmap.
- For red events: build the maneuver trade space, coordinate with flight dynamics and mission ops, submit the maneuver ephemeris for screening, and hold a go/no-go meeting.
  In the ESA Swarm‑B case: the alert came about 39 h out, coordination about 30 h out, commands were uplinked about 15 h out, and the manoeuvre was **aborted at the final go/no-go** after fresh GPS and radar data (https://www.esa.int/Enabling_Support/Operations/Anatomy_of_a_debris_incident).
- ESA says each alert that needs follow-up takes "hours of analysis" (https://www.esa.int/Space_Safety/Space_Debris/Automating_collision_avoidance).

### 1.5 Volumes
- **Per ESA LEO satellite:** **hundreds of alerts per week**, of which about **2 per satellite per week** need detailed analyst follow-up (ESA "Automating collision avoidance", URL above, and the ESA Kelvins announcement https://www.esa.int/Enabling_Support/Space_Engineering_Technology/AI_challenged_to_stave_off_collisions_in_space).
- **ESA SDO:** analyses "several hundred messages every day" and executes about **12 CAMs per year** across the fleet (https://www.esa.int/Space_Safety/Space_Debris/Reentry_and_collision_avoidance). Elsewhere ESA says "more than one per satellite per year" (Automating page).
- **Swarm:** several hundred potential conjunctions per year, typically about 10 critical (Anatomy of a debris incident).
- **Planet (2015–16):** about 1,700 CDMs between Jan 2015 and Jul 2016, about 850 of them intra-Planet. JSpOC reported 96 % of close approaches 28 h or more ahead (https://swfound.org/media/205611/smallsat2016_vittaldev_swf.pdf, and Planet FCC filings via search; numbers from search summaries).
- **Starlink** (SpaceX semi-annual reports to the FCC, via press):
  - 144,404 avoidance manoeuvres in Dec 2024–May 2025.
  - 148,696 in Jun–Nov 2025 (https://www.altagrove.com/grovewire-news/spacex-148696-starlink-collision-maneuvers-in-6-months-ending-nov-30-continued-issues-with-operators-slipshod-reporting).
  - **207,152** in Dec 2025–May 2026, about 1,140 per day, which press summarised as "every Starlink dodges almost weekly" (https://space.com/space-exploration/satellites/every-spacex-starlink-satellite-has-to-dodge-a-collision-almost-weekly-and-experts-fear-the-worst).
  - SpaceX's threshold is **3e-7**, versus the industry's 1e-4 (altagrove, above).
  - I did not read the primary FCC filings.
- **Space-Track total CDMs per day: UNVERIFIED.** Secondary sites claim about 75/day and about 1,600/week, and a news aggregator claims "200k/day (2020) → 1.3M/day (2026)" DoD conjunction messages. None of these is an authoritative source. **Do not put them on a slide.**
  The Space Force fact sheet points to Space-Track and says 47,000+ tracked objects (https://www.spaceforce.mil/About-Us/Fact-Sheets/Fact-Sheet-Display/Article/3740012/18th-space-defense-squadron/).
- **CARA event rates per payload per year:** given as bar charts by HBR (0.1–10 m) and regime (LEO1 < 500 km, LEO2 500–750 km, LEO3/4 750–1400 km). These are images in App. F and I could not extract the numbers.
  CARA reprocessed **1.93 million conjunctions from 2021–2022** for the multistep Pc study (App. N.5), which shows the scale of one agency's archive.

### 1.6 Cost of a manoeuvre
- **Fuel / Δv:** a Canadian DRDC study found that **0.01–0.05 m/s** burns avoid most LEO conjunctions if applied ≥ 20 h before TCA, with the cost rising sharply in the last 24 h (https://publications.gc.ca/collections/collection_2016/rddc-drdc/D68-6-113-2013-eng.pdf).
  One dissertation reports about 0.5 m/s strategic versus more than 8 m/s emergency (https://www.ideals.illinois.edu/items/139700; abstract only).
  CARA App. Q.8 says the late option in its example needed **twice** the Δv of the early option.
- **Mission downtime:** instruments are shut down around the burn (Swarm example). Larger burns may need a **restoration manoeuvre** to return to the reference orbit (CARA App. M.5).
- **Analyst time:** "hours of analysis" per followed-up alert (ESA). A maneuver needs coordination across flight dynamics, the debris office and mission ops (Swarm example).
- **Sensitivity:** if CARA moved from Pc to more conservative metrics, maneuvers would multiply by ×~1 (Alfano MaxPc), ×~8 (Elkantassi–Davison) or ×9 (3σ ellipsoid overlap). CARA calls an order-of-magnitude increase unbearable without major mission disruption (App. M.5).
  **This is our best citation for "false alarms are a real cost".**

### 1.7 False-alarm burden: what the evidence supports
- **Untracked objects (CARA App. M.1):** about 300,000 objects > 1 cm exist but only 30–50k are catalogued, so roughly 85 % of collision-capable conjunctions cannot be seen at all. That argues against over-conservatism on the ones we can see.
- **Our own data (Uriot Fig. 12, my reading of a garbled figure, treat as approximate):** among train events whose latest risk at ≥ 2 d before TCA was ≥ −6, about 70 stayed high and about 269 ended low. That is roughly 79 % **false alarms for LRP**.
  Among events that started low, about 23 ended high, the "anomalous" misses.
- **All top-10 Kelvins teams** missed the same 11 of 150 high-risk test events. Those events show big risk jumps and high uncertainties, for example t_sigma_rdot (Uriot §5.4 and Fig. 14).
  **This is our "visible failure case with reason".**

---

## 2. ESA Space Debris Office and the Kelvins Collision Avoidance Challenge (2019)

### 2.1 ESA SDO practice (Uriot §2, ESA pages)
- The SDO supports Aeolus, Cluster II, CryoSat‑2, Swarm A/B/C, seven Copernicus Sentinels and third-party customers.
  CDMs come from CSpOC. Risk is recomputed in-house with methods such as **Alfriend–Akella** (Uriot §2, ref. [17]).
  So the dataset's `risk` is ESA's own Pc, **not** the CDM's `COLLISION_PROBABILITY`.
- Reaction thresholds are 1e-5 to 1e-4 by mission, and notification starts one order of magnitude lower (Uriot §4.1).
- ESA alerts control teams about 3 d before TCA and makes the final decision about 1 d before, with a 1e-4 threshold (ESA Kelvins announcement).
- ESA's automation programme is **CREAM** (Collision Risk Estimation and Automated Mitigation), run by GMV with Guardtime. It aims to cut operator workload and false alerts, and as of 2025 is in ground tests (https://www.innovationnewsnetwork.com/space-traffic-management-enters-a-new-era-with-esas-cream-project/60737/).

### 2.2 The dataset (Uriot §3)
- Raw database: 15,321 events and 199,082 CDMs from 2015–2019, with 103 attributes and 1–23 CDMs per event (mean 13).
  High-risk final events: 30 with r ≥ −4, 131 with r ≥ −5 and 515 with r ≥ −6.
- Pre-filtering:
  - Kept only events with max Pc (Pc after covariance scaling) above 1e-15.
  - Removed intra-constellation events and events with zero relative velocity.
  - **Removed all CDMs before any manoeuvre epoch.** So events where ESA did manoeuvre do not show the pre-burn history. This is a selection bias to mention.
- Anonymisation: time is given as time_to_tca, states are relative to the target, missions become `mission_id` and events get a random ID.
- Release: launched **16 Oct 2019** and ran for two months, with **96–97 teams** (the paper says both) and 862 submissions.

### 2.3 Column meanings (official Kelvins data page, https://kelvins.esa.int/collision-avoidance-challenge/data/)
- `risk`: log10 of ESA's self-computed Pc at that CDM.
- `max_risk_estimate`: maximum Pc obtained by scaling the combined covariance. `max_risk_scaling`: the scale factor that achieves it.
- `time_to_tca` (days), `miss_distance` (m), `relative_speed` (m/s), `relative_position_{r,t,n}` (m) and `relative_velocity_{r,t,n}` (m/s), all in the target's RTN frame.
- `mahalanobis_distance` is not described on the page.
- `c_object_type`: categorical. `geocentric_latitude` (deg) of the conjunction point. `azimuth` and `elevation` (deg) of the relative velocity vector.
- Space weather:
  - `F10`: 10.7 cm solar flux.
  - `F3M`: 81-day mean of F10.7.
  - `AP`: daily planetary geomagnetic amplitude.
  - `SSN`: Wolf sunspot number.
- Per object, `t_` = target (ESA satellite) and `c_` = chaser:
  - Uncertainties: `sigma_{r,t,n}` (m), `sigma_{rdot,tdot,ndot}` (m/s), and **correlation** coefficients `ct_r, cn_r, cn_t, crdot_*, ctdot_*, cndot_*`.
  - `position_covariance_det`, `cd_area_over_mass` (m²/kg) and `cr_area_over_mass`.
  - `span` (m): object size used in the risk computation, with a 2 m minimum for the chaser. Uriot calls t_span "the diameter".
  - `rcs_estimate` (m²) and `sedr` (W/kg).
  - Orbit: `h_apo`, `h_per` (km), `ecc`, `j2k_inc` (deg), `j2k_sma` (km).
  - OD: `actual_od_span` and `recommended_od_span` (days), `obs_available`, `obs_used`, `residuals_accepted`, `time_lastob_start/end` (days, relative to CDM creation) and `weighted_rms`.

### 2.4 Official metric (Uriot §4.3), implement exactly
- High risk means r ≥ −6 (log10 Pc).
- `L = MSE_HR / F2`. Here MSE_HR = (1/N*) Σ_{i: r_i ≥ −6} (r_i − r̂_i)², taken only over the true high-risk events.
  F2 = 5·p·q / (4p + q), computed on high vs low over **all** events.
- Clipping: predictions below −6 are set to −6.001. This is the optimal value for any predicted-low event, and the reported scores use it.
- Eligible test events have ≥ 2 CDMs, a final CDM within 1 d of TCA, and a first CDM ≥ 2 d before TCA. All CDMs within 2 d are removed, and the final CDM is the target.
  Of 515 high-risk events, only 216 were eligible: 150 went to test and 66 to train.

### 2.5 Results (Uriot Table 3, final test set)
| Team | L | MSE_HR | F2 |
|---|---|---|---|
| **LRP baseline** (r̂ = r₋₂ if ≥ −6, else −6.001) | **0.694** | 0.513 | 0.739 |
| sesc (1st) | 0.556 | 0.407 | 0.733 |
| dietmarw (2nd) | 0.571 | 0.437 | **0.765** |
| Magpies (3rd) | 0.585 | 0.441 | 0.753 |
| Vidente | 0.610 | 0.436 | 0.714 |
| DeCRA | 0.615 | 0.457 | 0.743 |
| Valis / DunderMifflin | 0.628 | 0.467 / 0.451 | 0.744 / 0.718 |
| Constant r̂ = −5 (CRP) | 2.5 | — | — |

- Only **12 of 97 teams beat LRP**, and most of the top teams needed about 20 days to do it. All top-10 teams improved MSE_HR, but **many did not improve F2** over LRP.
- **sesc's winning method is a hand-tuned rule cascade on the latest risk r₋₂, not ML.** Applied in this order:
  1. Map r₋₂ ∈ [−6.04, −6.00) to −5.95, r₋₂ ∈ [−6.40, −6.04) to −5.60, and r₋₂ ∈ [−7.30, −6.40) to −5.00. This promotes borderline-low events for recall.
  2. Force low (−6.00001) when `c_object_type == PAYLOAD`, `t_span < 0.5`, or `miss_distance > 30000`.
  3. Clip highs: r₋₂ ∈ [−4, −3.5) to −4, and r₋₂ ≥ −3.5 to −3.5.
  The team admitted probing the leaderboard ("trial-and-error"). Their earlier attempts with feature extraction, AutoML via genetic programming and random forests scored L between 0.83 and 1.0.
- **Magpies:** a Manhattan-LSTM siamese ensemble flags "low-now, high-later" anomalies from 10 features. The 7 chosen CDM features are time_to_tca, max_risk_estimate, max_risk_scaling, mahalanobis_distance, miss_distance, c_position_covariance_det and c_obs_used, plus the number of CDMs and the mean and std of risk.
  It predicts −5.35 for flagged anomalies and otherwise behaves like LRP.
- **Post-hoc LightGBM (§6):**
  - Setup: 8,293 train events with one CDM each (the last one at ≥ 2 d), 100 numeric features, target = r − r₋₂ (quantile-transformed), and LightGBM 2.2.3 with n_estimators 25, feature_fraction 0.25 and learning_rate 0.05.
  - Of 5,000 models, **only 1 beat LRP on both train and test**. Its test L was 0.684, a 1.4 % gain.
  - Train-to-test Spearman correlation is negative for 1/F2, which means overfitting.
  - Over 10,000 stratified re-splits, an 80/20 split gives about +1.1 % mean test gain over LRP on 1/F2.
- **The key finding to quote:** the naive latest-risk forecast is a strong predictor of the final risk. Expect ML to add only a few percent, so present ML as *calibrating uncertainty / flagging likely jumps*, not as *replacing Pc*.
- **Most-missed events have high uncertainties.** Uriot suggests outputting a **risk distribution instead of a point prediction** (§5.4), which supports our "uncertainty visible" rubric item.

---

## 3. Pc computation: implementable formulas

### 3.1 Encounter-plane (B-plane) construction (CARA App. N.2; Foster & Estes 1992; Chan 2008)
Inputs at TCA: relative position **r** and relative velocity **v**, both secondary minus primary (CDM convention: Object2 relative to Object1, in Object1's RTN frame).
Also the 3×3 position covariances C₁ and C₂ in a common frame, and the hard-body radius R.

```
C   = C1 + C2                              # assumes independence (App. N assumption 1)
ŷ   = v / |v|                               # along relative velocity (the "soda straw")
ẑ   = (r × v) / |r × v|
x̂   = ŷ × ẑ                                 # lies along r because r ⟂ v at TCA
P   = [x̂ᵀ; ẑᵀ]                              # 2×3 projector onto the encounter plane
C2d = P C Pᵀ                                # 2×2 combined covariance in the plane
m   = P r = (|r|, 0)                        # miss vector in the plane
```
**Foster 2D Pc (App. N, eq. N-1):**
Pc = 1/√det(2πC2d) ∬_{‖ρ‖≤R} exp(−½ (ρ−m)ᵀ C2d⁻¹ (ρ−m)) dρ

**Alfano 1-D form (eq. N-2), the recommended implementation.**
1. Diagonalise C2d → σx ≥ σy, and rotate m to (x_m, y_m) in the principal axes, both ≥ 0.
2. Compute
   Pc = 1/(√(8π) σx) ∫_{−R}^{R} [erfc(y₋) − erfc(y₊)] · exp(−(u + x_m)² / (2σx²)) du,
   with y± = (y_m ± √(R² − u²)) / (√2 σy).
Use the erfc difference for small Pc (Elrod 2019). Gauss–Chebyshev quadrature is fine while R < 4σx; otherwise use adaptive quadrature (App. N.2).

**Upper bound, no integration (eq. N-3/N-4):**
Psq = ¼ [erf(X₊) − erf(X₋)] [erf(Y₊) − erf(Y₋)], with X± = (x_m ± R)/(√2 σx) and Y± = (y_m ± R)/(√2 σy).
CARA notes that **USSPACECOM ASW CDMs often report Psq** as the Pc.

**Chan (2008) series (equivalent-area / Rician).**
With u = R²/(σx σy) and v = x_m²/σx² + y_m²/σy² (the squared Mahalanobis distance of the miss):
Pc ≈ e^{−v/2} Σ_{m=0}^{M} [vᵐ / (2ᵐ m!)] · [1 − e^{−u/2} Σ_{k=0}^{m} uᵏ / (2ᵏ k!)]
M = 10–20 terms is enough. Accuracy degrades when R is large relative to σy.
I wrote this from memory of Chan 2008 and verified it numerically against the Alfano integral (table below). I did not read Chan's book.

**Maximum Pc (Alfano 2005b, "dilution").**
Scale the covariance by k² and use the small-R approximation Pc(k) ≈ R²/(2k² σx σy) · e^{−v/(2k²)}.
This is maximised at k*² = v/2, which gives
**Pc_max ≈ R² / (e · σx σy · v)**.
(This is my derivation under the small-R approximation. The CARA SDK has the exact `FrisbeeMaxPc.m` and `DilutionMaxPc.m`.)
`max_risk_estimate` and `max_risk_scaling` in Kelvins are ESA's version of this quantity.
**Dilution test:** if k* > 1, the actual covariance is larger than the one that maximises Pc. The event is then in the *dilution region*, and a low Pc does **not** mean safe (App. M.4). The UI should show this as a badge.

**Numerical cross-check** (`scratchpad/scripts/pc.py`; inputs x_m, y_m, σx, σy, R in metres, principal axes):
| case (x_m, y_m, σx, σy, R) | Alfano/Foster | Chan (12 terms) | Psq | Pc_max (k*) |
|---|---|---|---|---|
| (100, 50, 200, 50, 10) | 2.676e-3 | 2.674e-3 | 3.406e-3 | 2.94e-3 (0.79) |
| (500, 0, 1000, 20, 15) | 4.638e-3 | 4.952e-3 (+7 %; R/σy = 0.75) | 5.775e-3 | 1.66e-2 (0.35) |
| (20, 5, 30, 10, 5) | 2.871e-2 | 2.905e-2 | 3.625e-2 | 4.41e-2 (0.59) |
| (1500, 200, 800, 100, 20) | 5.922e-5 | 5.854e-5 | 7.578e-5 | 2.45e-4 (1.94 → dilution) |
Use these values as unit tests. NASA's Alfano and Omitron test-case CDMs are better regression tests: https://github.com/nasa/CARA_Analysis_Tools, under `DataFiles/SampleCDMs/AlfanoTestCase01..11.cdm` and `OmitronTestCase_Test0{1..8}*.cdm`.
The reference MATLAB code is in `DistributedMatlab/ProbabilityOfCollision/{Pc2D_Foster,PcElrod,PcCircle,Pc2D_Hall,Pc3D_Hall,PcMultiStep}.m`.

### 3.2 Hard-body radius conventions (App. N.10)
- Pc scales roughly with R², so ×3 on R gives ×9 on Pc. **Do not pad R for conservatism.** Lower the threshold instead.
- Historic practice was a fixed 20 m. CARA says nearly all operators have dropped it because it produces false alarms.
- **Today's options, from simplest to most precise:**
  1. Circumscribing sphere around the primary plus a secondary size. This is the most common option.
  2. Equivalent circle of the maximum projected area. This is CARA's recommended middle ground.
  3. Actual projection into the conjunction plane, which needs attitude and a CAD model.
- **Uncertain secondary size:** R_eff = √((R₁ + R̄₂)² + σ_R2²) (eq. N-30). RCS-based sizes are uncertain by a factor of 2–3.
  About 98 % of unknown secondaries have R̄₂ < 35 cm. The default for unknowns is **1.5 m**.
- **18/19 SDS:** computes the probability of being within a combined "exclusion radius", conservative by design (SFS Handbook Annex A).
  Size appears as `AREA_PC` (m²), and an operator-supplied HBR goes in a CDM COMMENT field.
- **In our dataset:** use `t_span` and `c_span` (c_span ≥ 2 m). Likely R = (t_span + c_span)/2 if span is a diameter, which is Uriot's wording. **Calibrate:** recompute Pc both ways and keep the convention that best reproduces `risk`.

### 3.3 Assumptions and when they fail (App. N.2 and N.3–N.5)
1. **Independence of C₁ and C₂.** This fails for drag-correlated objects that share atmospheric-density forecast error (App. N.8.3).
2. **Rectilinear motion and static covariance during the encounter.** This needs high relative speed. CARA notes that relative speed **alone** cannot tell when the assumption fails.
   Typical LEO is about 10–14 km/s. CARA's ~3 m/s example was underestimated **×330** by 2D Pc (App. N.3).
   Our data has relative_speed from **4 to 17,138 m/s**. Flag events with low v_rel, for example < 100 m/s. That cut-off is my heuristic, not a CARA number. Those events get "2D Pc not valid → refuse/escalate".
3. **Gaussian positions and negligible velocity uncertainty.** These fail with long propagation, defective covariances or solar storms (App. P.9).
4. **Temporally isolated event.** This fails for long co-orbiting interactions.

How often this matters: on 1.79 M conjunctions, CARA's multistep algorithm used 2D‑Pc for **94.7 %**, 2D‑Nc for 5.0 %, 3D‑Nc for 0.3 %, and Monte Carlo for 0.006 % + 0.002 %. Another 7.3 % could not be processed because of OD or filter conditions (App. N.5).
**So 2D Pc with a usage-violation flag is the right tool for a hackathon.**

### 3.4 Covariance realism and defects (App. I, N.8, P)
- **Realism test:** with truth data available, normalised errors εᵀC⁻¹ε should follow a χ² distribution with 3 DoF. Test the fit with Cramér–von Mises or Anderson–Darling (App. I.4).
  We have no truth, so we cannot run this test. Say that explicitly.
- **Non-positive-semidefinite covariances:** clip negative eigenvalues to zero or a small positive value (App. N.8.2).
  Building C from sigmas and correlations, Σᵢⱼ = ρᵢⱼ σᵢ σⱼ, can produce NPD matrices, so clip them and log it.
- **Default covariance:** position variances of (10 R_E)² mean a general-perturbations state with 1–2 km error, which is **not actionable** (App. P.7). The CCSDS field is `COVARIANCE_METHOD = DEFAULT`.
  → Dataset sentinel σ = 63,781,363 m = 10 × 6,378,136.3 m. **Rule: if any position sigma equals the sentinel, the event is not actionable from this CDM. Show it as a refusal with this citation.**
- **Probability dilution (App. M.4):** a low Pc with a huge covariance is not evidence of safety. CARA's practice is to accept a low-Pc dilution case like an untracked-object risk, and to use MaxPc for high-consequence events.

### 3.5 Recomputing Pc from the Kelvins columns (proposal, an approximation that must be labelled)
1. Target covariance in target RTN (3×3) from `t_sigma_{r,t,n}` and `t_ct_r, t_cn_r, t_cn_t`, using Σᵢⱼ = ρᵢⱼσᵢσⱼ.
2. Chaser covariance comes in the **chaser's** RTN frame, which we must rotate into the target RTN frame.
   - Approximate the radial axes as equal, because the positions differ by only a few km.
   - Approximate the chaser velocity in target RTN as v_c ≈ (0, v_t, 0) + v_rel. Here v_t = √(μ(2/r − 1/a)) with a = `t_j2k_sma`, μ = 398600.4418 km³/s², and r ≈ a. The target orbits are near-circular (check `t_ecc`), so v_t ≈ √(μ/a).
   - Then T̂_c = normalise(v_c − (v_c·R̂)R̂) and N̂_c = R̂ × T̂_c.
3. C = C_t + R_c C_c R_cᵀ. Project with **r** = `relative_position_{r,t,n}` and **v** = `relative_velocity_{r,t,n}`. Then compute Pc with R from the spans.
4. **Validate:** report the fraction of CDMs where |log10 Pc_ours − risk| < 0.5. If the fraction is low, say so and fall back to showing ESA's `risk` as the authoritative Pc.
   **Success criterion is not guaranteed.** I have not run this.

---

## 4. CDM standard (CCSDS 508.0-B-1) and how the dataset maps to it

### 4.1 Structure (Blue Book, Tables 3-1 to 3-4)
- **Header:** `CCSDS_CDM_VERS`, `CREATION_DATE`, `ORIGINATOR` (e.g. JSPOC, ESA SST), `MESSAGE_FOR`, `MESSAGE_ID`.
- **Relative metadata/data:**
  - `TCA` (UTC, obligatory) and `MISS_DISTANCE` (m, obligatory).
  - `RELATIVE_SPEED` (m/s), `RELATIVE_POSITION_R/T/N` and `RELATIVE_VELOCITY_R/T/N`, defined as Object2 relative to Object1, in Object1's RTN frame.
  - `START/STOP_SCREEN_PERIOD`, `SCREEN_VOLUME_FRAME` (RTN or TVN), `SCREEN_VOLUME_SHAPE` (ELLIPSOID or BOX), `SCREEN_VOLUME_X/Y/Z` and `SCREEN_ENTRY/EXIT_TIME`.
  - `COLLISION_PROBABILITY` (0–1) and `COLLISION_PROBABILITY_METHOD`. The method values come from the SANA registry, https://sanaregistry.org/r/cdm_cpm, e.g. FOSTER-1992, CHAN-1997, ALFANO-2005; **I did not open the registry, so verify the exact spellings**.
- **Per-object metadata (Object1 and Object2):**
  - `OBJECT`, `OBJECT_DESIGNATOR`, `CATALOG_NAME`, `OBJECT_NAME` and `INTERNATIONAL_DESIGNATOR`.
  - **`OBJECT_TYPE` ∈ {PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, OTHER}.**
  - Operator contact fields.
  - `EPHEMERIS_NAME` (NONE means the originator's own OD).
  - **`COVARIANCE_METHOD` ∈ {CALCULATED, DEFAULT}** and **`MANEUVERABLE` ∈ {YES, NO, N/A}**.
  - `REF_FRAME` plus the force-model fields.
- **OD parameters:** `TIME_LASTOB_START/END`, `RECOMMENDED_OD_SPAN`, `ACTUAL_OD_SPAN`, `OBS_AVAILABLE/USED`, `TRACKS_AVAILABLE/USED`, `RESIDUALS_ACCEPTED` (%) and `WEIGHTED_RMS`.
- **Additional parameters:** `AREA_PC`, `AREA_DRG` and `AREA_SRP` (m²), `MASS`, `CD_AREA_OVER_MASS` and `CR_AREA_OVER_MASS` (m²/kg), `THRUST_ACCELERATION` and `SEDR` (W/kg).
- **State vector:** `X, Y, Z` (km) and `X_DOT, Y_DOT, Z_DOT` (km/s). 18 SDS normally gives these in ITRF (SFS Handbook Annex A).
- **Covariance:** RTN, lower-triangular, 6×6 obligatory and up to 9×9 with drag, SRP and thrust terms.
  The entries are **variances and covariances** such as `CR_R` (m²), `CT_R`, `CT_T`, `CN_R`, `CN_T`, `CN_N`, then `CRDOT_*` (m²/s) and `CRDOT_RDOT` (m²/s²).
- 18 SDS CDMs add **COMMENT** lines such as apogee, perigee, inclination, exclusion-volume radius, operator HBR, covariance scale factor and DCP density-sensitivity vectors (SFS Handbook CDM table).
- Related:
  - ISO adopted the standard as ISO 19389:2014.
  - CCSDS 508.0-P-1.1 is a draft revision.
  - TraCSS (US Commerce) publishes its own CDM spec v2.1 (Jan 2026), https://www.space.commerce.gov/wp-content/uploads/2026/01/TraCSS-Spec-001-v2.1_CDM.pdf. I did not read it.

### 4.2 Kelvins column → CDM keyword map
| Kelvins column | CDM source | Transform / note |
|---|---|---|
| `time_to_tca` | `TCA − CREATION_DATE` | days. Absolute times were removed. |
| `miss_distance`, `relative_speed` | `MISS_DISTANCE`, `RELATIVE_SPEED` | same units (m, m/s) |
| `relative_position_{r,t,n}`, `relative_velocity_{r,t,n}` | `RELATIVE_POSITION_*`, `RELATIVE_VELOCITY_*` | same, in target RTN |
| `c_object_type` | Object2 `OBJECT_TYPE` | the dataset also has `TBA` (177 rows) |
| `t_/c_sigma_r` etc. | `sqrt(CR_R)`, `sqrt(CT_T)`, `sqrt(CN_N)`, `sqrt(CRDOT_RDOT)` … | sigma, not variance |
| `t_/c_ct_r`, `cn_r`, `cn_t`, `crdot_*` … | `CT_R/(σ_t σ_r)` etc. | **correlations**, not covariances |
| sigma = 63,781,363 m | `COVARIANCE_METHOD = DEFAULT` (inferred) | 10 R_E. Not actionable (CARA P.7). |
| `*_obs_available/used`, `*_residuals_accepted`, `*_weighted_rms`, `*_actual_od_span`, `*_recommended_od_span`, `*_time_lastob_start/end` | OD-parameter keywords of the same names | lastob times converted to days before the CDM |
| `*_cd_area_over_mass`, `*_cr_area_over_mass`, `*_sedr` | same keywords | — |
| `*_span`, `*_rcs_estimate` | `AREA_PC` / exclusion-radius COMMENT / ESA size DB | **derivation unknown** |
| `*_j2k_sma/ecc/inc`, `*_h_apo/h_per` | derived from the state vector or the 18 SDS COMMENT lines | J2000 osculating (assumed) |
| `risk`, `max_risk_estimate`, `max_risk_scaling`, `mahalanobis_distance`, `*_position_covariance_det` | **ESA-computed**, not in the CDM | `risk` is log10 Pc, floored at −30 |
| `geocentric_latitude`, `azimuth`, `elevation` | derived from the states | — |
| `F10`, `F3M`, `AP`, `SSN` | external space-weather indices | not CDM fields |
| `mission_id`, `event_id` | anonymised | — |
| **missing** | `CREATION_DATE`, `TCA`, `OBJECT_DESIGNATOR`, `X..Z_DOT`, `SCREEN_VOLUME_*`, `COLLISION_PROBABILITY(_METHOD)`, `MANEUVERABLE`, `EPHEMERIS_NAME` | We cannot reconstruct absolute orbits or know whether a manoeuvre happened (manoeuvred events were truncated). |

For the "trace to timestamped signal" rubric item, our internal CDM record should keep CCSDS names, for example `CdmRecord{TCA_REL_DAYS, MISS_DISTANCE, ..., OBJECT2.COVARIANCE_METHOD}`. The audit ledger can then say "CDM #k (T−3.2 d): OBJECT2 COVARIANCE_METHOD=DEFAULT → rule P.7".

---

## 5. The named user, the status quo, pain and cost

### 5.1 Primary persona (the one our data is literally about)
**A conjunction analyst in an EO/science operator's space-debris or flight-dynamics office.** The model is the ESA SDO supporting Sentinel, CryoSat and Swarm. The dataset is that office's 2015–2019 CDM archive, covering 19 missions in our train file.
- **Job:** triage 8-hourly CDM drops, keep about 2 events per satellite per week under watch, and recommend go/no-go at about T−1 d for events above 1e-4. Coordinate with flight dynamics and mission ops.
- **Tools today:**
  - Space-Track CDMs from 18/19 SDS, with in-house Pc recomputation (ESA's own tools, e.g. CORAM/DRAMA family, **UNVERIFIED naming**).
  - The NASA CARA SDK (free MATLAB), https://github.com/nasa/CARA_Analysis_Tools.
  - Commercial SSA: LeoLabs radar catalogue and CDM ingestion (https://leolabs.space/conjunction-alerts/), and Ansys STK Conjunction Analysis / AdvCAT (https://help.agi.com/stk/Content/stk/ObjectMap_AdCAT.htm).
- **Pain:**
  - Volume: hundreds of alerts per week per satellite.
  - Analysis time: hours per followed-up alert.
  - Volatility: risk jumps late, and all top-10 Kelvins teams missed 11 of 150 high-risk events.
  - Inability to act on dilution cases.
  - Commit-early versus wait (CARA Q.8).
- **Cost of error:**
  - False alarm: Δv, a science outage, a possible restoration burn, and staff hours.
  - Miss: loss of the satellite and debris generation.

### 5.2 Secondary persona
**A small-satellite operator with no CA staff, relying on the free Space-Track basic service.** Emergency CDMs and a CAN email arrive only when TCA ≤ 3 d, miss ≤ 1 km and Pc ≥ 1e-4 in LEO (SFS Handbook Table 5).
So they hear about an event late, with one number and no context. Planet's early CA relied on JSpOC and differential drag, which needs about 28 h of lead time.
US civil SSA is moving from Space-Track to Commerce's **TraCSS**, but as of Feb 2026 it had 17 pilot organisations and uncertain funding (https://www.space.commerce.gov/tracss/; https://spacepolicyonline.com/news/aerospace-industry-urges-congress-not-to-kill-tracss/).

### 5.3 Problem-realism slide: facts we can cite safely
- ESA: hundreds of alerts per satellite per week, about 2 needing an analyst, more than 1 manoeuvre per satellite per year, and a final decision about 1 d before TCA (ESA pages above).
- NASA: decisions are made 0.5–1 d before TCA. More conservative metrics would raise manoeuvres ×8–9, which CARA calls unbearable (CARA Vol 2, App. F and M.5).
- Starlink: 207,152 avoidance manoeuvres in 6 months (Dec 2025–May 2026) at a 3e-7 threshold (SpaceX FCC report via press).
- 2009 Iridium‑33/Cosmos‑2251 and 2007 FY‑1C: their debris is still on orbit today. CelesTrak still tracks 1,980 FY‑1C, 584 COSMOS‑2251 and 109 IRIDIUM‑33 fragments as of 30 Oct 2026.
- Kelvins: 97 ML teams, and only 12 beat "use the latest CDM's risk" (Uriot 2022).
- I avoided long verbatim quotes. A short public-domain line from NASA for the slide: CARA's purpose is "to take prudent measures, at reasonable cost, to improve safety of flight" (Vol 2 App. M.1, a US government work).

---

## 6. Visualisation conventions: what an expert expects

### 6.1 CARA's own display flow (Vol 2 App. Q), and whether the Kelvins data supports it
| # | Display | What it shows | Kelvins support |
|---|---|---|---|
| Q.2 | Primary / secondary "vital statistics" tables | period, perigee/apogee, inc, EDR, RCS, tracks/day, obs in span, time of last obs, WRMS, ballistic coeff., SRP coeff. | **Yes**: `*_h_per/h_apo/j2k_inc/sedr/rcs_estimate/obs_used/time_lastob_*/weighted_rms/cd_area_over_mass/cr_area_over_mass`. Colour WRMS against 1.5 / 2.0 / 5.0. |
| Q.3 | Ground trace with point of closest approach, v_rel and approach angle | geography of the TCA | **Partial**: latitude and relative-velocity azimuth/elevation only. Longitude is unknown, so it must be **labelled "illustrative longitude"**. |
| Q.4 | **Conjunction-plane plot** | miss vector on x, HBR circle, 1σ/2σ/3σ combined-covariance ellipses | **Yes, reconstructed** (§3.5). Collapse onto the x-axis means in-track dominates. |
| Q.5 | **Pc time history** | Pc vs time-to-TCA, red threshold line, markers filled when the update had new tracking | **Yes**: `risk` vs `time_to_tca`. "New tracking" ≈ `time_lastob_end` changed between CDMs (inferred). Add a vertical MCP line at 1 d and a Kelvins cut line at 2 d. |
| Q.6 | Total miss-distance history | miss vs time | **Yes** |
| Q.6.1 | **RTN component miss with ±1σ** | radial / in-track / cross-track miss with shrinking σ | **Yes**: `relative_position_*` and √(t_σ² + c_σ²) per axis. This is an approximation for the chaser, so label it. |
| Q.6.2 | Normalised consistency plot | Δmiss / previous σ, ±1σ band, about 68 % inside | **Yes**, computable between consecutive CDMs |
| Q.7 | Space weather (Dst forecast, −75 nT line) | storm risk to TCA | **Partial**: `AP`, `F10`, `F3M` per CDM. Dst is absent. Use ap ≥ 48 (≈ Kp 5, NOAA G1) as a storm flag (standard Kp↔ap table, https://kp.gfz-potsdam.de/en/; my mapping, not a CARA number). |
| Q.7.2 | Drag-scale trade-space heatmap | log10 Pc vs ±0.5 log primary/secondary drag scale; "ridge" means safe to dismiss | **No real data**. Possible only as a labelled sensitivity sketch. |
| Q.8 | **Maneuver trade-space heatmap** | Pc after a burn vs Δv size × burn time, with 1e-7 and 3e-6 contours | **Approximate**: a tangential Δv applied Δt before TCA shifts the in-track position by about −3 Δv·Δt (secular Clohessy–Wiltshire/Hill term) → shift the miss in the B-plane → recompute Pc. Label it "first-order". Do not present it as a maneuver plan. |

### 6.2 Other industry conventions
- **Risk-evolution plot:** this is the Kelvins / Uriot Fig. 13 style, log10 risk vs time_to_tca with linear interpolation between CDMs and −6/−4 threshold lines.
- **3D views:**
  - STK colours the primary and secondary **error ellipsoids green → yellow → red** as the objects approach (AGI Conjunction Analysis spec sheet, https://www.agi.com/getmedia/a8841ba8-e6d0-457e-b6e5-a8f9a7897645/Conjunction-Analysis-Product-Specsheet.pdf).
  - LeoLabs offers a free 3D globe of LEO objects ("last 24 hours") and a conjunction visualisation stream, `platform.leolabs.space/visualizations/conjunctions/stream` per a 2019 talk, https://www.nasa.gov/wp-content/uploads/2019/10/space_portal_michael_nicolls.pdf. **Current state UNVERIFIED.**
  - CelesTrak **SOCRATES** lists the top public conjunctions by max probability and minimum range (https://celestrak.org/SOCRATES/).
- **Colour semantics:** CARA uses green (any), yellow (> 1e-7) and red (> 1e-4). We can map those to the brief's Safe, Elevated and Critical tiers, and show the thresholds as lines, not just colours.

### 6.3 What a judge or expert would look for on our screen
1. Pc shown with its uncertainty: a dilution badge, MaxPc and the k* value.
2. A data-quality verdict with its reason: default covariance, WRMS above the limit, stale last observation.
3. The time axis anchored on TCA, with the MCP line.
4. The B-plane with the HBR circle.
5. The rule that fired and the CDM it came from.
6. A "human decides" control at the MCP, with the trade-off between waiting and committing.

---

## 7. Public data to make the globe live

### 7.1 CelesTrak GP (OMM)
- **Query:** `https://celestrak.org/NORAD/elements/gp.php?GROUP=<name>&FORMAT=json`.
  - FORMAT can be TLE/3LE, 2LE, XML, KVN, JSON, JSON-PRETTY or CSV. **The default has been CSV since 2026‑05‑09**, so always pass `FORMAT=json`.
  - Other query keys are CATNR, INTDES, NAME and SPECIAL (GPZ, GPZ-PLUS, DECAYING) (https://celestrak.org/NORAD/documentation/gp-data-formats.php).
- **Groups:** `active`, `stations`, `visual`, `last-30-days`, `analyst`, `starlink`, `oneweb`, `kuiper`, `qianfan`, `gps-ops`, `weather`, `resource`, `sar`, `planet`, `cubesat`.
  Debris groups are **`fengyun-1c-debris`, `iridium-33-debris` and `cosmos-2251-debris`**. `cosmos-1408-debris` is **not** listed any more (https://celestrak.org/NORAD/elements/).
- **Counts I pulled (one request each, 31 Oct 2026; epochs up to 2026‑10‑31):**
  | group | objects | epoch spread |
  |---|---|---|
  | active | 16,846 (16,037 with mean motion > 11.25 rev/d, i.e. LEO-ish) | 2026‑10‑06 to 10‑31 |
  | starlink | 11,157 | 10‑27 to 10‑30 |
  | fengyun-1c-debris | 1,980 | 10‑01 to 10‑30 |
  | cosmos-2251-debris | 584 | 10‑04 to 10‑30 |
  | iridium-33-debris | 109 | 10‑09 to 10‑30 |
  | stations | 20 | 10‑30 |
  Raw sizes: active.json 7.1 MB, starlink.json 4.7 MB. Ship them gzipped or slim the fields.
- **OMM JSON fields:** `OBJECT_NAME, OBJECT_ID, EPOCH, MEAN_MOTION, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY, EPHEMERIS_TYPE, CLASSIFICATION_TYPE, NORAD_CAT_ID, ELEMENT_SET_NO, REV_AT_EPOCH, BSTAR, MEAN_MOTION_DOT, MEAN_MOTION_DDOT`. The mandatory OMM fields CENTER_NAME, REF_FRAME, TIME_SYSTEM and MEAN_ELEMENT_THEORY are omitted in JSON.
- **TLE is no longer enough:** CelesTrak ran out of 5-digit catalogue numbers on **2026‑07‑11**. The SATCAT is at 101227, and the maximum NORAD ID in `active` is 101227, so new objects have **no TLE**. Use OMM.
- **Usage policy (2026‑05‑22, https://celestrak.org/usage-policy.php):**
  - Download only what you need, **once per update**. GP data updates every 2 h; SOCRATES every 10–11 h.
  - Repeat requests get **HTTP 403** until the next update.
  - More than 50 errors (301/403/404) in 2 h can get the IP **firewalled**.
  - Use `.org`, not `.com`.
  - Stop on any non-200 response and tell a human.
  - The page has no explicit licence, attribution or redistribution terms. **Attribute "Data: CelesTrak (celestrak.org)" in the UI anyway.**
- **Design rule:** commit a snapshot (e.g. `data/celestrak/2026-10-31/{group}.json`) and have the server refresh it at most every 2 h behind a cache. The demo must never depend on live fetches; this is a robustness point.
- **Space-Track** also offers GP and needs a free account (https://www.space-track.org). Its CDMs need operator credentials. Do not use it in the demo.

### 7.2 SGP4 in the browser
- **satellite.js 7.1.0** (MIT, released 2026‑07‑23, npm) has `json2satrec(omm)` ("preferred for new applications"), `twoline2satrec`, `propagate(satrec, date)` (returns `null` on failure; see `satrec.error`), `gstime`, `eciToEcf` and `eciToGeodetic` (https://unpkg.com/satellite.js@7.1.0/README.md).
- Pin the version exactly. 7.1.0 is more than two weeks old, so it is acceptable for the artifact CDN rule. For the 3D side, three is at 0.187.0 (released 2026‑10‑30, too new to pin; use ≤ 0.184.0) and globe.gl at 2.46.2 (2026‑08‑22).
- **Performance:** about 17k objects × SGP4 per frame is heavy on the main thread. Propagate in a Web Worker at 1–2 Hz and interpolate linearly per frame. Use one `THREE.Points` or `InstancedMesh` buffer, with colour by group (debris clouds tinted). Engineering suggestion, not measured.
- **Honesty rule (CARA App. E):** SGP4/TLE has no covariance and is too noisy for CA. It is fine for the *ambient* globe, **never** for Pc.
  The conjunctions shown with Pc come from the Kelvins CDMs on reconstructed geometry, and the UI must label them so ("illustrative orbit reconstructed from sma/ecc/inc + conjunction latitude; true RAAN unknown").

### 7.3 Placing a Kelvins event on the globe (documented reconstruction)
- **Target orbit:** a = `t_j2k_sma`, e = `t_ecc`, i = `t_j2k_inc`.
  - Argument of latitude at TCA: u = asin(sin φ / sin i), where φ = `geocentric_latitude`. Choose the ascending or descending branch from the sign of the radial/normal velocity; if that is unknown, hash `event_id`.
  - RAAN: a deterministic hash of (split, event_id), **labelled synthetic**.
- **Chaser orbit:** passes within `miss_distance` of the same point with velocity v_c = v_t + v_rel (in target RTN → ECI).
  Its plane normal is r × v_c. **Check:** the resulting inclination should match `c_j2k_inc`. Show the residual as a reconstruction-quality badge.
- Animate both orbits so they converge on the TCA marker, and draw the B-plane inset (§6.1 Q.4) at the conjunction point.

---

## 8. Open items and things I could not verify
- CARA Vol 1 (Best Practices) was blocked (403), so I quote no Vol 1 text. The thresholds above come from Vol 2 App. F and Q.
- 18 SDS criteria print as "e^-4" and "e^-7" in the extracted PDF text. I read these as 1e-4 and 1e-7, which is consistent with CARA's usage.
- Total Space-Track CDM volumes are not verified.
- The values of the SANA `COLLISION_PROBABILITY_METHOD` registry were not read.
- How ESA derived `*_span` and `*_rcs_estimate` is unknown. HBR convention: calibrate it against `risk`.
- The Magpies Fig. 12 counts (8530/23/70/269) come from a garbled PDF figure, so they are approximate.
- The Chan series was written from memory and checked numerically, not against the book.
- LeoLabs' current public conjunction visualisation was not checked live.

## 9. Source list
- NASA CA Handbook Rev 2 Vol 2 (Aug 2026): https://www.nasa.gov/wp-content/uploads/2026/08/ca-handbook-volume2-nasa-sp-20205011318-rev2-vol2.pdf
- NASA CA Handbook Rev 2 Vol 1 record: https://ntrs.nasa.gov/citations/20260000453
- NASA CARA SDK: https://github.com/nasa/CARA_Analysis_Tools
- Uriot et al. 2022: https://arxiv.org/abs/2008.03069 · https://doi.org/10.1007/s42064-021-0101-5
- Kelvins data page: https://kelvins.esa.int/collision-avoidance-challenge/data/
- ESA Kelvins announcement: https://www.esa.int/Enabling_Support/Space_Engineering_Technology/AI_challenged_to_stave_off_collisions_in_space
- ESA Automating collision avoidance: https://www.esa.int/Space_Safety/Space_Debris/Automating_collision_avoidance
- ESA Reentry and collision avoidance: https://www.esa.int/Space_Safety/Space_Debris/Reentry_and_collision_avoidance
- ESA Anatomy of a debris incident: https://www.esa.int/Enabling_Support/Operations/Anatomy_of_a_debris_incident
- ESA CREAM: https://www.innovationnewsnetwork.com/space-traffic-management-enters-a-new-era-with-esas-cream-project/60737/
- CCSDS 508.0-B-1: https://ccsds.org/Pubs/508x0b1e2c2.pdf
- 18/19 SDS Spaceflight Safety Handbook v1.7: https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf
- TraCSS: https://www.space.commerce.gov/tracss/ · CDM spec v2.1: https://www.space.commerce.gov/wp-content/uploads/2026/01/TraCSS-Spec-001-v2.1_CDM.pdf
- Starlink manoeuvre counts: https://www.altagrove.com/grovewire-news/spacex-148696-starlink-collision-maneuvers-in-6-months-ending-nov-30-continued-issues-with-operators-slipshod-reporting · https://space.com/space-exploration/satellites/every-spacex-starlink-satellite-has-to-dodge-a-collision-almost-weekly-and-experts-fear-the-worst
- Δv studies: https://publications.gc.ca/collections/collection_2016/rddc-drdc/D68-6-113-2013-eng.pdf · https://www.ideals.illinois.edu/items/139700
- Planet CA (SmallSat 2016): https://swfound.org/media/205611/smallsat2016_vittaldev_swf.pdf
- LeoLabs: https://leolabs.space/conjunction-alerts/ · https://www.nasa.gov/wp-content/uploads/2019/10/space_portal_michael_nicolls.pdf
- AGI/Ansys STK CA: https://help.agi.com/stk/Content/stk/ObjectMap_AdCAT.htm · https://www.agi.com/getmedia/a8841ba8-e6d0-457e-b6e5-a8f9a7897645/Conjunction-Analysis-Product-Specsheet.pdf
- CelesTrak: https://celestrak.org/NORAD/documentation/gp-data-formats.php · https://celestrak.org/NORAD/elements/ · https://celestrak.org/usage-policy.php · https://celestrak.org/SOCRATES/
- satellite.js 7.1.0: https://unpkg.com/satellite.js@7.1.0/README.md · https://www.npmjs.com/package/satellite.js
- Kp/ap: https://kp.gfz-potsdam.de/en/
