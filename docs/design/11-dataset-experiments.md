# 11 — Dataset experiments (grounding the design in numbers)

Status: real experiments run on 31 Oct 2026 on `train_data.csv` (162,634 CDMs / 13,154 events) and `test_data.csv`
(24,484 CDMs / 2,167 events). Nothing under `Downloads` or the reference repo was touched.

**Scripts** (all run with `venv/bin/python -I <script>`), outputs saved next to them:

| Part | Script | Raw output |
|---|---|---|
| shared loader, tiers, sentinel cleaning, event features | `SCRATCH/exp/w2/common.py` | — |
| 2D Pc implementation | `SCRATCH/exp/w2/pc2d.py` | — |
| A target definition | `SCRATCH/exp/w2/a_target.py` | `SCRATCH/exp/w2/out/a_target.txt` |
| B baselines + GBM | `SCRATCH/exp/w2/b_baselines.py` | `SCRATCH/exp/w2/out/b_baselines.txt` |
| C physics | `SCRATCH/exp/w2/c_physics.py`, `c_span.py`, `c_final.py`, `c_physics2.py` | `SCRATCH/exp/w2/out/c_*.txt` |
| D live feed / calendar | `SCRATCH/exp/w2/d_timeline.py`, `d_concurrency.py`, `d_calendar.py` | `SCRATCH/exp/w2/out/d_*.txt` |
| E orbits | `SCRATCH/exp/w2/e_orbits.py` | `SCRATCH/exp/w2/out/e_orbits.txt` |
| F stories | `SCRATCH/exp/w2/f_stories.py`, `f_stories2.py` | `SCRATCH/exp/w2/out/f_stories*.txt` |

`SCRATCH` = `<scratchpad>`.
(Note: a second, concurrent run of this same task wrote files into `SCRATCH/exp/` directly; the numbers in this
document all come from the isolated `SCRATCH/exp/w2/` copies.)
Libraries: Python 3.13, pandas 3.0.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1. LightGBM was installed but cannot load on
this Mac (`libomp.dylib` missing), so every model is sklearn `HistGradientBoosting*` (300 iterations, lr 0.05, 31 leaves, min 20 samples per leaf, `random_state=0`).

External reference used for the Kelvins setup and metric: Uriot et al., "Spacecraft Collision Avoidance Challenge: design and
results of a machine learning competition", arXiv:2008.03069 (https://arxiv.org/abs/2008.03069), sections 3–5.

---

## Headline findings (read this if nothing else)

1. **The `risk` column can be recomputed almost exactly from physics.** A standard 2D Pc (encounter-plane projection
   of the combined covariance, chaser covariance rotated into the target RTN frame, integrated over a disc) with
   **hard-body radius = (t_span + c_span)/2** (no fitting) reproduces `risk` with a **median error of 0.059 dex**
   (0.006 dex for risk ≥ −4, 0.013 dex for risk in [−6, −4)). Pc ≥ 1e-4 threshold: recall 0.918, precision 0.945.
   Brief-tier agreement 98.97 % of rows. So the "evaluator" can be **physics first**, and the dataset validates it.
2. **The 13 "allowed" columns cannot compute Pc.** With only the radial sigmas, an isotropic 2D Pc has Spearman 0.34
   against `risk` (vs 0.94 for full covariance) and is off by hundreds of orders of magnitude, because the
   dominant uncertainty is along-track (`c_sigma_t` median ~4 km vs `c_sigma_r` median 68 m). A GBM on the 13
   columns does better (MAE 3.7 dex) but still 6x worse than physics. Design consequence: read the 13 columns as
   "what the dashboard shows", not as a hard limit on the evaluator. We use full covariance for Pc and say so.
3. **Persistence is a strong baseline.** The tier at the last CDM ≥ 2 days out equals the final tier for 92.9 % of
   events. Critical recall 0.820 at 113 false alarms (see B for the ML comparison).
4. **The brief's tiers are dominated by miss distance, not Pc.** Of 551 critical final events (Kelvins-eligible),
   549 are critical because miss < 1 km and only 6 because Pc ≥ 1e-4. 571 of 905 events whose final miss is
   < 1 km have Pc ≤ 1e-10 (331 are at the −30 floor). The rules disagree, and the UI has to show why.
5. **Live-feed replay works:** CDMs arrive every ~8 h (median 8.07 h, 74 % of gaps 6–10 h), ~3 per day, for up to 7
   days. Space-weather values (F10, F3M, SSN) act as a **day fingerprint**: about 1 change per event-day, and a
   relative calendar solved from them has lag-1 autocorrelation 0.999 for F3M (random order: −0.03).
6. **3D orbits can be rebuilt and checked.** Put the target on a circular orbit through `geocentric_latitude`, add
   the RTN relative state, and the derived chaser inclination matches `c_j2k_inc` with a **median error of 0.041°**
   (99.9 % within 1°). Its semi-major axis matches with a 12 km median error. `azimuth` = atan2(v_n, −v_t) and
   `elevation` = −asin(v_r/|v|) of the relative velocity, exactly.

---

## A. Target definition (Kelvins replication on train)

Setup, following Uriot et al. §3: inputs are the CDMs with `time_to_tca ≥ 2` d. The target is the risk (and miss distance)
of the **final** CDM. Eligible events have ≥1 CDM at ≥ 2 d and a final CDM at < 1 d.

| Quantity | Value |
|---|---|
| Eligible train events | **8,293** (of 13,154) |
| Latest input CDM time_to_tca | median 2.19 d (IQR 2.09–2.29, max 6.95) |
| Final CDM time_to_tca | median 0.165 d (≈ 4 h), IQR 0.08–0.26 d, min −0.144 d (after TCA) |
| (a) **High risk, final risk ≥ −6** | **66 events = 0.80 %** |
| Final risk at the −30 floor | 6,838 (82.5 %) |
| Latest input CDM already ≥ −6 | 308 |

**Risk ≥ −6, latest input CDM vs final CDM (counts):**

| latest ≥2 d \ final | < −6 | ≥ −6 |
|---|---|---|
| < −6 | 7,966 | **19** (late escalations) |
| ≥ −6 | **261** (de-escalations) | 47 |

The naive forecast catches 47 of 66 high-risk events (recall 0.71), with 261 false positives.

**(b) Brief tiers (final risk AND final miss distance):**

| Tier | Final (target) | % | Latest ≥ 2 d CDM |
|---|---|---|---|
| Safe | 6,138 | 74.0 % | 6,040 |
| Elevated | 1,604 | 19.3 % | 1,688 |
| Critical | **551** | 6.6 % | 565 |

What drives each tier (final): critical comes from risk ≥ −4 in **6** events and from miss < 1 km in **549** (4 meet both).
Elevated comes from risk in [−6, −4) in 34 events and from miss 1–5 km in 1,593.
For all 13,154 train events (not only eligible ones) the final tiers are 9,843 / 2,403 / 908.

**Tier transition matrix, latest ≥ 2 d CDM → final CDM (counts, row %):**

| from \ to | safe | elevated | critical |
|---|---|---|---|
| safe (6,040) | 5,885 (97.4 %) | 145 (2.4 %) | 10 (0.2 %) |
| elevated (1,688) | 235 (13.9 %) | 1,364 (80.8 %) | 89 (5.3 %) |
| critical (565) | 18 (3.2 %) | 95 (16.8 %) | 452 (80.0 %) |

Tier changed in **592 events (7.1 %)**: 244 went up and 348 went down. Miss distance from latest to final: median change +8 m,
IQR −331…+387 m, 5–95 % range −4.1…+4.1 km, median |log10 ratio| 0.015.

Test split, for comparison (latest CDM only; truth is hidden): tiers 1,481 / 500 / 186, latest risk ≥ −6 in 178 events, latest time_to_tca 2.00–6.87 d.
Test has 18 missions. Mission 13 appears in train only, so a leave-mission-out split is the right check of generalisation.

---

## B. Baselines on grouped holdouts

Protocol: 5-fold out-of-fold predictions on the 8,293 eligible events.
"By event" = `KFold(5, shuffle, seed 0)` over events (each event is one row, so this is grouped by `event_id`).
"By mission" = `GroupKFold(5)` on `mission_id`, so whole missions are held out.
Kelvins metric (Uriot et al. eq. 1): F2 on the class `final risk ≥ −6`; MSE_HR computed only over true high-risk events with both values clipped at −6.001; L = MSE_HR / F2 (lower is better).

Feature sets (one row per event, built from the CDMs at ≥ 2 d):
- **A13**: the 13 allowed columns. That means latest values; deltas (last−previous, last−first); min, max and std; and the slope versus time_to_tca for miss, speed, r/t/n, sigmas and max_risk_estimate. It adds derived log-miss, combined sigma_r and |r|/sigma_r, plus n_cdm and first_ttca, for 84 features. **`risk` is not used** (it is not among the 13).
- **A13+risk**: A13 plus the risk history (91 features).
- **ALL**: latest value of all 100 numeric columns, plus trends for risk, mahalanobis, all sigmas, covariance determinants, obs_used and weighted_rms (233 features). mission_id is excluded.
The regressor is trained on max(final risk, −7). The tier classifier is a 3-class HGB, unweighted and also with `class_weight="balanced"`.

B_RESULTS_PLACEHOLDER

---

## C. Physics: recomputing Pc from the covariance

### C.1 Sentinel cleaning (63,781,363)
- 63,781,363 m = 10 × Earth radius (6,378,136.3 m). It is a "covariance unknown" placeholder, not a measurement. It appears as
  exactly `63781363.0` and `63781360.91` in **all three position sigmas of one object at once** (t-side 2,486 rows, c-side 2,652
  rows, both sides 9 rows). That is **5,129 train rows (3.2 %) across 1,313 events**. 481 events have it in their final CDM; 259 events have it in every CDM. Test has 659 such rows.
- On those rows `risk` is pinned to a narrow, meaningless band: mean −14.31, sd 0.34, range [−16.0, −13.4]. This is not
  −30 and not a real Pc. Most are PAYLOAD chasers (3,549 of 5,129).
- `c_sigma_t` has 24 more rows between 1e7 and 6e7 m (up to 76,548,834 m), which is also physically meaningless.
- **Rule** (`common.clean`): any position sigma ≥ 1e7 m → NaN, and set `cov_unknown = 1`. In the product, Pc for such a CDM is
  **"not computable — covariance unknown"**, never the dataset's −14.x. The tier then falls back to miss distance alone and
  carries an explicit "uncertainty unknown" badge. Velocity sigmas (`*_sigma_rdot` etc.) go up to 79,054 m/s, which is also unphysical.
  They are not used.

### C.2 2D Pc implementation (`pc2d.py`)
1. Build the 3×3 RTN position covariance per object from `x_sigma_r/t/n` and the correlation terms `x_ct_r, x_cn_r, x_cn_t`
   (verified in [−1, 1], so they are correlations, not covariances).
2. **Rotate the chaser covariance into the target RTN frame.** Each CDM gives each object's covariance in its own RTN frame. The chaser
   velocity in target RTN = (0, v_t, 0) + relative_velocity, with v_t from vis-viva on `t_j2k_sma`. Then chaser R ≈ target R,
   N_c = R × v_c, T_c = N_c × R.
3. C = C_t + C_c. The encounter plane is ⟂ relative velocity. Check: relative position ⟂ relative velocity
   (cos between them in [−0.0045, 0.0047] for 98 % of rows), and |rel_pos| = miss_distance to 3.5e-5.
4. Project C onto the plane (2×2) and integrate the bivariate normal centred at the miss vector over a disc of radius HBR.
   Use Gauss–Legendre (24 radial) × 48 angular nodes, summed in log space (logsumexp) so log10 Pc stays exact down to −300.

### C.3 Agreement with `risk` (157,468 clean rows)

| Variant | Pearson (risk > −30, floored) | Spearman | MAE dex (risk ≥ −10) | Median abs err dex | Implied HBR |
|---|---|---|---|---|---|
| no frame rotation, HBR 5 m | 0.751 | 0.821 | 5.99 | — | ~5 m |
| rotation, HBR 5 m | 0.902 | 0.927 | 1.18 | — | 3.71 m |
| rotation, HBR 10 m | 0.900 | 0.927 | 1.40 | — | 3.68 m |
| rotation, HBR 20 m | 0.894 | 0.925 | 1.86 | — | 3.61 m |
| rotation, single fitted HBR 3.71 m | 0.902 | 0.927 | 1.16 | 0.663 | — |
| rotation, per-mission fitted HBR (split-half) | — | 0.940 | 0.69 on held-out | 0.061 | 1.0–16.7 m |
| **rotation, HBR = (t_span + c_span)/2, no fitting** | **0.908** | **0.939** | **0.741** | **0.059** | — |

- Frame rotation is essential: MAE drops from 5.99 to 1.18 dex.
- The per-mission fitted HBRs (e.g. mission 1: 11.6 m, mission 18: 14.8 m, missions 22–24: 1.05 m) track the median `t_span`
  per mission (21 m, 27.7 m, 0.1 m). `t_span`/`c_span` are documented by Uriot et al. as "size of the target satellite used by
  the collision risk computation" [m]. `c_span` is 2 m for 75 % of rows and 3 m median for PAYLOAD chasers. **HBR = (t_span + c_span)/2
  has bias −0.000 dex, so it is the dataset's own convention.** The single fitted HBR of ~3.7 m is meaningless; use spans.

Final variant, by risk band:

| risk band | n | MAE dex | median abs err | bias |
|---|---|---|---|---|
| (−30, −20) | 10,594 | 3.71 | 0.63 | +0.01 |
| [−20, −10) | 33,868 | 2.20 | 0.41 | +0.01 |
| [−10, −6) | 35,594 | 0.77 | 0.08 | 0.00 |
| [−6, −4) | 9,467 | 0.68 | 0.013 | 0.00 |
| **[−4, 0)** | **716** | **0.144** | **0.006** | 0.00 |

Threshold agreement: at Pc ≥ 1e-6, TP 9,332 / FN 851 / FP 952 (recall 0.916, precision 0.907). At Pc ≥ 1e-4, TP 657 / FN 59 / FP 38
(recall 0.918, precision 0.945). Brief-tier agreement per row: 98.97 % (safe↔elevated swaps 729 + 845; elevated↔critical 25 + 26;
no safe↔critical). Values that hit the −30 floor agree on 98.0 % of rows.

**Where it disagrees** (|err| > 1 dex: 4,571 rows, 10.0 % of risk ≥ −10):
- **Far, tail-dominated encounters**: miss > 20 km has MAE 2.5 dex, against **0.065 dex for miss ≤ 5 km**. Tiny Pc values come from
  Gaussian tails and are sensitive to every approximation; operationally they do not matter.
- **Missions 6 and 7** hold 28.5 % of the bad rows but are 10.7 % of all rows (MAE 5.63 and 2.13). Example: event 4200 (mission 6)
  has risk −4.99 at miss 56.9 km with c_sigma_t 105.8 km, where our Pc is ~1e-479. ESA's value there implies a much larger effective covariance
  (maybe a covariance scaling or a different method). We cannot reproduce it and say so.
- **Mission 14** has bias −2.25 dex (11 events only). Its t_span (0.63 m) is probably not what was used.
- **PAYLOAD / ROCKET BODY chasers**: MAE 4.75 / 3.65 dex, against 0.74 for DEBRIS and 0.47 for UNKNOWN.
- Without missions 6 and 7: MAE 0.414 dex, median 0.051, 60.9 % of rows within 0.1 dex.

### C.4 Crude Pc from the 13 allowed columns only
- Isotropic 2D Gaussian with σ² = t_sigma_r² + c_sigma_r², Pc ≈ HBR²/(2σ²)·exp(−d²/2σ²): **Spearman 0.34, Pearson 0.19,
  MAE ≈ 6,168 dex, floor agreement 0.51**. At −6 it finds 933 of 10,183 high-risk rows. It fails because the miss vector lies mostly
  along T/N, where the sigma is km-scale (`c_sigma_t` median 12.8 km among risk ≥ −10 rows), not the radial tens of metres.
- The dilution upper bound Pc_max = HBR²/(e·d²) (miss only) correlates 0.535 with `max_risk_estimate`. 99.65 % of rows have
  risk ≤ max_risk_estimate, so `max_risk_estimate` is the dilution maximum, an upper bound.
- Learned crude model (row-level HGB on the 13 columns including max_risk_estimate, GroupKFold by event): MAE 3.73 dex,
  Spearman 0.69, recall at −6 4,276/10,183. **That is about 5x the physics MAE (0.74), and recall at −6 is 0.42 versus 0.92 for physics.**
- A hybrid (13 columns + the physics Pc as a feature) reaches MAE 0.46 dex and Spearman 0.97. Physics carries the signal.

---

## D. Replaying CDMs as a live feed

| Quantity | Value |
|---|---|
| CDMs per event | median 13, IQR 5–20, p90 21, max 23. 225 events have a single CDM |
| First CDM time_to_tca | median 6.60 d (p25 4.63, p95 6.92) |
| Last CDM time_to_tca | median 0.24 d. 9,463 events end < 1 d; 391 end after TCA (ttca < 0) |
| **Spacing between consecutive CDMs** | **median 8.07 h**, mean 9.06, IQR 7.0–9.2 h, p95 15.0, p99 30.3. 74.4 % within 6–10 h, 0.1 % < 1 h, 2.1 % > 24 h |
| Spacing vs horizon | flat: 8.05–8.09 h median in every band (<1 d, 1–2, 2–4, 4–7 d) |
| Events per mission | 1: 2,322 · 5: 2,253 · 15: 1,769 · 7: 1,765 · 2: 1,497 · 6: 1,108 · 19: 671 · 3: 486 · 9: 410 · 4: 265 · 10: 163 · 18: 120 · 16: 115 · 20: 70 · 22: 43 · 24: 40 · 23: 38 · 14: 11 · 13: 8 |
| CDMs with 0 ≤ ttca < 1 d | 25,623 (9,445 events) |
| Events whose final CDM is inside the last 24 h | 9,463. Final tier safe 6,999 / elevated 1,841 / critical 623. 621 have miss < 1 km and 2,440 have miss < 5 km |

**No absolute epochs exist.** Uriot et al. §3 state that timestamps were converted to time-to-TCA and event ids were randomised. But
**space weather is a usable day fingerprint**:
- 1,220 distinct (F10, F3M, SSN) tuples, or 1,496 with AP. Within an event the tuple changes **0.96 times per event-day** (median),
  so these values are stamped per CDM creation day.
- Treating each tuple as a node and each within-event change as the constraint `pos(b) − pos(a) = Δcreation time` gives one
  connected graph (51,671 constraints). The least-squares solution has residual median 0.30 d, and along that order F3M has lag-1
  autocorrelation 0.999 (shuffled control −0.03). F3M runs from 157 down to 70, matching the solar cycle 24 decline. The solved span (~490 d) is
  **shorter than the 2015–2019 collection period**, probably because identical tuples on different dates merge (solar minimum). So treat it as
  an ordering hint, not a calendar. Matching to a public F10.7 history (e.g. CelesTrak SW-All.csv, https://celestrak.org/SpaceData/)
  could date events. That was not tried (no download made).
- Concurrency proxy (same day-key, approximate): live events per mission-day median 3 (p90 13). Across all missions, median 33 events
  per day-key. "Critical-looking" events (any CDM with miss < 1 km or risk ≥ −4) median 2 per day-key, p90 7.

### Proposed 24 h operations-day synthesis (what is real vs synthetic)
1. **Pick a cast.** Choose one operator, i.e. one mission (default mission 1, 2,322 events; or mission 15, which holds the best
   stories), plus 30–40 events: about 25 routine, 8–10 elevated and 3–5 from the story list in F. This matches the measured ~33 live events
   per day-key and ~2–7 critical-looking ones.
2. **Synthetic TCA epochs (SYNTHETIC).** Sim day D0 = 00:00–24:00 "SIM UTC". Give each event a TCA_e = D0 + u_e, with u_e in
   [+2 h, +7 d], so the window contains early-warning, mid-life and final-day events. Spread the story events so that one
   decision lands every ~3–4 sim-hours.
3. **CDM arrival times (REAL relative timing).** Each CDM is created at TCA_e − time_to_tca×24 h, using the dataset's own spacing.
   During the day the feed emits the CDMs whose creation time falls in [D0, D0+24 h], about 3 per event. At 8x speed that is one CDM every ~2–4 s on
   screen. Every row shows `source: train event 2563, CDM 3/3, ttca 0.44 d`.
4. **"24 h danger window" (REAL rule on SYNTHETIC clock).** At sim time τ, count the events with TCA_e − τ < 24 h whose latest CDM has
   miss < 1 km, or < 5 km for an "approach" count. This is the brief's "how many come within the dangerous distance".
5. **Labelling.** Every synthetic quantity (TCA wall-clock, the order of events across days, globe longitude/RAAN from E) carries a
   `SIM` tag in the UI and in the ledger. Real quantities (risk, miss, sigmas, ttca, spacing, F10/AP) carry `CDM`.
   Optionally, space-weather day ordering (above) can sort events across days with honest wording.

---

## E. Orbit reconstruction for the 3D globe

What the columns give (verified on the 13,154 final CDMs):
- `t_/c_ j2k_sma/ecc/inc`: shape and tilt. `h_apo = a(1+e) − 6378.137` and `h_per = a(1−e) − 6378.137` exactly (error 0.000 km),
  so they add nothing beyond sma/ecc. Targets: sma 6,675–7,211 km, ecc ≤ 0.021, inc 87–99° (polar/SSO LEO).
  Chasers: 393 events have sma > 8,000 km and 447 have ecc > 0.1 (HEO/GTO debris passing through LEO).
- **No RAAN, argument of perigee, mean anomaly or epoch.** The orbit plane orientation and the phase are unknown.
- `geocentric_latitude` = latitude of the conjunction (|lat| ≤ both inclination limits in 100 % of events).
- **`azimuth` = atan2(v_rel,n, −v_rel,t)** (corr +1.000, MAE 0.00°) and **`elevation` = −asin(v_rel,r/|v_rel|)** (MAE 0.001°). So
  these are the approach direction of the relative velocity in the target's local frame, not a ground-station look angle.
- The altitude bands [h_per, h_apo] of target and chaser overlap (within 5 km) in 100 % of events, which is consistent.

**Algorithm (`e_orbits.py`, validated):**
1. Conjunction point P: latitude φ = `geocentric_latitude`, longitude λ = **synthetic** (seeded by event id or chosen for camera
   framing), radius r = a_t (circular approximation; ecc ≤ 0.021 gives ≤ ±150 km error, typically ±15 km).
2. Target heading at P: sin ψ = cos i_t / cos φ (ψ from north). There are two branches, northbound (ψ) and southbound (π − ψ).
   Speed comes from vis-viva. This gives the target state (r_t, v_t) in an ECI frame anchored at P, and RAAN and argument of latitude follow.
3. Chaser state = target state + RTN→ECI(relative_position, relative_velocity). This is **exact relative geometry from the CDM**.
4. Pick the branch that reproduces the chaser's `c_j2k_inc` and `c_j2k_sma`, then convert both states to Keplerian elements
   and propagate two-body (Kepler) ±1 orbit for drawing. Both objects pass through P at TCA by construction, with the right miss vector
   and crossing angle.
5. **Validation:** the derived chaser inclination matches `c_j2k_inc` with **median error 0.041°, p90 0.121°, 99.9 % < 1°**. Derived chaser sma
   matches with median error 12.1 km (p90 64.8 km, 81 % < 50 km), and ecc with median error 0.0011. The branch choice matters: the wrong branch gives an 18.8° median
   inclination error. Both branches fit within 1° in only 5.8 % of events. The target inclination is reproduced in 99.4 % of events.

**Honest limits** (show in the UI as "reconstructed geometry"):
- Longitude, RAAN and the absolute time are synthetic. The ground track is plausible, not the real one, and Earth rotation phase is arbitrary.
- The circular-orbit radius for the target causes the chaser sma residual (median 12 km). A 1-D search over the conjunction radius in
  [r_per, r_apo] and over the flight-path angle would reduce it. This was not done.
- Two-body propagation only (no J2/drag). It is fine for ±1 revolution of animation, but not for days. The relative geometry is exact only at TCA.
- The miss vector (tens of metres to tens of km) is invisible at globe scale. The 3D view needs a zoomed "encounter-plane" inset
  with the covariance ellipse, and that inset is fully data-driven from C.2.
- Chasers on HEO/GTO (393 events with sma > 8,000 km) produce long ellipses that need camera handling.

---

## F. Story events for demo scenarios (train, event_id → trajectory as (time_to_tca d, risk log10 Pc, miss km))

**(1) Late escalation, safe → critical.** Event **2563** (mission 15, DEBRIS, 3 CDMs):
(2.43, −6.32, 8.05) → (1.41, −6.31, 1.63) → (0.44, **−4.19, 0.14 km**). Safe at the 2-day decision point, critical by miss at the end.
Alternatives: **5253** (mission 2) sits at −30 for 4 days, jumps to −17.8 at 2.6 d, then to **−4.80 at 1.6 d** and ends at −5.17 / 1.26 km.
**2115** (mission 15): −15 to −17 until 2.25 d, then −6.21 → −5.28 / 1.72 km. **8767** (mission 15, steady climb):
(6.67, −8.57, 2.59) → (5.59, −6.69, 1.53) → (4.59, −5.45, 1.05) → (3.59, −3.43, 0.12) → … → (0.63, **−2.69, 0.04 km**). Final Pc ≈ 2e-3, the second-highest final Pc in train.
No eligible event goes from all-early risk ≤ −8 to final ≥ −5 *without* a miss-distance warning, so a fully hidden escalation does not occur in the data.

**(2) Looks scary early, falls to −30** (388 eligible events have early max ≥ −5 and final −30). **2427** (mission 18, DEBRIS, 21 CDMs):
(6.85, −4.01, 0.29) … (5.18, **−1.44, 0.04 km**) (4.86, −1.49, 0.04) → (4.39, −15.35, 0.14) → … → (2.76, **−30, 0.25**) … (0.24, −30, 0.41).
This is the highest risk in the whole train set, and it collapses. **7465** (mission 14, PAYLOAD): −4.5 for 5 days, −1.83 at 2.17 d, then −10.8 at 1.85 d and −30 from 0.70 d,
with miss 0.07–0.19 km throughout. **10378** (mission 7): oscillates between −2.0 and −28.7 before settling at −30 / 0.67 km. It is a noisy-feed showcase.

**(3) Miss < 1 km but tiny Pc (tier-rule disagreement).** 905 train events end with miss < 1 km. 806 of them have Pc < 1e-6, 571 have Pc ≤ 1e-10 and 331 sit at −30.
**8756** (mission 4, DEBRIS): miss 0.02–0.12 km for all 21 CDMs, while risk goes from −2.46 (4.30 d) down to −19.33 / −18.27 at the end.
So miss < 1 km makes it critical, and the probability says it is negligible. **4797** (mission 2): ends (0.02, −16.29, 0.08 km). **2142** (mission 5, PAYLOAD): (0.44, −15.42, 0.08 km).
Converse (Pc ≥ 1e-4 at the final CDM, all 13): 1148, 8767, 8863, 12555, 9383, 3555, 9051, 5185, 6341, 11180, 5773, 9361, 12506. 9361 has
Pc 1.2e-4 at **3.93 km** miss, so it is critical by Pc while outside the 1 km rule.

**(4) Sentinel / huge covariance.** 1,054 events contain some sentinel rows and 259 contain only sentinel rows. **2299** (mission 7, PAYLOAD):
c_sigma_r = SENTINEL for 13 CDMs, with risk frozen at −14.29 while miss drifts 1.87 → 33 km. Then a real covariance arrives (c_sigma_r 29.1 m at
1.58 d, then 15.0, 12.6, 7.3 m) and risk drops to −30. This is the ideal "uncertainty unknown → known" demo. **2759** (mission 7): sentinel from CDM 2
onward, risk pinned at −14.40, miss ~30 km. Largest non-sentinel c_sigma_t: event 5393 (9,739 km, UNKNOWN, final miss 3.9 km, risk −8.3).

**(5) High space weather.** Row AP: median 6, p99 44, max 108. F10: median 72, p99 135, max 246. **6623** (mission 1, DEBRIS):
AP 12 → **108** → 47, F10 117→114, with CDMs (5.66, −4.90, 17.93) (4.66, −5.78, 10.43) (3.23, −4.75, 11.22). Elevated by Pc (≥ 1e-6) at an 11 km miss.
**3280** (mission 3): AP 70, 70, 30, 13. Risk −7.06 → −5.36. F10 = 246 (max) in events 8696 (final −7.60, 1.82 km), 6461 and 690.
**Honest note:** over eligible events, corr(max AP, |final − latest risk|) = 0.001. Storms do not visibly move risk in this
data, so the space-weather story is context ("drag uncertainty up"), not a predictor.

**(6) Physics disagreement, to show as a failure case.** **4200** (mission 6): (6.89, −4.99, 56.88 km) (6.54, −5.29, 34.44) (6.27, −30, 56.87).
ESA's risk is −4.99 at 57 km miss. The full-covariance 2D Pc is ~1e-479. The evaluator should flag "reported risk inconsistent with covariance".

**Test set** (no truth, good for "live" predictions): the highest latest risk events are test 891 (mission 1, 2.18 d, −2.89, 503 m),
1568 (mission 3, −2.99, 148 m), 1448 (mission 2, −3.04, 104 m), 2059 (mission 4, −3.10, 60 m) and 2091 (mission 2, −3.14, 236 m). Key test events
as `test:<id>`, because ids collide with train.
