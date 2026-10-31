# Cost ratio R = C_miss / C_fa for the CRITICAL alert cut-off

Status: **predetermined.** Written 31 Oct 2026, before any sweep result was produced or read. Only the label file
(`data/labels.parquet`: final tier, final risk, final miss) was read, to get the collision probability of events that end CRITICAL.
No model output, OOF file or score was opened.

**Recommendation: R = 10.** The alert rule is: flag CRITICAL at T−2 d when the expected cost `R·FN + FP` is lowest.
On a calibrated `p_critical`, that is the same as flagging when `p_critical ≥ 1/(1+R) = 0.091`.

| | R_low | R_central | R_high | Recommended |
|---|---|---|---|---|
| Value | **2.4** (Monte Carlo P10) | **6.0** (point central; MC median 5.1) | **10.3** (MC P90) | **10** |

---

## 1. What the two errors mean in our product

The product looks at an event at the **T−2 d decision point**, using only CDMs with ttca ≥ 2 d. It flags the event if it predicts
that the event will end CRITICAL. The brief's tier on the final CDM decides CRITICAL: risk ≥ −4 **or** miss < 1 km.

- **False alarm (FP):** the event is flagged CRITICAL at T−2 d, but its final CDM is ELEVATED or SAFE.
  Operationally, an analyst spends time reviewing it. The analyst may ask 18/19 SDS (or a commercial radar) for extra tracking. A share of these events
  move on to **manoeuvre planning**, as in the Swarm-B case where a "team of teams" planned the burn and then aborted it at go/no-go.
  A small share end in an **unnecessary executed manoeuvre**, which costs an instrument outage.
  A false alarm **does not** cause a collision. The events that end non-critical carry almost no risk: their mean final Pc is
  1.8e-7 (ELEVATED) and 3.5e-9 (SAFE).
- **Missed critical (FN):** the event ends CRITICAL but was not flagged at T−2 d. The operator's own screening still finds it
  later, from the CDMs inside the last 2 days. So a miss is **late discovery, not a collision.** The cost is the difference between a planned
  response and a rushed one: overtime and on-call staff, analysis compressed into less than a day before the manoeuvre commitment point (MCP, 0.5–1 d before TCA),
  about 2× the Δv, an outage that cannot be scheduled around acquisitions, and possibly a restoration burn.
  There is also a **small chance that the late response cannot be completed in time.** In that case the event goes unmitigated and the satellite
  carries its collision probability.
- A correct alert (TP) and a correct non-alert (TN) are the baseline. TP review work happens in both worlds, so it does not count as a cost of the miss.
  C_miss and C_fa are therefore **incremental** costs.

## 2. Evidence

| Quantity | Value used | Source |
|---|---|---|
| Loaded engineer/analyst labour rate | $80/h (median $80k salary × overhead 2) | NASA OTPS 2023, *Cost and Benefit Analysis of Orbital Debris Remediation*, "Operator Cost Model" |
| Analyst time per high-interest warning | 1 h average for civil science (half automated, 2 h otherwise); 2 h for civil operational; 2 h for stations | OTPS 2023, App. A (Civil Science Long; Civil Operational; Human Spaceflight) |
| Labour per executed manoeuvre | ~10 h (civil); ~100 h for a crewed station | OTPS 2023, App. A |
| Lost ops per manoeuvre | ~10 min of science at $28M/yr ($53/min) → about $530 | OTPS 2023, Table 18 |
| Outage in practice | ESA: a 15 s manoeuvre once interrupted spacecraft operations for 8 h. Instruments are switched off for CAMs | ESA case study for UNOOSA (2021); ESA "The cost of avoiding collision" (2021) |
| Propellant | CAM burns are usually posigrade, so they count towards orbit maintenance. Propellant cost ≈ 0 | OTPS 2023, App. A |
| Late vs early Δv | the late option needed 2× the Δv; 0.01–0.05 m/s is enough at ≥ 20 h; cost rises steeply inside 24 h | CARA Handbook Rev 2 Vol 2 App. Q.8; DRDC 2013 (via `12-domain-research.md` §1.6) |
| Alert volume | hundreds of alerts per satellite per week; ~2 per satellite per week need follow-up with "hours of analysis"; > 1 manoeuvre per satellite per year | ESA "Automating collision avoidance" |
| Manoeuvre rate | 3–4 CAMs per mission per year | ESA "The cost of avoiding collision" (2021) |
| Late-response feasibility | Swarm-B: alert ~39 h out, commands uplinked ~15 h out, go/no-go ~12.5 h out, **only two ground-contact slots** | ESA "Anatomy of a debris incident" |
| ESA funds work on late commanding | CREAM-2, €800k, "late commanding paths" for late CA decisions | Astroscale/ESA press release (2022) |
| Satellite value (LEO EO) | Sentinel-1A €280M (1B ~half); Sentinel-3C/3D €225M each; launch €52–82M | SpaceNews contract reports (search summaries) |
| Loss per collision used by NASA | civil science $270M hardware + $140M lost ops; civil operational $820M | OTPS 2023, Table 5 |
| Pc of events that end CRITICAL (our data) | **mean final Pc = 6.1e-6.** Only 6 of 551 have risk ≥ −4; 549 are critical because miss < 1 km. 18 have Pc ≥ 1e-5 and 52 have Pc ≥ 1e-7. The median is at the 1e-30 floor. Σ Pc over all 551 = 0.0034 | `bakeoff/data/labels.parquet` (final CDM; train events only) |
| A real red event | Pc ≈ 1e-4 is the action threshold (NASA/ESA) | CARA Vol 2 App. F; ESA |

USD and EUR are treated as equal. That is within ±10 %, which is small next to the ranges below.

## 3. Cost model

```
C_fa   = h_review·rate + track + f_plan·h_plan·rate + f_exec·(h_plan·rate + outage)
C_miss = h_rush·rate·k_ot + f_mnvr·late_extra + p_late·Pc·L
R      = C_miss / C_fa
```

| Parameter | Low | Central | High | Basis |
|---|---|---|---|---|
| rate (€/h, loaded) | 80 | 100 | 150 | OTPS $80/h. European ops-centre contractor rates are higher (**assumption**) |
| h_review (h per false flag) | 1 | 3 | 8 | OTPS 1–2 h per warning; ESA "hours of analysis" |
| track (€ per extra tracking request) | 0 | 100 | 500 | 18/19 SDS tasking is free, so this is staff time. The high value allows for a commercial radar (**assumption**, no public price) |
| f_plan (share of FPs that reach manoeuvre planning) | 0 | 0.2 | 0.5 | Swarm-B planned and then aborted (**assumption**) |
| h_plan (h) | 10 | 10 | 30 | OTPS 10 h per manoeuvre; Swarm multi-team |
| f_exec (share of FPs that end in an executed CAM) | 0 | 0.02 | 0.10 | operators decide at the MCP on fresher CDMs (**assumption**) |
| outage (€ per executed CAM) | 530 | 25,000 | 46,000 | OTPS 10 min @ $28M/yr; ESA 8 h interruption @ $28M/yr and @ $50M/yr |
| h_rush (extra person-hours, late vs planned) | 10 | 30 | 60 | Swarm timeline; CARA MCP 0.5–1 d (**assumption**) |
| k_ot (overtime / on-call multiplier) | 1.0 | 1.5 | 2.0 | **assumption** |
| f_mnvr (share of late criticals that still need a CAM) | 0.05 | 0.2 | 0.5 | most criticals are miss-only with tiny Pc (our data) |
| late_extra (€ extra cost of a late vs early CAM) | 500 | 10,000 | 50,000 | 2× Δv (CARA Q.8); unscheduled outage; restoration burn (CARA M.5) |
| p_late (share of misses not mitigated in time) | 0.02 | 0.10 | 0.30 | Swarm had 2 contact slots; CREAM exists for this problem (**assumption**) |
| Pc (collision probability if unmitigated) | 1e-6 | 6.1e-6 | 1e-4 | data mean; high = red threshold, covering Pc dilution and covariance realism (CARA App. M) |
| L (€ loss per collision) | 150M | 410M | 1,000M | Sentinel-1B; OTPS civil science $270M + $140M; OTPS civil operational $820M plus debris externality |

### Results

| Scenario (all parameters at that level) | C_fa | C_miss | R | C_miss breakdown: rush / late CAM / risk |
|---|---|---|---|---|
| Low (cheap, automated) | €80 | €830 | 10.3 | 800 / 25 / 3 |
| **Central** | **€1,120** | **€6,750** | **6.0** | 4,500 / 2,000 / 250 |
| High (expensive, high-value) | €9,000 | €73,000 | 8.1 | 18,000 / 25,000 / 30,000 |

**Monte Carlo** (200k draws, each parameter drawn independently from a triangular low/central/high distribution, on a log scale for the multiplicative ones, seed 20261031):
R percentiles P5 / P10 / P25 / **P50** / P75 / **P90** / P95 = 2.0 / **2.4** / 3.5 / **5.1** / 7.4 / **10.3** / 12.5.
P(R > 50) = 1e-5 and P(R > 100) = 0. The nearest grid value is 5 in 72 % of draws, 10 in 25 %, 20 in 3 %, 50 in < 0.1 % and 100 in 0 %.

**One-at-a-time sensitivity** (the others held at central). Two parameters move R most:
- f_exec: R goes from 11.3 to 2.1. Unnecessary executed manoeuvres are the main false-alarm cost.
- late_extra: R goes from 4.3 to 13.2.

Pc and L have little effect (5.8–9.5 and 5.9–6.3), **because a miss is late discovery and not a collision**.

**Stress case (not our label):** a genuine red event with Pc = 1e-4 that is never mitigated (p_late = 1) gives R ≈ 42.
That is the regime where R = 50 would apply. In this dataset only 6 of 551 criticals look like that.

**Sanity check on the base rate:** 551 critical and 7,742 non-critical events.
- At R < 14, "flag nothing" costs less than "flag everything". So at R = 10, a model earns its keep only if it is informative.
- At R ≥ 14, trivially flagging all events would already beat flagging none. This is a further reason not to choose R ≥ 20 on this label.

## 4. Recommendation

**R = 10** (alert when calibrated `p_critical ≥ 0.091`, or choose the cut-off that minimises `10·FN + FP` on training-fold OOF predictions).

Justification for a judge:
1. On public operator cost data (NASA OTPS labour, outage and loss values; ESA alert volumes and the Swarm-B timeline; CARA's 2× late Δv), a missed CRITICAL costs about €7k against about €1.1k for a false alarm, so R ≈ 6.
   This holds because missing it at T−2 d means a rushed late response, not a collision. Events that end CRITICAL in this dataset have a mean Pc of only 6e-6.
2. We take the **90th percentile** of the uncertainty distribution, R ≈ 10, instead of the median, because the user explicitly puts catching criticals ahead of false alarms.
   It also covers costs we did not put a price on: the debris externality to other operators, regulatory duty of care, and Pc under-estimation.
3. Going higher is not supported by evidence. R = 20 or more is reached in < 3 % of draws, and only under the assumption that missed events go unmitigated with red-level Pc.
   Also, at R ≥ 14 "flag every event" already beats "flag none" on this label, so the threshold would stop discriminating.

**Interaction with the recall target (95 %, floor 90 %).** R sets the cost-optimal cut-off. The recall floor is a separate **constraint**.
1. Report the recall at the R = 10 optimum.
2. If that recall is below 90 %, move the cut-off down to the lowest point that reaches 90 % recall, and report the extra false alarms this costs, as `ΔFP = 10·ΔFN_saved − Δcost`.
3. If the floor is not met, do **not** change R after seeing results, because R is frozen here.
4. Report R = 5 and R = 20 alongside as sensitivity rows.

## 5. Caveats
- The share parameters (f_plan, f_exec, p_late, h_rush, k_ot) are expert-style assumptions bounded by the cited anecdotes. No public dataset gives them.
- OTPS figures are US civil (2020 USD) and are used as a proxy for ESA. The Sentinel contract values come from search summaries of SpaceNews articles, not the primary documents.
- The brief's miss < 1 km rule makes "critical" much broader than a Pc ≥ 1e-4 "red" event (CARA App. M.2 / App. E).
  If the product switched to a Pc-only label, C_miss would grow (Pc ≈ 1e-4 per event), and R should be re-derived in the 20–50 range **before** looking at results.
- Alert fatigue is a real but unpriced false-alarm cost. CARA App. M.5 calls a ×10 rise in manoeuvres unbearable. Leaving it out biases R upward, which is conservative for catching criticals.

## 6. Sources
- NASA OTPS (2023), Cost and Benefit Analysis of Orbital Debris Remediation: https://www.nasa.gov/wp-content/uploads/2023/03/otps_-_cost_and_benefit_analysis_of_orbital_debris_remediation_-_final.pdf (text-extracted; Operator Cost Model, Table 5, Table 18, App. A)
- NASA CA Handbook Rev 2 Vol 2 (App. F, M, Q.8): https://www.nasa.gov/wp-content/uploads/2026/08/ca-handbook-volume2-nasa-sp-20205011318-rev2-vol2.pdf
- ESA, Automating collision avoidance: https://www.esa.int/Space_Safety/Space_Debris/Automating_collision_avoidance
- ESA, The cost of avoiding collision (2021): https://www.esa.int/ESA_Multimedia/Images/2021/02/The_cost_of_avoiding_collision
- ESA, Anatomy of a debris incident (Swarm-B): https://www.esa.int/Enabling_Support/Operations/Anatomy_of_a_debris_incident
- ESA case study for UNOOSA space sustainability (2021): https://spacesustainability.unoosa.org/sites/spacesustainability.unoosa.org/files/files/documents/2021/Sep/fnl_template_for_case_study_submissions_jan._2021_esa.pdf (quoted via search snippet; the direct fetch failed)
- ESA, The cost of space debris (OECD 5–10 % of mission cost): https://www.esa.int/Space_Safety/Space_Debris/The_cost_of_space_debris
- Astroscale / ESA CREAM-2 contract: https://www.astroscale.com/en/news/astroscale-awarded-800k-euro-esa-contract-to-boost-collision-avoidance-maneuver-capability-on-congested-orbital-highways
- Sentinel costs (SpaceNews): https://spacenews.com/?p=14910 · https://spacenews.com/?p=57589 · launch: https://europeanspaceflight.com/?p=6390
- DRDC Δv study: https://publications.gc.ca/collections/collection_2016/rddc-drdc/D68-6-113-2013-eng.pdf
- Project domain notes: `docs/design/12-domain-research.md` §1.4–1.7, §5
- Cost model: the formulas and parameters in §3 reproduce every number above. The script (numpy, run once) was
  `/private/tmp/claude-501/-Users-nguyenphuongdiep--claude-spares-spare-gJnPMY/55ec3528-3cbe-4220-b821-8753fcac3a6b/scratchpad/costmodel.py`, which is a temporary path.
