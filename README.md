# ORBITAL DODGE

**Collision-avoidance triage for a satellite conjunction analyst.** Airbus track, Challenge 1, Autonomous SG 2026.

Every ~8 h a new batch of conjunction messages (CDMs) arrives. ORBITAL DODGE recomputes the collision probability from the covariance, applies the CARA/ESA tier rules, forecasts which events will still be **critical** at closest approach, and hands the analyst a suggested action with its evidence. A human approves every action. All of it runs inside a 3D Earth simulator.

## Headline results

Measured on the organizer's official labels (`challenge1_event_labels.xlsx`, 11,942 usable events). The baseline is **tier persistence**: the tier of the latest message ≥ 2 days out is taken as the final tier.

| Organizer 80/20 split by event (5 seeds) | Persistence | **ORBITAL DODGE** |
|---|---|---|
| Critical events caught | 85.1 % | **93.7 %** |
| Expected cost per 1000 events (10 × missed + false alarms) | 110.8 [93–127] | **80.1** [68–93] |
| False alarms per test set (~2,389 events) | **24.8** | 89.4 |
| AUC (critical) | 0.970 | **0.988** |
| Macro-F1 (3 tiers) | 0.895 | 0.895 |

- **Held-out missions** (a harder test): recall **94.8 % vs 86.6 %**, cost **72.8 vs 104.3**.
- **The trade-off is explicit.** We accept more false alarms to catch more criticals. The ratio of 10 false alarms per missed critical is derived from ESA/CARA cost evidence in [docs/COST_RATIO.md](docs/COST_RATIO.md).
- **Physics.** Our 2D Pc, computed from the full covariance, reproduces the dataset's `risk` with a median error of **0.054 dex** and **98.97 % tier agreement**. Our tier labels match the organizer's on all 13,154 events.

## What's in it

| Layer | Job | Evidence |
|---|---|---|
| **Physics** | 2D Pc from the CDM covariance; refuses when the covariance is a default (sentinel) | [`server/riskengine/pc.py`](server/riskengine/pc.py), golden tests |
| **Rules** | Brief tiers (Pc OR miss distance), data-quality gates, allowed actions per tier | [`server/riskengine/rules.py`](server/riskengine/rules.py) |
| **ML forecast** | P(final tier = CRITICAL) at T−2 d. It adds attention and never suppresses an alert | [`registry/forecaster_organizer/`](registry/forecaster_organizer/), [docs/BAKEOFF.md](docs/BAKEOFF.md) |
| **Decision model** | TypeSafe **Jev** (official API) picks among rule-allowed actions with per-action probabilities, compared against an expected-cost rule | [`server/decisions/typesafe.py`](server/decisions/typesafe.py) |
| **LLM** | Claude explains with placeholder numbers; a verifier rejects any number not in the fact sheet | [`server/llm/`](server/llm/) |
| **Ledger** | Hash-chained audit trail of every signal, decision and approval | `GET /api/ledger/verify` |
| **3D simulator** | Realistic Earth, ~19.5k CelesTrak objects, reconstructed focal pair, encounter plane, risk fan | [`game/`](game/) |
| **Live sky** | Our own GPU screening of today's catalogue (28,686 objects, 72 h), validated against CelesTrak SOCRATES | [research/livesky/REPORT.md](research/livesky/REPORT.md) |

## Model search (all on held-out data, with CIs)
- **Bake-off:** 8 model families under a pre-registered rule. LightGBM won narrowly over TabPFN, a GRU, risk futures (an MDN with 10k Monte Carlo paths) and EBM. See [docs/BAKEOFF.md](docs/BAKEOFF.md).
- **Cost sweep:** ~300 configs on Triologay (4 GPUs, 56 cores) using nested by-mission CV.
- **False-alarm push:** 13 further candidates (physics features, ensembles, per-tier thresholds). None beat the served model beyond noise. 1.1.0, which adds physics features, ties it and is kept as an alternative.
- **Risk futures and conformal alerts:** the 95 % target gives a realised recall of **94.9 %** on held-out missions. See [research/bigml/REPORT.md](research/bigml/REPORT.md).

## Where it fails (shown in the product)
- **Late escalations:** train:2563 is SAFE at T−2.4 d and CRITICAL at 136 m at T−0.44 d. Its out-of-fold forecast was 1.1 %. Physics catches it after the commitment point.
- **Pc-driven criticals are rare** (6 of 551), so the forecaster barely learns them (train:9361).
- **Default covariance (sentinel) rows:** Pc is refused and automation is blocked (train:2299).
- **Physics vs reported risk disagree** on missions 6/7 (train:4200). The UI flags the disagreement.
- **Reconstructed geometry:** the 3D orbits are plausible reconstructions and are labelled as such. The encounter plane and Pc use only CDM data.

## Run it
```bash
make setup        # Python venv + npm install
make precompute   # clean → physics Pc → features → train → scenarios (needs data/raw/, not in repo)
make dev          # backend :8000 + game :5173
```
Keys go in a gitignored `.env` (see `.env.example`). Without keys the system runs in labelled mock/template mode.

Data: the ESA Kelvins CDM dataset as provided by the organizers. It is not redistributed here.
