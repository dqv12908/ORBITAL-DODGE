# ORBITAL DODGE

**Collision-avoidance triage for the analyst who protects a satellite.** Airbus track, Challenge 1, Autonomous SG 2026.

Hundreds of collision warnings (CDMs) reach a satellite operator every week. Almost all are harmless. ORBITAL DODGE turns them into a short, ranked to-do list, with a suggested action and the evidence behind it. The human still makes the call.

![ORBITAL DODGE during a simulated duty shift](docs/screenshots/readme/hero.png)

## We beat the baseline

The baseline is the standard rule: *the tier two days before the encounter is the final tier.* We scored both on the **organizer's official labels** (11,942 events):

| | Baseline | **ORBITAL DODGE** |
|---|---|---|
| Critical events caught 2 days ahead (organizer 80/20 split, 5 seeds) | 85.1 % | **93.7 %** |
| Critical events caught 2 days ahead (missions never seen in training) | 86.6 % | **94.8 %** |
| Expected cost per 1,000 events (a miss = 10 false alarms) | 110.8 | **80.1** (−28 %) |
| AUC | 0.970 | **0.988** |

- **The trade is deliberate:** more false alarms, chosen with a cost ratio (R = 10) that we derived from ESA/CARA cost evidence before looking at scores ([docs/COST_RATIO.md](docs/COST_RATIO.md)).
- **Recompute it in one command** from committed predictions, with no data or training needed: `server/.venv/bin/python research/organizer/recompute_table.py`.

## What the analyst sees

| | |
|---|---|
| ![Inspector](docs/screenshots/readme/inspector.png) | **Inspector, at the moment it matters.** Event `train:2563` turns CRITICAL 10.5 h before closest approach (miss 136 m). **Encounter plane:** the debris position cloud (1σ/2σ/3σ) against our 3.55 m hard body. **Risk evolution:** the physics Pc for each warning, plus a fan of 10,000 simulated futures. |
| ![Decision card](docs/screenshots/readme/approval.png) | **The decision card**, shown only when a human must decide. One headline, three evidence tiles, **one recommendation** and how sure the AI is, the two strongest reasons, a short **AI advisor** note, and buttons. The full audit trail is folded under *Details*. |
| ![Statistics](docs/screenshots/readme/stats.png) | **Statistics, out-of-fold only.** ROC (AUC 0.988), precision–recall, reliability, and **expected cost vs threshold at R = 10**, with the chosen cut-off and the baseline marked. |
| ![Live sky](docs/screenshots/readme/livesky.png) | **Live sky.** Our own GPU screening of **today's real catalogue**: 28,686 objects over 72 h in about 2.5 min. Given the same orbit data, it finds 97 % of CelesTrak SOCRATES's events. |

More screens (RTN chart, dashboard, ledger, desk, scenarios) are in [docs/screenshots/readme/](docs/screenshots/readme/).

## How it works: one warning, ten steps

1. **Clean.** Flag broken data, such as the fake "unknown uncertainty" placeholder.
2. **Physics computes Pc** from both objects' uncertainty clouds. It matches ESA's own value to a median 0.06 on the log scale (98.97 % tier agreement). It refuses on broken data.
3. **Rules assign the tier**, using the brief's thresholds: the worse of the Pc rule and the miss-distance rule.
4. **ML forecast:** the chance this event ends CRITICAL. It can only *add* attention, never dismiss an alert.
5. **Gate.** Code decides which actions are allowed. A safe, low-risk event is logged automatically and nothing else happens.
6. **Two advisors.** A cost rule (expected loss) and **TypeSafe Jev**, a decision model that returns a probability for each action. Jev only sees the actions the rules allow.
7. **Verdict.**
   - **Automatic** only if the event isn't critical, Jev is at least 70 % sure, and both advisors agree.
   - **Escalate** to the analyst otherwise.
   - **Block** on broken data.
8. **Claude writes a 2–3 sentence advisor note.** Every number is checked against the facts; a fixed template is used if the check fails.
9. **The analyst decides.**
10. **Hash-chained ledger.** Every step is tamper-evident.

On a live replay, this policy handles **12 of 18** flagged decisions automatically, up from 0. The 6 it still escalates are the 2 critical events and one genuinely ambiguous event.

## How we chose the model

![Triologay, our GPU server](docs/images/triologay-server.webp)
*Triologay, our own server, reached over Tailscale: 56 cores, 2× V100 16 GB, a 32 GB PG503 and an RTX 2060.*

- **About 540 configurations in total:**
  - a pre-registered bake-off of **8 models**
  - **120** Optuna trials
  - a nested, cost-based sweep of **403 configs** (gradient boosters, CatBoost, GRU/LSTM/transformer, TabPFN, EBM, forests)
  - **13** false-alarm experiments
- **Bake-off results** (all models at the same false-alarm budget, missions held out):

  | Model | Recall |
  |---|---|
  | **LightGBM, all features** | **84.2 %** |
  | TabPFN | 83.8 % |
  | GRU, risk futures, EBM | 83.1 % |
  | Baseline | 82.0 % |

  These are before cost-threshold optimisation.
- **Winner:** LightGBM on all features, retrained on the organizer's 11,942 events, with the alert threshold set by cost. That gives the headline table above.
- **What compute didn't fix:** extra features, ensembles and smarter thresholds could not cut false alarms beyond noise, and we report that as a tie.

## Trust, and where it fails

**Trust**
- Every tier names the rule that fired.
- Pc is refused on broken covariance.
- Training events only ever show **out-of-fold** forecasts. The in-sample value for `train:2563` would be 0.88; we show the honest 0.011.
- Decision-model numbers are never shown as collision probabilities.
- Critical events always go to a human, and "prepare manoeuvre" is never automatic.

**Where it fails**
- **Late escalations:** 43 of 834 criticals are still missed 2 days ahead (the baseline misses 112). `train:2563` looks safe at T−2 d and turns critical at 136 m. Physics catches it on the late warning, and the card shows the low forecast, so the analyst can see the miss.
- **About 1 alert in 3 is false:** precision is 64 %, against 85 % for the baseline. That's the cost of catching more.
- **Default covariance (`train:2299`):** Pc is refused and the event is blocked.
- **Physics vs ESA disagreement (`train:4200`):** the disagreement is flagged on the card.
- **Not yet proven:** that it is safer than an expert. Jev has not been validated offline.

## Rubric → evidence

| Criterion | Evidence |
|---|---|
| Problem realism (20) | Named user and workflow; cited status quo (ESA, CARA); cost ratio derived from evidence. [Report §1](docs/REPORT.md#1-the-user-and-the-problem) |
| Technical soundness (25) | Code, physics, rules, ML and AI each do one job; Pc validated against ESA's values; **168/168 tests**; end-to-end runs with 0 HTTP errors. [Report §3–4](docs/REPORT.md#3-architecture-and-division-of-labour), [§11](docs/REPORT.md#11-how-we-tested-it-locally) |
| Honest evaluation (20) | Official labels, two splits, bootstrap CIs, pre-registered bake-off, leakage audits; ties and losses reported. [Report §5–6](docs/REPORT.md#5-machine-learning-the-full-process), [docs/BAKEOFF.md](docs/BAKEOFF.md) |
| Trust & traceability (20) | Refusals, out-of-fold forecasts, number-checked LLM, hash-chained ledger, human approval. [Report §7–8](docs/REPORT.md#7-trust-and-traceability) |
| Impact (10) | 791 vs 722 of 834 criticals caught on unseen missions; cost −30 %. [Report §12](docs/REPORT.md#12-impact-before-and-after-for-the-analyst) |

## Run it

```bash
make setup        # Python venv + npm install
make precompute   # needs the organizer's CSVs in data/raw/ (not redistributed)
make test         # 168 backend tests
make dev          # backend :8000 + app :5173
```

API keys go in a gitignored `.env` (see `.env.example`). Without keys, the app runs in a clearly labelled mock mode.

**Everything else** (every number with its source file, the full ML process, all audits, the local test runs, the repo map) is in the full technical report, **[docs/REPORT.md](docs/REPORT.md)**, with a 5-minute judge path in [JUDGING.md](JUDGING.md).
