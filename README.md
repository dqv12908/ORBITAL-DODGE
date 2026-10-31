# ORBITAL DODGE 🛰️

### Two days before closest approach, ORBITAL DODGE catches **94.8 %** of the encounters that end critical. The standard baseline catches **86.6 %**.

*Measured on the organizer's official labels, on missions the model never saw in training.*

![ORBITAL DODGE during a simulated duty shift](docs/screenshots/readme/hero.png)

**In 10 seconds:**
- 🛰️ **The problem:** a satellite analyst gets hundreds of collision warnings a week. Almost all are noise, and the real decision has to be made about a day before the encounter.
- ⚡ **What we built:** physics, rules, ML and an AI decision model turn that flood into a short, ranked to-do list, with a recommendation and the evidence. All of it runs in a live 3D Earth simulator.
- 🧑‍🚀 **Who decides:** always the human. Every step lands in a tamper-evident ledger.
- 📈 **The result:** we beat the baseline on every headline number that matters to the operator.

Airbus track · Challenge 1 ("AI-driven satellite collision avoidance") · Autonomous SG 2026 · built in one day, 31 Oct 2026.

---

## 📈 We beat the baseline

The baseline is the standard rule: *whatever tier the event has two days out is its final tier.* It's harder to beat than it sounds: in ESA's Kelvins competition, only 12 of 97 teams beat its close cousin, the "latest risk" baseline. We scored both on the **organizer's official labels**, 11,942 events:

| | Baseline | **ORBITAL DODGE** |
|---|---|---|
| Critical events caught 2 days ahead (organizer's 80/20 split, 5 seeds) | 85.1 % | **93.7 %** |
| Critical events caught 2 days ahead (missions never seen in training) | 86.6 % | **94.8 %** |
| Criticals still missed, unseen missions (out of 834) | 112 | **43** |
| Expected cost per 1,000 events, unseen missions (one miss = 10 false alarms) | 104.3 | **72.8** (−30 %) |
| AUC, unseen missions | 0.969 | **0.988** |

The honest trade-off: **we raise more false alarms, on purpose.** Missing a real collision risk is far worse than an extra analyst review. We set that ratio (R = 10) from ESA and CARA cost evidence *before* looking at any model scores ([docs/COST_RATIO.md](docs/COST_RATIO.md)).

> 🔁 **Don't trust us, check it:** `server/.venv/bin/python research/organizer/recompute_table.py` rebuilds that table from the committed predictions in one command, with no data or training needed.

---

## 👀 What the analyst actually sees

### The inspector: the moment `train:2563` turns critical
![Inspector](docs/screenshots/readme/inspector.png)
Event `train:2563` looked safe two days out. Then, **10.5 hours before closest approach**, a new warning puts the debris **136 m** away. The inspector shows it all:
- **Encounter plane.** The debris's uncertainty cloud (1σ / 2σ / 3σ) against our satellite's 3.55 m hard body, zoomed ×135 so you can actually see it.
- **Risk evolution.** Our physics Pc for every warning, plus a fan of **10,000 simulated futures** from our risk-futures model.

### The decision card: only when a human has to decide
![Decision card](docs/screenshots/readme/approval.png)
- One headline and three evidence tiles.
- **One recommendation**, and how sure the AI is.
- The two strongest reasons it's asking you.
- A short note from the **AI advisor** in plain English.
- Buttons.

The full audit trail is folded under *Details*. In a live replay, the system handles **12 of 18** flagged decisions on its own. The 6 it brings to you are the 2 critical events and one genuinely ambiguous event.

### Statistics, out-of-fold only
![Statistics](docs/screenshots/readme/stats.png)
ROC (AUC 0.988), precision–recall, reliability, and the chart that drives everything: **expected cost against alert threshold at R = 10**, with our chosen cut-off and the baseline marked.

### Live sky: today's real catalogue, screened by us
![Live sky](docs/screenshots/readme/livesky.png)
We propagated **28,686 real objects** over the next 72 hours and screened every pair on our own GPU in about **2.5 minutes**. Given the same orbit data, we find **97 %** of the events in CelesTrak's SOCRATES reports.

<details>
<summary><b>More screens:</b> RTN chart, ops dashboard, ledger, analyst desk, scenario picker</summary>

![RTN](docs/screenshots/readme/inspector-rtn.png)
*Relative position split into radial, along-track and cross-track, with ±1σ bands. At T−2.4 d the along-track uncertainty was ±22.7 km, which is why "6.3 km apart" meant nothing yet.*

![Dashboard](docs/screenshots/readme/dashboard.png)
*The 24 h ops dashboard: events within 1 km and 5 km, the tiers, the verdicts and auto-generated insights.*

![Ledger](docs/screenshots/readme/ledger.png)
*The hash-chained ledger: every signal, decision and click, verified end to end (`CHAIN ✓`).*

![Desk](docs/screenshots/readme/desk.png)
*The analyst desk: the operator's policy in plain language, with the tier-mode switch and the live operating curve.*

![Scenarios](docs/screenshots/readme/scenarios.png)
*Nine scenarios, each teaching one thing, including one called "Where we fail".*
</details>

---

## ⚙️ How it works: one warning, ten steps

1. **Clean.** Catch broken data, such as the fake "unknown uncertainty" placeholder in about 3 % of rows.
2. **Physics computes Pc** from both objects' uncertainty clouds. It reproduces ESA's own values to a **median 0.06 on the log scale, with 98.97 % tier agreement**. On broken data it refuses instead of guessing.
3. **Rules assign the tier** with the brief's thresholds: the worse of the Pc rule and the miss-distance rule.
4. **ML forecast:** the chance this event ends CRITICAL. It can only *add* attention, never dismiss an alert.
5. **Gate.** Code decides which actions are even allowed. Safe and boring events are logged automatically.
6. **Two advisors.** A cost rule (expected loss) and **TypeSafe Jev**, a decision model that returns a probability for each action. Jev only ever sees the allowed actions.
7. **Verdict.**
   - **Automatic** only if the event isn't critical, Jev is at least 70 % sure, and both advisors agree.
   - **Escalate** to the analyst otherwise.
   - **Block** on broken data.
8. **Claude writes a 2–3 sentence advisor note.** Every number is checked against the facts. If one is wrong, a fixed template is shown instead.
9. **The analyst decides.**
10. **Ledger.** Every step is hash-chained, so tampering shows.

Physics computes, rules decide what's allowed, ML forecasts, AI advises, and a human signs off. No layer does another layer's job.

**Want the deep dive on the models?** Every experiment, number and lesson is in **[ML.md](ML.md)**.

---

## 🔥 The ML grind

![Triologay, our GPU server](docs/images/triologay-server.webp)
*Meet Triologay, the box that did the heavy lifting: 56 cores, 2× V100 16 GB, a 32 GB PG503 and an RTX 2060, reached over Tailscale.*

We didn't just train one model and hope. We ran **about 540 configurations**:

| Stage | What | Configs |
|---|---|---|
| Pre-registered bake-off | LightGBM, TabPFN, GRU, risk futures (10,000 Monte Carlo paths per event), EBM, HGB | 8 |
| Hyper-parameter tuning | LightGBM with Optuna | 120 |
| Nested cost sweep | Gradient boosters, CatBoost, GRU/LSTM/transformer, TabPFN, EBM, forests | 403 |
| False-alarm push | Physics features, ensembles, smarter thresholds | 13 |

**The bake-off.** Every model got the same false-alarm budget, and the missions were held out:

| Model | Critical events caught |
|---|---|
| 🥇 **LightGBM, all features** | **84.2 %** |
| 🥈 TabPFN | 83.8 % |
| GRU · risk futures · EBM | 83.1 % |
| Baseline | 82.0 % |

*These are before cost-threshold optimisation.*

**Winner:** LightGBM on all features. We retrained it on the organizer's 11,942 events and set the alert threshold by cost, which gives the headline numbers above.

**Bonus results** from the same server:
- **Conformal alerts:** at a 95 % recall target, 94.9 % on unseen missions.
- **An optimal decision policy** by backward induction.
- **Our own screening of the live catalogue.**

**What compute couldn't buy:** fewer false alarms. Extra physics features, ensembles and per-tier thresholds all came out as a statistical tie, and we report it that way. The data's ceiling is the alert threshold, not the model.

---

## 🧠 Things that surprised us

- **ESA's risk numbers are reproducible from first principles.** Our 2D Pc, from the covariance alone, lands within 0.06 on the log scale of ESA's own value.
- **"Critical" is mostly about distance, not probability.** 549 of 551 critical events are critical because the miss distance is under 1 km, often with a tiny Pc. The card shows both sub-tiers so the analyst sees the disagreement.
- **Our AI handed *everything* back at first.** Jev escalated 18 of 18 decisions. The cause was the policy, not Jev: our cost rule kept recommending tracking hours before the encounter, when tracking can no longer help. Fixing that brought it to **12 of 18 handled automatically**, with the right 6 still going to the human.
- **The organizer released official labels mid-day.** We re-scored everything against them, and our labels matched on **13,154 of 13,154** events.

---

## 🛡️ Trust, and where it fails

**Built-in honesty**
- Every tier names the rule that fired.
- Pc is refused on broken covariance.
- Training events only ever show **out-of-fold** forecasts. The in-sample value for `train:2563` would be 0.88; we show the honest 0.011.
- Decision-model numbers are never shown as collision probabilities.
- Critical events always go to a human, and "prepare manoeuvre" is never automatic.

**Where it fails, and how the analyst knows**
- **Late escalations still happen.** 43 of 834 criticals are missed 2 days ahead (the baseline misses 112). `train:2563` is one: it's safe at T−2 d and critical at 136 m. Physics catches it on the late warning, and the card shows the low forecast, so the miss is visible.
- **About 1 alert in 3 is false.** Precision is 64 %, against 85 % for the baseline. That's the price of catching more.
- **Broken covariance (`train:2299`):** Pc is refused and the event is blocked. **Physics vs ESA disagreement (`train:4200`):** it's flagged on the card.
- **Not yet proven:** that it beats an expert analyst. Jev hasn't been validated offline.

---

## 🧪 How we tested it

- **168 backend tests**, all passing. They cover:
  - physics property tests
  - golden tests against the dataset's risk
  - policy rules (critical is never automatic)
  - the LLM number and style checker
  - ledger tamper detection
  - replay determinism
- **Leakage audits.** We replaced every warning from the last 2 days with random noise and got bit-identical features. Shuffled labels gave chance-level AUC.
- **End-to-end runs** in headless Chrome against a real backend: 1,805 events streamed, 0 HTTP errors, ledger verified.
- **Live calls** to TypeSafe Jev (`jev-1.14.0`, about 0.3 s per decision) and Claude, with labelled mock mode when no keys are set.

Details are in [docs/REPORT.md §11](docs/REPORT.md#11-how-we-tested-it-locally).

---

## ✅ Rubric → evidence

| Criterion | Evidence |
|---|---|
| Problem realism (20) | Named user and workflow; cited status quo (ESA, CARA); cost ratio derived from evidence. [Report §1](docs/REPORT.md#1-the-user-and-the-problem) |
| Technical soundness (25) | Each layer does one job; Pc validated against ESA's values; **168/168 tests**; end-to-end runs with 0 HTTP errors. [Report §3–4](docs/REPORT.md#3-architecture-and-division-of-labour), [§11](docs/REPORT.md#11-how-we-tested-it-locally) |
| Honest evaluation (20) | Official labels, two splits, bootstrap CIs, pre-registered bake-off, leakage audits; ties and losses reported. [Report §5–6](docs/REPORT.md#5-machine-learning-the-full-process), [docs/BAKEOFF.md](docs/BAKEOFF.md) |
| Trust & traceability (20) | Refusals, out-of-fold forecasts, number-checked LLM, hash-chained ledger, human approval. [Report §7–8](docs/REPORT.md#7-trust-and-traceability) |
| Impact (10) | 791 vs 722 of 834 criticals caught on unseen missions; cost −30 %. [Report §12](docs/REPORT.md#12-impact-before-and-after-for-the-analyst) |

A 5-minute hands-on path for judges is in [JUDGING.md](JUDGING.md).

---

## 🚀 Run it

```bash
make setup        # Python venv + npm install
make precompute   # needs the organizer's CSVs in data/raw/ (not redistributed)
make test         # 168 backend tests
make dev          # backend :8000 + app :5173 — pick "The late CDM" and hit 8×
```

API keys go in a gitignored `.env` (see `.env.example`). Without keys, everything runs in a clearly labelled mock mode.

**Want every number with its source file?** The full technical report is **[docs/REPORT.md](docs/REPORT.md)**: data, physics, the full ML process, all audits, the repo map, limits and next steps.
