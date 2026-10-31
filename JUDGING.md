# Judging guide: criteria → evidence

| Criterion (weight) | Claim | Where to verify |
|---|---|---|
| **Problem realism (20)** | The user is the duty conjunction analyst. Status quo: hundreds of alerts per satellite per week, about 2 needing hours of analysis (ESA). The cost ratio is derived from evidence | [docs/design/40-final-design.md §1](docs/design/40-final-design.md), [docs/COST_RATIO.md](docs/COST_RATIO.md) |
| **Technical soundness (25)** | Physics computes Pc and rules compute tiers. ML only forecasts and adds attention. Jev picks among rule-allowed actions. The LLM only explains, and a number verifier checks it | `server/riskengine/`, `server/decisions/`, `server/llm/verify.py`; 156 backend tests (`make test`) |
| **Honest evaluation (20)** | Baseline is persistence. Splits are by event (80/20) and by held-out mission, with bootstrap CIs. Pre-registered bake-off. Leakage audits, including a noise-substitution test. The false-alarm push reports "no significant gain" | README tables, [docs/BAKEOFF.md](docs/BAKEOFF.md), `research/organizer/results.json`, `research/fa/` |
| **Trust & traceability (20)** | Every tier shows the rule that fired. Pc is refused when the covariance is a default. Forecasts are always out-of-fold for training events. Decision probabilities are labelled as decision-model probabilities. Hash-chained ledger | `GET /api/ledger/verify`, ops.log, approval card, `server/tests/test_perf_honesty.py` |
| **Impact (10)** | Critical recall goes from 85.1 % to 93.7 % with expected cost down about 28 % (organizer 80/20); on held-out missions, 86.6 % to 94.8 % | `research/organizer/results.json` |
| **Failure cases** | Late escalation (train:2563), Pc-driven criticals (train:9361), default covariance (train:2299), physics vs reported disagreement (train:4200) | Scenario picker: "The late CDM", "Rules disagree", "Covariance unknown", "Where we fail" |

**Quick check (5 min):** `make dev`, open http://localhost:5173, then:
1. Pick **The late CDM** and run at 8×.
2. Watch the CRITICAL tier fire by miss distance and the approval card appear.
3. Open the inspector to see the encounter plane, risk fan and forecast source.
4. Open Dashboard → Ledger and check that verify shows ✓.
