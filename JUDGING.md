# Judging guide: criteria → evidence

Every number below is in the [README](README.md) and indexed in [`docs/FACTS.json`](docs/FACTS.json).

| Criterion (weight) | Claim | Where to verify |
|---|---|---|
| **Problem realism (20)** | The user is the duty conjunction analyst. Status quo: hundreds of alerts per satellite per week, about 2 needing hours of analysis (ESA). The cost ratio R = 10 is derived from evidence (central 6.0, MC P90 10.3) | [docs/design/40-final-design.md §1](docs/design/40-final-design.md), [docs/COST_RATIO.md](docs/COST_RATIO.md) |
| **Technical soundness (25)** | Physics computes Pc and rules compute tiers. ML only forecasts and adds attention. Jev picks among rule-allowed actions. The LLM only explains, and a number verifier checks it. 168 backend tests pass. Headless E2E runs in mock mode: 1,805 SSE events, 48 verdicts, 0 HTTP errors, ledger verify OK over 729 entries | `server/riskengine/`, `server/decisions/`, `server/llm/verify.py`; `make test`; [docs/evidence/e2e/summary.json](docs/evidence/e2e/summary.json) |
| **Honest evaluation (20)** | Baseline is persistence. Splits are by event (80/20, 5 seeds) and by held-out mission, with bootstrap CIs. The bake-off rule was written before scoring and sha256-pinned (the order is our attestation, README §5.3). Leakage audits, including a noise-substitution test. Macro-F1 is a tie; persistence wins the final ≥ 2 d slice. The false-alarm study ended in a tie | README §5–§6, [docs/BAKEOFF.md](docs/BAKEOFF.md), `research/organizer/results.json`, `research/fa/`, `research/bakeoff/audit/` |
| **Trust & traceability (20)** | Every tier shows the rule that fired. Pc is refused when the covariance is a default. Forecasts are always out-of-fold for training events. Jev's per-action probabilities and its separate confidence field are labelled as decision-model numbers, not collision probabilities. Hash-chained ledger | `GET /api/ledger/verify`, ops.log, approval card, `server/tests/test_perf_honesty.py` |
| **Impact (10)** | Critical recall at T−2 d goes from 85.1 % to 93.7 %, cost per 1,000 events from 110.8 to 80.1 (−28 %, organizer 80/20). On held-out missions: 86.6 % → 94.8 % (722 → 791 of 834), cost 104.3 → 72.8 (−30 %). The price: false alarms 126 → 439, precision of a CRITICAL alert 85 % → 64 % | `research/organizer/results.json`; recompute with `server/.venv/bin/python research/organizer/recompute_table.py` |
| **Time discipline (5)** | 12 commits on 2026-10-31, 11:37 → 14:21, before the README; the submission time is the final commit. 3-minute demo script in README §13 | `git log` |
| **Failure cases** | Late escalation (`train:2563`: forecast 1.1 %, ends CRITICAL at 136 m), 43 of 834 criticals still missed 2 d ahead, Pc-driven criticals (`train:9361`), default covariance (`train:2299`), physics vs reported disagreement (`train:4200`) | Scenario picker: "The late CDM", "Rules disagree", "Covariance unknown", "Where we fail"; README §8 |

**Quick check without the dataset (1 min):** `server/.venv/bin/python research/organizer/recompute_table.py` prints the held-out-missions table (persistence 722 caught / 126 false alarms / 104.3 per 1,000 vs ORBITAL DODGE 791 / 439 / 72.8).

**Hands-on check (5 min):** `make dev`, open http://localhost:5173, then:
1. Pick **The late CDM** and run at 8×.
2. Watch the CRITICAL tier fire by miss distance and the approval card appear.
3. Open the inspector to see the encounter plane, risk fan and forecast source.
4. Open Dashboard → Ledger and check that verify shows ✓.

Without API keys the app runs in labelled mock mode (decisions = softmax of the cost rule, template text).
