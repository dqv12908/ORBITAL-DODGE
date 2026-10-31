# ORBITAL DODGE — backend

FastAPI, one process. SIM clock + CDM replay → physics Pc → rules → forecast → policy gate → decision model →
policy verdict → human approval → explanation, all on one SSE bus that also feeds a hash-chained SQLite ledger.

## Run

```bash
cd server && .venv/bin/uvicorn app.main:app --port 8000          # live keys from ../.env if present
cd server && C1_MOCK_AI=1 .venv/bin/uvicorn app.main:app --port 8000   # offline demo: mock decisions, template text
```

Boot takes ~5–30 s (loads `data/processed/*.parquet` and the model). The boot scenario (`late-cdm`) is loaded **paused**.
Vite proxies `/api` to `:8000`. If `game/dist` exists it is served at `/`.

Tests (backend suites, ~60 s; the engine owns `test_pc*`, `test_rules`, `test_forecast`):

```bash
cd server && .venv/bin/python -m pytest -q tests/test_policy.py tests/test_verifier.py tests/test_ledger.py \
  tests/test_providers.py tests/test_replay.py tests/test_contract.py tests/test_api.py
```

Regenerate the SSE contract schema after changing `app/schema.py`:
`cd server && .venv/bin/python -m app.schema` → `shared/schema/events.schema.json` (a test fails if it is stale).

## Environment (`C1_` prefix, read from the repo-root `.env`; keys also accepted without the prefix)

| Var | Default | Meaning |
|---|---|---|
| `C1_OPENAI_API_KEY` | – | OpenAI Decisions API (`gpt-6-luna`). Absent → provider skipped |
| `C1_OPENROUTER_API_KEY` | – | Jev via OpenRouter `typesafe/jev-router` (k=5 samples → **vote shares**) |
| `C1_JEV_API_KEY` | – | alternative key for the same Jev provider |
| `C1_ANTHROPIC_API_KEY` | – | Claude `claude-sonnet-5-5` explanations (verified); absent → template only |
| `C1_MOCK_AI` | auto | `1` forces mock decisions + template text. Auto-mock when no decision provider key is set |
| `C1_REPLAY` | 0 | `1` serves decisions only from `var/cache/decisions` (miss → mock + `system.degraded`) |
| `C1_LLM_ENABLED` | 1 | `0` disables the LLM even with a key |
| `C1_SEED` | 1729 | default seed when a scenario has none |
| `C1_DECISIONS_TIMEOUT_S` / `C1_DECISIONS_BUDGET_S` | 5 / 12 | per-provider timeout / whole-chain budget |
| `C1_LLM_TIMEOUT_S` | 8 | Claude timeout (1 retry) |
| `C1_JEV_SAMPLES` | 5 | Jev votes per decision |
| `C1_APPROVAL_HOLD_MAX_S` | 45 | real seconds a pending approval may hold the SIM clock |
| `C1_BOOT_SCENARIO` | late-cdm | scenario loaded (paused) at boot |
| `C1_DB_PATH` | `var/orbitaldodge.sqlite` | ledger / idempotency / approvals / policy versions |
| `C1_CACHE_DIR` | `var/cache` | `decisions/` and `llm/` response caches (by request hash) |
| `C1_DATA_DIR` | `data` | root of `processed/` and `raw/` |

Keys are `SecretStr`; `/api/health` shows only `ok | down | unconfigured | configured (disabled)`.

## Modules

| Path | Job |
|---|---|
| `app/main.py` | app, lifespan (build context, load boot scenario, start SimLoop), CORS for :5173, JSON 404 catch-all |
| `app/settings.py` | pydantic-settings |
| `app/sse.py` | bus: envelope v1, seq, 2,000-event ring buffer for `Last-Event-ID`, 15 s `: ping`, snapshot on connect; appends ledger rows **before** broadcasting |
| `app/simloop.py` | SimClock (4 Hz, 1× = 3 sim-min/s, 2/4/8×, holds), CDM release at `tca_sim − ttca`, `outcome.revealed` at TCA, the per-CDM pipeline, KPIs, state |
| `app/engine_adapter.py` | the only door into `riskengine` + artifacts; every capability has a fallback (raw CSV / local Pc port / local rules / persistence forecast / built-in 2563 fixture) |
| `app/scenarios.py` | `shared/scenarios/*.json` + built-in fallback catalogue |
| `app/approvals.py` | pending → approved / rejected / expired (superseded by a newer CDM or TCA reached) |
| `app/api/{sim,ops,data}.py` | REST |
| `app/schema.py` | pydantic discriminated union of all SSE envelopes → JSON Schema |
| `decisions/policy.py` | gate (allowed actions) + verdict AUTO/ESCALATE/BLOCK with verbatim reason strings |
| `decisions/expected_cost.py` | the cost-rule baseline (loss table = stated assumptions) |
| `decisions/provider.py` | chain OpenAI Decisions → Jev router → mock, opaque choices `o1..oN`, disk cache, 12 s budget |
| `decisions/{openai_decisions,jev,mock}.py` | providers |
| `llm/{factsheet,templates,verify,explain,ask}.py` | fact sheet, template text, number verifier, Claude with `{{placeholders}}`, ask-the-desk |
| `ledger/ledger.py` | SQLite WAL hash chain + side tables |
| `registry/registry.py` | model list/card, sha256 check, eval merge |

## Honesty notes the UI should keep

- Forecast source per CDM (`forecast.updated.data.p_source`): **out-of-fold** value at the T−2 d CDM of training events;
  live model only for events it never trained on (test split, non-eligible); otherwise persistence-empirical.
  In-sample model values on training events are never shown (they are overfit: train:2563 in-sample 0.88 vs OOF 0.005).
  `registry/forecaster_bakeoff/predictions.parquet` is preferred at the T−2 d CDM when present (`model` names it).
- `decision.made.probability_kind`: `vote share (n=5)` for Jev (no native probabilities; routed model + upstream
  provider of every sample in `routed`), `mock: softmax(-E[loss]/2) of the cost rule` in mock mode. Never a collision probability.
- Explanations: template first (always), then `source: "llm"` only if the verifier passed.
