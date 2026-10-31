# 13 — Technology research: 3D Earth simulator + robust backend

Status: research and recommendation only (no product code). Written 31 Oct 2026.
Inputs: `00-context.md`, the reference repo (`agentic-factory`: `game/`, `server/`, `shared/`, `docs/`), live npm/PyPI registry queries, local benchmarks on this machine, and the web sources cited inline.

How facts were checked:
- **[reg]**: version and publish date read from `registry.npmjs.org/<pkg>` or `pypi.org/pypi/<pkg>/json` on 31 Oct 2026.
- **[local]**: run on this Mac (Apple M2, Node v20.11.1, Python 3.13.15, uv installed, **no Docker**). Scratch code is in `SCRATCH/techcheck/`.
- **[web]**: an external page, with its URL.
- **[unsure]**: not verified. Treat it as a claim to check.

---

## 0. Final recommended stack

| Layer | Choice | Pinned version | Why (short) |
|---|---|---|---|
| Build tool | Vite | **6.4.5** | Vite 7.x/8.x need Node ^20.19 or >=22.12, and this machine has **Node 20.11.1** [reg][local]. Vite 6.4.5 supports Node ^18, ^20 and >=22 and was published 21 Oct 2026 [reg]. |
| 3D | three.js, `WebGLRenderer`, plain JS (no React) | **three@0.186.1** | Same stack as the reference repo (`three ^0.169`, hand-written `World.js`). 0.187.0 came out 30 Oct, one day ago, and **breaks the postprocessing peer range** (npm ERESOLVE, verified) [reg][local]. |
| Post-FX | pmndrs `postprocessing` | **6.39.5** | Merges Bloom, Outline, SMAA and Vignette into one pass. Peer dependency is `three >=0.168 <0.187` [reg]. |
| Camera | `camera-controls` (or the reference repo's own easing) | **3.1.2** | `setLookAt(..., true)` and `fitToSphere` give fly-to with damping. It only warns about Node>=22 at install (browser library) [local]. |
| Real-catalog propagation (optional layer) | `satellite.js` | **7.1.0** | SGP4 in JS plus a WASM bulk API. Benchmarks are in §2 [reg][local]. |
| Dev aids | `stats-gl` 4.2.3, `lil-gui` 0.21.0 | as listed | FPS/GPU timing and tuning knobs during the build [reg]. |
| Contract to TS types | `openapi-typescript` | **7.13.0** (dev) | FastAPI `/openapi.json` becomes `schema.d.ts`, used through JSDoc in plain-JS `game/` [reg][web]. |
| Backend | **One FastAPI service** (Python 3.13) that serves the API, SSE and the built game | `fastapi[standard]==0.162.0`, `uvicorn[standard]==0.54.0`, `starlette==1.10.0` (resolved) | Physics, ML and evaluation are Python. Native `fastapi.sse.EventSourceResponse` / `ServerSentEvent` import cleanly in 0.162.0 [local][web]. |
| Models and config | pydantic v2, pydantic-settings | `pydantic==2.14.2`, `pydantic-settings==2.16.2` | Typed contracts, a discriminated union for SSE events, env config [reg]. |
| Data | Parquet (pyarrow) plus **DuckDB** for analytics; the audit ledger as a hash-chained table in the same DuckDB file, owned by one process | `pyarrow==26.0.0`, `duckdb==1.5.6` (not 2.0.0, see §3.3), `pandas==3.1.0` | Columnar, single-file, no server. The CSVs are 234 MB + 35 MB [local]. |
| ML | LightGBM plus scikit-learn (calibration, metrics), joblib | `lightgbm==4.7.0`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `numpy==2.5.4`, `scipy==1.18.1` | Tabular, imbalanced (66 high-risk Kelvins-eligible events). Fast to train on a CPU [reg]. |
| Logging and resilience | structlog, orjson, tenacity | `structlog==26.1.0`, `orjson==3.13.0`, `tenacity==9.2.1` | JSON logs with `request_id` and `event_key` context. Bounded retries [reg]. |
| LLM | Anthropic SDK: **`claude-sonnet-5-5`** for live explanations, **`claude-opus-5-5`** for the precomputed insight brief | `anthropic==2.3.0` | `messages.parse` exists in 2.3.0 [local]. Prices are in §4 [skill docs]. |
| Decision model | OpenAI Decisions API (`gpt-6-luna`) as primary, TypeSafe Jev as secondary, then Claude structured output, then a deterministic mock | `openai==3.42.0` (has `client.decisions.create`) [local] | One `DecisionProvider` protocol. The policy layer still enforces the hard CARA/ESA rules. |
| Tests | pytest, hypothesis, schemathesis | `pytest==9.2.0`, `hypothesis==6.170.14`, `schemathesis==4.32.7` | Unit tests, property tests of the physics, and OpenAPI contract fuzzing [reg]. |
| Transport | **SSE** (server to client) plus REST POST for commands | n/a | Same pattern as the reference repo's `events.js` / `net.js`. Reasons in §5. |
| Deploy | One Dockerfile (Node 22 build stage, then `python:3.13-slim` with uv) on Render (`render.yaml` like the reference repo). Local run without Docker: `uv run` plus `vite build`. | n/a | Docker is not installed locally, so the local path cannot depend on it. |

The whole Python set resolves together under Python 3.13 (`uv pip compile`, exit 0) [local].

---

## 1. 3D globe: options and recommendation

### 1.1 Criteria and scoring (5 = best)

| Criterion | Plain three.js custom Earth | globe.gl / three-globe | CesiumJS | deck.gl GlobeView | react-three-fiber (+drei) |
|---|---|---|---|---|---|
| Stylized pastel/toy look matching the reference repo | **5**: full control over materials, shares `palette.js` ideas | 3: `globeMaterial()` can be swapped [web], but layers impose their own look | 1: photoreal by design. Fighting the imagery, terrain and lighting costs hours | 2: map-style flat layers, limited styling [web] | 5: same three.js underneath |
| 10–30k orbiting objects | **5**: one `Points`/`InstancedMesh` draw call with a custom vertex shader (GPU Kepler) | 3: `particlesData` exists (the satellites example uses it) [web]. Positions come from CPU accessors, with no published benchmark [web] | 4: `PointPrimitiveCollection` handles tens of thousands. Kayhan's `sgp4.gl` demo pairs CesiumJS with GPU SGP4 [web] | 4: `ScatterplotLayer` on the GPU, but GlobeView is **experimental**: no camera rotation, some layers unsupported [web: deck.gl GlobeView docs] | 5: the same `InstancedMesh`/shader approach |
| Orbit trails | 5: `Line2`/`LineMaterial` (examples/jsm) or an analytic ellipse | 3: `pathsData` follows the surface, not 3D orbits | 4: `PolylineCollection`/`PathGraphics` | 3: `PathLayer` | 5 |
| Camera fly-to choreography | 5: `camera-controls` or the reference repo's `easeInOutCubic` framing (`World.js`) | 3: `pointOfView({lat,lng,altitude}, ms)` only | 5: `camera.flyTo` built in | 2: GlobeView camera always points at the Earth centre, north up [web] | 4: drei `CameraControls` |
| Post-processing (bloom, outline) | **5**: pmndrs `postprocessing` 6.39.5 | 3: possible through `renderer()` access, but fiddly | 2: own `PostProcessStage`, limited stylization | 1 | 5: `@react-three/postprocessing` 3.2.2 |
| One person, one day, continuity with the reference repo | **5**: same `World` / `Director` / UI split, plain JS, Vite | 4: quickest first pixels, then a ceiling | 2: about 30 MB of static assets, Ion token or offline imagery, its own build config | 2 | 3: adds React 19 (`@react-three/fiber@9.8.2` peers `react >=19 <19.4`) [reg]. The reference repo's HUD is vanilla DOM, so this would be a rewrite |
| **Total** | **35** | 22 | 20 | 14 | 32 |

**Recommendation: plain three.js (0.186.1, `WebGLRenderer`) with a hand-built stylized Earth, in the reference repo's architecture.**
- `game/src/world/World.js` keeps the reference repo's public-API style: `setEvents`, `focusEvent(id)`, `flyTo(target)`, `setTimeWarp(x)`, `screenPos(id)`.
- A `Director` turns SSE events into world calls.
- The DOM HUD, log and dashboard are reused nearly verbatim.

R3F scores close but adds React and a UI rewrite for no visual gain. globe.gl is the fallback only if the custom Earth is not looking good by mid-morning; it accepts a custom `globeMaterial(ShaderMaterial)` [web].

**Why not `WebGPURenderer`/TSL right now:**
- The TSL API is still churning: r186 removed several TSL exports, and r187 renamed `getProjectionDepth()` and `ProxyNodeMaterial` [web: three.js releases r186, r187].
- pmndrs `postprocessing` is WebGL-only.
- three.js says `WebGLRenderer` gets no new large features [web: threejs.org manual], but it is stable and the reference repo uses it.

Revisit this after the hackathon.

### 1.2 The stylized Earth recipe

Units: 1 scene unit = 1 Earth radius (6378.137 km). The scene is Y-up, ECI is Z-up, so map ECI (x, y, z) to three (x, z, −y) in one helper, `eciToScene()`, and nowhere else.

| Element | Implementation | Cost |
|---|---|---|
| Globe body | `IcosahedronGeometry(1, 6)` gives a faceted low-poly look. Colour comes from an equirect **land mask** (Natural Earth 110 m land, public domain, rasterized to 1024×512 PNG at build time) quantized to two pastel tones (ocean `#7ea6d6`, land `#8fcfc8`, from the reference repo's `palette.js`). Toon shading uses a `ShaderMaterial` with a 3-step ramp on `dot(N, sunDir)`. | 1 draw call |
| Night lights | The same shader samples a city-lights texture (NASA Black Marble, public domain [unsure on exact file/licence page, check before shipping]). It is blended in with `smoothstep(0.05, -0.15, dot(N, sunDir))` and tinted warm yellow `#f5c445`. | Same pass |
| Atmosphere glow | A `SphereGeometry(1.03, 64, 32)` with `side: BackSide`, additive blending, fresnel `pow(1.0 - dot(N, V), 3.0)`, colour `#a8d4ec`. | 1 draw call |
| Clouds | `SphereGeometry(1.012)`, an alpha cloud texture or simplex-noise FBM in the fragment shader, slow independent rotation. Use soft white at 0.35 opacity, not photoreal. | 1 draw call |
| Terminator and sun | A directional light plus `sunDir` computed from sim time with a simple solar ephemeris. satellite.js exposes `SunPositionCalculator` [web]. | n/a |
| Stars | 4,000 `Points` on a large sphere, size-attenuated. | 1 draw call |
| Ground track or ring under a selected object | Reuse the reference repo's `ringGlowTexture` and pulse logic (`machines.js`) for the red pulsing alert ring. | n/a |
| Earth rotation | Rotate the Earth group by **GMST(simTime)** while orbits stay in ECI. Forgetting this is a classic bug: tracks drift against continents. | n/a |

### 1.3 Objects, trails and the "wow" moment

- **Background population (10–30k)**: one `THREE.Points` with a custom `ShaderMaterial`. Kepler propagation happens in the vertex shader (§2.3). Colour comes from an object-type attribute: DEBRIS, PAYLOAD, ROCKET BODY, UNKNOWN. `frustumCulled = false`, because the shader moves vertices and the CPU bounding sphere is wrong. `depthWrite = false` with additive blending to avoid sort artefacts. Point size is attenuated by distance and clamped to ≤ 8 px (WebGL point-size limits vary by GPU).
- **Hero objects (≤ 50)**: the conjunction pairs from the CDM data. They use an `InstancedMesh` of a toy satellite: rounded box body (`RoundedBoxGeometry`, as in `palette.js`) plus two panel boxes, and a debris "rock" from a low-detail `IcosahedronGeometry(1, 0)`. They are propagated on the CPU in float64 (§2.2) so the close approach is exact on screen.
- **Trails**: for heroes, draw the **analytic orbit ellipse** (256 segments) as `Line2` + `LineMaterial` (linewidth 2 px, dashed for "reconstructed"). Add a short fading comet tail (ring buffer of the last 64 positions, alpha attribute). Background objects get no trails; at 20k that is noise.
- **Conjunction choreography**: (1) camera `fitToSphere` around both objects (`camera-controls`, 1.6 s transition); (2) time-warp eases from 600× to 1× approaching TCA; (3) the miss-distance line and label are drawn with `CSS2DRenderer` (examples/jsm), showing "412 m, Pc 3.1e-5, ELEVATED"; (4) the covariance ellipsoid (a scaled sphere aligned to the RTN frame with sigma_r/t/n, 50% alpha) intersects or misses; (5) a `SelectiveBloom` pulse on the critical pair; (6) when the operator clicks "Plan manoeuvre", a small delta-v arrow appears and the post-burn orbit is drawn in green.
- **Post-processing budget**: `EffectComposer` → `RenderPass` → one `EffectPass(Bloom{mipmapBlur, luminanceThreshold 0.6, intensity 1.2}, Outline{selection: heroes}, SMAA, Vignette)`. Effects merged into one `EffectPass` share a single fullscreen pass, which is the main reason to prefer pmndrs over the three/examples `UnrealBloomPass` chain. Adaptive pixel ratio: `min(devicePixelRatio, 1.75)`, as in the reference repo's `World.js`.
- **Picking**: do not GPU-pick 20k points. Only heroes and a handful of "notable" objects are clickable (raycast against the `InstancedMesh`). Background hover is disabled, or done through a worker-side spatial hash at 4 Hz.

### 1.4 Honesty requirement for the visuals

The dataset has **no RAAN, argument of perigee, mean anomaly or absolute epoch** (`00-context.md`), so hero orbits are *reconstructed*: sma, ecc and inc from `t_/c_j2k_*`, plus a RAAN and phase chosen so the two objects meet at the CDM's geocentric latitude, azimuth and elevation at TCA. The background population is either synthetic (sampled from the dataset's sma/ecc/inc distributions) or a CelesTrak snapshot.

The UI must label this. Suggested HUD badge: `ORBITS: RECONSTRUCTED FOR ILLUSTRATION · RISK NUMBERS: FROM CDM DATA`. Hero trails use dashed lines. This also fits the rubric's "where does this fail" question.

---

## 2. Orbit propagation in the browser

### 2.1 Measured numbers (Apple M2, Node 20.11.1, 20,000 objects, warm JIT) [local]

| Method | Time per frame (one epoch, 20k objects) | Notes |
|---|---|---|
| `satellite.js` 7.1.0 `propagate()` pure JS | **36–40 ms** (about 1.8–2.0 µs/object) | Over the 16.7 ms budget at 60 fps even if it were the only work. |
| satellite.js WASM `BulkPropagator`, `createSingleThreadRuntime()` | **20.9 ms** | About 1.9× faster than pure JS, still too slow per frame. 60 epochs in one batch took 1,038 ms (17 ms/epoch). |
| satellite.js WASM `createMultiThreadRuntime({threadsCount: 4})` | **62.1 ms** | *Slower* in this test (thread overhead). In browsers it also needs `SharedArrayBuffer`, which means COOP/COEP cross-origin isolation [web: satellite.js bulk docs]. |
| Plain JS two-body Kepler, 4 Newton iterations | about 10 ms | Not optimized. Shows that even Kepler on the CPU eats most of a frame. |
| GPU Kepler in a vertex shader | well under 1 ms | 20k vertices × about 60 flops is trivial for any integrated GPU (estimate). |

API gotchas found while benchmarking:
- `BulkPropagator` needs calculator **instances**: `calculators: [new EciBaseCalculator()]`, not the class. Passing the class throws `Cyclic dependency detected`.
- Raw output buffers are reused on every `run()` call [web].
- `dispose()` on the multi-thread runtime calls `exit(0)` in Node.

The satellite.js docs themselves say pure JS can beat WASM and that you should measure [web: https://shashwatak.github.io/satellite-js/docs/bulk-propagation].

### 2.2 Recommended approach: three tiers

1. **Heroes (≤ 50 objects): CPU float64 every frame on the main thread.**
   - Two-body Kepler plus J2 secular drift (RAAN and argument-of-perigee rates), using the reconstructed elements from §1.4.
   - Cost is about 50 × 0.5 µs, which is negligible.
   - float64 keeps the close-approach geometry exact. TCA is pinned so the pair meets at the CDM miss vector, scaled for visibility (miss distances are 100 m–5 km, invisible at globe scale, so the encounter inset uses a local "zoom box" at 1:1000).
2. **Background (10–30k): GPU Kepler in the vertex shader.**
   - Per-instance attributes: `a` (Earth radii), `e`, `i`, `Ω0`, `ω0`, `M0`, `n` (rad/s), `dΩ/dt` and `dω/dt` (J2 secular, precomputed on the CPU in float64), and `type`.
   - That is 9 floats × 30k = about 1 MB of GPU memory. Fine.
3. **Optional "real sky" layer: satellite.js in a Web Worker.**
   - Load a bundled CelesTrak OMM JSON snapshot with `json2satrec`, propagate in the worker at **1 Hz sim-time keyframes** (WASM single-thread, about 21 ms per keyframe for 20k), and post a transferable `Float32Array`.
   - The vertex shader does Hermite or linear interpolation between keyframe k and k+1 using `uAlpha`.
   - At high time-warp, use 1 keyframe per N sim-seconds. At 3600× the motion is mostly the orbit itself, so linear interpolation of positions cuts across the arc. Use GPU Kepler (tier 2) above 60× instead.
   - Bundle the snapshot; never fetch live during the demo. CelesTrak enforces limits on large GP sets such as the Active list, with 403/429 responses, and `FORMAT=json` must now be passed explicitly (CSV became the default on 9 May 2026) [web: https://celestrak.org/usage-policy.php, https://www.celestrak.org/NORAD/documentation/sup-gp-queries.php]. [unsure: current Active-list size. Check at download.]

### 2.3 GPU Kepler vertex shader: design and pitfalls

```glsl
uniform float uDt;   // seconds since uEpoch (CPU float64 → small float32)
attribute float aA, aE, aI, aRaan0, aArgp0, aM0, aN, aRaanDot, aArgpDot;
void main() {
  float M = mod(aM0 + aN * uDt, 6.2831853);
  float E = M + aE * sin(M);                 // starter
  for (int k = 0; k < 5; k++) E -= (E - aE*sin(E) - M) / (1.0 - aE*cos(E));
  // perifocal → ECI with Ω = aRaan0 + aRaanDot*uDt, ω = aArgp0 + aArgpDot*uDt; then eciToScene
}
```

Pitfalls:
- **float32 time.** Never pass absolute Julian dates or Unix seconds to the GPU. Keep `simTime` as a float64 JS number, and pass `uDt = simTime − uEpoch`.
  - **Rebase** every 6 sim-hours: recompute `M0 ← (M0 + n·Δ) mod 2π`, `Ω0 ← Ω0 + Ω̇·Δ` and `ω0 ← ω0 + ω̇·Δ` on the CPU in float64, then re-upload. That is 3 × 30k floats, under 1 ms.
  - Without rebasing, `aN * uDt` exceeds 1e3 rad after about 10 days. float32 (about 7 significant digits) then loses about 1e-4 rad, about 0.7 km at LEO. That is invisible, but phase error grows and jitter appears at very high warp.
- **float32 positions** are fine at globe scale. A GPU-SGP4 study found FP32 error under 1 km median over two weeks, small next to SGP4's own error of about 1 km/day [web: https://arxiv.org/abs/2603.27830]. Close-approach geometry still belongs on the CPU (tier 1).
- **High eccentricity**: the dataset has GTO-like objects. A fixed 5 Newton iterations with the `E = M + e·sin M` starter is fine for e < 0.8. Clamp e to ≤ 0.95 and drop hyperbolic or decayed objects at load.
- **Frames**: ECI vs ECEF (rotate Earth by GMST, §1.2); Z-up vs Y-up (one helper only).
- **satellite.js specifics**: it uses WGS72 constants since v6 [web: satellite.js migration notes]. `propagate()` returns `false` or an error code for decayed objects (filter on load). Allocating a `Date` per object per frame creates GC churn: reuse one `Date`, or use `sgp4(satrec, minutesSinceEpoch)`.
- **Culling and blending**: `frustumCulled = false`; `depthWrite = false`; render background points *before* atmosphere glow (renderOrder) so the glow does not wash them out.

### 2.4 Time-warp clock

`SimClock` (game side, float64):
- State: `{ epochMs, rate ∈ {0, 1, 60, 600, 3600}, simTimeMs }`. `tick(realDt)` advances by `realDt × rate`.
- The HUD shows UTC sim time and "T−hh:mm to TCA" for the focused event, plus the reference repo's 1×/2×/4×/8× buttons relabelled 1×/60×/600×/1h/s.
- **The server owns the authoritative sim clock** and broadcasts `clock.sync` (§5) every 5 s and on every rate change. The client extrapolates in between, corrects drift with a 250 ms ease, and never jumps backwards.
- Deterministic replays use the same clock with a fixed seed (§3.6).

---

## 3. Backend

### 3.1 Architecture options

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| A. **Single FastAPI service**: physics, ML, rules, decisions, LLM, ledger, SSE; serves `game/dist` | One process, one language for the scored logic, one deploy, one owner of the DuckDB file. Native SSE in FastAPI 0.162 [local]. Pydantic models give OpenAPI to TS types for free. | Reuses only the *patterns* of the reference repo's Express code, not the code. | **Recommended** |
| B. Python "risk engine" plus Node/Express gateway with SSE (reference style) | Reuses `events.js`, `ledger.js` and `decide.js` almost verbatim. | Two services, two runtimes, an internal HTTP hop, duplicated schemas (zod vs pydantic), two failure domains. Decision and LLM code would live away from the evidence they explain. | Only if a teammate owns Node. Here there is one builder. |
| C. Node only (ML via ONNX/precomputed) | One runtime. | Pc physics, LightGBM training, calibration and the evaluation harness are all Python-native. Moving them adds a day of risk. | Rejected |

Rubric fit: "right tool per job" (25 pts) is easier to show when physics, rules and statistics are one Python package (`riskengine/`) with tests, and the LLM and decision layers sit visibly *on top* as explain/triage.

### 3.2 Proposed module layout (Option A)

```
server/
  pyproject.toml          # uv-managed, pins from §0
  app/main.py             # FastAPI app, lifespan starts SimLoop, mounts /api and static game/dist
  app/settings.py         # pydantic-settings
  app/api/{events,conjunctions,decisions,ledger,models,health}.py
  app/sse.py              # Broadcaster: asyncio fan-out, ring buffer for Last-Event-ID replay
  riskengine/
    cdm.py                # cleaning (sentinel 63,781,363 → NaN), unit normalization, (split,event_id) keys
    pc.py                 # 2D Pc (Foster/Alfano-style encounter-plane integral) + max-Pc fallback when covariance unknown
    rules.py              # CARA/ESA tiers: pure function → (tier, rule_id, reason)
    features.py, model.py # LightGBM + isotonic calibration; predict final-CDM risk tier
    evaluate.py           # baselines (last-CDM persistence = Kelvins baseline), metrics, failure cases
  decisions/              # DecisionProvider protocol + openai_decisions, jev, claude_structured, mock
  llm/                    # explain(): structured output → slot-filled text → number verifier → fallback
  ledger/                 # hash-chained append-only log
  registry/               # model cards + versioned artifacts
  tests/                  # pytest + hypothesis + schemathesis
shared/schema/            # exported openapi.json + events.schema.json (generated, committed)
game/                     # Vite + three (reference-style World/Director/UI)
```

### 3.3 Data layer

| Option | Fit | Notes |
|---|---|---|
| **Parquet + DuckDB** | **Analytics** (CDM history, 24 h windows, evaluation tables) and the ledger | Convert the CSVs once to `data/processed/cdm.parquet` (zstd). DuckDB queries Parquet in place. Native file locking allows **one read-write process**; multi-process writes "would need to be implemented in application logic" [web: https://duckdb.org/docs/stable/connect/concurrency.html]. Run uvicorn with `--workers 1`. That is fine here: the sim loop and SSE need a single process anyway. |
| SQLite | Ledger only | Stdlib, WAL mode, triggers can block UPDATE/DELETE. A good alternative if DuckDB causes trouble, but it means a second store. |
| Postgres | Overkill | Network service, credentials, no gain for a single-node demo. |

**Version pin: `duckdb==1.5.6` (28 Sep 2026), not 2.0.0 (21 Oct 2026, 10 days old).**
- 2.0.0 changes the Python API. Verified locally: `con.sql(...).fetchone()` now raises "Relations no longer have fetchone()". `con.execute(...).fetchone()` still works.
- Both versions resolve with the rest of the stack [local].
- If 2.0 is chosen anyway, use `con.execute()` everywhere.

**Append-only audit ledger with a hash chain** (table `ledger`):
- Columns: `seq BIGINT PK`, `ts_utc`, `sim_time`, `kind` (`signal|rule|ml|decision|llm|human|system`), `event_key`, `actor`, `payload_json` (canonical: orjson with `OPT_SORT_KEYS`), `prev_hash`, `hash = sha256(prev_hash ‖ seq ‖ ts ‖ kind ‖ payload_json)`. DuckDB's `sha256()` exists [local], but hash in Python so the logic is testable.
- Writes go through one `Ledger.append()` coroutine guarded by an `asyncio.Lock`. Nothing else issues UPDATE or DELETE.
- `GET /api/ledger/verify` recomputes the chain and returns `{ok, first_bad_seq}`. The UI shows a "chain verified" tick, which is cheap and strong evidence for the "trust & traceability" criterion (20 pts).
- Every decision row references the input evidence hashes (`cdm_rows_hash`, `model_version`, `rule_id`, `provider`, `prompt_hash`), so a judge can trace any warning back to a timestamped CDM and the rule applied.
- Export: `GET /api/ledger.jsonl`.

**Model registry** (files, no MLflow):
- Layout: `registry/<model_name>/<semver>/model.joblib` + `model_card.json` + `metrics.json`; `registry/<model_name>/CURRENT` holds the version string.
- The card holds: `sha256` of the artifact, training-data hash (Parquet file hash plus filter SQL), feature list and order, label definition (final-CDM tier), split (train events by `(split, event_id)`; validation held out **by mission** to avoid leakage [design choice]), metrics vs baseline, calibration curve, known failure modes, `created_at`, library versions.
- On load, recompute the sha256 and refuse to serve on mismatch (`/api/health` shows `model: degraded`).
- joblib/pickle is only safe for files you produced. Never load uploaded models.

### 3.4 Config, logging, idempotency

- **pydantic-settings** `Settings(BaseSettings)`:
  - `model_config = SettingsConfigDict(env_file=".env", env_prefix="C1_")`.
  - Fields: `mock_ai: bool` (auto-true when no keys, as in the reference repo's `config.js`), `llm_provider`, `anthropic_model_explain="claude-sonnet-5-5"`, `anthropic_model_brief="claude-opus-5-5"`, `decisions_provider_order=["openai","jev","claude","mock"]`, timeouts, `data_dir`, `ledger_path`, `seed=1729`, `time_warp_default=600`.
  - Secrets use `SecretStr`. `/api/health` reports *which* providers are configured, never the keys.
- **structlog**: JSON renderer in prod, console renderer in dev. Use `structlog.contextvars` for `request_id` (middleware; echoes `X-Request-ID`), `event_key` and `provider`. Log every external call with `latency_ms`, `outcome` (`ok|timeout|refusal|error|fallback`) and token usage. These logs also feed the dashboard's "AI calls" tile.
- **Idempotency**:
  - Commands (`POST /api/events/{key}/decision`, `/approve`, `/override`) require an `Idempotency-Key` header. Store `(key, request_hash, response_json)` in an `idempotency` table. Same key with the same body replays the stored response; same key with a different body returns 409.
  - Deterministic IDs: `uuid5(NAMESPACE, f"{split}:{event_id}:{cdm_index}")` for CDMs and `uuid5(..., f"{event_key}:{decision_round}")` for decisions, so replays produce the same IDs.

### 3.5 Background tasks

- Use the FastAPI `lifespan` with an `asyncio.TaskGroup` that runs the `SimLoop` at 4 Hz wall time. It advances the sim clock, releases CDMs whose `time_to_tca` is reached, runs rules, ML and decisions, and emits events.
- CPU-heavy steps (Pc integration over many events, model batch scoring) go through `await asyncio.to_thread(...)` so the event loop and SSE never stall.
- Training runs **offline** (`uv run python -m riskengine.train`), never in the request path.
- No Celery or Redis.

### 3.6 Deterministic replays

- Single seed (`numpy.random.default_rng(settings.seed)`). The sim clock is injected, never `datetime.now()` inside the engine.
- Scenario files (`scenarios/*.json`) list the events, start offset and warp, like the reference repo's scenario picker.
- **LLM and decision responses are cached on disk by request hash**: `sha256(provider, model, prompt_version, canonical_input)`. A `REPLAY=1` mode serves only from cache and fails closed to the mock. This gives byte-identical demo runs and offline rehearsal.
- `GET /api/replay/{scenario}` re-emits the ledger for a scenario at a chosen speed. This is also the judge-facing "show me how you got here".

### 3.7 Testing

- **pytest** unit tests: CDM cleaning (sentinel value, null sigmas, F10/AP nulls), tier rules (table-driven, one test per row of the brief's thresholds including the boundaries 1e-6, 1e-4, 1.0 km, 5.0 km), ledger chain (tamper one row, then `verify` reports that seq).
- **hypothesis** property tests for `pc.py`:
  - `0 ≤ Pc ≤ 1`.
  - Symmetric when the two objects are swapped.
  - Invariant to rotating the encounter-plane coordinates when the covariance is rotated consistently.
  - Monotone non-increasing in miss distance along a fixed direction for a fixed isotropic covariance.
  - Pc → 0 as σ → 0 with miss > hard-body radius, and Pc → 1 as σ → 0 with miss < HBR.
  - Pc increases with HBR.
  - Agreement within 1% with a brute-force Monte Carlo (1e6 samples) on 20 random cases (`@settings(max_examples=20, deadline=None)`).
  - For golden values, NASA CARA publishes a 2D Pc module (Foster/Estes method) [web: https://software.nasa.gov/software/GSC-18016-1]. [unsure: whether its test vectors are easy to extract in time; if not, Monte Carlo is the reference.]
- **Rules vs model**: a test asserts that the policy layer *overrides* the ML/decision output whenever a hard rule fires (Pc ≥ 1e-4 or miss < 1 km means CRITICAL and ESCALATE, whatever the model says).
- **Contract**:
  1. Pydantic models are the source of truth. `scripts/export_schema.py` writes `shared/schema/openapi.json`, plus `events.schema.json` built with `TypeAdapter(SseEvent).json_schema()` for the SSE discriminated union, which OpenAPI does not cover.
  2. `npx openapi-typescript shared/schema/openapi.json -o game/src/types/api.d.ts`.
  3. `game/` stays plain JS with `// @ts-check` and JSDoc `@type {import('./types/api').components['schemas']['Conjunction']}`.
  4. CI-style test: generated files are up to date (`git diff --exit-code`), and the frontend fixture events in `game/fixtures/*.json` validate against `events.schema.json` (pytest + `jsonschema`).
  5. **schemathesis** fuzzes the running API from `openapi.json`: no 500s, and responses match the schemas.

### 3.8 Containerisation, deploy, offline mode

- **Dockerfile** (multi-stage):
  1. `node:22-alpine`: `npm ci && npm run build` for `game/`.
  2. `python:3.13-slim` with `COPY --from=ghcr.io/astral-sh/uv` and `uv sync --frozen --no-dev`. Copy `game/dist`, `registry/` and the processed Parquet. `CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`.
  - Docker is **not installed locally**, so local dev is `uv run uvicorn ... --reload` plus `vite` with a `/api` proxy (the same `vite.config.js` proxy pattern as the reference repo).
- **Render**: a `render.yaml` web service like the reference repo's, with `healthCheckPath: /api/health` and secrets as `sync: false`.
  - Free web services spin down after 15 min idle and take about 1 min to wake [web: Render pricing articles, e.g. https://livemy.app/blog/render-pricing]. Use a paid instance (from $7/month [web]) for judging day, or hit `/api/health` before presenting.
  - Since 24 Feb 2026, WebSocket messages count as activity [web]. [unsure: whether an open SSE stream counts.]
- **Fly.io**: per-second Machine billing [web: https://fly.io/pricing.md]. Third-party pricing reports conflict, so [unsure]. It would be the fallback if Render misbehaves (`fly launch` with the same Dockerfile).
- **Offline/mock mode**:
  - `C1_MOCK_AI=1` (auto when no keys): decisions come from the deterministic mock (returns the ML/physics prior as in the reference repo's `mock()`), and LLM text comes from templates.
  - `REPLAY=1` serves cached real responses.
  - The game works with no network: bundled textures, bundled orbit snapshot, no CDN at runtime.
  - The demo laptop runs everything from `localhost`.

---

## 4. LLM layer and decision-model layer

### 4.1 LLM for grounded explanations

| Option | Price (input/output per MTok) | Use |
|---|---|---|
| `claude-sonnet-5-5` | $2 / $10 | **Live per-event explanation**, operator Q&A. Fast. Set `effort: "low"` explicitly. |
| `claude-opus-5-5` | $4 / $20 | **Precomputed** daily "insights brief" over the 24 h window (run once per scenario, cached). Set effort `medium`, which is its default. |
| `claude-haiku-5-5` | $0.10 / $0.50 | Optional speech-bubble one-liners (the reference repo's `say()`). |
| OpenAI GPT (chat/structured) | n/a | Already the reference repo's fallback. Not needed if Claude is primary. Keep it as the decision-model fallback via the Decisions endpoint instead. |

Prices are from the Claude API skill's model table (cached 6 Oct 2026). [unsure: re-check https://platform.claude.com/docs/en/about-claude/pricing.md on the day.] Cost is trivial: about 2k input + 300 output tokens per explanation is about $0.007 on Sonnet 5.5.

API facts that bite (from the bundled Claude API skill; recalled patterns are stale):
- Structured output goes through `output_config: {format: ...}`, best via `client.messages.parse(...)`. This exists in `anthropic==2.3.0` [local].
- **Forced `tool_choice` (`any`/`tool`) returns a 400** on Opus 5.5 and Sonnet 5.5. Use structured outputs or `auto` with `strict: true`.
- **`temperature`/`top_p`/`top_k` return a 400 on every model since 21 Oct 2026.** Do not try to get determinism from sampling. Determinism comes from caching (§3.6).
- No assistant prefill (400).
- Opus 5.5 thinking cannot be disabled (`effort` is the only lever). On Sonnet 5.5, `{type: "disabled"}` returns a 400; use `{type: "between_tools"}` or low effort.
- Check `stop_reason == "refusal"` before reading content. The server-side `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) handles classifier refusals.
- Prompt caching: the minimum cacheable prefix is 512 tokens on Opus 5.5 and Sonnet 5.5. Put the frozen system prompt, glossary, tier table and output schema first with `cache_control`; the per-event fact sheet goes last. Verify with `usage.cache_read_input_tokens`.
- Timeouts: Python SDK `timeout` is in seconds (default 10 min) with `max_retries` default 2. Set `client.with_options(timeout=8.0, max_retries=1)` for live calls; the worst-case wall clock is timeout × (retries+1) = 16 s.

**Grounding and number-verification guardrail (three lines of defence):**
1. **Fact sheet in, slots out.** The server builds a `FactSheet` (pydantic) per event, e.g. `pc_final=3.1e-5`, `miss_m=412`, `tier="ELEVATED"`, `rule_id="ESA-ELEV-PC"`, `ml_p_critical=0.22 (CI 0.15–0.31)`, `n_cdms=13`, `trend="decreasing"`, `cdm_ts=[...]`. The LLM returns structured JSON: `{headline, explanation, recommendation, uncertainty_note, cited_facts: [fact_key]}`, with **numbers written only as placeholders** like `{{pc_final}}`. The server renders them with fixed formatting. A hallucinated number cannot reach the screen through this path.
2. **Verifier.** After rendering, a regex extracts every numeral (including `1e-4`, `10^-4`, `412 m`, `0.4 km`) and checks each against the fact-sheet values after unit normalization (relative tolerance 1%, or an exact match at the displayed precision). Any unmatched number, or any `cited_facts` key not in the sheet, rejects the output. The ledger records `llm.rejected` with the reason.
3. **Deterministic fallback text.** A Jinja-free f-string template per tier is always rendered first and sent immediately. The LLM text replaces it via an `explanation.ready` SSE event when it passes (or arrives within 8 s). The UI badge shows `AI-written · verified` vs `template`.
- Refusal or timeout: keep the template and log `outcome=refusal|timeout`.
- **Caching**: a disk cache keyed by `sha256(model, prompt_version, factsheet_canonical_json)`. Explanations are pure functions of the fact sheet, so re-renders, replays and the offline demo cost nothing.

### 4.2 Decision model: OpenAI Decisions API vs TypeSafe Jev

| Aspect | OpenAI Decisions API | TypeSafe Jev |
|---|---|---|
| Shape | `POST /v1/decisions {model:"gpt-6-luna", input, questions:[{type: predicate\|choice\|score, name, instructions, choices[{value,description}] \| levels[{label,description}]}]}` → `answers[]` with `probability` / `choice+probabilities+confidence` / `score+probabilities+confidence` / `refusal` (verified live 9 Oct 2026 by the reference repo). **Python SDK `openai==3.42.0` has `client.decisions.create(input, model, questions, prompt_cache_key, prompt_cache_options, safety_identifier, timeout)`** [local]. | Typed questions `noul` (P(yes)), `choice` (`criteria` map, 1–255 options), `score` (`criteria` list, 2–10 levels); request field `state` instead of `input`; `questions` is an **object keyed by name**; response `answers` is an object with `probabilities` keyed by label/index, plus `legend` for score [web: https://docs.aimlapi.com/api-references/decision-models/typesafe/jev]. Hosted endpoint reported as `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest` (→ jev-1.13.0) [web, third-party; unsure]. AIMLAPI lists `typesafe/jev` (→ 1.14). Context window 262,144 tokens. |
| Latency | Vendor claim of about 150 ms vs about 1.6 s for the same model via the regular API [web: https://www.orcarouter.ai/blog/openai-decisions-api-gpt-6-luna] [unsure: measure]. | No figures published. The OpenClaw docs use a 30 s timeout in examples [web: https://docs.openclaw.ai/concepts/decision-models]. |
| Price | One report says $0.10 per M input tokens; another says no rate is published [web: techenclave vs orcarouter]. [unsure] | AIMLAPI sample: about $0.0000225 for a 406-in/73-out call [web]. |
| Images | Supported (reported). | `assets` experimental. Text is enough here. |
| Caveat | A shared interface does not make probabilities interchangeable across providers; calibration needs its own eval [web: OpenClaw docs]. | Same caveat. |

**One interface (Python port of the reference repo's `decide.js`):**

```python
class DecisionRequest(BaseModel):
    kind: Literal["predicate", "choice", "score"]
    name: str
    question: str
    options: list[Option]          # value, description
    evidence: str                  # rendered FactSheet + CDM trend lines (never the prior)
    prior: dict[str, float] | None # physics/ML prior, used by mock + shown next to model
class DecisionResult(BaseModel):
    answer: str | bool
    probabilities: dict[str, float]  # normalized to sum 1
    confidence: float
    provider: Literal["openai-decisions", "jev", "claude-structured", "mock"]
    latency_ms: int
    refused: bool = False
    request_hash: str
class DecisionProvider(Protocol):
    name: str
    async def decide(self, req: DecisionRequest, *, timeout_s: float) -> DecisionResult | None: ...
```

- Adapters translate the request: `options` become `choices[{value:"o1",description}]` for OpenAI (as `decide.js` does), and `criteria{o1: description}` plus `questions{name: ...}` / `state` for Jev. Both normalize probabilities as in `finish()`.
- **Order**: OpenAI Decisions (timeout 5 s), then Jev (5 s), then Claude Sonnet 5.5 structured output `{probabilities: {...}}` (8 s), then mock. Total budget: 12 s, enforced with `asyncio.timeout`.
- **Failure policy**: timeout, 5xx or 429 moves to the next provider and is logged. **A refusal is not a "no"**: it is recorded as `refused=True` and the policy layer returns `ESCALATE: model declined` (human decides). This matches OpenClaw's guidance not to treat failure as a negative answer [web].
- **Where it is used**: one triage question per flagged event, for example `choice: {escalate_now, monitor_next_cdm, dismiss}`, with physics Pc, ML forecast and CDM trend as evidence.
  - `policy.py` (port of the reference repo's AUTO/ESCALATE/BLOCK) overrides: any CRITICAL by rule means ESCALATE regardless; `dismiss` is never AUTO for ELEVATED; low confidence (< 0.6) means ESCALATE.
  - The decision model "earns its place" only if, on held-out events, it beats the naive rule baseline on the operator-workload metric (escalations per true high-risk event). Otherwise the UI shows it as "advisory" with its measured score.
- **Mock**: deterministic, returns the prior mapped to options (as in the reference repo), with an optional `force_uncertain` 0.54/0.46 split to demonstrate escalation.

---

## 5. Real-time transport: SSE vs WebSocket

| Factor | SSE | WebSocket |
|---|---|---|
| Direction needed | Server to client events. Commands are rare and fit REST POST with idempotency keys. | Bidirectional (not needed). |
| Reconnect and resume | `EventSource` auto-reconnects and sends `Last-Event-ID`. The server replays from a ring buffer. | Hand-rolled. |
| Server support | FastAPI 0.162 native `EventSourceResponse` / `ServerSentEvent(data, event, id, retry, comment)` [web: https://fastapi.tiangolo.com/reference/sse/][local]. | Starlette WebSocket, more code. |
| Proxies and PaaS | Plain HTTP. Needs keep-alive comments every 15 s; FastAPI sends none automatically [web]. | Upgrade handling, idle timeouts. |
| Limits | Max **6 connections per browser per domain** on HTTP/1.1 (Chrome/Firefox "won't fix"); about 100 streams on HTTP/2 [web: https://developer.mozilla.org/en-US/docs/Web/API/EventSource]. Use one `EventSource` per tab. | No such limit. |
| Continuity | Same as the reference repo (`events.js`, `net.js`). | New. |

**Recommendation: SSE.** Position data never goes over the wire: the client propagates orbits itself (§2), so SSE carries only low-rate semantic events, about 1–20 messages per second at 600× warp.

**Envelope (versioned):**

```json
{ "v": 1, "seq": 1842, "id": "1842", "type": "conjunction.flagged",
  "at": "2026-10-31T09:12:03.120Z", "sim_time": "2026-11-02T04:00:00Z",
  "event_key": "train:4512", "trace": {"ledger_seq": 9911, "request_id": "..."},
  "data": { ... } }
```

- **Types** (dot-namespaced, mirrored in the UI log tabs ALL/ALERTS/AGENT):
  - `state.snapshot` (sent on connect, as in the reference repo's `write(res, {type:'state'})`)
  - `clock.sync`
  - `cdm.received`
  - `risk.assessed` (Pc, tier, rule_id)
  - `ml.forecast`
  - `decision.requested`, `decision.made`
  - `policy.applied` (AUTO/ESCALATE/BLOCK with reason)
  - `explanation.ready`
  - `human.action`
  - `ledger.appended`
  - `system.degraded` (provider down, model hash mismatch)
- **Versioning rules**:
  - Within `v: 1`, changes are additive only (new optional fields, new types). Clients ignore unknown types and fields.
  - A breaking change bumps to `v: 2`, served at `/api/events?v=2` while v1 stays alive for the session.
  - The pydantic discriminated union (`Field(discriminator="type")`) is the single source of truth, exported to `events.schema.json` and tested against fixtures (§3.7).
  - `seq` is monotone per server boot, and `id = str(seq)` for `Last-Event-ID`. On a gap (client `seq` jumps), the client calls `GET /api/state` to resync.
- **Keepalive**: `ServerSentEvent(comment="ping")` every 15 s (the reference repo uses `: ping` every 15 s). `retry: 2000`.

---

## 6. Risks and mitigations (tech-specific)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Node 20.11.1 vs modern tooling (Vite 7/8 refuse to run) | Certain if unpinned | Pin Vite 6.4.5. Optionally install Node 22 (no nvm or brew on this machine, so that would be a manual install). |
| three 0.187 and postprocessing peer conflict | Certain if unpinned | Pin three@0.186.1. Do not use `--legacy-peer-deps`. |
| Custom Earth shader looks bad by midday | Medium | Time-box the globe to 2 h. Fallback: three-globe `globeMaterial` with the same pastel shader, or plain `MeshToonMaterial` plus a land texture. |
| 20k CPU propagation drops frames | Certain on the main thread (36–40 ms measured) | GPU Kepler for the background. CPU only for ≤ 50 heroes. |
| DuckDB 2.0 API breaks | Verified | Pin 1.5.6. Single writer process (`--workers 1`). |
| Beta Decisions API slow, down or refusing | Medium | Provider chain, 12 s budget, refusal means ESCALATE, `REPLAY` cache, mock. |
| LLM invents numbers | Medium without guardrail | Slots, verifier and template fallback (§4.1). |
| Render cold start during judging | High on the free tier | Paid instance, or warm-up ping, or run locally. |
| Reconstructed orbits mistaken for real | Medium (judges) | HUD badge, dashed trails, documented in the architecture doc. |

## 7. Open questions for the other design docs

1. Synthetic background vs a bundled CelesTrak snapshot. Synthetic is offline-safe and consistent with "reconstructed"; real looks better. Default: synthetic from dataset distributions, with real as a toggle if time allows.
2. Whether the decision model's triage question beats the rule baseline. This must be measured in the evaluation doc before it is promoted from "advisory".
3. Pc method choice and what to do when covariance is the 63,781,363 sentinel: max-Pc bound vs "insufficient data → ESCALATE". This belongs in the risk-engine doc, but it changes the API schema (`pc_method`, `pc_status` enum).

## Sources

- npm registry: https://registry.npmjs.org/three, /postprocessing, /vite, /satellite.js, /camera-controls, /globe.gl, /three-globe, /cesium, /@deck.gl/core, /@react-three/fiber, /openapi-typescript
- PyPI JSON API: https://pypi.org/pypi/fastapi/json (and the same for each Python package in §0)
- three.js releases r186 and r187: https://github.com/mrdoob/three.js/releases/tag/r186, https://github.com/mrdoob/three.js/releases/tag/r187
- WebGPURenderer manual: https://threejs.org/manual/pages/webgpurenderer
- satellite.js bulk propagation: https://shashwatak.github.io/satellite-js/docs/bulk-propagation; migration notes: https://shashwatak.github.io/satellite-js/docs/migrating
- GPU SGP4 FP32 accuracy (jaxsgp4): https://arxiv.org/abs/2603.27830
- Kayhan sgp4.gl: https://kayhan.space/newsroom/kayhan-space-launches-free-gpu-accelerated-satellite-propagation-tool-for-real-time-visualization/aid/16
- deck.gl GlobeView limitations: https://deck.gl/docs/api-reference/core/globe-view
- globe.gl / three-globe: https://github.com/vasturiano/globe.gl, https://npmjs.com/package/three-globe
- FastAPI SSE reference: https://fastapi.tiangolo.com/reference/sse/
- DuckDB concurrency: https://duckdb.org/docs/stable/connect/concurrency.html
- MDN EventSource: https://developer.mozilla.org/en-US/docs/Web/API/EventSource
- openapi-typescript: https://github.com/openapi-ts/openapi-typescript
- CelesTrak usage policy and GP queries: https://celestrak.org/usage-policy.php, https://www.celestrak.org/NORAD/documentation/sup-gp-queries.php
- NASA CARA 2D Pc software: https://software.nasa.gov/software/GSC-18016-1
- Decision models (Jev and OpenAI comparison): https://docs.openclaw.ai/concepts/decision-models; Jev API shape: https://docs.aimlapi.com/api-references/decision-models/typesafe/jev
- Decisions API pricing and latency reports: https://techenclave.com/news/items/openai-opens-decisions-api-priced-at-0-10-per-million-input-tokens, https://www.orcarouter.ai/blog/openai-decisions-api-gpt-6-luna
- Render free-tier spin-down: https://livemy.app/blog/render-pricing, https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026.md; Fly pricing: https://fly.io/pricing.md
- Claude model IDs, prices and API behaviour: the bundled `claude-api` skill (model table cached 2026-10-06); live page https://platform.claude.com/docs/en/about-claude/pricing.md
- Reference repo files read: `game/package.json`, `server/package.json`, `render.yaml`, `server/src/events.js`, `game/src/net.js`, `game/src/world/World.js`, `game/src/world/palette.js`, `server/src/agent/decide.js`, `server/src/agent/ledger.js`, `server/src/config.js`, `docs/DECISIONS_API.md`
