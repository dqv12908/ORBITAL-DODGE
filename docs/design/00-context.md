# Context pack — Challenge 1 (Airbus track, Autonomous SG 2026)

Hackathon day: 31 Oct 2026 (NUS). One builder (Claude) does everything. Pitch is 3 minutes; earlier submission scores a little (5%).

## The ask (from the team)
Build the Challenge 1 system ("AI-driven satellite collision avoidance") PLUS a 3D simulator of the entire Earth
with a dynamic, "wow factor" visual experience, in the visual and simulator style of
https://github.com/Aboyang/agentic-factory (cloned at SCRATCH/ref/agentic-factory, see below).
The backend must be really robust and well designed. Layering wanted: decision model (OpenAI Decisions API /
TypeSafe Jev, "trendy"), then LLM, then machine learning.

Full brief text: docs/brief.txt (Challenge 1 is page 1-2; rubric pages 7-8).

### Challenge 1 requirements (verbatim gist)
- Collision risk evaluator: reads conjunction telemetry (CDMs), calculates collision probability (Pc) for satellite–debris pairs,
  auto-flags high-risk events needing manoeuvre planning.
- Risk classification backed by probability, per NASA CARA handbook (NASA/SP-20205011318) and ESA SDO:
  Safe: Pc < 1e-6 AND miss > 5.0 km; Elevated: 1e-6 <= Pc < 1e-4 OR miss 1.0–5.0 km; Critical: Pc >= 1e-4 (risk >= -4.0) OR miss < 1.0 km.
- Dashboard UI: trajectories, collision risk scores, miss distances; warnings: in a 24 h frame how many come within the
  dangerous distance; generate insights.
- Data: "Can use only columns event_id, time_to_tca, miss_distance, relative_speed, relative_position_r/t/n, c_object_type,
  c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP for simplicity purposes" (ambiguous: restriction vs simplification —
  design must decide and justify). `risk` = log10(Pc); map to tiers to train/validate. Train on train_data, predict test_data.

### Rubric (weights)
Problem realism 20 (named user, status quo, cost) · Technical soundness 25 (works end to end, right tool per job:
LLM for reading/explaining/interaction; solvers/rules/statistics for detection, compliance) · Honest evaluation 20
(baseline, metrics, visible failure case with reason) · Trust & traceability 20 (trace to timestamped signal/applied rule,
uncertainty visible, human decides, refusal) · Impact 10 (before/after vs baseline) · Time discipline 5.
Judges: "reward sound restraint, not presentation polish"; "Where does this fail, and how would your user know?"

## Dataset facts (profiled by me, verified)
Location: data/raw/Challenge 1 Airbus /{train_data.csv,test_data.csv}  (note trailing space in folder name)
It is the ESA Kelvins Collision Avoidance Challenge dataset (103 columns, identical schema in train/test).
- train: 162,634 CDM rows, 13,154 events, 19 missions; 1–23 CDMs per event (median 13). time_to_tca from 6.99 d down to −0.15 d.
- test: 24,484 rows, 2,167 events, 18 missions. time_to_tca ONLY in [2.0, 6.99] d — i.e. test contains only the early CDMs
  (the Kelvins setup: predict the risk of the final CDM, which is hidden). test `risk` column is present but is the risk of
  those early CDMs, not the target. Event IDs in test are renumbered 0–2166 and collide numerically with train IDs — key by (split, event_id).
- risk = log10(Pc), floored at −30 (63% of train events end at −30). Range up to −1.44.
- Final-CDM tiers in train (using brief thresholds on final risk + final miss distance): safe 9,843 · elevated 2,403 · critical 908.
  BUT only 13 train events have final risk >= −4; critical is dominated by miss_distance < 1 km. risk >= −6 at final: 365 events.
- Kelvins-eligible train events (has a CDM >= 2 d before TCA and final CDM < 1 d): 8,293; of these only 66 are high-risk (final risk >= −6).
- miss_distance in metres (1%: 256 m, median 12.4 km, max 67 km); relative_speed m/s (4 – 17,138).
- Sigmas: sentinel 63,781,363 (= 10 × Earth radius in m) marks unknown covariance — must be cleaned. c_sigma_* null in 11 rows. F10/AP null in ~4%.
- Full covariance info IS present in the 103 columns: t_/c_ sigma_r/t/n (+rdot/tdot/ndot), correlation terms (ct_r, cn_r, cn_t ...),
  relative_velocity_r/t/n, mahalanobis_distance, position_covariance_det, orbital elements t_/c_ j2k_sma/ecc/inc, h_apo/h_per,
  geocentric_latitude, azimuth, elevation, observation stats (obs_used, weighted_rms, od_span), SSN, F3M, max_risk_scaling.
  No RAAN / argument of perigee / mean anomaly / absolute epoch → real orbits cannot be reconstructed exactly; a 3D globe needs
  a documented, plausible reconstruction (sma/ecc/inc + conjunction latitude/azimuth/elevation) labelled as such.
- object types (rows): DEBRIS 94k, UNKNOWN 51k, PAYLOAD 15k, ROCKET BODY 2k, TBA 177.
- corr(max_risk_estimate, risk) ≈ 0.22 (train rows).

## Reference repo (style + architecture to emulate)
Cloned at <agentic-factory clone>
"Wrench-bot Factory" — hackathon winner. Three.js stylized low-poly 3D world (pastel, soft lighting, rounded toy-like models),
dark glassy HUD bar (clock, scenario picker, KPIs, 1×/2×/4×/8× speed), speech bubbles over agents, red pulsing alert rings,
plant.log terminal panel (ALL/ALERTS/AGENT tabs, mono font), manager dashboard modal with KPI tiles and tables, scenario picker
(each scenario demonstrates one idea), Manager's Desk (live policy knobs), chaos tools.
Backend: Express API + SSE events, simulation engine (hidden truth → signals), orchestrator pipeline, decide.js (OpenAI Decisions API
typed choices with probabilities), policy.js (hard code-enforced rules: AUTO / ESCALATE / BLOCK with reason strings), ledger (audit trail),
mock mode (MOCK_AI=1) to run offline. Docs: ARCHITECTURE, AGENT, SIMULATION, UI_SPEC, SIM_SPEC, API, JUDGING (criteria→evidence matrix).
Key idea it sells: "decision model that earns its place" — shown next to a naive rule baseline.

### OpenAI Decisions API — verified shape (from that repo, live call 9 Oct 2026, public beta, model gpt-6-luna)
POST https://api.openai.com/v1/decisions  {model, input: text, questions:[{type:'predicate'|'choice'|'score', name, instructions, choices:[{value,description}] | levels:[{label,description}]}]}
→ {answers:[{type, name, probability | choice+probabilities+confidence | score+probabilities+confidence} | {type:'refusal'}], usage}
TypeSafe Jev (launched 15 Sep 2026) is the competing "decision model" (Choice/Score primitives, per-option probabilities).

## Tooling on this machine
Python venv with pandas/pyarrow/pypdf: python (run with -I).
macOS, Node presumably available. Repo: github.com/dqv12908/ORBITAL-DODGE.
