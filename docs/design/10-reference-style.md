# 10 — Reference style extraction: "Wrench-bot Factory" → orbital collision avoidance

Source repo (read-only): `SCRATCH/ref/agentic-factory` where
`SCRATCH=<scratchpad>`.
All `file:line` refs below are relative to that repo root. Everything here was read from the code, the docs and the 10
screenshots in `docs/screenshots/` (looked at one by one). Where docs and code disagree, **the code wins** (noted inline).

> Important drift: `docs/SIM_SPEC.md:290-292` and `docs/UI_SPEC.md:47` still say "low-res pixel look, render at 1/3
> resolution, `image-rendering: pixelated`, Press Start 2P titles, VT323 text". The shipped code does **not** do that:
> `game/src/world/World.js:1-5,96-106` renders full-res, antialiased, ACES, soft shadows and forces `image-rendering:auto`;
> `game/src/ui/ui.css:1-6` says "Inter for everything you read, VT323 only in the terminal log, Press Start 2P only for the
> title-screen logo". The screenshots confirm the smooth look. Copy the **code**, not those two spec lines.

---

## 0. Versions and stack (exact)

| Item | Value | Where |
|---|---|---|
| three | `^0.169.0` (lock: **0.169.0**) | `game/package.json:11`, `package-lock.json:1973-1974` |
| vite | `^5.4.10` (lock: **5.4.21**) | `game/package.json:14`, `package-lock.json:2067-2068` |
| express | `^4.21.1` (lock 4.22.3) | `server/package.json:13`, lock `:1198` |
| dotenv | `^16.4.5` (lock 16.6.1) | `server/package.json:12` |
| viem | `^2.21.45` (only for onchain USDC; we drop it) | `server/package.json:14` |
| concurrently | `^9.0.1` (root devDep) | `package.json:14-16` |
| Node | `>=20` (Render pins 22) | `package.json:5-7`, `render.yaml:12-13` |
| Module system | ESM (`"type": "module"`) both workspaces | `game/package.json:4`, `server/package.json:4` |
| Workspaces | npm workspaces `["server","game"]`; `npm run dev` = concurrently server+vite; `dev:mock` = `MOCK_REAP=1 MOCK_AI=1` | `package.json:4,9-10` |
| Dev proxy | Vite `:5173` proxies `/api` → `:8787`; `fs.allow ['..']` so the game imports `../shared/contract.js` | `game/vite.config.js:3-8` |
| Rule | "no other dependencies. Never add npm packages." No vendor SDKs: OpenAI/Reap called with plain `fetch` | `docs/SIM_SPEC.md:6-7`, `docs/ARCHITECTURE.md:33-35` |

Our plan should mirror this: `three@0.169.0` (or a newer pinned r1xx, but keep the import paths
`three/examples/jsm/...` which r169 supports), `vite@5.4.x`, `express@4.x`, `dotenv@16.x`, Node 20+, no SDKs.

---

## 1. Visual language

### 1.1 Overall look (from the screenshots)

- **World:** bright, clean low-poly diorama — light grey concrete floor tiles, corrugated blue-grey walls with
  big pale-blue windows, pastel toy-like machines (mint, sky blue, butter yellow, coral orange, lime green), yellow/black
  hazard stripes, stack-light "andon" towers. Seen from a 3/4 perspective (~39° elevation, ~9° azimuth). Not pixelated.
- **HUD:** dark, near-black glassy panels (`rgba(21,25,34,0.95)` + `blur(10px)`) floating over the bright world,
  12 px radius, 1 px hairline border, soft deep shadow. Strong contrast: bright diorama, dark chrome.
- **Accents:** Nord palette — frost cyan `#88c0d0` (primary accent, active speed button, "ALL" tab), aurora green
  `#a3be8c` (good / revenue / approve), yellow `#ebcb8b` (warn / attention / "needs your OK"), red `#d06b75`
  (bad / down / reject), purple `#b48ead` (agent identity, "AI LIVE" chip).
- **States speak in 3 places at once:** 3D (lamp + floor halo + sparks/smoke), screen-anchored DOM chips
  (hint chip `✖ E-ARM-310`, floating text `DOWN -$60/min`) and the terminal log line colour.

Screenshot index (what each one shows):
`01-scenarios` full-screen picker (pixel-art robot + Press Start 2P logo, 6 cards with coloured top bar + tag pill,
"TEACHES" box, step strip 1–6). `02-factory` idle view: top bar, `plant.log` bottom-right, robot on its cyan pad.
`03-breakdown` red floor halo + red lamp + `DOWN -$60/min` pill + `E-ARM-310` hint chip + white speech bubble + top-centre
agent chip "Robot Arm · Diagnosing… 1/7". `04-diagnosis` agent popover left: model bars vs faint "fault-code rule" bars,
"EVIDENCE FROM THE LOG", agent quote box, listings with PICK badge; yellow "?" thought bubble over the robot.
`05-sensors` machine popover right: DEGRADED/CRITICAL chips, alarm rows with coloured left border, parts list, expanded
part with threshold gauge (green→amber→red zones + white marker), sparkline; amber floor halo. `06-approval` centred
modal with 4 px yellow top border, "APPROVAL NEEDED" chip, total `$69.69`, "WHY IT ASKS YOU" bullets, `−$35/min while you
decide`, big green Approve + red Reject; the world dims behind. `07-fixed` agent popover "Back online" receipt card +
7 completed steps; green chip "Fixed". `08-terminal` expanded log 660×430. `09-dashboard` manager dashboard modal: 6 KPI
tiles, tabs with count pills, table with badges. `10-onchain-tx` block explorer (not ours to copy).

### 1.2 Colour tokens — DOM (`game/src/ui/ui.css:8-43`)

```css
:root {
  --bg: #12151c;            --surface: rgba(21, 25, 34, 0.95);  --surface-solid: #171b24;
  --surface-2: #1f2430;     --surface-3: #2a3040;               --surface-4: #343b4d;
  --line: rgba(216,222,233,0.09);  --line-2: rgba(216,222,233,0.17);  --line-3: rgba(216,222,233,0.28);
  --text: #eceff4;  --text-2: #d8dee9;  --muted: #a3adbf;  --dim: #7f8a9e;
  --good: #a3be8c;  --warn: #ebcb8b;  --bad: #d06b75;  --bad-text: #f2a0a8;
  --accent: #88c0d0;  --info: #81a1c1;  --agent: #b48ead;  --maint: #81a1c1;  --ink: #0e1117;
  --font-ui: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  --font-term: 'VT323', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  --font-pixel: 'Press Start 2P', ui-monospace, monospace;
  --gutter: 14px;  --bar-h: 46px;  --pop-top: calc(var(--gutter) + var(--bar-h) + 12px);
  --radius: 12px;  --radius-sm: 8px;
  --shadow: 0 14px 36px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.35);
  --shadow-sm: 0 6px 16px rgba(0,0,0,0.45);
}
```
Extra literal colours used in components: chip "maint" `#5e81ac`; mainttxt `#9cb8dc`; text on coloured buttons
`#18210f` (on good), `#2a2008` (on warn), `#0f1e24` (on accent), `#11141a` (scenario Start button);
maroon-on-white inside light bubble `#1a1f2b` on `#fbfcfe`, bubble "who" label `#5e81ac` (`ui.css:283-294`).
Terminal colours: bg `rgba(8,10,13,0.94)`, rows `#b9c1cd`, time `#6c7689`, level `#8d97a8`, AGENT `#88d4e6`,
WARN `#f0cf88`, ERROR `#f08b95`, FAULT `#ff8f99`/msg `#ffb3b9` on `rgba(208,107,117,0.12)`, prompt `#7fb87f`,
cursor block `#a3be8c` (`ui-shell.css:155-213`).

Semantic tints are always the token at low alpha: good `rgba(163,190,140,.08–.18)`, warn `rgba(235,203,139,.1–.2)`,
bad `rgba(208,107,117,.12–.25)`, accent `rgba(136,192,208,.08–.16)`, agent `rgba(180,142,173,.08–.16)`.

### 1.3 Colour tokens — 3D (`game/src/world/palette.js:10-21`, `environment.js:9`)

```js
export const C = {
  n0: '#2f3542', n1: '#3d4556', n2: '#4a5367', n3: '#5b667d',      // Nord polar night (lifted)
  s0: '#dfe4ec', s1: '#eaeef4', s2: '#f4f6f9',                      // snow storm
  f0: '#8fcfc8', f1: '#7cc7dc', f2: '#7ea6d6', f3: '#4f7fc4',      // frost
  red: '#e0525f', orange: '#f0874e', yellow: '#f5c445', green: '#78c25f', purple: '#a77fd0',
  bg: '#1a1f2b', ink: '#242933', steel: '#a9b2c0', glass: '#a8d4ec', sky: '#c6dbe6',
  wood: '#c98d58', woodDark: '#94633d', cardboard: '#d2a868', tape: '#efdcaa',
  skin: '#efc6a0', ground: '#272d39', asphalt: '#30363f', rubber: '#22252c',
};
export const LAMP = { green: '#5dff6e', amber: '#ffb52e', red: '#ff3448', blue: '#3fa4ff', off: '#20242d' };
export const FOG = '#1f2430';   // scene background + Fog(55,135)
```
"Nord-based palette, lifted and saturated so machines read clearly against the floor under ACES tone mapping"
(`palette.js:8-9`). Lamp colours are emissive (MeshBasic, `toneMapped:false`) so they read as light sources.
Selection ring `#7fe6ff`, hover ring `#ffffff` @0.6 (`World.js:212,230`). Burst colours `#7dff8a` / `#ffd04a`
(`World.js:673-674`). Dim lamp segments: `{green:'#24332a', amber:'#3b3426', red:'#3b272b', blue:'#242f40'}` (`machines.js:17`).

### 1.4 Fonts (`game/index.html:12-13`, `ui.css:1-6,32-34`)

Google Fonts: `Inter:wght@400;500;600;700`, `Press Start 2P`, `VT323`, `display=swap`, with `preconnect`.
- **Inter** — all panels; body 15px/1.45, `font-feature-settings:'cv11','ss01'`, antialiased (`ui.css:46-52`).
  Numbers always `font-variant-numeric: tabular-nums`.
- **VT323** — terminal log only, 19px/1.08 (`ui-shell.css:186`).
- **Press Start 2P** — title logo only (`ui-shell.css:309-311`): `clamp(24px,3.2vw,40px)`, `--warn` with hard text
  shadow `3px 3px 0 #d08770, 5px 5px 0 #000`; subtitle 0.55em, `letter-spacing:.5em`, `--accent`.
- Micro-labels everywhere: 11–12px, weight 700, `letter-spacing:.06–.08em`, uppercase, `--muted`
  (e.g. `.tb-k` `ui-shell.css:46`, `.ap-sub` `ui-agent.css:95`).

### 1.5 Panel recipe (copy exactly)

| Element | Background | Border | Radius | Shadow | Blur | Ref |
|---|---|---|---|---|---|---|
| Popover `.pop` | `--surface` | `1px --line-2` | 12 | `--shadow` | `blur(10px)` | `ui.css:147-152` |
| Top bar `.tb` | `--surface` | `1px --line-2` | 12 | `--shadow-sm` | 10px | `ui-shell.css:37-43` |
| Menu `.menu` | `--surface` | `1px --line-2` | 12 (items 9) | `--shadow` | 10px | `ui.css:166-176` |
| Modal `.mdl` | `--surface-solid` | `1px --line-2` | 14 | `--shadow` | none; backdrop `rgba(8,10,14,.55)` | `ui.css:201-215` |
| Approval `.ap-modal` | `--surface-solid` | `1px --line-2` + `border-top:4px --warn` (`--good` if within limits) | 16 | `--shadow` | backdrop `.6` | `ui-agent.css:251-260` |
| Terminal `.term` | `rgba(8,10,13,.94)` | `1px rgba(216,222,233,.16)` | 10 | `--shadow` | none | `ui-shell.css:155-161` |
| Card (scenario) | `--surface-solid` | `1px --line-2`, 4px coloured top bar `::before` | 14 | `0 8px 24px rgba(0,0,0,.35)` | — | `ui-shell.css:336-349` |
| KPI tile | `--surface-2` | `1px --line` | 10 | — | — | `dashboard.css:12-15` |
| Chip | `rgba(216,222,233,.12)`; filled variants per state | none | 5 | — | — | `ui.css:101-111` |
| Pill (float/hint) | state colour solid | 1px transparent | 999 | `--shadow-sm` | — | `ui.css:298-324` |

Animations: `pop-in` 0.14–0.18s (`translateY(-4px) scale(.985)` → none), `fade-in`, `blink` (opacity .35), status dots
use `steps(1)` blinking; bad hint chip pulses a 6px ring (`hint-pulse 1.6s`, `ui.css:319,324`); attention chip pulses a
yellow ring (`ap-ring 1.6s`, `ui-agent.css:5,35`). `prefers-reduced-motion` kills everything (`ui.css:361-363`).
Buttons: 600 14px, radius 8, `--surface-3` default, `.good/.bad/.accent` solid fill with dark text, `:active` 1px down
(`ui.css:75-92`). Range sliders: 6px track `--surface-4`, 18px `--warn` thumb with 3px `--surface-solid` border (`ui.css:124-128`).

### 1.6 HUD layout (`docs/UI_SPEC.md:8-25`, `ui-shell.css:8-18,36-152`)

Default screen = **3D world full window + 3 things**. Everything else is a popup.
1. **Top bar** top-left at 14px gutter, height 46px, one line, segments separated by 1px hairlines inset 10px:
   `● 08:06 Stopped` (lamp 10px dot with 3px halo; green/amber/red-blinking/blue/grey) · scenario name ▾ (opens picker) ·
   KPIs label-over-value (`REVENUE $14.3k` green, `DOWNTIME −$540` red, `PARTS $32.95`) · `BUDGET $967 LEFT` + 5px meter
   (green → warn → bad) · speed group `❚❚ 1× 2× 4× 8×` (34×28 buttons, active = accent fill; pause active = warn fill) ·
   `☰ Menu` (red 8px badge when something wants attention). Values flash green/red on change (`shell-flash-up/down 1.4s`).
   Responsive drop-outs at 1440/1320/1100/900 px (`ui-shell.css:133-152`).
2. **Agent status chip** top-centre under the bar, one per open incident: round 30px icon + `Robot Arm · Diagnosing… 1/7`
   + 7 progress dashes (14×4px; done=green, on=accent pulsing, fail=red). Tones: busy (accent border), attention
   (warn border + ring pulse, label "Needs your OK"), bad, good ("Fixed") (`ui-agent.css:8-43`, `agentpanel.js:20-39`).
3. **Terminal log** bottom-right (details §1.7).
4. In-world overlays: speech bubble over the robot, floating texts, hint chips over degraded/down machines only.
5. First-run nudge pill bottom-centre "Click any machine to see its sensors" (`ui.css:327-332`, `UI.js:113`).
6. Toasts bottom-left, 4px coloured left border (`ui.css:335-350`). Connection-lost banner top-centre red pill (`ui.css:353-357`).

Popups (one at a time; Esc / ✕ / click-outside): machine popover **right** (380px), agent popover **left** (430px),
☰ menu (300px) → centred modals (Scenarios, Manager's Desk, Chaos, Spare parts, Dashboard). When the machine popover is
open the log slides left by 392px (`ui-shell.css:16-18`). Keyboard: `Esc` close, `M` menu, `D` dashboard, `L` log size,
`Space` pause, `1-4` speed (`UI.js:590-620`, `hud.js:389-395`).
The world camera is told what screen edges the UI covers: `world.setViewInsets({top,right,bottom,left})` and a default
guess `{top:64, bottom:min(60,8%h)}` plus log room (`World.js:305-309,314-359,732-735`).

### 1.7 Terminal log (`game/src/ui/plantlog.js`, `ui-shell.css:154-219`)

- Size 410×210 (small), 660×430 (large, `L`), minimized = header only. Header 36px: blinking green LED + `plant.log`
  (+ unread count pill) · tab group `ALL · ALERTS · AGENT` in a dark pill (active = accent fill) · size + minimize icons.
- Row format: `08:06 FAULT E-ARM-310 Robot Arm Gripper position error – Position error 5.09° (limit 5°)`
  → spans `.tl-t` time, `.tl-lv` level (min-width 5.2ch), `.tl-code`, `.tl-m` machine (underlined, clickable → opens the
  machine popover), `.tl-msg`. Levels: `LOG_LEVELS = ['INFO','AGENT','WARN','ERROR','FAULT']` (`shared/contract.js:285`);
  ALERTS = rank ≥ WARN; AGENT = level AGENT (`plantlog.js:240-241`).
- New rows flash cyan 1.6s; FAULT rows get a red tinted background; prompt line `wrench-bot@plant:~$` + blinking
  block cursor at the bottom (`plantlog.js:61`). Max 400 rows (`plantlog.js:9`); autoscroll unless scrolled up, then a
  "jump to latest" pill (red if alerts). Prefs (filter/size/min) in `localStorage['wrenchbot.term.v1']`.
- Agent writes structured AGENT lines with short codes: `INC`, `DIAG`, `SOURCE`, `QUOTE`, `POLICY`, `APPROVAL`, `ORDER`,
  `PAY`, `VERIFY`, `PAYOUT`, `FIXED`, `RESOLVED` (seen in screenshots 07/08, `JUDGING.md` §2.4).

### 1.8 Speech bubble, floating text, hint chips (`ui.css:282-324`, `UI.js:494-517,693-781`)

- **Bubble:** white `#fbfcfe`, text `#1a1f2b` 15px/500, radius 12, padding `9px 13px 11px`, max-width 320, shadow
  `0 8px 22px rgba(0,0,0,.45)`, tail = 14px rotated square at `--tail` x, label `WRENCH-BOT` 11px uppercase `#5e81ac`.
  **Typewriter** at ~71 chars/s (`ceil(elapsed/14ms)`), lifetime `clamp(len×55ms, 3.5s, 9s)`, fades 250ms. Re-positioned
  every rAF from `world.screenPos('robot')`, clamped inside the stage and below the top bar (y ≥ top+64).
- **Floating text:** pill 13px/700, rises 46px over `FLOAT_MS = 2600` with fade in/out, stacks 30px per concurrent float,
  avoids the agent chips. Kinds: `bad` (`DOWN -$60/min`), `good` (`RUNNING`), `warn` (`WARN GRIPPER SERVO`), `info`
  (`PART REPLACED`), `maint` (`REPAIRING`) (`director.js:355-368`).
- **Hint chips:** only over degraded (amber `⚠ W-ARM-310 +2`) or down (red `✖ E-ARM-310`, pulsing) machines; clickable →
  machine popover. Positioned 12px above `screenPos(machineId)` each frame.
- **3D "?" thought bubble** (camera-facing canvas sprite) over the robot whenever any incident is unsure
  (confidence < threshold) or the UI needs a human (`director.js:421-424`, `robot.js:115-149`).

### 1.9 Alert rings (3D) (`machines.js:42-48,135-169`, `palette.js:179-194`, `effects.js:48-60,126-132`)

- **Status halo:** a 6.2×6.2 floor plane with a radial "ring" canvas texture (transparent centre 0.18, bright band at
  0.74, fade to 0 at 1.0), `MeshBasicMaterial` additive-looking via transparency, `toneMapped:false`, `depthWrite:false`,
  `renderOrder 1`, y=0.075. Down = red, opacity blinks 1 / 0.5 at 2.2 Hz (55% duty); degraded = amber, `0.68 ± 0.14·sin(3t)`;
  maintenance = blue `0.55 ± 0.08·sin(2t)`; running = hidden.
- **Lamp flare:** soft radial sprite (`softGlowTexture`) on the lit lamp, scale 2.3 down / 1.9 warn / 1.4 ok —
  explicitly "cheap stand-in for bloom" (`machines.js:90-96`).
- **Event rings:** pool of 6 `RingGeometry(0.88,1,64)` floor rings, expand `from→to` over `dur` with ease-out and
  opacity `(1-k)·0.9`. Down: red 0.5→2.6 in 0.6s + 16 sparks + 3 soot; back running: green 0.6→2.8 in 0.8s + twinkles;
  maintenance: blue 0.6→2.4 in 0.7s; part replaced: 64 confetti + green `#7dff8a` 0.6→3.4 + yellow `#ffd04a` 0.3→2.4
  (`machines.js:115-128`, `World.js:668-675`).
- **Selection reticle:** cyan `#7fe6ff` ring r 2.05–2.3 with 4 tick marks rotating 0.6 rad/s, pulse `scale 1+0.04·pulse`,
  opacity `0.6+0.35·pulse` at 4 rad/s; white hover ring (`World.js:209-234,516-529`).
- **Down machine shake:** bursts of 0.45s every 1.6s, amplitude 0.08 (`machines.js:188-198`).

### 1.10 Dashboard modal (`dashboard.css`, `dashboard.js`, screenshot 09)

`width:min(1320px,90vw); height:min(85vh,1000px)`. Header title 19px/700 + one-line subtitle. Row of KPI tiles
(`grid auto-fit minmax(170px,1fr)`, gap 10): label 12px/600 uppercase muted, value 24px/700 tabular (coloured per
tile), small caption; one wide tile with a 10px stacked split bar (auto green / manager info-blue / blocked red) + legend.
Sticky tab bar: uppercase 13px/700 tabs with count pills (active pill = accent), right side "● Live" blinking + `Export CSV`
+ `Clear history`. Tables: 14px, header 12px uppercase on `--surface-2`, rows 10×12 padding, hover tint, blocked rows red
tint with struck-through amount and red reason line, badge pills (`b-reap` info, `b-chain` good, `demo` warn uppercase).
Timeline tab: 32px round tone-coloured icon + title + detail + `·`-separated meta. New live entries flash cyan 2.6s.
Empty state: dashed box, round icon, bold title, 460px paragraph. Data from `GET /api/ledger` + `ledger.entry` SSE.

### 1.11 Scenario picker (`scenarios.js`, `ui-shell.css:295-395`, screenshot 01)

Full-screen overlay `radial-gradient(ellipse at 50% 20%, rgba(38,44,58,.97) 0%, rgba(10,12,17,.985) 70%)`, z 100,
inner max 1180px. Head: canvas pixel-art robot sprite (115×90, `image-rendering:pixelated`, `drop-shadow(4px 4px 0 #000)`,
bob `steps(2)` 1.6s) + pixel logo + 17px subtitle. Numbered step strip (`1 Read sensors ▸ 2 Read the plant log ▸ …`,
22px accent circles). Label row `CHOOSE A SCENARIO` (warn, uppercase) ↔ "Each one starts a fresh shift at 08:00".
Grid `auto-fill minmax(320px,1fr)` gap 14 → cards: per-scenario `--scn-c` colour (warn/bad/accent/good) drives the 4px top
bar, the outline tag pill (`AUTO-BUY`, `ROOT CAUSE`, `PREDICTIVE`, `BUDGET`, `ALLOWLIST`, `FREE PLAY`), the "TEACHES" label
and the `Start ▸` button; 10×10 two-colour pixel icon per card (`scenarios.js:42-61`). Footer: mode badges
(`Reap offline · AI live · Card enrolled`, `hud.js:40-54`) and "Arrows move · Enter starts". Whole card is the button; no
hover transforms ("a card that moves under the pointer can swallow the click", `ui-shell.css:334-335`). Shown on first load;
not closable until a scenario was picked.

### 1.12 Inspector / machine popover (`inspector.js`, `ui-shell.css:221-293`, screenshot 05)

Header: name 20px/700 + status chips (`RUNNING/DEGRADED/DOWN/MAINTENANCE`, `CRITICAL` outline tag) + cost line
(`$60/min when stopped`, red when bleeding). Alarm rows (code bold + text, 3px coloured left border). Caption
"Live sensor data — this is what Wrench-bot reads." One collapsible row per part: status dot (blinking if fault), name,
its most abnormal reading on the right. Expanded part: `WARNING FOR 2 MIN` chip, code + fault text, "Also drags down: …"
(cross-effects), then per signal: label + big value (18px/700 coloured by status) + unit; **direction-aware threshold
gauge** (8px, three zones ok/warn/fail tinted at .2/.24/.42 alpha, fill bar, 3px white marker); caption
`lower is worse · nominal 24.1 · warn 23.2 · fail 19.5 V`; 34px canvas sparkline (`HIST = 90` samples, dpr ≤ 2,
seeded from `GET /api/sim/history/:id`). Gauges extend 15% past fail (`OVER = 1.15`, `inspector.js:12-13`).

### 1.13 Agent popover + approval card (`agentpanel.js`, `ui-agent.css:45-281`, screenshots 04/06/07)

- Header: CSS-drawn robot avatar (34×28, eyes scanning when busy), `WRENCH-BOT` in warn 800 letter-spaced, state line
  (`Needs your OK` in warn), provider chip (`AI LIVE` agent purple). Scenario brief collapsible with "TEACHES" box.
- Card: kind chips (`BREAKDOWN` red / `PREDICTIVE` amber + `FIXED` green), title 17px, meta line (`down 10 min · −$600`),
  7-segment strip, then **step tracker** (`Diagnose → Source → Quote → Approve → Ship → Repair → Verify`), each row
  `18px marker | 74px label | summary` (done = filled green ring, on = accent pulsing, fail = red, todo = dim).
- **Diagnosis block:** legend `▬ Decisions API ▬ fault-code rule`; per candidate a grid `name | bars | % (+ prior % small)`;
  model bar 8px accent (top) / 55% accent (others), prior bar 4px `rgba(216,222,233,.4)` underneath. "EVIDENCE FROM THE
  LOG" bullet list (warn/bad coloured), agent quote box (3px purple left border), source line
  `openai-decisions · 88% sure (below your 75% bar) · view signals`. Ruled-out parts shown struck through.
- **Receipt** (quote): rows label/value, dashed separator, total in green 16px. **Policy verdict** box with 3px left border
  (green AUTO / warn ESCALATE / red BLOCK) + reasons list.
- **Resolved card:** green tinted "Back online", root cause, grid Parts/Labor/Downtime/Total cost, "approved by you · order #…".
- **Error card:** red tinted, chip + hint from `ERROR_HINTS` (8 codes, each with **Retry**; BLOCKED also "Open Manager's
  Desk") (`agentpanel.js:47-56`).
- **Approval modal** (`ApprovalModal`, `agentpanel.js:1172-1300`): chip `APPROVAL NEEDED`; title `Needs your decision`
  (or `Within your limits — confirm … (one tap)` with green top border); sub "Wrench-bot stopped before paying: this order is
  outside the rules you set."; incident line; item row with 46px white image tile; `Total $69.69` 24px/800 + "USD incl.
  shipping + tax"; `WHY IT ASKS YOU` bullet reasons (verbatim policy reason strings); `Diagnosis: Network switch at 95%
  confidence`; **live cost of waiting** `The Packer is down: −$35/min while you decide (waited N min, $X)`
  (`agentpanel.js:1266`); buttons: full-width green `Approve $69.69` + red `Reject`; footnote explaining offline stand-in.
  Pops by itself on `checkout.approval_required` — "the one essential action" (`director.js:247-251`).

### 1.14 Manager's Desk and Chaos (`desk.js`, `chaos.js`, `ui.css:217-264`)

- **Desk:** one-sentence policy in a cyan-tinted box: "Wrench-bot pays on its own when the order is at most **$60**, it is
  at least **75%** sure, the store is trusted and the budget covers it. Anything else comes to you." (`desk.js:160-162`).
  Sliders: `AUTO-APPROVE LIMIT` 0–300 step 5, `MONTHLY BUDGET` 100–3000 step 50 (+ budget bar), `CONFIDENCE NEEDED`
  0.5–0.95 step 0.05 (`desk.js:10-14`); value in warn 20px/700. Toggle switch "Predictive maintenance" (38×22). Trusted
  store checkboxes; a store that blocked an order is tinted red + chip `BLOCKED AN ORDER`, list flashes 3× warn ring.
  Never overwrites a control the user is dragging (`desk.js:1-3`). `PUT /api/policy` on change.
- **Chaos:** selects MACHINE / PART (+ live status chip and current readings), 2-up mode cards `Sudden` / `Gradual`
  (selected = warn border), full-width red `BREAK` button; note "This only changes the simulation; the agent still has to
  notice the failure from signals and the plant log" (`chaos.js:1-2`). Refused with 409 BUSY during a repair.

---

## 2. 3D rendering approach

| Aspect | Exact setting | Ref |
|---|---|---|
| Renderer | `WebGLRenderer({antialias:true, powerPreference:'high-performance'})`, `outputColorSpace = SRGBColorSpace`, `toneMapping = ACESFilmicToneMapping`, exposure 1.0 | `World.js:96-102` |
| Shadows | `PCFSoftShadowMap`; one shadow caster (sun), map 2048², ortho box ±21×±16, near 2 far 60, `bias -0.0005`, `normalBias 0.025` | `World.js:103-104,154-161` |
| Env lighting | `PMREMGenerator.fromScene(new RoomEnvironment(), 0.04)`, `scene.environmentIntensity = 0.42` (in try/catch) | `World.js:108-117` |
| Lights | `HemisphereLight('#eef3ff','#5a5266',1.25)` + sun `DirectionalLight('#fff3e0',2.5)` at (-6,20,12) + cool fill `DirectionalLight('#b4c8ff',0.6)` at (16,7,-5). **No point lights** ("expensive per pixel") | `World.js:152-166`, `machines.js:1-3` |
| Background/fog | `scene.background = FOG #1f2430`, `Fog(FOG,55,135)`; ground plane with radial gradient fading into fog | `World.js:72-73`, `environment.js:80-95` |
| Materials | `MeshStandardMaterial` everywhere: roughness 0.68 / metalness 0.02 (matte), metals (`steel`) 0.42/0.35. **Not toon, not flat-shaded**: smooth normals on bevelled boxes. Emissive things (lamps, screens, LEDs, windows, particles) = `MeshBasicMaterial({toneMapped:false})` | `palette.js:59-85` |
| Geometry | Procedural primitives only, no GLTF assets: `RoundedBoxGeometry(w,h,d,1,r)` with `r=min(0.06, 0.22·min)`, plain `BoxGeometry` for <0.05 or textured faces, `CylinderGeometry` with ≥18 segments (≤4 = deliberate prism). All geometries cached by key in a `Map` | `palette.js:23-57` |
| Helpers | `box()` (bottom at y), `boxC()` (centred), `cyl(axis)`, `glyph(rows,px)` voxel text from `'#'` strings | `palette.js:107-131,196-212` |
| Textures | All canvas-drawn (`canvasTex(w,h,draw)` supersampled to ~256px, `LinearMipmapLinear`, anisotropy 8, sRGB) — floor tiles with deterministic speckle, wall ribs, screens, ring/glow sprites | `palette.js:133-194`, `environment.js:46-70` |
| Per-object hover | `matSet()` gives each machine its own material set so hover sets `emissive` `(0.13,0.18,0.26)` without touching shared caches | `palette.js:87-98`, `machines.js:20,131-133` |
| Camera | `PerspectiveCamera(FOV 30, near 0.5, far 400)`; default yaw 0.15 rad, pitch 0.68 rad; pose = `{target, dist, yaw, pitch}` | `World.js:34-37,119` |
| Auto-framing | `#fitPose()` iterates 40× projecting a list of `FIT_POINTS` through a scratch camera to fit them inside the UI-free rectangle (insets) with margin 0.97 | `World.js:43-50,314-359` |
| Controls | `OrbitControls`: damping 0.08, rotate 0.45, zoom 0.9, pan 0.8, `screenSpacePanning=false`, `zoomToCursor=true`, distance 8–44, polar 0.32–1.3, azimuth −0.55–1.55; orbit target clamped to a box | `World.js:168-196,417-427` |
| Camera choreography | `#flyTo(pose,dur)` with `easeInOutCubic` lerp of target/dist/yaw/pitch; intro swoop 2.2s from `dist×1.55, yaw+0.45, pitch+0.2`; context views (home ↔ dock) flown automatically 1.3–1.6s; **auto-framing stops once the user drags/zooms** (`userMoved`), double-click empty floor = `resetView()` | `World.js:361-415` |
| Picking | invisible generous hitboxes (`MeshBasicMaterial({visible:false})` 3.7×4.7×3.8) + Raycaster; click vs drag threshold 6px; pointer cursor on hover | `machines.js:18-19,49-52`, `World.js:236-300` |
| Screen anchoring | `screenPos(name)` projects a world point → CSS px (null if behind camera); canvas rect re-measured every 250ms | `World.js:712-728` |
| Post-processing | **None.** Bloom faked with soft glow sprites and emissive basic materials | `machines.js:90` |
| Particles | `Effects`: two `InstancedMesh` pools (320 glow cubes + 200 soft icosahedron puffs), per-instance colour, capped, zero allocation per frame, `frustumCulled=false`. Kinds: spark, smoke, soot, confetti, poof, twinkle, dust with gravity/drag/bounce/grow params | `effects.js:10-46,139-181` |
| Performance | adaptive pixel ratio: cap 2, averaged over 90 frames after 4s warm-up; if avg > 1/40 s, step down 0.25 to 1 (never up) | `World.js:97-99,448-467` |
| Robustness | animation loop wrapped in try/catch, logs first 3 failures only; `dt` clamped to 0.05; camera fly uses wall time (≤0.2); world fails → no-op `Proxy` + "3D view unavailable" message, UI still works | `World.js:478-514`, `game/src/main.js:15-27` |
| Animation | each actor has `update(dt,t)`; machine process motion synced to the belt phase; robot walks waypoints with speed `clamp(dist/1.7,3.5,9.5)` so every walk ≲2s (agent waits ≤2.5s); idle bob, eye-scan beam, `easeOutBack` pop-ins | `World.js:478-509`, `robot.js:170-183`, `machines.js:9-10` |

Director pattern (`game/src/director.js`): the browser runs **no game logic**. `Director.handle(evt)` switches on
`EVENTS.*` and calls only the narrow **World API** (`setMachineStatus`, `robotGoTo`, `spawnTruck`, `technicianArrive`,
`partReplaced`, `focusMachine`, `screenPos`, …, `World.js:7-16`) and the UI facade, each call wrapped in `call()` try/catch
(`director.js:10-18`). Time mapping: `spawnTruck(etaMinutes / sim.speed)` real seconds (`director.js:269-276`). Floating
texts come from diffing consecutive `sim.tick` snapshots (`director.js:354-368`). Robot choreography: incident created →
machine; searching → terminal; delivery arrived → dock; resolved/error → home or the next open incident (`director.js:399-418`).
Boot: `GET /api/state` retried with backoff and a boot screen ("Connecting to the plant…") until the server answers
(`director.js:41-54`, `index.html:17-31`).

---

## 3. Simulator patterns

- **Clock:** server `setInterval(TICK_MS=1000)`; each tick advances `dt = speed` game-minutes, `SPEEDS=[1,2,4,8]`,
  sub-stepped in ≤1-minute slices so log timestamps/waits stay exact; paused → `step(0)` but still broadcasts `sim.tick`
  (`engine.js:9-10,552-558`, `SIMULATION.md` §2.1). Shift clock `HH:MM` from 08:00. Seeded `mulberry32` PRNG reseeded on
  preset load; `loadPreset(id,{seed})` replays exactly.
- **Clock holds (key idea):** while the agent works in *real* time (model calls, catalog, quote, payment, waiting for the
  manager) the game clock is **held** (`sim.hold(incidentId,true)`), so a slow API never shows up as extra downtime at 8×.
  Only `ctx.minutes(n)` (shipping/travel/repair/test run) releases it; the snapshot exposes `held:true` so the UI can tell
  "agent working" from "user paused" (`engine.js:118-124,443-455`, `orchestrator.js:268-274,302-310`). Animation pauses
  shrink by `1/√speed`.
- **Hidden truth → observed signals (information barrier):** each part has hidden `health∈[0,1]` that "never leaves
  engine.js". `d(h)=clamp((0.7−h)/0.7,0,1)^1.6`; signal `value = nominal + (fail−nominal)·frac + noise(σ=0.8% span)` where
  `frac` = own `d(h)` + Σ weighted `d()` of *upstream* parts (`effects`) capped at 1.15; status from the **noiseless** frac
  (`fault ≥ 0.97`, `warn ≥ warnFrac`) to avoid flapping (`docs/SIM_SPEC.md:38-68`, `SIMULATION.md` §1). The agent sees
  only `telemetryContext()` (value, thresholds, severity, `trend10m`, warn minutes) + last 20 log lines. Chaos injection
  writes **no** log line ("chaos is invisible, the plant only shows the symptoms").
- **Log rules written by the engine** (WARN on ok→warn naming the worst signal; repeat every 5 game-min with trend if moving
  ≥3% of span, else every 15; FAULT with limit; INFO cleared; ERROR `M-DOWN`; INFO `M-UP`; `MAINT`; `SHIFT`)
  (`SIM_SPEC.md:94-111`, `engine.js:12-16`).
- **Scenarios:** declarative `PRESETS` in the shared contract: `{id, name, tagline, teaches, policy overrides, initial
  health, wear, failureEvery, script:[{at,target,health}|{at,target,degradeTo,over}]}` (`shared/contract.js:180-250`).
  Each preset = **one idea** (Auto-buy, Root cause, Predictive, Budget, Allowlist, Free play). Loading a preset cancels
  all incidents, resets policy, KPIs, log and clock. Boot loads `first-failure` **paused** until the user picks
  (`server/src/index.js:259-265`).
- **One failure at a time:** `busy()` true while anything is down/faulted/ramping or an incident is open → chaos refused
  (409 BUSY), scripted events held until 2 calm minutes, free-play scheduler waits (`engine.js:421-441`).
  Free play: weighted-by-`1/life` random part every 6–12 calm minutes, 30% sudden / 70% gradual.
- **Chaos tools:** `POST /api/sim/fault {machineId, componentId, mode:'sudden'|'gradual'}` → health 0 now or ramp to 0
  over 12 min (`index.js:134-145`, `engine.js:22`).
- **Baseline inside the sim:** `diagnostics.js` naive "System 1" prior `score = 0.6·maxSeverity + 0.4·hasFaultCode`,
  `softmax(score/0.25)`; deliberately trusts fault codes so it fails in Brownout (`diagnostics.js:9,27-37`). It is the
  offline mock *and* the faint comparison bar in the UI, but is **withheld from the model** (measured: with the prior the
  model copied it — servo 0.95; without it, PSU 0.59) (`AGENT.md` §5.3, `decide.js:80-82`).
- **Ground-truth verification:** after the fix, `verify()` trusts `sim.isHealthy()` (a status check, not a health peek);
  the model's yes/no is only shown. Wrong fix → part ruled out → re-diagnose, `MAX_ATTEMPTS = 3` (`orchestrator.js:26,322-371`).
- **Director/camera choreography:** see §2 (fly-to poses, intro swoop, dock view while a delivery is on, robot walks
  timed ≤2.5 s to match the server's `ctx.pause(2500)`, `orchestrator.js:373-381`).

---

## 4. Backend patterns

- **Single process, all state** (`server/src/index.js`): sim + agent + ledger + pollers in one Express app; serves
  `game/dist` statically in prod (one Render web service, `render.yaml`, health check `/api/health`).
- **SSE event bus** (`server/src/events.js`, 52 lines): `sseHandler` sets `text/event-stream`, `no-cache`, keep-alive,
  `flushHeaders`, **sends a full `state` snapshot on connect**, `: ping` comment every 15s; `emit(type,data,incidentId)`
  builds `{type, incidentId, data, at}` → writes `data: <json>\n\n` to every client and calls `onEmit` listeners (ledger)
  in try/catch. `setSnapshot(fn)` injects the STATE builder.
- **Client resilience** (`game/src/main.js:29-95`): stale-stream watchdog (no message for 8s → reconnect; server ticks
  every 1s even when paused), hidden tab closes its EventSource after 20s (browsers allow ~6 HTTP/1.1 connections per host;
  see MDN https://developer.mozilla.org/en-US/docs/Web/API/EventSource), any click/keypress reopens; connection banner.
- **Event contract** in `shared/contract.js:254-282`: 27 named constants — `state`,
  `sim.tick {sim}`, `sim.preset`, `log.entry {entry}`, `incident.created/cancelled/resolved/error`, `agent.thinking {text}`,
  `agent.diagnosis {componentId, probabilities, prior, confidence, provider, attempt, evidence[], explanation}`,
  `policy.decision {action, reasons, total, confidence}`, `checkout.approval_required`, …, `policy.updated`, `ledger.entry`.
  The contract file also holds MACHINES (signals with nominal/warn/fail/digits, effects) and PRESETS and is imported by
  **both** server and game (Vite `fs.allow ['..']`).
- **REST** (`index.js:81-240`): `GET /api/events` (SSE), `GET /api/state`, `GET /api/health` (mode flags),
  `POST /api/sim/preset|speed|pause|resume|fault`, `GET /api/sim/history/:id`, `GET /api/sim/logs?machineId&limit&minLevel`,
  `GET /api/incidents`, `POST /api/incidents/:id/retry`, `GET|PUT /api/policy`, `GET /api/ledger`, `POST /api/ledger/clear`,
  `POST /api/mock/approve/:id`. Validation returns 400 with a JSON `{error}` and lists valid values; catch-all
  `/api` 404 JSON; error middleware converts bad JSON to 400 `Invalid JSON body`; `process.on('unhandledRejection')` guard
  (`index.js:240-257`). **No endpoint creates incidents** — only the monitor does.
- **Orchestrator pipeline** (`orchestrator.js`): monitor listens to `sim.on('machine.down')` and per-minute warnings
  (predictive after `PREDICTIVE_AFTER_MIN=3`); `startIncident` → `launch()` with one `AbortController` per incident;
  `context()` gives `guard(promise)`, `sleep`, `pause`, `minutes(n)`, `step(name)` (freezes while paused), `think(text)`,
  `log(code,msg)`; `run()` loop: diagnose → source → quote → **policy** → pay → ship → repair → verify → resolve/re-diagnose;
  failures become `IncidentStop(code,msg)` → `incident.error {code,message,retryable:true}` + ERROR log line + toast.
  Error codes: `BLOCKED, NO_PARTS, NO_QUOTE, NO_CARD, CHECKOUT_EXPIRED/TIMEOUT/FAILED, GAVE_UP` (`ARCHITECTURE.md` §7.3).
- **decide.js — one interface for every typed decision** (`decide.js:1-57`):
  `decide({kind:'choice'|'score'|'yesno', question, options|levels, context})` →
  `{answer, probabilities:{opt:p}, confidence, provider}`. Provider chain: `openaiDecisions` → `jev` (stub) →
  `gptStructured` (Chat Completions + strict `json_schema`) → `mock`; `MOCK_AI` short-circuits to mock; every failure
  logged `[decide] <provider> failed: …`; `finish()` clamps, normalises to 4 dp, argmax = answer, max p = confidence.
  Decisions API call: `POST https://api.openai.com/v1/decisions`, `model:'gpt-6-luna'`, `input` = plain text built by
  `contextText()` (each context key on its own line; telemetry as one line per part:
  `Part [warn 3m]: Signal 21.2V (nominal 24.1, warn 23.2, fail 19.5, severity 0.63, trend -2.52)`), one question:
  `predicate` for yes/no, `score` with `levels:[{label}]`, `choice` with `choices:[{value:'o1',description:<option>}]`
  (**opaque values o1…oN so any option text is safe**), `AbortSignal.timeout(20s)`, `refusal` → throw → next provider
  (`decide.js:63-98`). Root-cause guidance string appended for choice questions (`decide.js:100-103`). Free-text `say()`
  (≤18 words) only for bubbles/explanations, with deterministic fallback; never feeds a decision (`decide.js:224-244`).
- **Policy (code, never model)** (`policy.js`): `DEFAULT_POLICY` frozen; `EDITABLE` whitelist; `clean()` type-validates
  every patch; `reset(overrides)` per preset; `evaluate({total, merchant, confidence})` → BLOCK first (allowlist, single
  reason), else collect reasons (over remaining budget / over auto-approve limit / `Not sure enough (59% < 75%)`) → any
  reason = ESCALATE, none = AUTO with a "Within limits: …" reason. **Reason strings are user-facing and shown verbatim**
  in the log, modal and dashboard. Spend recorded only after money moves.
- **Ledger / audit** (`ledger.js`): built purely by observing `onEmit` events (pipeline can't forget to record);
  entries `{id, at ISO, t, clock, preset, incidentId, machineId, type, title, detail, amount, approval, confidence,
  provider, simulated, …}`, max 1000, persisted debounced 1s to `server/.cache/ledger.json` via tmp+rename, `totals`
  computed on read; CSV export client-side (`shared/contract.js:296-313`).
- **Mock mode & config** (`config.js`): dotenv from repo root `.env`; `mockAi = MOCK_AI==='1' || !OPENAI_API_KEY`
  (same for Reap) — *missing key ⇒ mock automatically*; `JUDGE_MODE` for the public demo (live AI, simulated checkout,
  idle reset to the first scenario after 10 min); mode flags exposed in `/api/health` and `STATE.mode` and rendered as
  badges ("AI live" / "AI mock"). `.env.example` has only empty placeholders; secrets `sync:false` in `render.yaml`.
- **Shared contract discipline:** `SIM_SPEC.md` assigned files to builders and "don't change signatures"; the contract
  is the single source of truth for both sides.

---

## 5. Docs / judging patterns

- **README** (`README.md`): badges row (integration · model) → one-line value prop → hero screenshot → **"Project
  highlights" table `Claim | Where to verify`** (each row = bold claim + file links or tx links) → problem paragraph →
  "What happens when X" 2-column screenshot/text table → authority table `Who | Can do | Enforced where` → integrations
  table `System | Status (Live/Simulated) | Proof` → scenarios table `Scenario | What you'll see` → mermaid architecture →
  quick start (3 commands + `dev:mock` with no keys) → "What we learned" (verified API quirks) → Limitations.
- **JUDGING.md** ("for hackathon judges and for LLMs asked to score this repository… If a claim has no pointer, ignore
  it"): TL;DR (6 bullets) + mermaid loop → **§1 criteria → evidence matrix**: each judging criterion quoted verbatim, then
  numbered claims `T1…T15 | Claim | Evidence (code: path → function) | Observe (what you see in the app)`; a states table
  (every approval/error state → trigger in code → exact UI wording); observed-run table (scenario → verdict path → result,
  with date and mode) → §2 verify in 5 minutes (live demo / local; scenario → claim → on screen → log → dashboard; server
  log lines to watch; curl commands) → §3 live vs simulated with proof → §4 originality (comparison table + "five design
  decisions, with the reasons") → §5 honest limitations table `Limitation | Where | Impact` + what we'd build next → §6
  quick reference for code reviewers `Question | Read`.
- **AGENT.md:** exact model input printed verbatim; measured ablation table (what the model received → top pick → p);
  worked example minute by minute; Path A vs Path B comparison with costs. **SIMULATION.md:** information-barrier mermaid,
  time model, formulas, "Why diagnosis is genuinely hard", "Reproduce the numbers", "Where the code differs from SIM_SPEC".
  **ARCHITECTURE.md:** key-numbers table, event catalogue, two clocks, cancellation, fallback chains mermaid + table
  `Dependency | Primary | Fallback | Trigger | Visible to the user as`, error-code table.

These map almost 1:1 onto our rubric (Technical soundness, Honest evaluation, Trust & traceability): reuse the
criterion-quoted matrix with `Claim | Evidence | Observe` and add an "Honest evaluation" section with baseline vs model
metrics and a named failure case.

---

## 6. Translation table — factory → whole-Earth orbital collision avoidance

Design rule carried over: **one shared contract, hidden truth on the server, the agent and UI see only observed CDMs,
policy in code, every step an event + an ops.log line, one idea per scenario, model shown next to a naive baseline.**

| Factory element (ref) | Orbital analogue (proposal) | Notes / data mapping |
|---|---|---|
| Factory floor diorama (`environment.js`) | Stylised **Earth globe** in space: smooth sphere (`SphereGeometry`/`IcosahedronGeometry`, MeshStandard, canvas-drawn pastel continents), thin atmosphere glow sprite (same `softGlowTexture` trick), dark `#1f2430`-ish space backdrop with sparse star points, day/night via the "sun" DirectionalLight | Keep ACES + Hemisphere + sun + fill + RoomEnvironment 0.42; no post-processing; scale 1 unit = 1 Earth radius (6,371 km) with satellites/debris exaggerated (label it) |
| Production line / belt items flowing | Background **debris field**: thousands of instanced points/cubes on orbits, flowing while sim runs, frozen when paused | `InstancedMesh` pool pattern from `effects.js`; counts by object type (DEBRIS/UNKNOWN/PAYLOAD/ROCKET BODY) |
| Machine (4, `MACHINES`) | **Protected satellite / mission** (4–6 featured of the 19 train missions) | `mission_id`; orbit from `t_j2k_sma/ecc/inc` (+ assumed RAAN/arg — documented reconstruction, see 00-context) |
| Part / component | **Conjunction event** (satellite–object pair) on that satellite | `(split, event_id)` key; chaser `c_object_type` |
| Signals with nominal/warn/fail | **CDM fields** with tier thresholds: `risk` (log10 Pc) warn −6, fail −4; `miss_distance` warn 5,000 m, fail 1,000 m ("lower is worse"); plus `time_to_tca`, `relative_speed`, `c_sigma_r`, `t_sigma_r`, `mahalanobis_distance` | Direction-aware gauge already exists in `inspector.js` |
| Hidden part `health` | **Hidden final CDM** (the true risk/miss at TCA), never sent to UI or model until TCA passes | Exactly the Kelvins setup: test only shows CDMs ≥2 days out |
| Signal noise / cross-part `effects` | CDM-to-CDM update noise; covariance inflation (`c_sigma_*`), space weather `F10`/`AP` raising drag uncertainty | "Effects" = drivers that move risk without the alarm field saying so |
| Andon lamp green/amber/red/blue | Satellite status: **Safe** green / **Elevated** amber / **Critical** red (brief thresholds) / **Manoeuvre planned** blue | Reuse `LAMP` colours |
| Red pulsing floor halo | Pulsing **ring/billboard at the TCA point** on the orbit + red orbit arc segment | Ring texture from `ringGlowTexture()` billboarded |
| Sparks/smoke on down machine | Glints/"encounter ellipsoid" flicker at the conjunction point | Same `Effects` kinds |
| Fault code `E-ARM-310` / warn `W-ARM-310` | Event codes, e.g. `CRIT-PC E-<mission>-<event>` / `ELEV-MISS W-…` | Keep the E/W prefix convention |
| Wrench-bot (agent avatar + speech bubble) | Ops-agent mascot ("CARA-bot"-like) — a small satellite/drone that flies to the conjunction; bubble anchored to it | Name TBD; don't impersonate NASA CARA |
| Technician (human contractor) | **Flight-dynamics analyst / operator** who approves and executes the manoeuvre | Human decides (rubric) |
| Truck delivery / dock (`spawnTruck(eta/speed)`) | **Ground-station pass / uplink window**: beam from a ground station to the satellite; ETA in game time scaled by speed | Gives the "wait for real-world time" step |
| Procurement terminal | Screening console / mission-control desk | Agent "walks" there when querying |
| Spare-parts catalog (Reap) | Manoeuvre option catalogue (e.g. along-track/radial burns with Δv, post-burn miss) — or drop | Keep only if time allows |
| Purchase / checkout | **Manoeuvre request** (costs Δv/propellant, mission downtime) | |
| Policy AUTO / ESCALATE / BLOCK | **AUTO** = auto-log & keep monitoring (safe, confident); **ESCALATE** = flag for analyst (elevated/critical, or confidence < bar, or Δv over limit); **BLOCK/REFUSE** = data unfit (sentinel sigma 63,781,363, NaN covariance, stale CDM) → "cannot assess" | Reason strings verbatim, e.g. `Miss distance 812 m < 1,000 m`, `Not sure enough (62% < 75%)`, `Covariance unknown (sentinel)` |
| Monthly budget / auto-approve limit | Δv budget per mission / max auto-plannable Δv; alert budget per 24 h (analyst workload) | |
| Trusted store allowlist | Trusted data quality gate (e.g. `obs_used`, `od_span`, `weighted_rms` thresholds) / object types | |
| Confidence threshold 0.75 | Classifier tier confidence threshold | Same slider |
| Downtime cost $/min | Exposure: hours to TCA ticking down; "−18 h to TCA while you decide" | Replaces "−$60/min while you decide" |
| Shift clock 08:00, 1×/2×/4×/8× | UTC mission clock + **T-minus to TCA**; e.g. 1 tick = N hours of CDM time | 7 days of CDMs at 8 h/tick ≈ 21 s per event |
| Clock holds during real-time work | Same: hold sim time while model/LLM calls run | Copy verbatim |
| `plant.log` + `wrench-bot@plant:~$` | **`ops.log`** + `cara@ops:~$` (name TBD); levels INFO/AGENT/WARN/ERROR/CRITICAL | ALL · ALERTS · AGENT tabs unchanged |
| Inspector machine popover | Satellite inspector: list of its conjunctions, each with risk/miss gauges + **sparkline of the CDM sequence vs time_to_tca** | "This is what the model reads" caption |
| Fault-code rule (faint bar) | **Naive baseline** bar: e.g. "last-CDM rule" (final risk = latest observed risk) or threshold on `max_risk_estimate` | Kelvins baseline was the latest-CDM prediction (Uriot et al. 2021, https://arxiv.org/abs/2008.03069) — verify before citing in pitch |
| Brownout (alarm points at wrong part) | Scenario where the early CDM looks safe but the trend/covariance says it's closing (or vice-versa: high `max_risk_estimate` but huge sigma) | "The alarm field misleads" story |
| verify() against `sim.isHealthy()` | **Reveal final CDM at TCA** → was the flag right? closed loop, counts TP/FP/FN | Honest evaluation in-app |
| Re-diagnose up to 3 attempts | Re-assess on each new CDM; flag history kept | |
| Chaos tools (sudden/gradual break) | Inject: shrink miss distance, inflate/sentinel covariance, solar storm (F10/AP spike), drop a CDM update | Chaos invisible to the agent except via symptoms |
| Manager's Desk | **Risk policy desk**: Pc thresholds (1e-6 / 1e-4), miss thresholds (5 / 1 km), confidence bar, Δv/alert budget, refuse-on-sentinel toggle | One-sentence rule text at top |
| Manager dashboard | **Ops dashboard**: tiles = events in next 24 h within dangerous distance, critical/elevated counts, model vs baseline precision/recall, refusals, avg lead time; tabs Alerts · Agent activity · Refusals; CSV export | Rubric "dashboard + 24 h warnings + insights" |
| Ledger (`ledger.json`) | Audit ledger of every flag, decision, approval, refusal with the CDM timestamp and rule applied | Rubric "trace to timestamped signal / applied rule" |
| Scenario picker (6 cards, one idea each) | e.g. Quiet day (all safe) · Close approach (miss < 1 km) · Creeping Pc (trend) · Bad covariance (refusal) · Alert storm (24 h triage) · Where the model fails (visible failure case) · Free play | Tag pills: AUTO-LOG, ESCALATE, TREND, REFUSAL, TRIAGE, FAILURE CASE |
| Hint chips over degraded machines | Chips over satellites only when elevated/critical: `⚠ ELEV 2.1 km` / `✖ CRIT Pc 3e-4` | |
| Floating text `DOWN -$60/min` | `CRITICAL Pc 3.2e-4`, `TCA −18 h`, `MANOEUVRE PLANNED` | |
| Part replaced confetti + green ring | "Risk cleared" green ring burst at the satellite after manoeuvre/TCA pass | |
| Camera dock view while delivery | Camera flies to the conjunction location (fit both objects + Earth limb), back to global view; user drag stops auto-framing | Reuse `#fitPose` with FIT_POINTS = objects |

---

## 7. Things to copy verbatim vs change

Copy: CSS tokens and component recipes (§1.2, §1.5), terminal component, bubble/float/hint rAF loop, director + narrow
World API + `call()` wrapper + no-op Proxy fallback, renderer/light/material settings, `Effects` pools, adaptive pixel
ratio, `events.js` SSE bus with snapshot-on-connect and `onEmit` ledger, `decide()` provider chain with opaque choice
values and `finish()`, `policy.evaluate()` shape with verbatim reason strings, preset schema, clock holds, AbortController
per incident, mock-by-missing-key config, README `Claim | Where to verify`, JUDGING criterion matrix.

Change: factory geometry → globe/orbits; money KPIs → risk KPIs; Reap/Kwal/chain modules → delete; "fault-code rule" →
our documented naive baseline; verification uses the revealed final CDM; add explicit model metrics (baseline vs model)
and a visible failure case, which the reference lacked as numbers.
