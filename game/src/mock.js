// FIXTURE MODE (?mock=1, or automatically when /api/health fails).
// A tiny stand-in for the backend that replays canned SSE sequences built from the real CDM rows of
// train:2563 and train:8767 (values copied from the dataset; tiers per the brief thresholds; decision = mock cost rule).
// It is labelled "FIXTURE" in the HUD. The real backend owns all of this logic.
import { elementsThroughPoint, gmst } from './orbit.js';

const D2R = Math.PI / 180;
const DAY = 86400e3;
const iso = (ms) => new Date(ms).toISOString().replace('.000Z', 'Z');

// Rows: [ttca_d, risk(log10, ESA reported), miss_m, rtn_m[3], t_sigma[r,t,n], c_sigma[r,t,n], tier, pc_tier, miss_tier, rule_ids, reasons]
const EVENTS = {
  'train:2563': {
    mission: 15, chaser_type: 'DEBRIS', rel_speed: 9281, lat: 81.65, az: 51.77, el: 0.01,
    t: { a_km: 7084.63, e: 0.000333, i_deg: 92.0587 }, c: { a_km: 7079.45, e: 0.000852, i_deg: 98.3973 },
    rows: [
      [2.433107, -6.3229, 8050, [401.9, 6315.0, 4976.6], [6.73, 149.23, 2.44], [520.49, 22708.84, 109.49], 'SAFE', 'SAFE', 'SAFE', [], ['Pc 4.8e-7 < 1e-6', 'Miss 8.05 km > 5 km']],
      [1.412072, -6.3096, 1629, [443.2, 1230.3, 971.5], [5.07, 46.48, 2.35], [520.48, 22192.50, 109.28], 'ELEVATED', 'SAFE', 'ELEVATED', ['R-MISS-ELEV'], ['Miss 1.63 km < 5 km (miss-distance screen)']],
      [0.439538, -4.1880, 136, [95.0, 78.6, 58.8], [4.87, 15.35, 2.42], [144.09, 683.25, 116.34], 'CRITICAL', 'ELEVATED', 'CRITICAL', ['R-MISS-CRIT', 'R-PC-ELEV'], ['Miss 136 m < 1 km (miss-distance screen)', 'Pc 6.5e-5 ≥ 1e-6 (Pc tier ELEVATED)']],
    ],
    final: { tier: 'CRITICAL', log10_pc: -4.188, miss_m: 136 }, persistence: 'SAFE',
  },
  'train:8767': {
    mission: 15, chaser_type: 'DEBRIS', rel_speed: 14940, lat: 80.11, az: 4.89, el: 0.0,
    t: { a_km: 7084.84, e: 0.0003, i_deg: 92.0734 }, c: { a_km: 7077.04, e: 0.0009, i_deg: 86.3306 },
    rows: [
      [6.673443, -8.5698, 2591, [146.3, -222.4, -2577.3], [10.77, 168.42, 2.31], [54.89, 6965.05, 56.70], 'ELEVATED', 'SAFE', 'ELEVATED', ['R-MISS-ELEV'], ['Miss 2.59 km < 5 km (miss-distance screen)']],
      [5.594763, -6.6893, 1534, [25.0, -133.0, -1528.2], [10.39, 148.44, 2.27], [100.87, 4918.60, 59.57], 'ELEVATED', 'SAFE', 'ELEVATED', ['R-MISS-ELEV'], ['Miss 1.53 km < 5 km (miss-distance screen)']],
      [4.587397, -5.4481, 1054, [98.5, -87.5, -1045.9], [8.06, 115.17, 2.13], [99.01, 4681.21, 64.44], 'ELEVATED', 'ELEVATED', 'ELEVATED', ['R-PC-ELEV', 'R-MISS-ELEV'], ['Pc 3.6e-6 ≥ 1e-6', 'Miss 1.05 km < 5 km']],
      [3.586509, -3.4299, 117, [-32.4, -5.4, -113.0], [7.45, 88.85, 2.26], [43.38, 3006.16, 61.32], 'CRITICAL', 'CRITICAL', 'CRITICAL', ['R-PC-CRIT', 'R-MISS-CRIT'], ['Pc 3.7e-4 ≥ 1e-4', 'Miss 117 m < 1 km (miss-distance screen)']],
      [2.589827, -3.5722, 216, [-38.7, 21.5, 211.9], [4.71, 53.82, 2.08], [43.74, 3468.71, 61.33], 'CRITICAL', 'CRITICAL', 'CRITICAL', ['R-PC-CRIT', 'R-MISS-CRIT'], ['Pc 2.7e-4 ≥ 1e-4', 'Miss 216 m < 1 km']],
      [1.655315, -2.9062, 36, [-34.5, -5.0, -10.1], [4.28, 23.04, 2.17], [43.74, 768.66, 63.18], 'CRITICAL', 'CRITICAL', 'CRITICAL', ['R-PC-CRIT', 'R-MISS-CRIT'], ['Pc 1.2e-3 ≥ 1e-4', 'Miss 36 m < 1 km']],
      [0.633984, -2.6880, 38, [-17.6, 2.9, 33.8], [2.46, 11.41, 2.02], [36.53, 255.29, 62.20], 'CRITICAL', 'CRITICAL', 'CRITICAL', ['R-PC-CRIT', 'R-MISS-CRIT'], ['Pc 2.1e-3 ≥ 1e-4', 'Miss 38 m < 1 km']],
    ],
    final: { tier: 'CRITICAL', log10_pc: -2.688, miss_m: 38 }, persistence: 'CRITICAL',
  },
};

const SCENARIOS = [
  { id: 'ops-day', title: 'Ops day, mission 15', tag: 'TRIAGE', order: 1, teaches: '24 h danger window, queue ranking, approvals.', tagline: 'A full duty shift: ~35 conjunctions on one Earth-observation satellite.', cast: [] },
  { id: 'late-cdm', title: 'The late CDM', tag: 'LATE', order: 2, teaches: 'Nothing at T−2 d foresaw it; physics catches it after the commitment point.', tagline: 'Safe two days out. Then a CDM with a 136 m miss lands.', cast: ['train:2563'], d0: '2026-10-31T06:00:00Z', tca: { 'train:2563': '2026-10-31T20:30:00Z' } },
  { id: 'steady-climb', title: 'Steady climb', tag: 'ESCALATE', order: 3, teaches: 'Full pipeline, early PREPARE, the dive.', tagline: 'Miss shrinks from 2.6 km to 117 m over three days.', cast: ['train:8767'], d0: '2026-10-31T06:00:00Z', tca: { 'train:8767': '2026-11-04T06:00:00Z' } },
  { id: 'scary-then-nothing', title: 'Scary, then nothing', tag: 'WAIT', order: 4, teaches: 'Commit early vs wait (CARA App. Q.8).', tagline: 'The highest risk in the archive collapses to nothing.', cast: [] },
  { id: 'rules-disagree', title: 'Rules disagree', tag: 'RULES', order: 5, teaches: 'The tier-mode knob: miss-distance vs Pc rules.', tagline: '20 m miss with Pc 1e-19; 3.9 km miss with Pc 1.2e-4.', cast: [] },
  { id: 'cov-unknown', title: 'Covariance unknown', tag: 'REFUSAL', order: 6, teaches: 'Refusal → resolution.', tagline: '13 CDMs with a default covariance, then a real one.', cast: [] },
  { id: 'failure', title: 'Where we fail', tag: 'FAILURE CASE', order: 7, teaches: 'How the user knows when the system is wrong.', tagline: 'Ours 1e-479 vs ESA 1e-5 at 57 km.', cast: [] },
  { id: 'blind-test', title: 'Blind test', tag: 'BLIND', order: 8, teaches: 'Forecasts on unseen data.', tagline: 'Five test events, truth hidden.', cast: [] },
  { id: 'free-play', title: 'Free play', tag: 'FREE', order: 9, teaches: 'Input beyond the script.', tagline: 'Seeded random cast from any mission.', cast: [] },
];

/** Encounter-plane projection of the combined (diagonal RTN) covariance. Fixture only. */
function encounter(row, ev) {
  const [, , miss, rtn, ts, cs] = row;
  const az = ev.az * D2R, el = ev.el * D2R;
  const v = [-Math.sin(el), -Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az)];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const rv = dot(rtn, v);
  let e1 = [rtn[0] - rv * v[0], rtn[1] - rv * v[1], rtn[2] - rv * v[2]];
  const n1 = Math.hypot(...e1) || 1;
  e1 = e1.map((x) => x / n1);
  const e2 = [v[1] * e1[2] - v[2] * e1[1], v[2] * e1[0] - v[0] * e1[2], v[0] * e1[1] - v[1] * e1[0]];
  const C = [0, 1, 2].map((k) => ts[k] ** 2 + cs[k] ** 2);
  const q = (a, b) => a[0] * b[0] * C[0] + a[1] * b[1] * C[1] + a[2] * b[2] * C[2];
  const cov = [[q(e1, e1), q(e1, e2)], [q(e1, e2), q(e2, e2)]];
  return { miss_xy_m: [n1 || miss, 0], cov_m2: cov, hbr_m: 5 };
}
function sigmas(cov) {
  const [[a, b], [, c]] = cov;
  const tr = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  return { sx: Math.sqrt(tr + d), sy: Math.sqrt(Math.max(tr - d, 0)), theta: 0.5 * Math.atan2(2 * b, a - c) / D2R };
}

export class MockServer {
  constructor() {
    this.subs = [];
    this.seq = 0;
    this.ledger = [];
    this.policy = { tier_mode: 'BRIEF', tau_watch: 0.0634, confidence_bar: 0.7, version: 'p1' };
    this.speed = 1; this.paused = true; this.held = false; this.hold_reason = null;
    this.simMs = Date.parse('2026-10-31T06:00:00Z');
    this.scn = null; this.queue = []; this.approvals = new Map(); this.events = new Map();
    this.idem = new Map();
    setInterval(() => this.#step(0.25), 250);
    setInterval(() => this.#emit('clock.tick', this.#clock()), 1000);
  }
  subscribe(fn) { this.subs.push(fn); fn(this.#env('state.snapshot', this.#state())); }

  #env(type, data, event_key = null) {
    return { v: 1, seq: ++this.seq, type, at: new Date().toISOString(), sim_time: iso(this.simMs), event_key, ledger_seq: null, data };
  }
  #emit(type, data, key = null, ledgerKind = null) {
    const e = this.#env(type, data, key);
    if (ledgerKind) {
      const prev = this.ledger.at(-1)?.hash || '0'.repeat(16);
      const hash = (Math.abs(hashStr(prev + type + JSON.stringify(data))) >>> 0).toString(16).padStart(8, '0') + (this.ledger.length + 1).toString(16).padStart(8, '0');
      const entry = { seq: this.ledger.length + 1, ts_wall: e.at, sim_time: e.sim_time, kind: ledgerKind, event_key: key, actor: type === 'human.action' ? 'analyst' : 'system', type, hash, prev_hash: prev, payload: data };
      this.ledger.push(entry);
      e.ledger_seq = entry.seq;
      for (const fn of this.subs) fn(e);
      for (const fn of this.subs) fn(this.#env('ledger.appended', { seq: entry.seq, hash, kind: ledgerKind }, key));
      return;
    }
    for (const fn of this.subs) fn(e);
  }
  #log(level, code, msg, key = null) { this.#emit('log.entry', { level, code, msg }, key); }
  #clock() { return { sim_time: iso(this.simMs), speed: this.speed, paused: this.paused, held: this.held, hold_reason: this.hold_reason }; }
  #state() {
    return {
      fixture: true,
      clock: this.#clock(),
      scenario: this.scn ? { id: this.scn.id, title: this.scn.title, tag: this.scn.tag, teaches: this.scn.teaches, d0: this.scn.d0 } : null,
      cast: this.scn ? this.scn.cast.map((k) => this.#summary(k)) : [],
      approvals: [...this.approvals.values()].filter((a) => a.status === 'pending'),
      policy: this.policy,
      mode: { decisions: 'mock', llm: 'template', fixture: true },
    };
  }
  #summary(key) {
    const s = this.events.get(key);
    const ev = EVENTS[key];
    const last = s?.released.at(-1);
    return { event_key: key, mission_id: ev.mission, chaser_type: ev.chaser_type, tca_sim: iso(s.tca), n_known: s.released.length,
      tier: last != null ? ev.rows[last][6] : null, latest_miss_m: last != null ? ev.rows[last][2] : null, latest_log10_pc: last != null ? ev.rows[last][1] : null };
  }

  // ─── REST ─────────────────────────────────────────────────────────────
  async handle(method, path, body) {
    await new Promise((r) => setTimeout(r, 30));
    const p = path.split('?')[0];
    const m = (re) => p.match(re);
    let r;
    if (p === '/health') return { ok: true, mode: { decisions: 'mock', llm: 'template', fixture: true }, model: { name: 'tier_forecaster', version: 'fixture', sha_ok: true }, ledger: { seq: this.ledger.length, verified: true } };
    if (p === '/state') return this.#state();
    if (p === '/scenarios') return SCENARIOS.map((s) => ({ ...s, event_keys: s.cast, playable: s.cast.length > 0 }));
    if (p === '/sim/scenario' && method === 'POST') return this.#load(body?.id);
    if (p === '/sim/speed') { this.speed = body?.speed || 1; this.paused = false; this.#emit('clock.tick', this.#clock()); return this.#clock(); }
    if (p === '/sim/pause') { this.paused = true; this.#emit('clock.tick', this.#clock()); return this.#clock(); }
    if (p === '/sim/resume') { this.paused = false; this.#emit('clock.tick', this.#clock()); return this.#clock(); }
    if (p === '/sim/next-cdm') {
      const next = this.queue.filter((q) => q.at > this.simMs).sort((a, b) => a.at - b.at)[0];
      if (next) this.simMs = next.at - 1000;
      this.paused = false;
      return this.#clock();
    }
    if ((r = m(/^\/conjunctions\/([^/]+)\/geometry$/))) return this.#geometry(decodeURIComponent(r[1]));
    if ((r = m(/^\/conjunctions\/([^/]+)$/))) return this.#conj(decodeURIComponent(r[1]));
    if ((r = m(/^\/approvals\/([^/]+)$/)) && method === 'POST') return this.#approve(decodeURIComponent(r[1]), body);
    if (p === '/policy' && method === 'GET') return this.policy;
    if (p === '/policy' && method === 'PUT') {
      const diff = {};
      for (const k of Object.keys(body || {})) if (this.policy[k] !== body[k]) diff[k] = [this.policy[k], body[k]];
      Object.assign(this.policy, body, { version: 'p' + (this.ledger.length + 1) });
      this.#emit('policy.updated', { version: this.policy.version, diff, actor: 'analyst', policy: this.policy }, null, 'human');
      this.#log('INFO', 'POLICY', `policy ${this.policy.version}: ${Object.entries(diff).map(([k, [a, b]]) => `${k} ${a}→${b}`).join(', ')} (fixture: tiers not re-run)`);
      return this.policy;
    }
    if (p === '/ledger') return { entries: this.ledger.slice().reverse(), n: this.ledger.length };
    if (p === '/ledger/verify') return { ok: true, n: this.ledger.length, first_bad_seq: null };
    if (p === '/dashboard') return this.#dashboard();
    if (p === '/livesky' || (r = m(/^\/futures\/([^/]+)$/))) {
      const url = p === '/livesky' ? '/fixture/livesky.json' : `/fixture/futures/${r[1]}.json`;
      const res = await fetch(url);
      if (!res.ok) { const e = new Error('not found'); e.status = 404; throw e; }
      return res.json();
    }
    const err = new Error('Not available in fixture mode'); err.status = 404; throw err;
  }

  #load(id) {
    const s = SCENARIOS.find((x) => x.id === id);
    if (!s || !s.cast.length) { const e = new Error('Fixture mode plays only "The late CDM" and "Steady climb". Start the backend for the rest.'); e.status = 400; throw e; }
    this.scn = s; this.approvals.clear(); this.events.clear(); this.queue = [];
    this.simMs = Date.parse(s.d0); this.paused = false; this.held = false; this.hold_reason = null; this.speed = 2;
    let pre = 0;
    for (const key of s.cast) {
      const ev = EVENTS[key];
      const tca = Date.parse(s.tca[key]);
      this.events.set(key, { tca, released: [], revealed: false });
      ev.rows.forEach((row, idx) => {
        const at = tca - row[0] * DAY;
        if (at <= this.simMs) pre++;
        this.queue.push({ key, idx, at });
      });
      this.queue.push({ key, idx: -1, at: tca, outcome: true });
    }
    this.#emit('scenario.loaded', { id: s.id, title: s.title, tag: s.tag, teaches: s.teaches, seed: 1729, d0: s.d0, cast: s.cast, preloaded_cdms: pre }, null, 'system');
    this.#log('INFO', 'SCN', `scenario ${s.id} loaded · cast ${s.cast.join(', ')} · FIXTURE (backend offline)`);
    // preloaded history: released quietly (no agent flow)
    for (const q of this.queue.filter((q) => !q.outcome && q.at <= this.simMs).sort((a, b) => a.at - b.at)) { this.#release(q, true); q.done = true; }
    this.#emit('state.snapshot', this.#state());
    return this.#state();
  }

  #step(dtReal) {
    if (!this.scn || this.paused || this.held) return;
    const next = this.simMs + dtReal * this.speed * 180e3; // 1x = 3 sim-min per real second
    const due = this.queue.filter((q) => !q.done && q.at <= next).sort((a, b) => a.at - b.at);
    if (due.length) {
      const q = due[0];
      this.simMs = q.at;
      q.done = true;
      if (q.outcome) this.#reveal(q.key); else this.#release(q, false);
      return;
    }
    this.simMs = next;
  }

  #cdmData(key, idx) {
    const ev = EVENTS[key], row = ev.rows[idx];
    return { cdm_id: `${key}:${idx}`, cdm_index: idx, n_known: idx + 1, ttca_d: row[0], created_sim: iso(this.simMs), miss_m: row[2], rel_speed_ms: ev.rel_speed,
      rtn_m: row[3], c_object_type: ev.chaser_type, flags: [], source: { split: 'train', event_id: +key.split(':')[1], row: idx } };
  }
  #riskData(key, idx) {
    const ev = EVENTS[key], row = ev.rows[idx];
    const enc = encounter(row, ev), s = sigmas(enc.cov_m2);
    return { cdm_id: `${key}:${idx}`, pc: { status: 'OK', log10: row[1], method: '2D-Foster-GL24x48', hbr_m: enc.hbr_m, sigma_plane_m: [s.sx, s.sy], theta_deg: s.theta, mahalanobis: row[2] / Math.max(s.sx, 1), dilution: false },
      reported_log10: row[1], disagree_dex: 0,
      tier: { tier: row[6], pc_tier: row[7], miss_tier: row[8], mode: this.policy.tier_mode, rule_ids: row[9], reasons: row[10], citations: ['brief thresholds'] } };
  }

  #release(q, quiet) {
    const { key, idx } = q, ev = EVENTS[key], row = ev.rows[idx];
    const st = this.events.get(key);
    st.released.push(idx);
    const n = ev.rows.length;
    this.#emit('cdm.received', this.#cdmData(key, idx), key, 'signal');
    this.#log('CDM', 'CDM', `${key} CDM ${idx + 1}/${n} T−${row[0].toFixed(2)} d miss ${fmtM(row[2])} · source train row ${idx}`, key);
    const risk = this.#riskData(key, idx);
    this.#emit('risk.assessed', risk, key, 'physics');
    const lv = row[6] === 'CRITICAL' ? 'CRIT' : row[6] === 'ELEVATED' ? 'WARN' : 'INFO';
    this.#log(lv, 'TIER', `${key} ${row[6]}${row[9].length ? ' by ' + row[9][0] : ''} · Pc ${pcStr(row[1])} · ${row[10][0]}`, key);
    if (row[0] >= 2) {
      this.#emit('forecast.updated', { model_version: 'fixture', valid: true, p_tiers: null, p_critical: null, calib_bin: null,
        baseline: { persistence_tier: row[6], p_critical_empirical: row[6] === 'CRITICAL' ? 0.8 : row[6] === 'ELEVATED' ? 0.053 : 0.002 }, watch: false, rank: 1 }, key, 'ml');
    }
    if (quiet) return;
    if (row[6] === 'SAFE') { this.#emit('policy.verdict', { decision_id: null, verdict: 'AUTO', action: 'MONITOR', reasons: ['SAFE: log and keep monitoring'] }, key, 'rule'); return; }
    this.#agentFlow(key, idx);
  }

  async #agentFlow(key, idx) {
    const ev = EVENTS[key], row = ev.rows[idx], st = this.events.get(key);
    const crit = row[6] === 'CRITICAL';
    const ttcaH = (st.tca - this.simMs) / 3600e3;
    this.held = true; this.hold_reason = 'AI working';
    this.#emit('clock.tick', this.#clock());
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    await wait(900);
    const allowed = crit ? ['REQUEST_TRACKING', 'PREPARE_MANOEUVRE'] : ['MONITOR', 'REQUEST_TRACKING', 'PREPARE_MANOEUVRE'];
    const reasons = [...row[10]];
    if (ttcaH < 24) reasons.push(`T−${ttcaH.toFixed(1)} h: inside commitment window`);
    this.#emit('policy.gated', { allowed, must_escalate: crit, blocked: false, reasons }, key, 'rule');
    this.#log('INFO', 'GATE', `${key} allowed ${allowed.join(' / ')}${crit ? ' · must escalate' : ''}`, key);
    await wait(700);
    const decision_id = `d-${key}-${idx}`;
    this.#emit('decision.requested', { decision_id, provider_chain: ['mock'], request_hash: 'fixture' }, key, 'decision');
    await wait(1300);
    const p = crit ? 0.8 : 0.053;
    const L = { MONITOR: [100, 0], REQUEST_TRACKING: [30, 1], PREPARE_MANOEUVRE: [5, 8] };
    const el = Object.fromEntries(allowed.map((a) => [a, +(L[a][0] * p + L[a][1] * (1 - p)).toFixed(2)]));
    const ex = allowed.map((a) => Math.exp(-el[a] / 2)), z = ex.reduce((s, x) => s + x, 0);
    const probs = Object.fromEntries(allowed.map((a, k) => [a, +(ex[k] / z).toFixed(4)]));
    const answer = allowed.reduce((b, a) => (probs[a] > probs[b] ? a : b));
    const base = allowed.reduce((b, a) => (el[a] < el[b] ? a : b));
    this.#emit('decision.made', { decision_id, provider: 'mock', answer, probabilities: probs, confidence: probs[answer], evidence_sufficient_p: null, refused: false, latency_ms: 4,
      baseline: { action: base, expected_loss: el }, agree: answer === base }, key, 'decision');
    this.#log('AGENT', 'DEC', `mock (cost rule) ${short(answer)} ${(probs[answer] * 100).toFixed(0)}% · cost rule ${short(base)} · ${answer === base ? 'agree' : 'disagree'}`, key);
    await wait(500);
    const verdict = crit ? 'ESCALATE' : 'ESCALATE';
    const vreasons = crit ? reasons : [...reasons, 'Fixture: elevated events come to you'];
    this.#emit('policy.verdict', { decision_id, verdict, action: answer, reasons: vreasons }, key, 'rule');
    this.#log('WARN', 'POLICY', `${key} ${verdict} · ${vreasons.at(-1)} · awaiting analyst`, key);
    const approval_id = `a-${key}-${idx}`;
    const appr = { approval_id, decision_id, event_key: key, options: allowed, suggested: answer, ttca_h: ttcaH, mcp_passed: ttcaH < 24, reasons: vreasons, status: 'pending' };
    this.approvals.set(approval_id, appr);
    this.#emit('approval.required', appr, key, 'system');
    this.hold_reason = 'waiting for you';
    this.#emit('clock.tick', this.#clock());
    await wait(400);
    const text = `${key}: latest CDM (T−${row[0].toFixed(2)} d) puts the miss at ${fmtM(row[2])} with a physics Pc of ${pcStr(row[1])}. `
      + `The rules make it ${row[6]} (${row[9].join(', ') || 'no rule'}). The suggested next step is ${short(answer).toLowerCase()}; a human decides.`;
    this.#emit('explanation.ready', { cdm_id: `${key}:${idx}`, source: 'template', verified: true, text }, key, 'llm');
  }

  #approve(id, body) {
    const a = this.approvals.get(id);
    if (!a) { const e = new Error('unknown approval'); e.status = 404; throw e; }
    if (a.status !== 'pending') { const e = new Error('APPROVAL_ALREADY_DECIDED'); e.status = 409; throw e; }
    a.status = 'decided'; a.action = body?.action;
    this.#emit('human.action', { approval_id: id, action: body?.action, actor: 'analyst', note: body?.note || '', wait_sim_min: 0 }, a.event_key, 'human');
    this.#log('AGENT', 'HUMAN', `${a.event_key} analyst chose ${short(body?.action)} · ledger #${this.ledger.length}`, a.event_key);
    if (![...this.approvals.values()].some((x) => x.status === 'pending')) { this.held = false; this.hold_reason = null; }
    this.#emit('clock.tick', this.#clock());
    return { ...a };
  }

  #reveal(key) {
    const ev = EVENTS[key];
    const verdict = ev.persistence === 'CRITICAL' ? 'TP' : 'FN';
    this.#emit('outcome.revealed', { final: ev.final, forecast_was: { p_critical: null, persistence_tier: ev.persistence }, verdict }, key, 'outcome');
    this.#log(verdict === 'FN' ? 'CRIT' : 'INFO', 'OUTCOME', `${key} TCA passed · final ${ev.final.tier} miss ${fmtM(ev.final.miss_m)} · persistence at T−2 d said ${ev.persistence} → ${verdict}`, key);
  }

  #geometry(key) {
    const ev = EVENTS[key], st = this.events.get(key);
    if (!ev) { const e = new Error('unknown event'); e.status = 404; throw e; }
    const tca = st?.tca ?? this.simMs;
    const lon = ((hashStr(key) % 360) + 360) % 360 - 180; // synthetic longitude seeded by event key
    const ra = lon + gmst(tca) / D2R;
    const target = { ...elementsThroughPoint({ ...ev.t, lat_deg: ev.lat, ra_deg: ra, branch: 0 }), e: ev.t.e };
    const chaser = { ...elementsThroughPoint({ ...ev.c, lat_deg: ev.lat, ra_deg: ra, branch: 1 }), e: ev.c.e };
    return { event_key: key, tca_sim: iso(tca), reconstructed: true, target, chaser, chaser_type: ev.chaser_type,
      conjunction: { lat_deg: ev.lat, lon_deg: lon, alt_km: ev.t.a_km - 6378.137 },
      residuals: { inc_res_deg: 0.04, sma_res_km: 12.1 },
      encounter: ev.rows.map((row, k) => ({ cdm_index: k, ...encounter(row, ev) })) };
  }
  #conj(key) {
    const st = this.events.get(key);
    if (!st) { const e = new Error('unknown event'); e.status = 404; throw e; }
    return { event_key: key, cdms: st.released.map((i) => this.#cdmData(key, i)), risks: st.released.map((i) => this.#riskData(key, i)), tca_sim: iso(st.tca) };
  }
  #dashboard() {
    const cast = this.scn ? this.scn.cast.map((k) => this.#summary(k)) : [];
    return { fixture: true, window: '24h', tiles: {}, next24h: cast, insights: ['Fixture mode: numbers come from the canned sequence, not the backend.'] };
  }
}

function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h; }
const fmtM = (m) => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);
const pcStr = (l) => (l == null ? '—' : l < -10 ? '< 1e-10' : (10 ** l).toExponential(1));
const short = (a) => ({ MONITOR: 'MONITOR', REQUEST_TRACKING: 'TRACK', PREPARE_MANOEUVRE: 'PREPARE' }[a] || a);
