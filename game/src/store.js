// Client-side view state built only from server events (Director pattern). No risk logic.

export const S = {
  mode: { fixture: false, decisions: '?', llm: '?' },
  connected: false,
  clock: { simMs: Date.parse('2026-10-31T06:00:00Z'), speed: 1, paused: true, held: false, hold_reason: null, wallAt: performance.now() },
  scenario: null,
  scenarios: [],
  events: new Map(), // key -> E
  focus: null,
  approvals: new Map(),
  policy: { tier_mode: 'BRIEF', tau_watch: 0.1, confidence_bar: 0.75 },
  ledger: { seq: 0, n: 0, last: null },
  log: [],
  liveSky: null,
  liveSkyOn: false,
};

const subs = new Map();
export function on(topic, fn) { if (!subs.has(topic)) subs.set(topic, []); subs.get(topic).push(fn); }
export function emit(topic, ...a) { for (const fn of subs.get(topic) || []) { try { fn(...a); } catch (e) { console.error('[store]', topic, e); } } }

export function ev(key) {
  if (!key) return null;
  let e = S.events.get(key);
  if (!e) {
    e = { key, cdms: [], risks: [], forecast: null, gated: null, decision: null, decisionPending: false, verdict: null, approval: null,
      explanation: null, outcome: null, geometry: null, futures: null, human: null, tcaMs: null, chaser_type: null, mission: null, steps: {} };
    S.events.set(key, e);
  }
  return e;
}
export const latestRisk = (e) => e?.risks.filter(Boolean).at(-1) || null;
export const latestCdm = (e) => e?.cdms.filter(Boolean).at(-1) || null;
export function tierOf(e) {
  if (!e) return null;
  if (e.human && /PREPARE/.test(e.human.action || '')) return 'PLANNED';
  const r = latestRisk(e);
  return r?.tier?.tier || e.summaryTier || null;
}

/** Interpolated SIM time (server ticks at 1 Hz; 1x = 3 sim-min per real second). */
export function simNow() {
  const c = S.clock;
  if (c.paused || c.held) return c.simMs;
  const dt = (performance.now() - c.wallAt) / 1000;
  return c.simMs + Math.min(dt, 2) * (c.rate ? c.rate * 1000 : c.speed * 180e3);
}

// ─── formatting (the formatter refuses to show a Pc as a percent) ─────
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export function fmtPc(log10, { status } = {}) {
  if (status && status !== 'OK' && status !== 'NPD_CLIPPED') return status === 'COV_UNKNOWN' ? 'no covariance' : status;
  if (log10 == null || !Number.isFinite(log10)) return '—';
  if (log10 < -10) return '< 1e-10';
  const ex = Math.floor(log10), m = 10 ** (log10 - ex);
  const mm = m.toFixed(1) === '10.0' ? '1.0' : m.toFixed(1);
  const e2 = m.toFixed(1) === '10.0' ? ex + 1 : ex;
  return `${mm}×10${String(e2).split('').map((c) => SUP[c]).join('')}`;
}
export const fmtPcShort = (l) => (l == null || !Number.isFinite(l) ? '—' : l < -10 ? '<1e-10' : (10 ** l).toExponential(1));
export const fmtMiss = (m) => (m == null ? '—' : m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);
export const fmtPct = (p) => (p == null ? '—' : p > 0 && p < 0.001 ? '<0.1%' : p < 0.095 && p > 0 ? `${(p * 100).toFixed(1)}%` : `${Math.round(p * 100)}%`);
export const pad = (n) => String(n).padStart(2, '0');
export function fmtSim(ms, withDate = true) {
  const d = new Date(ms);
  const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  return withDate ? `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${hm}` : hm;
}
export function fmtDur(ms) {
  const neg = ms < 0; ms = Math.abs(ms);
  const h = Math.floor(ms / 3600e3), m = Math.floor((ms % 3600e3) / 60e3);
  const s = h >= 48 ? `${(h / 24).toFixed(1)} d` : `${h} h ${pad(m)} min`;
  return neg ? `−${s}` : s;
}
export function fmtTminus(ms) {
  const neg = ms < 0; ms = Math.abs(ms);
  const h = Math.floor(ms / 3600e3), m = Math.floor((ms % 3600e3) / 60e3);
  return `${neg ? 'T+' : 'T−'}${pad(h)}:${pad(m)}`;
}
export const ACTION_LABEL = { MONITOR: 'Keep monitoring', REQUEST_TRACKING: 'Request tracking', PREPARE_MANOEUVRE: 'Prepare manoeuvre options', REJECT: 'Reject' };
export const ACTION_SHORT = { MONITOR: 'MONITOR', REQUEST_TRACKING: 'TRACK', PREPARE_MANOEUVRE: 'PREPARE', REJECT: 'REJECTED' };
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
