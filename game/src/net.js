// Talking to the server (or to the fixture when the backend is down / ?mock=1).
// SSE: dedupe by seq, reconnect with Last-Event-ID, on a seq gap call /api/state.

const TYPES = ['state.snapshot', 'clock.tick', 'scenario.loaded', 'cdm.received', 'risk.assessed', 'forecast.updated',
  'policy.gated', 'decision.requested', 'decision.made', 'policy.verdict', 'approval.required', 'human.action',
  'explanation.ready', 'log.entry', 'outcome.revealed', 'policy.updated', 'ledger.appended', 'system.degraded'];

export const net = { mock: null, connected: false, lastSeq: 0 };

export class ApiError extends Error {
  constructor(status, body) {
    const e = body?.error;
    super(typeof e === 'string' ? e : e?.message || `HTTP ${status}`);
    this.status = status; this.code = e?.code; this.hint = e?.hint; this.body = body;
  }
}

const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : 'k-' + Math.random().toString(36).slice(2) + Date.now().toString(36));

/** api('GET', '/state'); api('POST', '/approvals/x', body, {idem: true}) */
export async function api(method, path, body, { idem = false, key } = {}) {
  if (net.mock) return net.mock.handle(method, path, body);
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (idem) headers['Idempotency-Key'] = key || uuid();
  const res = await fetch(`/api${path}`, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, json);
  return json;
}
export const newIdemKey = uuid;

/** Is the real backend up? */
export async function probeBackend() {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 2500);
    const r = await fetch('/api/health', { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

/**
 * Subscribe to the event stream. handler(envelope). onGap() is called when a seq gap is seen
 * (the caller re-syncs with GET /api/state). onConn(bool) reports connection state.
 */
export function connectEvents(handler, { onGap, onConn } = {}) {
  if (net.mock) { net.mock.subscribe(handler); onConn?.(true); return { close() {} }; }
  let es = null, last = performance.now(), closed = false, everOpen = false;
  const deliver = (raw, lastEventId) => {
    last = performance.now();
    let evt;
    try { evt = JSON.parse(raw); } catch { return; }
    if (!evt || typeof evt !== 'object') return;
    const seq = evt.seq ?? (lastEventId ? +lastEventId : null);
    if (seq != null && Number.isFinite(seq)) {
      if (seq <= net.lastSeq && evt.type !== 'state.snapshot') return; // duplicate after reconnect
      if (net.lastSeq && seq > net.lastSeq + 1 && evt.type !== 'state.snapshot') onGap?.(net.lastSeq, seq);
      net.lastSeq = Math.max(net.lastSeq, seq);
      if (evt.type === 'state.snapshot') net.lastSeq = seq;
    }
    handler(evt);
  };
  const open = () => {
    if (closed) return;
    const q = net.lastSeq ? `?lastEventId=${net.lastSeq}` : '';
    es = new EventSource(`/api/events${q}`);
    es.onmessage = (m) => deliver(m.data, m.lastEventId);
    // the server may name its events; listen to every known type too (deduped by seq)
    for (const t of TYPES) es.addEventListener(t, (m) => deliver(m.data, m.lastEventId));
    es.onopen = () => { everOpen = true; last = performance.now(); net.connected = true; onConn?.(true); };
    es.onerror = () => { if (everOpen) { net.connected = false; onConn?.(false); } };
  };
  // stale-stream watchdog: no message for 8 s (server ticks at 1 Hz) -> reconnect with Last-Event-ID
  const wd = setInterval(() => {
    if (!es || performance.now() - last < 8000) return;
    console.warn('[net] stream silent, reconnecting');
    onConn?.(false);
    es.close(); es = null; last = performance.now();
    open();
  }, 2000);
  // A page kept in the back/forward cache (or a reloaded tab) would keep its stream open and, with the browser's
  // 6-connections-per-host HTTP/1.1 limit, starve the live page's stream. Release it on pagehide, reopen on pageshow.
  window.addEventListener('pagehide', () => { es?.close(); es = null; });
  window.addEventListener('pageshow', (e) => { if (e.persisted && !closed && !es) { last = performance.now(); open(); } });
  open();
  return { close() { closed = true; clearInterval(wd); es?.close(); } };
}
