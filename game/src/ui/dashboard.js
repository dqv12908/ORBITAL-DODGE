// Ops dashboard (reference layout): 24 h tiles, insights, tabs Next 24 h · Alerts · Decisions · Refusals · Evaluation · Ledger.
// Every number comes from the server (GET /dashboard, /eval, /ledger) or from server events already in the store.
import { S, on, emit, tierOf, latestRisk, latestCdm, fmtPc, fmtMiss, fmtPct, fmtSim, fmtDur, esc, ACTION_SHORT, simNow } from '../store.js';
import { mountStats } from './gg/stats.js';
import { verdictBar } from './gg/plots.js';

const TABS = [
  ['next', 'Next 24 h'], ['alerts', 'Alerts'], ['decisions', 'Decisions'], ['refusals', 'Refusals'], ['eval', 'Evaluation'], ['stats', 'Statistics'], ['ledger', 'Ledger'],
];
const TIER_VAR = { SAFE: 'var(--good)', ELEVATED: 'var(--warn)', CRITICAL: 'var(--bad)', PLANNED: 'var(--info)' };
const ms = (s) => (s ? Date.parse(s) : null);
const ICO_EMPTY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';

const tierBadge = (t) => (t ? `<span class="dsh-badge" style="color:${TIER_VAR[t] || 'var(--muted)'};border-color:${TIER_VAR[t] || 'var(--line-2)'}">${esc(t)}</span>` : '<span class="muted">—</span>');
const keyLink = (k) => (k ? `<a class="dsh-key" data-key="${esc(k)}">${esc(k)}</a>` : '—');
const arr = (x) => (Array.isArray(x) ? x : x && typeof x === 'object' ? Object.values(x) : []);
const pick = (o, ...ks) => { for (const k of ks) if (o && o[k] != null) return o[k]; return null; };

function empty(title, text, small = false) {
  return `<div class="dsh-empty${small ? ' small' : ''}"><span class="dsh-empty-ico">${ICO_EMPTY}</span><b>${esc(title)}</b><p>${esc(text)}</p></div>`;
}

/** Rows derived from the store when /dashboard does not provide them (server-provided fields only). */
function storeRows() {
  const now = simNow();
  return [...(S.cast || [])].map((k) => {
    const e = S.events.get(k); if (!e) return null;
    const r = latestRisk(e), c = latestCdm(e);
    return {
      event_key: k, tca_sim: e.tcaMs ? new Date(e.tcaMs).toISOString() : null, tier: tierOf(e),
      latest_miss_m: c?.miss_m ?? e.summary?.latest_miss_m ?? null,
      latest_log10_pc: r?.pc?.log10 ?? e.summary?.latest_log10_pc ?? null, pc_status: r?.pc?.status,
      pc_tier: r?.tier?.pc_tier, miss_tier: r?.tier?.miss_tier, rule_ids: r?.tier?.rule_ids, reasons: e.verdict?.reasons || r?.tier?.reasons,
      p_critical: e.forecast?.p_critical ?? null, verdict: e.verdict?.verdict, action: e.verdict?.action || e.decision?.answer,
      provider: e.decision?.provider, confidence: e.decision?.confidence, agree: e.decision?.agree, awaiting: !!e.approval,
      outcome: e.outcome?.verdict, ttca_ms: e.tcaMs ? e.tcaMs - now : null,
    };
  }).filter(Boolean);
}

export class Dashboard {
  constructor(root, { api } = {}) {
    this.root = root; this.api = api;
    this.el = null; this.tab = 'next';
    this.data = null; this.evalData = undefined; this.ledgerData = null; this.verify = null;
    this.lastRefresh = 0; this.pending = 0;
    this.onKey = (e) => { if (e.key === 'Escape') this.close(); };
    const live = () => this.#throttled();
    on('ledger', live); on('event', live); on('approvals', live);
  }

  get isOpen() { return !!this.el; }

  open(tab = 'next') {
    this.tab = tab;
    if (!this.el) {
      this.el = document.createElement('div');
      this.el.className = 'mdl-back';
      this.el.innerHTML = `
        <div class="mdl mdl-dash" role="dialog" aria-label="Ops dashboard">
          <div class="mdl-head"><div><h2>Ops dashboard — next 24 h</h2><p class="dsh-sub"></p></div>
            <button class="pop-x" type="button" aria-label="Close">✕</button></div>
          <div class="mdl-body dsh-body"><div class="dsh">
            <div class="dsh-cards"></div>
            <ul class="dsh-insights"></ul>
            <div class="dsh-bar"><div class="dsh-tabs"></div>
              <div class="dsh-tools"><span class="dsh-live"><i></i>Live</span><button type="button" class="small dsh-csv">Export CSV</button></div></div>
            <div class="dsh-pane"></div>
          </div></div>
        </div>`;
      this.root.appendChild(this.el);
      this.el.querySelector('.pop-x').addEventListener('click', () => this.close());
      this.el.addEventListener('pointerdown', (e) => { if (e.target === this.el) this.close(); });
      this.el.querySelector('.dsh-csv').addEventListener('click', () => this.#csv());
      this.el.addEventListener('click', (e) => {
        const a = e.target.closest('.dsh-key');
        if (a) { e.preventDefault(); const k = a.dataset.key; this.close(); emit('inspect', k); return; }
        const t = e.target.closest('.dsh-tab');
        if (t) { this.tab = t.dataset.tab; this.#render(); this.#loadTab(); return; }
        if (e.target.closest('.dsh-verify')) this.#verify();
      });
      document.addEventListener('keydown', this.onKey);
    }
    this.#render();
    this.#refresh();
  }

  close() {
    if (!this.el) return;
    document.removeEventListener('keydown', this.onKey);
    this.el.remove(); this.el = null;
  }

  #throttled() {
    if (!this.el) return;
    const wait = 1000 - (performance.now() - this.lastRefresh);
    if (wait <= 0) { this.#refresh(); return; }
    clearTimeout(this.pending);
    this.pending = setTimeout(() => this.#refresh(), wait);
  }

  async #refresh() {
    this.lastRefresh = performance.now();
    try { this.data = await this.api('GET', '/dashboard?window=24h'); } catch { this.data = this.data || null; }
    await this.#loadTab(true);
    this.#render();
  }

  async #loadTab(quiet) {
    if (this.tab === 'eval' && this.evalData === undefined) {
      try { this.evalData = await this.api('GET', '/eval'); } catch (e) { this.evalData = { __err: e.message, __status: e.status }; }
    }
    if (this.tab === 'ledger') {
      try { const r = await this.api('GET', '/ledger?limit=200'); this.ledgerData = Array.isArray(r) ? r : r?.entries || r?.items || []; } catch (e) { this.ledgerData = { __err: e.message }; }
    }
    if (!quiet) this.#render();
  }

  // ─── data shaping ────────────────────────────────────────────────────
  #rows(name) {
    const d = this.data || {};
    const from = d[name] || d.tables?.[name];
    if (Array.isArray(from)) return from;
    const rows = storeRows();
    if (name === 'next24h') return rows.filter((r) => r.ttca_ms == null || (r.ttca_ms > -3600e3 && r.ttca_ms < 24 * 3600e3)).sort((a, b) => (a.ttca_ms ?? 9e15) - (b.ttca_ms ?? 9e15));
    if (name === 'alerts') return rows.filter((r) => r.tier === 'CRITICAL' || r.tier === 'ELEVATED');
    if (name === 'decisions') return rows.filter((r) => r.verdict);
    if (name === 'refusals') return rows.filter((r) => r.pc_status && r.pc_status !== 'OK');
    return [];
  }

  #tiles() {
    const d = this.data || {};
    const t = d.tiles;
    const get = (...ks) => {
      if (t && !Array.isArray(t)) return pick(t, ...ks);
      if (Array.isArray(t)) { const x = t.find((y) => ks.includes(y.id) || ks.includes(y.key)); return x ? x.value ?? x : null; }
      return null;
    };
    const rows = storeRows();
    const in24 = rows.filter((r) => r.ttca_ms != null && r.ttca_ms > 0 && r.ttca_ms < 24 * 3600e3);
    const val = (v, fb) => (v != null ? (typeof v === 'object' ? v.value ?? v.n ?? JSON.stringify(v) : v) : fb);
    const critical = rows.filter((r) => r.tier === 'CRITICAL');
    const missOnly = critical.filter((r) => r.pc_tier && r.pc_tier !== 'CRITICAL').length;
    const verdicts = rows.map((r) => r.verdict).filter(Boolean);
    const split = get('decisions_split', 'verdicts') || {
      AUTO: verdicts.filter((v) => v === 'AUTO').length, ESCALATE: verdicts.filter((v) => v === 'ESCALATE').length, BLOCK: verdicts.filter((v) => v === 'BLOCK').length,
    };
    const score = get('scoreboard', 'forecast_scoreboard');
    const revealed = rows.filter((r) => r.outcome);
    return {
      w1: val(get('within_1km_24h', 'within_1km'), in24.filter((r) => r.latest_miss_m != null && r.latest_miss_m < 1000).length),
      w5: val(get('within_5km_24h', 'within_5km'), in24.filter((r) => r.latest_miss_m != null && r.latest_miss_m < 5000).length),
      crit: val(get('critical'), critical.length),
      critMiss: val(get('critical_miss_only'), missOnly),
      elev: val(get('elevated'), rows.filter((r) => r.tier === 'ELEVATED').length),
      refused: val(get('refused'), rows.filter((r) => r.pc_status && r.pc_status !== 'OK').length),
      awaiting: val(get('awaiting', 'awaiting_you'), S.approvals.size),
      score: score ? (typeof score === 'object' ? `${score.persistence_hits ?? '—'} / ${score.model_hits ?? '—'}` : score)
        : revealed.length ? `${revealed.filter((r) => r.outcome === 'TP' || r.outcome === 'TN').length} / ${revealed.length}` : '—',
      scoreCap: score ? 'persistence / model hits (revealed today)' : revealed.length ? 'persistence hits / revealed' : 'nothing revealed yet',
      split,
    };
  }

  // ─── rendering ───────────────────────────────────────────────────────
  #render() {
    if (!this.el) return;
    this.el.querySelector('.dsh-sub').textContent = `SIM ${fmtSim(simNow())} · ${S.scenario?.title || 'no scenario'}${S.mode?.fixture || this.data?.fixture ? ' · FIXTURE' : ''} · every row traces to a CDM and a rule`;
    const T = this.#tiles();
    const sp = T.split || {}; const tot = (sp.AUTO || 0) + (sp.ESCALATE || 0) + (sp.BLOCK || 0);
    const w = (n) => (tot ? `${(100 * (n || 0)) / tot}%` : '0');
    this.el.querySelector('.dsh-cards').innerHTML = `
      <div class="dsh-card"><span class="dsh-card-l">Within 1 km · next 24 h</span><span class="dsh-card-v" style="color:${T.w1 ? 'var(--bad-text)' : 'var(--text)'}">${esc(T.w1)}</span><small>the brief's dangerous distance</small></div>
      <div class="dsh-card"><span class="dsh-card-l">Within 5 km · next 24 h</span><span class="dsh-card-v" style="color:${T.w5 ? 'var(--warn)' : 'var(--text)'}">${esc(T.w5)}</span><small>latest CDM miss</small></div>
      <div class="dsh-card"><span class="dsh-card-l">Critical</span><span class="dsh-card-v" style="color:var(--bad-text)">${esc(T.crit)}</span><small>${esc(T.critMiss)} by miss only</small></div>
      <div class="dsh-card"><span class="dsh-card-l">Elevated</span><span class="dsh-card-v" style="color:var(--warn)">${esc(T.elev)}</span><small>tier rules</small></div>
      <div class="dsh-card"><span class="dsh-card-l">Refused</span><span class="dsh-card-v" style="color:var(--agent)">${esc(T.refused)}</span><small>Pc not computable</small></div>
      <div class="dsh-card"><span class="dsh-card-l">Awaiting you</span><span class="dsh-card-v" style="color:${T.awaiting ? 'var(--warn)' : 'var(--text)'}">${esc(T.awaiting)}</span><small>approvals open</small></div>
      <div class="dsh-card"><span class="dsh-card-l">Forecast scoreboard</span><span class="dsh-card-v">${esc(T.score)}</span><small>${esc(T.scoreCap)}</small></div>
      <div class="dsh-card dsh-card-wide"><span class="dsh-card-l">Verdicts</span>
        <div class="gg-plot gg-verdicts">${verdictBar(sp, { width: 300, base: 12 })}</div></div>`;
    const ins = arr(this.data?.insights);
    this.el.querySelector('.dsh-insights').innerHTML = ins.map((i) => {
      const text = typeof i === 'string' ? i : i.text || i.msg || '';
      const q = typeof i === 'object' ? i.query || '' : '';
      return `<li title="${esc(q)}">${esc(text)}</li>`;
    }).join('');
    const counts = { next: this.#rows('next24h').length, alerts: this.#rows('alerts').length, decisions: this.#rows('decisions').length, refusals: this.#rows('refusals').length,
      eval: null, ledger: S.ledger?.n || (Array.isArray(this.ledgerData) ? this.ledgerData.length : null) };
    this.el.querySelector('.dsh-tabs').innerHTML = TABS.map(([id, label]) => `<button type="button" class="dsh-tab${this.tab === id ? ' on' : ''}" data-tab="${id}">${label}${counts[id] != null ? `<span class="dsh-count">${counts[id]}</span>` : ''}</button>`).join('');
    const pane = this.el.querySelector('.dsh-pane');
    if (this.tab === 'stats') mountStats(pane, { api: this.api, dash: this.data }); else pane.innerHTML = this.#pane();
  }

  #table(head, rows, rowFn, cls = () => '') {
    return `<div class="dsh-table-wrap"><table class="dsh-table"><thead><tr>${head.map((x) => `<th${/^(Pc|Miss|P\(crit\)|Conf|#)/.test(x) ? ' class="dsh-num"' : ''}>${x}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((r) => `<tr class="${cls(r)}">${rowFn(r)}</tr>`).join('')}</tbody></table></div>`;
  }

  #pane() {
    const now = simNow();
    const tca = (r) => { const t = ms(r.tca_sim) ?? (r.ttca_ms != null ? now + r.ttca_ms : null); return t ? `${fmtSim(t, false)}<small>${fmtDur(t - now)}</small>` : '—'; };
    const pc = (r) => fmtPc(r.latest_log10_pc ?? r.log10_pc ?? r.pc?.log10, { status: r.pc_status });
    if (this.tab === 'next') {
      const rows = this.#rows('next24h');
      if (!rows.length) return empty('Nothing in the next 24 h', 'No conjunction in the current cast has its TCA inside the next 24 SIM hours.');
      return this.#table(['Event', 'TCA (SIM)', 'Tier', 'Miss', 'Pc (physics)', 'P(crit) forecast', 'Status'], rows, (r) =>
        `<td>${keyLink(r.event_key)}</td><td class="dsh-time">${tca(r)}</td><td>${tierBadge(r.tier)}</td><td class="dsh-num">${fmtMiss(r.latest_miss_m)}</td>` +
        `<td class="dsh-num">${pc(r)}</td><td class="dsh-num">${r.p_critical != null ? fmtPct(r.p_critical) : '<span class="muted">n/a</span>'}</td>` +
        `<td>${r.awaiting ? '<span class="dsh-badge" style="color:var(--warn);border-color:var(--warn)">AWAITING YOU</span>' : r.verdict ? esc(r.verdict) : '—'}</td>`,
      (r) => (r.tier === 'CRITICAL' ? 'is-bad' : ''));
    }
    if (this.tab === 'alerts') {
      const rows = this.#rows('alerts');
      if (!rows.length) return empty('No alerts', 'No event is ELEVATED or CRITICAL right now.');
      return this.#table(['Event', 'Tier', 'Pc tier', 'Miss tier', 'Rules', 'Why'], rows, (r) =>
        `<td>${keyLink(r.event_key)}</td><td>${tierBadge(r.tier)}</td><td>${tierBadge(r.pc_tier)}</td><td>${tierBadge(r.miss_tier)}</td>` +
        `<td class="dsh-mono">${esc((r.rule_ids || []).join(', '))}</td><td>${esc(arr(r.reasons).join(' · '))}</td>`, (r) => (r.tier === 'CRITICAL' ? 'is-bad' : ''));
    }
    if (this.tab === 'decisions') {
      const rows = this.#rows('decisions');
      if (!rows.length) return empty('No decisions yet', 'Flagged events get a decision-model suggestion next to the cost rule; SAFE events are logged automatically.');
      return `<p class="dsh-note">Decision confidence is the model's confidence in its choice — <b>not a collision probability</b>.</p>` +
        this.#table(['Event', 'Verdict', 'Action', 'Provider', 'Conf', 'Agrees w/ cost rule', 'Reasons'], rows, (r) =>
          `<td>${keyLink(r.event_key)}</td><td>${esc(r.verdict || '—')}</td><td>${esc(ACTION_SHORT[r.action] || r.action || '—')}</td><td>${esc(r.provider || '—')}</td>` +
          `<td class="dsh-num">${r.confidence != null ? fmtPct(r.confidence) : '—'}</td><td>${r.agree == null ? '—' : r.agree ? 'yes' : '<span class="badtxt">no</span>'}</td>` +
          `<td>${esc(arr(r.reasons).join(' · '))}</td>`);
    }
    if (this.tab === 'refusals') {
      const rows = this.#rows('refusals');
      if (!rows.length) return empty('No refusals', 'Every CDM in the cast has a usable covariance. A default covariance (σ = 63,781,363 m) would show here instead of a fake Pc.');
      return this.#table(['Event', 'Status', 'Tier', 'Why'], rows, (r) =>
        `<td>${keyLink(r.event_key)}</td><td><span class="dsh-badge" style="color:var(--agent);border-color:var(--agent)">${esc(r.pc_status || 'REFUSED')}</span></td><td>${tierBadge(r.tier)}</td><td>${esc(arr(r.reasons).join(' · '))}</td>`);
    }
    if (this.tab === 'eval') return this.#evalPane();
    if (this.tab === 'ledger') return this.#ledgerPane();
    return '';
  }

  #evalPane() {
    const d = this.evalData;
    if (d === undefined) return '<p class="dsh-note">Loading evaluation…</p>';
    if (!d || d.__err) return empty('Evaluation not available', d?.__err ? `GET /api/eval: ${d.__err}. Offline metrics come from the model registry.` : 'No metrics yet.');
    const out = ['<p class="dsh-note">Offline metrics from the model registry — <b>measured</b> on held-out data (OOF by event / by mission). Numbers are shown as the server reports them.</p>'];
    const fmtV = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(3)) : typeof v === 'boolean' ? (v ? 'yes' : 'no') : esc(String(v)));
    const isScalarObj = (o) => o && typeof o === 'object' && !Array.isArray(o) && Object.values(o).every((v) => v == null || typeof v !== 'object');
    const section = (title, obj, depth = 0) => {
      if (obj == null) return;
      if (Array.isArray(obj)) {
        if (!obj.length) return;
        if (obj.every((x) => typeof x !== 'object')) { out.push(`<div class="dsh-ev-h">${esc(title)}</div><ul class="dsh-list">${obj.map((x) => `<li>${fmtV(x)}</li>`).join('')}</ul>`); return; }
        const cols = [...new Set(obj.slice(0, 50).flatMap((x) => Object.keys(x || {})))].filter((c) => obj.every((x) => x?.[c] == null || typeof x[c] !== 'object')).slice(0, 9);
        out.push(`<div class="dsh-ev-h">${esc(title)}</div>` + this.#table(cols.map(esc), obj.slice(0, 60), (r) => cols.map((c) => `<td class="${typeof r?.[c] === 'number' ? 'dsh-num' : ''}">${r?.[c] == null ? '—' : /event_key|^key$/.test(c) ? keyLink(r[c]) : fmtV(r[c])}</td>`).join('')));
        return;
      }
      if (typeof obj !== 'object') { out.push(`<p class="dsh-note"><b>${esc(title)}:</b> ${fmtV(obj)}</p>`); return; }
      const scal = Object.entries(obj).filter(([, v]) => v == null || typeof v !== 'object');
      const nested = Object.entries(obj).filter(([, v]) => v && typeof v === 'object');
      // baseline-vs-model: nested objects that are all scalar maps with shared keys → one comparison table
      const cmp = nested.filter(([, v]) => isScalarObj(v));
      if (scal.length) out.push(`<div class="dsh-ev-h">${esc(title)}</div>` + this.#table(['Metric', 'Value'], scal, ([k, v]) => `<td>${esc(k)}</td><td class="dsh-num">${v == null ? '—' : fmtV(v)}</td>`));
      if (cmp.length >= 2) {
        const keys = [...new Set(cmp.flatMap(([, v]) => Object.keys(v)))].slice(0, 10);
        out.push(`<div class="dsh-ev-h">${esc(title)} · model vs baselines</div>` + this.#table(['', ...keys.map(esc)], cmp, ([n, v]) =>
          `<td><b>${esc(n)}</b></td>${keys.map((k) => `<td class="dsh-num">${v[k] == null ? '—' : fmtV(v[k])}</td>`).join('')}`, ([n]) => (/persist|baseline|lrp|naive/i.test(n) ? '' : 'dsh-model')));
      } else for (const [k, v] of cmp) section(`${title} · ${k}`, v, depth + 1);
      if (depth < 3) for (const [k, v] of nested.filter(([, v]) => !isScalarObj(v))) section(depth ? `${title} · ${k}` : k, v, depth + 1);
    };
    for (const [k, v] of Object.entries(d)) section(k, v, 0);
    return out.join('');
  }

  #ledgerPane() {
    const d = this.ledgerData;
    const v = this.verify;
    const chip = v ? (v.ok ? `<span class="chip good">chain ✓ ${(v.n ?? 0).toLocaleString()} entries</span>` : `<span class="chip bad">chain broken at #${esc(v.first_bad_seq)}</span>`) : '';
    const head = `<div class="dsh-ledger-h"><button type="button" class="small dsh-verify">Verify chain</button>${chip}<span class="muted">Append-only, hash-chained; built by observing the event bus.</span></div>`;
    if (d == null) return head + '<p class="dsh-note">Loading ledger…</p>';
    if (d.__err) return head + empty('Ledger not available', `GET /api/ledger: ${d.__err}`, true);
    if (!d.length) return head + empty('No ledger entries yet', 'Start a scenario: every CDM, assessment, decision and approval becomes a row.', true);
    const rows = [...d].sort((a, b) => (b.seq ?? 0) - (a.seq ?? 0));
    this.csvRows = rows;
    return head + this.#table(['#', 'SIM time', 'Kind', 'Event', 'Actor', 'Type', 'Hash'], rows, (r) =>
      `<td class="dsh-num">${esc(r.seq)}</td><td class="dsh-time">${r.sim_time ? fmtSim(ms(r.sim_time)) : '—'}</td><td>${esc(r.kind || '—')}</td><td>${keyLink(r.event_key)}</td>` +
      `<td>${esc(r.actor || '—')}</td><td>${esc(r.type || r.payload?.type || '—')}</td><td class="dsh-mono" title="${esc(r.hash)}">${esc(String(r.hash || '').slice(0, 12))}…</td>`);
  }

  async #verify() {
    try { this.verify = await this.api('GET', '/ledger/verify'); } catch (e) { this.verify = { ok: false, first_bad_seq: `? (${e.message})` }; }
    this.#render();
  }

  #csv() {
    let rows, cols;
    if (this.tab === 'ledger' && Array.isArray(this.ledgerData)) { rows = this.ledgerData; cols = ['seq', 'sim_time', 'kind', 'event_key', 'actor', 'type', 'hash', 'prev_hash']; }
    else {
      const name = { next: 'next24h', alerts: 'alerts', decisions: 'decisions', refusals: 'refusals' }[this.tab];
      if (!name) return;
      rows = this.#rows(name);
      cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    }
    const cell = (v) => { const s = v == null ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => cell(r[c])).join(','))].join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `orbital-dodge-${this.tab}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
}
