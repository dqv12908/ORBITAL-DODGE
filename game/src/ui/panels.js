// Inspector (right), agent popover (left), approval card (modal), live-sky list.
import { S, on, emit, ev, tierOf, latestRisk, latestCdm, simNow, fmtPc, fmtPcShort, fmtMiss, fmtPct, fmtDur, fmtTminus, fmtSim, esc, h, ACTION_LABEL, ACTION_SHORT } from '../store.js';
import { planeParams } from './charts.js';
import { GgEncounter, GgRisk, GgRtn } from './gg/inspector.js';
import { newIdemKey } from '../net.js';

const TIER_CLS = { SAFE: 'good', ELEVATED: 'warn', CRITICAL: 'bad', PLANNED: 'info', UNKNOWN: '' };
const chip = (t, extra = '') => (t ? `<span class="chip ${TIER_CLS[t] || ''}">${esc(t)}${extra}</span>` : '');

// ═══ Inspector ════════════════════════════════════════════════════════
export class Inspector {
  constructor(root, { world }) {
    this.world = world;
    this.el = root.appendChild(h(`<div class="pop pop-machine insp" hidden>
      <div class="mp">
        <div class="mp-head"><div class="mp-titles"><div class="mp-name"></div><div class="mp-tags"></div></div><button class="pop-x" title="Close (Esc)">✕</button></div>
        <div class="in-sec"><span class="ap-sub">Encounter plane</span><span class="in-tag">PHYSICS · from CDM covariance</span></div>
        <div class="in-enc gg-host"></div>
        <div class="in-cap">Encounter plane — from CDM covariance (physics). Not reconstructed. Ellipses 1/2/3σ of the combined covariance.</div>
        <div class="in-sec"><span class="ap-sub">Risk evolution</span><span class="in-tag">time to TCA →</span></div>
        <div class="in-risk gg-host"></div>
        <div class="in-sec"><span class="ap-sub">Relative position</span><span class="in-tag">RTN · from CDM</span></div>
        <div class="in-rtn gg-host"></div>
        <div class="in-vitals"></div>
        <div class="in-recon"></div>
        <div class="mp-caption">This is what the system reads: released CDMs only; future CDMs are never sent to the browser.</div>
      </div></div>`));
    this.enc = new GgEncounter(this.el.querySelector('.in-enc'));
    this.risk = new GgRisk(this.el.querySelector('.in-risk'));
    this.rtn = new GgRtn(this.el.querySelector('.in-rtn'));
    this.el.querySelector('.pop-x').addEventListener('click', () => this.close());
    on('inspect', (k) => this.open(k));
    on('focus', (k) => { if (!this.el.hidden) this.show(k); });
    on('event', (k) => { if (k === this.key) this.render(); });
    on('cdm', (k) => { if (k === S.focus && this.el.hidden && !this.userClosed) this.open(k); });
    on('scenario', () => { this.userClosed = false; });
    setInterval(() => { if (!this.el.hidden) this.render(); }, 1000);
  }
  open(key) { this.userClosed = false; this.show(key || S.focus); }
  show(key) {
    if (!key) return;
    this.key = key;
    this.el.hidden = false;
    document.getElementById('ui').dataset.open = 'machine';
    this.render();
  }
  close() { this.el.hidden = true; this.userClosed = true; delete document.getElementById('ui').dataset.open; }
  get isOpen() { return !this.el.hidden; }
  render() {
    const e = ev(this.key); if (!e) return;
    const r = latestRisk(e), c = latestCdm(e), t = tierOf(e);
    const n = e.cdms.filter(Boolean).length;
    this.el.querySelector('.mp-name').textContent = e.key;
    const ttca = e.tcaMs != null ? e.tcaMs - simNow() : null;
    this.el.querySelector('.mp-tags').innerHTML = `${chip(t)} ${r?.tier ? `<span class="mp-tag">Pc tier ${esc(r.tier.pc_tier)}</span><span class="mp-tag">miss tier ${esc(r.tier.miss_tier)}</span>` : ''}
      <span class="mp-cost">${n} CDM${n === 1 ? '' : 's'} released${ttca != null ? ` · TCA ${ttca > 0 ? 'in ' + fmtDur(ttca) : 'passed'}` : ''}${r?.tier?.mode ? ` · rules ${esc(r.tier.mode)}` : ''}</span>`;
    const i = c?.cdm_index ?? n - 1;
    const encs = e.geometry?.encounter || e.geometry?.encounter_plane || [];
    const enc = Array.isArray(encs) ? encs.find((x) => x.cdm_index === i) || encs[i] : null;
    const pp = r || c ? planeParams(enc, r, c) : null;
    this.enc.set(pp, t, this.key);
    if (this.key === S.focus) { try { this.world.setCovariance?.(pp); } catch { /* world optional */ } }
    this.#chart();
    const f = e.forecast;
    const rows = [
      ['Pc (physics)', fmtPc(r?.pc?.log10, r?.pc), r?.pc?.status && r.pc.status !== 'OK' ? r.pc.status : (r?.pc?.method || '')],
      ['ESA reported', r?.reported_log10 != null ? fmtPc(r.reported_log10) : '—', r?.disagree_dex != null ? `Δ ${(+r.disagree_dex).toFixed(2)} dex` : ''],
      ['Miss distance', fmtMiss(c?.miss_m), c?.ttca_d != null ? `at T−${(+c.ttca_d).toFixed(2)} d` : ''],
      ['Relative speed', c?.rel_speed_ms != null ? `${(c.rel_speed_ms / 1000).toFixed(2)} km/s` : '—', ''],
      ['Chaser', e.chaser_type || c?.c_object_type || '—', e.mission != null ? `mission ${e.mission}` : ''],
    ];
    const flags = [...(c?.flags || [])];
    this.el.querySelector('.in-vitals').innerHTML = rows.map(([k, v, s]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b><small>${esc(s)}</small></div>`).join('')
      + forecastBlock(e, c)
      + (r?.tier?.reasons?.length ? `<ul class="in-reasons">${r.tier.reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '')
      + (flags.length ? `<div class="in-flags">${flags.map((x) => `<span class="chip warn">${esc(x)}</span>`).join(' ')}</div>` : '');
    const res = e.geometry?.residuals || e.geometry || {};
    this.el.querySelector('.in-recon').innerHTML = e.geometry ? `<b>RECONSTRUCTED GEOMETRY</b> · longitude/RAAN synthetic${res.inc_res_deg != null ? ` · chaser inc residual ${(+res.inc_res_deg).toFixed(2)}°` : ''}${res.sma_res_km != null ? `, sma residual ${Math.round(res.sma_res_km)} km` : ''}` : '';
  }
  #chart() {
    const e = ev(this.key); if (!e) return;
    const pts = [];
    e.cdms.forEach((c, i) => {
      if (!c) return;
      const r = e.risks[i];
      pts.push({ ttca: c.ttca_d ?? (e.tcaMs && c.at_sim ? (e.tcaMs - Date.parse(c.at_sim)) / 86400e3 : 0), log10: r?.pc?.log10, reported: r?.reported_log10, miss_m: c.miss_m, status: r?.pc?.status });
    });
    const nowTtca = e.tcaMs != null ? (e.tcaMs - simNow()) / 86400e3 : null;
    this.risk.set({ points: pts, nowTtca, futures: e.futures, tier: tierOf(e), key: this.key });
    this.rtn.set(e);
  }
}

// ═══ Agent popover (left) ═════════════════════════════════════════════
const STEPS = [['CDM', 'New CDM released'], ['Pc', 'Physics Pc from covariance'], ['Tier', 'Tier rules'], ['Forecast', 'ML forecast (≥ 2 d only)'], ['Decision', 'Decision model + cost rule'], ['You', 'Human decides']];
export class AgentPanel {
  constructor(root) {
    this.el = root.appendChild(h(`<div class="pop pop-agent" hidden><div class="ap">
      <div class="ap-head"><div class="ap-avatar"><i></i><i></i></div><div class="ap-who"><div class="title">DODGE AGENT</div><div class="ap-state"></div></div><span class="chip agent mode-chip"></span><button class="pop-x">✕</button></div>
      <div class="ap-brief"></div>
      <div class="ap-scroll"></div></div></div>`));
    this.el.querySelector('.pop-x').addEventListener('click', () => this.close());
    for (const t of ['event', 'agent', 'focus', 'approvals']) on(t, () => { if (!this.el.hidden) this.render(); });
    on('scenario', () => this.#brief());
  }
  open(key) { this.key = key || S.focus; this.el.hidden = false; this.#brief(); this.render(); }
  close() { this.el.hidden = true; }
  toggle(key) { this.el.hidden ? this.open(key) : this.close(); }
  get isOpen() { return !this.el.hidden; }
  #brief() {
    const s = S.scenario;
    this.el.querySelector('.ap-brief').innerHTML = s ? `<div class="ap-brief-b" style="padding-top:12px"><div class="ap-tagline"><b>${esc(s.title || s.id)}</b>${s.tagline ? ' — ' + esc(s.tagline) : ''}</div>${s.teaches ? `<div class="ap-teaches"><span>Teaches</span>${esc(s.teaches)}</div>` : ''}</div>` : '';
  }
  render() {
    const key = this.key || S.focus;
    const e = key && ev(key);
    const busy = e && Object.values(e.steps).includes('on');
    this.el.querySelector('.ap-avatar').classList.toggle('busy', !!busy);
    const st = this.el.querySelector('.ap-state');
    st.textContent = e?.approval ? 'Needs your OK' : busy ? 'Working…' : 'Watching the sky';
    st.className = 'ap-state' + (e?.approval ? ' warn' : '');
    this.el.querySelector('.mode-chip').textContent = S.mode.fixture ? 'FIXTURE' : `DECISIONS ${String(S.mode.decisions || '?').toUpperCase()}`;
    const sc = this.el.querySelector('.ap-scroll');
    if (!e) { sc.innerHTML = '<div class="ap-idle-h good">All quiet</div><div class="ap-idle-p">No conjunction in focus. Pick a scenario to start.</div>'; return; }
    const t = tierOf(e), r = latestRisk(e), c = latestCdm(e);
    const strip = STEPS.map(([s]) => `<i class="${e.steps[s] === 'done' || e.steps[s] === 'skip' ? 'done' : e.steps[s] === 'on' ? 'on' : ''}"></i>`).join('');
    const stepSummary = {
      CDM: c ? `CDM ${c.n_known ?? (c.cdm_index ?? 0) + 1} · T−${(+c.ttca_d || 0).toFixed(2)} d · miss ${fmtMiss(c.miss_m)}` : 'waiting',
      Pc: r ? `Pc ${fmtPc(r.pc?.log10, r.pc)} (${r.pc?.status || '—'})` : '',
      Tier: r ? `${r.tier?.tier} by ${(r.tier?.rule_ids || []).join(', ') || '—'}` : '',
      Forecast: e.forecast ? (e.forecast.valid === false ? 'not applicable inside 2 d' : e.forecast.p_critical != null ? `P(crit) ${fmtPct(e.forecast.p_critical)}` : `persistence ${e.forecast.baseline?.persistence_tier || '—'}`) : (c && c.ttca_d < 2 ? 'not applicable inside 2 d' : ''),
      Decision: e.decision ? `${e.decision.provider} → ${ACTION_SHORT[e.decision.answer] || e.decision.answer}` : e.decisionPending ? 'asking…' : e.verdict?.verdict === 'AUTO' ? 'AUTO · no model call' : '',
      You: e.human ? `${ACTION_SHORT[e.human.action] || e.human.action} by ${e.human.actor || 'you'}` : e.approval ? 'waiting for you' : '',
    };
    const steps = STEPS.map(([s, lbl]) => {
      const stt = e.steps[s] === 'skip' ? 'done' : e.steps[s] || 'todo';
      return `<li class="ap-step ${stt}"><div class="ap-step-h"><i class="mk"></i><b>${s}</b><span title="${esc(lbl)}">${esc(stepSummary[s] || lbl)}</span></div></li>`;
    }).join('');
    sc.innerHTML = `<div class="ap-card"><div class="ap-card-h"><div class="ap-chips">${chip(t)}${e.verdict ? `<span class="chip ${e.verdict.verdict === 'AUTO' ? 'good' : e.verdict.verdict === 'BLOCK' ? 'bad' : 'warn'}">${esc(e.verdict.verdict)}</span>` : ''}</div>
      <div class="ap-title" data-k="${esc(key)}">${esc(key)} · ${esc(e.chaser_type || '')}</div>
      <div class="ap-meta"><span>${e.tcaMs ? 'TCA ' + fmtTminus(e.tcaMs - simNow()) : ''}</span><span>${e.cdms.filter(Boolean).length} CDMs</span></div>
      <div class="ap-strip" style="grid-template-columns:repeat(6,1fr)">${strip}</div></div>
      ${e.approval ? `<div class="ap-cta"><span class="chip warn">APPROVAL NEEDED</span>The clock is held until you decide. <a data-a="open-appr">Open the decision card</a></div>` : ''}
      <ul class="ap-steps">${steps}</ul>
      ${decisionBlock(e)}
      ${e.verdict ? `<div class="ap-sub">Policy verdict</div><div class="ap-verdict ${e.verdict.verdict === 'AUTO' ? 'good' : e.verdict.verdict === 'BLOCK' ? 'bad' : 'warn'}"><div class="ap-verdict-h"><b>${esc(e.verdict.verdict)} · ${esc(ACTION_SHORT[e.verdict.action] || e.verdict.action || '')}</b><span>rules in code</span></div><ul>${(e.verdict.reasons || []).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>` : ''}
      ${e.explanation ? `<div class="ap-sub">Explanation <span class="chip ${e.explanation.verified ? 'good' : ''}">${esc(e.explanation.source)}${e.explanation.verified ? ' · verified' : ''}</span></div><div class="ap-say">${esc(e.explanation.text)}</div>` : ''}
      ${e.outcome ? `<div class="ap-sub">Outcome at TCA</div><div class="ap-done"><div class="ap-done-h">Final ${esc(e.outcome.final?.tier)} · miss ${fmtMiss(e.outcome.final?.miss_m)}</div><div class="ap-done-c">${e.outcome.hidden ? 'Truth hidden (blind test event).' : e.outcome.verdict == null ? esc(e.outcome.note || 'No CDM at ≥ 2 d: nothing to score.') : `T−2 d call by persistence: ${esc(e.outcome.forecast_was?.persistence_tier || '—')} → <b>${esc(e.outcome.persistence_verdict || e.outcome.verdict)}</b>${e.outcome.model_verdict ? ` · model → <b>${esc(e.outcome.model_verdict)}</b>` : ''}`}</div></div>` : ''}
    </div>`;
    sc.querySelector('[data-a=open-appr]')?.addEventListener('click', () => emit('approval.open', e.approval));
    sc.querySelector('.ap-title')?.addEventListener('click', () => emit('inspect', key));
  }
}

function routedTxt(r) {
  if (!r) return '';
  const list = Array.isArray(r) ? r : [r];
  const names = [...new Set(list.map((x) => (typeof x === 'string' ? x : [x.model, x.provider].filter(Boolean).join(' @ '))).filter(Boolean))];
  return names.length ? ` · routed: ${esc(names.slice(0, 3).join(', '))}${names.length > 3 ? ' +' + (names.length - 3) : ''}` : '';
}
function costLine(f) {
  const t = f?.alert_threshold ?? f?.threshold ?? f?.tau, R = f?.cost_ratio_R ?? f?.cost_ratio;
  if (t == null && R == null) return '';
  const parts = [t != null ? `alert threshold t = ${(+t).toFixed(3)}` : null, R != null ? `cost ratio R = ${esc(+R % 1 ? (+R).toFixed(1) : R)} (a missed critical costs R false alarms)` : null].filter(Boolean);
  const al = f?.alert == null ? '' : f.alert ? ' · <b class="warntxt">ALERT</b> (p ≥ t)' : ' · below t';
  return `<div class="ap-src fc-cost">${parts.join(', ')}${al}</div>`;
}
/** Honest provenance of a forecast value: model name + where the number comes from (out-of-fold for train events). */
export function fcSource(f) {
  if (!f) return '';
  if (f.valid === false) return f.reason || 'not applicable inside 2 d: physics on the latest CDM governs';
  if (f.fallback || f.model === 'persistence-empirical') {
    const b = f.baseline;
    return `persistence-empirical${b?.persistence_tier ? ` (last tier ${b.persistence_tier})` : ''}${f.note ? ' · ' + f.note : ''}`;
  }
  let ps = String(f.p_source || '');
  const src = String(f.model || '').split(':')[0];
  if (src && ps.startsWith(src)) ps = ps.slice(src.length).replace(/^\s*\((.*)\)\s*$/, '$1').trim(); // 'forecaster_bakeoff (x)' -> 'x'
  const oof = /oof|out-of-fold/i.test(ps) && !/out-of-fold/i.test(ps) ? ' · out-of-fold' : '';
  const mission = /mission/i.test(ps) ? ' (by mission)' : '';
  return [f.model, ps ? ps + oof + mission : null].filter(Boolean).join(' · ');
}
/** Inspector: the served forecast with its source, the T−2 d value and the tier_forecaster OOF value when they differ. */
function forecastBlock(e, c) {
  const f = e.forecast, m = e.meta || {}, t2 = m.forecast_t2;
  const late = c?.ttca_d != null && c.ttca_d < 2;
  let head, val;
  if (f && f.valid !== false && f.p_critical != null) { head = 'Chance this ends CRITICAL'; val = fmtPct(f.p_critical); }
  else if (t2?.p_critical != null) { head = 'Forecast made at T−2 d'; val = fmtPct(t2.p_critical); }
  else { head = 'Chance this ends CRITICAL'; val = late ? 'n/a inside 2 d' : 'n/a'; }
  const src = f && f.valid !== false ? fcSource(f) : t2 ? [t2.model, t2.source === 'oof' || /^train:/.test(e.key) ? 'out-of-fold' : null].filter(Boolean).join(' · ') : late ? 'not applicable inside 2 d: physics governs' : '';
  const extra = [];
  if (t2?.p_critical != null && f && f.valid !== false && f.p_critical != null && Math.abs(t2.p_critical - f.p_critical) > 1e-6) extra.push(`served at T−2 d: ${fmtPct(t2.p_critical)}${t2.model ? ' (' + esc(t2.model) + ')' : ''}`);
  if (m.p_critical_oof != null && (t2?.p_critical == null || Math.abs(m.p_critical_oof - t2.p_critical) > 1e-6)) extra.push(`tier_forecaster out-of-fold: ${fmtPct(m.p_critical_oof)}`);
  return `<div class="in-fc"><span>${esc(head)} <em class="chip-mini st">STATISTICS</em></span><b>${esc(val)}</b>
    <small>${esc(src)}</small>${costLine(f || t2)}${extra.length ? `<small class="fc-x">${extra.join(' · ')}</small>` : ''}</div>`;
}

/** Decision bars (agent purple) vs faint cost-rule bar. Never on a Pc axis. */
export function decisionBlock(e, { footnote = true } = {}) {
  const d = e.decision;
  if (!d) return e.decisionPending ? '<div class="ap-sub">Decision model</div><div class="ap-wait">Asking the decision model (clock held)</div>' : '';
  const probs = d.probabilities || {};
  const el = d.baseline?.expected_loss || {};
  const acts = Object.keys(probs).length ? Object.keys(probs) : Object.keys(el);
  const maxL = Math.max(1, ...Object.values(el));
  const rows = acts.map((a) => {
    const p = probs[a] ?? 0;
    const base = d.baseline?.action === a;
    const lossW = el[a] != null ? 100 * (1 - el[a] / maxL) : 0;
    return `<div class="ap-prob ${a === d.answer ? 'top' : ''}"><span class="nm">${esc(ACTION_LABEL[a] || a)}${base ? ' <em class="cr-pick">rule</em>' : ''}</span>
      <span class="bars"><span class="bar dec"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span class="bar prior" title="expected loss ${el[a] ?? '—'} (lower is better)"><i style="width:${lossW.toFixed(1)}%"></i></span></span>
      <b>${fmtPct(p)}<em>${el[a] != null ? 'loss ' + el[a] : ''}</em></b></div>`;
  }).join('');
  return `<div class="ap-sub">Decision model <span class="chip agent">AI JUDGEMENT</span></div>
    <div class="ap-legend"><span><i class="k-model k-dec"></i>${esc(d.provider)}</span><span><i class="k-prior"></i>cost rule (expected loss, shorter = worse)</span></div>
    <div class="ap-probs">${rows}</div>
    <div class="ap-src">${esc(d.provider)}${d.probability_kind ? ` · <b>${esc(d.probability_kind)}</b>` : ''}${routedTxt(d.routed)}${d.latency_ms != null ? ` · ${d.latency_ms} ms` : ''} · ${d.agree ? 'agrees with cost rule' : `<span class="warntxt">disagrees with cost rule (${esc(ACTION_SHORT[d.baseline?.action] || '—')})</span>`}${d.refused ? ' · <span class="badtxt">refused</span>' : ''}</div>
    ${footnote ? '<div class="dec-foot">Decision model\'s confidence in its choice — <b>not a collision probability</b>.</div>' : ''}`;
}

// ═══ Approval card (modal) ════════════════════════════════════════════
export class ApprovalCard {
  constructor(root, { api, toast }) {
    this.root = root; this.api = api; this.toast = toast;
    on('approval.required', (a) => this.open(a));
    on('approval.open', (a) => this.open(a));
    on('approvals', () => { if (this.a && !S.approvals.has(this.a.approval_id)) this.close(); });
    on('event', (k) => { if (this.a?.event_key === k) this.#render(); });
    setInterval(() => this.#live(), 500);
  }
  open(a) {
    if (!a) return;
    this.a = a;
    this.key = newIdemKey(); // one Idempotency-Key per card: retries are safe
    if (!this.back) {
      this.back = this.root.appendChild(h('<div class="ap-modal-back"><div class="ap-modal"></div></div>'));
      this.back.addEventListener('pointerdown', (e) => { if (e.target === this.back) this.close(); });
    }
    this.#render();
  }
  close() { this.back?.remove(); this.back = null; this.a = null; }
  get isOpen() { return !!this.back; }
  #render() {
    if (!this.back || !this.a) return;
    const a = this.a, e = ev(a.event_key), r = latestRisk(e), c = latestCdm(e), t = tierOf(e);
    const opts = a.options || ['REQUEST_TRACKING', 'PREPARE_MANOEUVRE'];
    const btnCls = { PREPARE_MANOEUVRE: 'good', REQUEST_TRACKING: 'accent', MONITOR: '' };
    const order = ['PREPARE_MANOEUVRE', 'REQUEST_TRACKING', 'MONITOR'].filter((x) => opts.includes(x));
    const f = e.forecast;
    const m = this.back.querySelector('.ap-modal');
    m.innerHTML = `<div class="ap-m-h"><span class="chip warn">APPROVAL NEEDED</span><small>suggested: ${esc(ACTION_SHORT[a.suggested] || a.suggested || '—')}</small><button class="pop-x">✕</button></div>
      <div class="ap-m-title">${esc(a.event_key)} — needs your decision</div>
      <div class="ap-m-sub">The system stopped before acting: this event is outside what it may do on its own. Nothing commands a burn.</div>
      <div class="ap-cols"><div class="ap-col">
      <div class="ap-ev">
        <div><span>Pc <em class="chip-mini">PHYSICS</em></span><b class="tc-${t}">${fmtPc(r?.pc?.log10, r?.pc)}</b></div>
        <div><span>Miss</span><b>${fmtMiss(c?.miss_m)}</b></div>
        <div><span>Sub-tiers</span><b>Pc ${esc(r?.tier?.pc_tier || '—')} · miss ${esc(r?.tier?.miss_tier || '—')}</b></div>
        <div><span>Forecast <em class="chip-mini st">STATISTICS</em></span><b>${f && f.valid !== false && f.p_critical != null ? (c?.ttca_d != null && c.ttca_d < 2 ? `at T−2 d: ${fmtPct(f.p_critical)}` : `${fmtPct(f.p_critical)} ends CRITICAL`) : e.meta?.forecast_t2?.p_critical != null ? `at T−2 d: ${fmtPct(e.meta.forecast_t2.p_critical)}` : 'n/a inside 2 d'}</b></div>
      </div>
      <div class="ap-src fc-src">Forecast source: ${esc(f && f.valid !== false ? fcSource(f) : e.meta?.forecast_t2?.model ? e.meta.forecast_t2.model + (/^train:/.test(e.key) ? ' · out-of-fold' : '') : 'none (inside 2 d, physics governs)')}</div>
      ${costLine(f && f.valid !== false ? f : e.meta?.forecast_t2)}
      <div class="ap-sub">Why it asks you</div>
      <ul class="ap-m-reasons">${(a.reasons || e.verdict?.reasons || []).map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
      <div class="ap-m-wait"></div>
      <textarea class="ap-note" rows="2" placeholder="Note for the ledger (optional)"></textarea>
      <div class="ap-m-btns">${order.map((o) => `<button class="${btnCls[o]}" data-act="${o}">${esc(ACTION_LABEL[o] || o)}</button>`).join('')}<button class="ghost rej" data-act="REJECT" title="Reject the suggestion (logged)">Reject</button></div>
      </div><div class="ap-col">
      ${decisionBlock(e)}
      ${e.explanation ? `<div class="ap-sub">Explanation <span class="chip ${e.explanation.verified ? 'good' : ''}">${esc(e.explanation.source)}${e.explanation.verified ? ' · verified' : ''}</span></div><div class="ap-say">${esc(e.explanation.text)}</div>` : ''}
      </div></div>
      <small class="ap-m-foot">Your choice is written to the hash-chained ledger with the CDM row, rules and model versions. ${S.mode.fixture ? 'FIXTURE mode: canned sequence, backend offline.' : ''}</small>`;
    m.querySelector('.pop-x').addEventListener('click', () => this.close());
    m.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => this.#decide(b.dataset.act, b)));
    this.#live();
  }
  #live() {
    if (!this.back || !this.a) return;
    const e = ev(this.a.event_key);
    const w = this.back.querySelector('.ap-m-wait');
    const left = e?.tcaMs != null ? e.tcaMs - simNow() : (this.a.ttca_h ?? 0) * 3600e3;
    const txt = `<b>TCA in ${fmtDur(left)}</b> · ${this.a.mcp_passed || left < 86400e3 ? '<span class="warntxt">commitment window open</span>' : 'before the commitment window'} · clock held while you decide`;
    if (w.innerHTML !== txt) w.innerHTML = txt;
  }
  async #decide(action, btn) {
    const a = this.a;
    const note = this.back.querySelector('.ap-note')?.value || '';
    this.back.querySelectorAll('[data-act]').forEach((b) => (b.disabled = true));
    btn.textContent = 'Sending…';
    try {
      await this.api('POST', `/approvals/${encodeURIComponent(a.approval_id)}`, { action, note }, { idem: true, key: this.key });
      S.approvals.delete(a.approval_id);
      const e = ev(a.event_key); if (e.approval?.approval_id === a.approval_id) e.approval = null;
      emit('approvals');
      this.close();
    } catch (err) {
      this.toast?.(err.status === 409 ? `Already decided: ${err.message}` : err.message, 'bad', 'APPROVAL');
      this.#render();
    }
  }
}

// ═══ Live sky list ═════════════════════════════════════════════════════
export class LiveSkyPanel {
  constructor(root, { api, world, toast }) {
    this.api = api; this.world = world; this.toast = toast;
    this.el = root.appendChild(h(`<div class="pop live-pop" hidden>
      <div class="lv-head"><div><b>LIVE SKY</b><small class="lv-src"></small></div><button class="pop-x">✕</button></div>
      <div class="lv-sum"></div>
      <div class="lv-note">Real close approaches in the next 72 h, screened by ORBITAL DODGE from today's public CelesTrak catalogue and validated vs CelesTrak SOCRATES. <b>max Pc</b>: no covariance in public TLEs, so this is a worst-case screening value, not an operational Pc. Separate from the CDM replay. Colour: <span class="badtxt">&lt; 50 m</span> · <span class="warntxt">&lt; 200 m</span> · <span style="color:var(--accent)">&lt; 5 km</span> screened miss.</div>
      <div class="lv-list"></div></div>`));
    this.el.querySelector('.pop-x').addEventListener('click', () => this.toggle(false));
    on('live.pick', (k) => this.#select(k));
  }
  async toggle(force) {
    const onNow = force ?? !S.liveSkyOn;
    if (onNow && !S.liveSky) {
      try { S.liveSky = await this.api('GET', '/livesky'); } catch (e) { this.toast?.('Live sky unavailable: no data/livesky/latest.json yet', 'warn', 'LIVE SKY'); return; }
      this.world.setLiveSky(S.liveSky.conjunctions || []);
      this.#list();
    }
    S.liveSkyOn = onNow;
    this.el.hidden = !onNow;
    this.world.setLiveSkyVisible(onNow);
    this.world.setPopulationDim?.(onNow ? 0.55 : 1);
    emit('livesky');
  }
  #list() {
    const L = S.liveSky;
    this.el.querySelector('.lv-src').textContent = `generated ${String(L.generated_utc || '').replace('T', ' ').slice(0, 16)} UTC · ${L.n_objects?.toLocaleString?.() || '?'} objects · ${L.window_h || '?'} h window${L.validation?.socrates_overlap != null ? ` · SOCRATES overlap ${fmtPct(L.validation.socrates_overlap)}` : ''}`;
    const sm = L.summary || {};
    this.el.querySelector('.lv-sum').innerHTML = [['< 1 km · next 24 h', sm.next24h_lt1km_excl_low_rel_speed ?? sm.next24h_lt1km], ['< 5 km · next 24 h', sm.next24h_lt5km_excl_low_rel_speed ?? sm.next24h_lt5km], ['shown (closest per pair)', (L.conjunctions || []).length]]
      .map(([k, v]) => `<div><b>${v != null ? Number(v).toLocaleString() : '—'}</b><span>${k}</span></div>`).join('');
    const rows = (L.conjunctions || []).map((c, k) => ({ c, k })).sort((a, b) => (a.c.min_range_km ?? 99) - (b.c.min_range_km ?? 99));
    this.el.querySelector('.lv-list').innerHTML = rows.map(({ c, k }) => `<button class="lv-row" data-k="${k}">
      <span class="lv-dot" style="background:${(c.min_range_km ?? 99) < 0.05 ? 'var(--bad)' : (c.min_range_km ?? 99) < 0.2 ? 'var(--warn)' : 'var(--accent)'}"></span>
      <span class="lv-nm"><b>${esc(c.name_1 || c.norad_1)} × ${esc(c.name_2 || c.norad_2)}</b><small>${c.tca_utc ? esc(c.tca_utc.replace('T', ' ').slice(0, 16)) + ' UTC' : ''}${c.socrates_match ? ' · SOCRATES ✓' : ''}</small></span>
      <span class="lv-v"><b>${c.min_range_km != null ? (+c.min_range_km).toFixed(3) + ' km' : '—'}</b><small>${c.rel_speed_kms != null ? (+c.rel_speed_kms).toFixed(1) + ' km/s · ' : ''}${c.log10_pc_max != null ? 'max Pc ' + fmtPcShort(c.log10_pc_max) : ''}</small></span></button>`).join('') || '<div class="dsh-empty small"><b>No approaches in the file</b></div>';
    this.el.querySelectorAll('.lv-row').forEach((b) => b.addEventListener('click', () => this.#select(+b.dataset.k)));
  }
  #select(k) {
    this.el.querySelectorAll('.lv-row').forEach((b) => b.classList.toggle('on', +b.dataset.k === k));
    const p = this.world.livePos(k);
    if (p) this.world.flyToPoint(p, { dist: 1.6 });
  }
}
void fmtSim;
