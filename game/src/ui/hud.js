// HUD: top bar, status chips, hint chips + floating pills anchored to the 3D world, toasts, connection banner, menu.
import { S, on, emit, ev, tierOf, latestRisk, latestCdm, simNow, fmtSim, fmtTminus, fmtPc, fmtMiss, esc, h, ACTION_SHORT } from '../store.js';

const STEPS = ['CDM', 'Pc', 'Tier', 'Forecast', 'Decision', 'You'];
const TIER_CLS = { SAFE: 'good', ELEVATED: 'warn', CRITICAL: 'bad', PLANNED: 'info' };

export class Hud {
  constructor(root, { api, world, actions }) {
    this.root = root; this.api = api; this.world = world; this.actions = actions;
    this.#bar();
    this.chips = root.appendChild(h('<div class="ac-chips"></div>'));
    this.hints = root.appendChild(h('<div class="hint-layer"></div>'));
    this.floats = root.appendChild(h('<div class="float-layer"></div>'));
    this.toasts = root.appendChild(h('<div class="toasts"></div>'));
    this.conn = root.appendChild(h('<div class="conn" hidden>Reconnecting to mission control…</div>'));
    this.labels = root.appendChild(h(`<div class="world-labels">
      <div class="wl-recon" hidden><b>RECONSTRUCTED GEOMETRY</b><span>longitude/RAAN synthetic · objects not to scale</span></div>
      <div class="wl-obj wl-target" hidden></div><div class="wl-obj wl-chaser" hidden></div>
    </div>`));
    this.bgNote = root.appendChild(h(`<div class="bg-note"><i></i>Background: today's sky (CelesTrak, approx.) — <b>not used for risk</b><span class="bg-legend"></span></div>`));
    this.stage = root.appendChild(h('<div class="stage-note" hidden>Conjunction: historical ESA event (2015–19), re-staged on SIM clock</div>'));
    this.floatList = [];
    on('clock', () => this.#clock());
    on('state', () => { this.#scn(); this.#kpis(); this.#chips(); });
    on('scenario', () => { this.#scn(); this.#kpis(); this.#chips(); });
    on('event', () => { this.#kpis(); this.#chips(); });
    on('approvals', () => { this.#kpis(); this.#chips(); });
    on('tiers', () => this.#kpis());
    on('risk', (key, prev, t) => this.#onRisk(key, prev, t));
    on('human', (key, d) => this.float(`${ACTION_SHORT[d.action] || d.action} · BY YOU`, /PREPARE/.test(d.action) ? 'info' : 'good', 'tca'));
    on('outcome', (key, d) => this.float(`TCA PASSED${d.hidden ? ' · TRUTH HIDDEN' : d.verdict ? ' · ' + d.verdict : ''}`, d.verdict === 'FN' ? 'bad' : 'good', 'tca'));
    on('degraded', (d) => this.toast(`${d.component}: ${d.reason} → ${d.fallback}`, 'warn', 'DEGRADED'));
    const loop = () => { this.#frame(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    setInterval(() => { this.#clockText(); this.#kpis(); }, 500);
  }

  // ─── top bar ─────────────────────────────────────────────────────────
  #bar() {
    const slot = this.root.appendChild(h('<div class="slot-hud"></div>'));
    this.tb = slot.appendChild(h(`<div class="tb">
      <div class="tb-seg tb-time"><i class="tb-lamp"></i><span class="sim-tag">SIM</span><span class="tb-clock">—</span><span class="tb-state"></span></div>
      <button class="tb-seg tb-scn" data-a="scn" title="Choose a scenario"><span class="tb-scn-name">Choose scenario</span><i class="tb-chev"></i></button>
      <div class="tb-seg tb-money">
        <div class="tb-kpi k-1km" title="Events with TCA in the next 24 h and latest miss < 1 km"><span class="tb-k">≤1 km in 24 h</span><span class="tb-v">0</span></div>
        <div class="tb-kpi k-crit"><span class="tb-k">Critical</span><span class="tb-v">0</span></div>
        <div class="tb-kpi k-elev"><span class="tb-k">Elevated</span><span class="tb-v">0</span></div>
        <div class="tb-kpi k-ref"><span class="tb-k">Refused</span><span class="tb-v">0</span></div>
        <div class="tb-kpi k-await"><span class="tb-k">Awaiting you</span><span class="tb-v">0</span></div>
      </div>
      <div class="tb-seg tb-speed">
        <button class="tb-sp tb-sp-pause" data-sp="pause" title="Pause / resume (Space)"><i class="tb-ico is-pause"></i></button>
        <button class="tb-sp" data-sp="1">1×</button><button class="tb-sp" data-sp="2">2×</button>
        <button class="tb-sp" data-sp="4">4×</button><button class="tb-sp" data-sp="8">8×</button>
        <button class="tb-sp tb-next" data-sp="next" title="Jump to the next CDM">⏭</button>
      </div>
      <button class="tb-seg tb-live" data-a="live" title="Today's real close approaches (CelesTrak)"><i></i>LIVE SKY</button>
      <button class="tb-seg tb-menu" data-a="menu"><span class="tb-burger"><b></b><b></b><b></b></span><span class="tb-menu-t">Menu</span><i class="tb-badge" hidden></i></button>
    </div>`));
    this.tb.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.sp) return this.#speed(b.dataset.sp);
      if (b.dataset.a === 'scn') this.actions.scenarios();
      if (b.dataset.a === 'live') this.actions.liveSky();
      if (b.dataset.a === 'menu') this.#menu(b);
    });
    on('livesky', () => this.tb.querySelector('.tb-live').classList.toggle('is-on', S.liveSkyOn));
  }
  async #speed(sp) {
    try {
      if (sp === 'pause') await this.api('POST', S.clock.paused ? '/sim/resume' : '/sim/pause');
      else if (sp === 'next') await this.api('POST', '/sim/next-cdm');
      else {
        await this.api('POST', '/sim/speed', { speed: +sp });
        if (S.clock.paused) await this.api('POST', '/sim/resume');
      }
    } catch (err) {
      const g = this.tb.querySelector('.tb-speed');
      g.classList.remove('is-error'); void g.offsetWidth; g.classList.add('is-error');
      this.toast(err.message, 'bad', 'SIM');
    }
  }
  togglePause() { this.#speed('pause'); }
  setSpeed(s) { this.#speed(String(s)); }
  #menu(btn) {
    if (this.menuEl) { this.menuEl.remove(); this.menuEl = null; btn.classList.remove('is-on'); return; }
    const r = btn.getBoundingClientRect();
    const m = (this.menuEl = h(`<div class="menu" style="top:${r.bottom + 8}px;left:${Math.max(14, r.right - 300)}px">
      ${[['scn', 'Scenarios', 'Pick one idea to play', 'S'], ['desk', 'Analyst desk', 'Tier mode, watch threshold', 'P'], ['dash', 'Dashboard', 'Next 24 h, evaluation, ledger', 'D'],
        ['agent', 'Agent', 'Steps, decision bars, evidence', 'A'], ['live', 'Live sky', "Today's real close approaches", 'V']]
        .map(([k, t, s, key]) => `<button class="menu-item" data-m="${k}"><span class="menu-ico">${ICON[k]}</span><span class="menu-txt"><b>${t}</b><small>${s}</small></span><kbd>${key}</kbd></button>`).join('')}
      <div class="menu-foot"><div class="menu-modes">${modeBadges()}</div><div class="menu-keys"><kbd>Space</kbd> pause <kbd>1</kbd>–<kbd>4</kbd> speed <kbd>N</kbd> next CDM <kbd>F</kbd> fly to event <kbd>H</kbd> home <kbd>L</kbd> log size</div></div>
    </div>`));
    btn.classList.add('is-on');
    m.addEventListener('click', (e) => {
      const it = e.target.closest('.menu-item'); if (!it) return;
      ({ scn: this.actions.scenarios, desk: this.actions.desk, dash: this.actions.dashboard, agent: this.actions.agent, live: this.actions.liveSky })[it.dataset.m]?.();
      m.remove(); this.menuEl = null; btn.classList.remove('is-on');
    });
    document.body.appendChild(m);
    setTimeout(() => document.addEventListener('pointerdown', (e) => { if (this.menuEl && !this.menuEl.contains(e.target) && !btn.contains(e.target)) { this.menuEl.remove(); this.menuEl = null; btn.classList.remove('is-on'); } }, { once: true }), 0);
  }

  #clock() {
    const c = S.clock;
    const seg = this.tb.querySelector('.tb-time');
    seg.classList.toggle('is-paused', c.paused);
    seg.classList.toggle('is-held', c.held);
    this.tb.classList.toggle('is-paused', c.paused);
    seg.querySelector('.tb-state').textContent = c.held ? (c.hold_reason === 'waiting for you' || S.approvals.size ? '⧗ Waiting for you' : '⧗ AI working') : c.paused ? 'Paused' : '';
    for (const b of this.tb.querySelectorAll('.tb-sp')) {
      const sp = b.dataset.sp;
      b.classList.toggle('is-on', sp === 'pause' ? c.paused : !c.paused && +sp === c.speed);
    }
    this.tb.querySelector('.tb-sp-pause i').className = 'tb-ico ' + (c.paused ? 'is-play' : 'is-pause');
    this.#clockText();
  }
  #clockText() {
    this.tb.querySelector('.tb-clock').textContent = fmtSim(simNow());
  }
  #scn() {
    this.tb.querySelector('.tb-scn-name').textContent = S.scenario?.title || S.scenario?.id || 'Choose scenario';
  }
  #kpis() {
    const now = simNow();
    let k1 = 0, crit = 0, critMiss = 0, elev = 0, ref = 0;
    for (const k of S.cast || []) {
      const e = S.events.get(k); if (!e) continue;
      const r = latestRisk(e), c = latestCdm(e), t = r?.tier?.tier || e.summaryTier;
      const miss = c?.miss_m ?? e.summary?.latest_miss_m;
      if (e.tcaMs != null && e.tcaMs > now && e.tcaMs - now < 86400e3 && miss != null && miss < 1000) k1++;
      if (t === 'CRITICAL') { crit++; if (r?.tier?.pc_tier && r.tier.pc_tier !== 'CRITICAL') critMiss++; }
      if (t === 'ELEVATED') elev++;
      if (r?.pc?.status === 'COV_UNKNOWN' || (c?.flags || []).some((f) => /COV/.test(f))) ref++;
    }
    const set = (cls, v, sub) => {
      const el = this.tb.querySelector(`.${cls} .tb-v`);
      const txt = sub ? `${v}` : String(v);
      if (el.textContent !== txt) { el.classList.remove('is-up', 'is-down'); void el.offsetWidth; if (el.textContent !== '0' || v) el.classList.add(v > (+el.textContent || 0) ? 'is-down' : 'is-up'); el.textContent = txt; }
      if (sub != null) el.title = sub;
      el.closest('.tb-kpi').classList.toggle('is-hot', v > 0);
    };
    set('k-1km', k1); set('k-crit', crit, `${critMiss} critical by miss distance only`); set('k-elev', elev); set('k-ref', ref); set('k-await', S.approvals.size);
    this.tb.querySelector('.tb-badge').hidden = !S.approvals.size;
  }

  // ─── status chips (top-centre) ───────────────────────────────────────
  #chips() {
    const keys = new Set();
    if (S.focus) keys.add(S.focus);
    for (const a of S.approvals.values()) if (a.event_key) keys.add(a.event_key);
    this.chips.innerHTML = '';
    for (const k of keys) {
      const e = S.events.get(k); if (!e) continue;
      const t = tierOf(e);
      const done = STEPS.filter((s) => e.steps[s] === 'done' || e.steps[s] === 'skip').length;
      const onStep = STEPS.find((s) => e.steps[s] === 'on');
      let label, tone;
      if (e.approval) { label = 'Needs your OK'; tone = 't-attn'; }
      else if (e.outcome) { label = `TCA passed${e.outcome.hidden ? ' · truth hidden' : e.outcome.verdict ? ' · ' + e.outcome.verdict : ''}`; tone = e.outcome.verdict === 'FN' ? 't-bad' : 't-good'; }
      else if (e.human) { label = `${ACTION_SHORT[e.human.action] || e.human.action} · approved`; tone = 't-good'; }
      else if (onStep && onStep !== 'You') { label = `Assessing… ${done + 1}/6`; tone = 't-busy'; }
      else if (e.verdict?.verdict === 'AUTO') { label = 'Monitoring · AUTO'; tone = 't-good'; }
      else if (t === 'CRITICAL') { label = 'Critical'; tone = 't-bad'; }
      else { label = e.cdms.length ? `Watching · ${e.cdms.filter(Boolean).length} CDM` : 'Waiting for CDM'; tone = 't-busy'; }
      const dots = STEPS.map((s) => `<i class="${e.steps[s] === 'done' || e.steps[s] === 'skip' ? 'done' : e.steps[s] === 'on' ? 'on' : ''}" title="${s}"></i>`).join('');
      const chip = h(`<button class="ac-chip ${tone}" data-k="${esc(k)}">
        <span class="ac-ico">${ICON.sat}</span>
        <span class="ac-main"><span class="ac-line"><b>${esc(k)}</b><span class="ac-sep">·</span><span class="ac-label">${esc(label)}</span>${t ? `<span class="chip ${TIER_CLS[t] || ''}">${t}</span>` : ''}</span>
        <span class="ac-dots">${dots}</span><span class="ac-steps">${STEPS.map((s) => `<em class="${e.steps[s] || ''}">${s}</em>`).join('<s>→</s>')}</span></span></button>`);
      chip.addEventListener('click', () => (e.approval ? emit('approval.open', e.approval) : this.actions.agent(k)));
      this.chips.appendChild(chip);
    }
  }

  #onRisk(key, prev, t) {
    if (key !== S.focus) return;
    const r = latestRisk(ev(key));
    if (t === 'CRITICAL') this.float(`CRITICAL · Pc ${fmtPc(r?.pc?.log10, r?.pc)}`, 'bad', 'tca');
    else if (t === 'ELEVATED' && prev !== 'ELEVATED') this.float(`ELEVATED · ${fmtMiss(latestCdm(ev(key))?.miss_m)}`, 'warn', 'tca');
    else if (t && prev && t !== prev) this.float(t, 'good', 'tca');
  }

  // ─── world-anchored overlays ─────────────────────────────────────────
  float(text, kind = 'info', anchor = 'tca') {
    const el = this.floats.appendChild(h(`<div class="float ${kind}">${esc(text)}</div>`));
    this.floatList.push({ el, anchor, t0: performance.now(), n: this.floatList.length });
  }
  toast(msg, kind = 'info', chip) {
    const el = this.toasts.appendChild(h(`<div class="toast ${kind}">${chip ? `<span class="chip ${kind === 'bad' ? 'bad' : kind === 'warn' ? 'warn' : 'info'}">${esc(chip)}</span>` : ''}<span>${esc(msg)}</span></div>`));
    const kill = () => { el.classList.add('out'); setTimeout(() => el.remove(), 300); };
    el.addEventListener('click', kill);
    setTimeout(kill, 6000);
  }
  setConnected(ok) { this.conn.hidden = !!ok; }

  #frame() {
    const w = this.world;
    const focal = !!w.focal && !!S.focus;
    const e = S.focus ? S.events.get(S.focus) : null;
    // hint chip over the conjunction point (focal objects only)
    let hint = this.hints.querySelector('.hint');
    const t = tierOf(e);
    const r = latestRisk(e), c = latestCdm(e);
    let txt = null, cls = '';
    if (focal && e) {
      if (r?.pc?.status === 'COV_UNKNOWN') { txt = '⊘ NO COVARIANCE'; cls = 'is-warn'; }
      else if (t === 'CRITICAL') { txt = `✖ CRIT miss ${fmtMiss(c?.miss_m)}`; cls = 'is-bad'; }
      else if (t === 'ELEVATED') { txt = `⚠ ELEV ${r?.tier?.pc_tier === 'ELEVATED' ? 'Pc ' + fmtPc(r?.pc?.log10) : 'miss ' + fmtMiss(c?.miss_m)}`; cls = 'is-warn'; }
      else if (t === 'PLANNED') { txt = '◆ MANOEUVRE OPTIONS REQUESTED'; cls = 'is-info'; }
      else if (t === 'SAFE') { txt = `● SAFE ${fmtMiss(c?.miss_m)}`; cls = 'is-good'; }
    }
    const p = focal ? w.screenPos('tca') : null;
    if (txt && p) {
      if (!hint) { hint = this.hints.appendChild(h('<button class="hint"></button>')); hint.addEventListener('click', () => emit('inspect', S.focus)); }
      const tm = e.tcaMs != null ? ` · ${fmtTminus(e.tcaMs - simNow())}` : '';
      const full = txt + tm;
      if (hint.textContent !== full) hint.textContent = full;
      hint.className = `hint ${cls}`;
      hint.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y - 26)}px) translate(-50%, -100%)`;
    } else if (hint) hint.remove();

    // object labels + RECONSTRUCTED GEOMETRY tag
    const rec = this.labels.querySelector('.wl-recon');
    const lt = this.labels.querySelector('.wl-target'), lc = this.labels.querySelector('.wl-chaser');
    const pt = focal ? w.screenPos('target') : null, pc = focal ? w.screenPos('chaser') : null;
    rec.hidden = !focal || !p;
    this.stage.hidden = !focal;
    if (p && focal) rec.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y + 54)}px) translate(-50%, 0)`;
    const place = (el, pp, text, left) => {
      el.hidden = !pp; if (!pp) return;
      if (el.textContent !== text) el.textContent = text;
      el.style.transform = left ? `translate(${Math.round(pp.x - 14)}px, ${Math.round(pp.y - 10)}px) translate(-100%, 0)` : `translate(${Math.round(pp.x + 14)}px, ${Math.round(pp.y - 10)}px)`;
    };
    place(lt, pt, e ? `MISSION ${e.mission ?? e.summary?.mission_id ?? ''}`.trim() : '', true);
    place(lc, pc, e ? (e.chaser_type || 'CHASER') : '');

    // floats rise 46 px over 2.6 s
    const now = performance.now();
    this.floatList = this.floatList.filter((f) => {
      const k = (now - f.t0) / 2600;
      if (k >= 1) { f.el.remove(); return false; }
      const pp = w.screenPos(f.anchor) || { x: innerWidth / 2, y: innerHeight / 2 };
      const a = k < 0.12 ? k / 0.12 : k > 0.8 ? (1 - k) / 0.2 : 1;
      f.el.style.opacity = a;
      f.el.style.transform = `translate(${Math.round(pp.x)}px, ${Math.round(pp.y - 70 - k * 46 - f.n * 0)}px) translate(-50%, -100%)`;
      return true;
    });
  }
}

export function modeBadges() {
  const m = S.mode || {};
  const b = [];
  if (m.fixture) b.push('<span class="mode-badge is-warn">FIXTURE · backend offline</span>');
  b.push(`<span class="mode-badge ${m.decisions === 'live' ? 'is-good' : 'is-accent'}">Decisions ${esc(m.decisions || '?')}${m.decisions === 'mock' ? ' (cost rule)' : ''}</span>`);
  b.push(`<span class="mode-badge ${m.llm === 'live' ? 'is-good' : 'is-accent'}">LLM ${esc(m.llm || '?')}</span>`);
  b.push('<span class="mode-badge">Orbits reconstructed</span>');
  return b.join('');
}

export const ICON = {
  sat: '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="6" y="5.5" width="4" height="5" rx="1"/><rect x="0.5" y="6.5" width="4.5" height="3" rx=".5"/><rect x="11" y="6.5" width="4.5" height="3" rx=".5"/><rect x="7.4" y="2.5" width="1.2" height="3"/></svg>',
  scn: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/><rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/></svg>',
  desk: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 4h12M2 8h12M2 12h12"/><circle cx="5" cy="4" r="1.5" fill="currentColor"/><circle cx="10" cy="8" r="1.5" fill="currentColor"/><circle cx="7" cy="12" r="1.5" fill="currentColor"/></svg>',
  dash: '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="2" y="8" width="3" height="6" rx=".6"/><rect x="6.5" y="4" width="3" height="10" rx=".6"/><rect x="11" y="2" width="3" height="12" rx=".6"/></svg>',
  agent: '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="4" width="10" height="8" rx="2"/><rect x="7.3" y="1.5" width="1.4" height="2.5"/><circle cx="6" cy="8" r="1.2" fill="#171b24"/><circle cx="10" cy="8" r="1.2" fill="#171b24"/></svg>',
  live: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="8" cy="8" r="5.5"/><ellipse cx="8" cy="8" rx="2.5" ry="5.5"/><path d="M2.5 8h11"/></svg>',
};
