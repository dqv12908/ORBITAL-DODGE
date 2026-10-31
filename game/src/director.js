// Director: maps server events to world calls and UI topics. The browser runs no risk logic:
// every Pc, tier, verdict and reason shown comes from an event payload.
import { S, ev, emit, tierOf, latestRisk, latestCdm, simNow } from './store.js';
import { api } from './net.js';
import { scenePos, normElements, latLonToScene } from './orbit.js';

function call(fn, ...a) { try { return fn?.(...a); } catch (e) { console.error('[director]', e); } }
const ms = (s) => (s ? Date.parse(s) : null);

export class Director {
  constructor(world, ui) {
    this.world = world; this.ui = ui;
    this.geomLoading = new Set();
    // keep the world's sim time interpolated each frame
    const loop = () => { call(() => world.setSimTime(simNow())); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    world.onPick?.((key) => this.pick(key));
  }

  pick(key) {
    if (String(key).startsWith('live:')) { emit('live.pick', +key.slice(5)); return; }
    this.setFocus(key, { fly: true });
    emit('inspect', key);
  }

  // ─── state snapshot (connect, scenario load, seq gap) ───────────────
  applyState(st) {
    if (!st) return;
    const clk = st.clock || st.sim || (st.sim_time ? st : null);
    if (clk) this.#clock(clk);
    if (st.mode) S.mode = { ...S.mode, ...st.mode };
    if (st.fixture) S.mode.fixture = true;
    if (st.policy) S.policy = { ...S.policy, ...st.policy };
    const scn = st.scenario;
    if (scn) S.scenario = typeof scn === 'string' ? { id: scn, title: scn } : scn;
    const cast = st.cast || st.events || [];
    const keys = [];
    for (const c of cast) {
      const k = typeof c === 'string' ? c : c.event_key || c.key;
      if (!k) continue;
      keys.push(k);
      const e = ev(k);
      if (typeof c === 'object') {
        e.tcaMs = ms(c.tca_sim) ?? e.tcaMs;
        e.chaser_type = c.chaser_type ?? c.c_object_type ?? e.chaser_type;
        e.mission = c.mission_id ?? e.mission;
        e.summaryTier = c.tier ?? c.latest?.tier?.tier ?? c.latest?.tier ?? e.summaryTier;
        e.summary = c;
        if (c.role === 'focal' && !S.focus) S.focus = k;
      }
    }
    if (keys.length) S.cast = keys;
    const apprs = st.approvals ?? st.approvals_open;
    if (Array.isArray(apprs)) {
      S.approvals.clear();
      for (const a of apprs) {
        if (a.status && a.status !== 'pending') continue;
        S.approvals.set(a.approval_id, a);
        if (a.event_key) ev(a.event_key).approval = a;
      }
    }
    if (S.cast?.length) {
      if (!S.focus || !S.cast.includes(S.focus)) S.focus = S.cast[0];
      for (const k of S.cast) this.#loadEvent(k);
    }
    emit('state');
    emit('approvals');
    this.#refreshMarkers();
  }

  #clock(c) {
    S.clock = {
      simMs: ms(c.sim_time) ?? S.clock.simMs, speed: c.speed ?? S.clock.speed, rate: c.sim_per_real_s ?? null, paused: !!c.paused, held: !!c.held,
      hold_reason: c.hold_reason ?? null, wallAt: performance.now(),
    };
    emit('clock');
  }

  async #loadEvent(key) {
    if (this.geomLoading.has(key)) return;
    this.geomLoading.add(key);
    try {
      const [g, hist] = await Promise.all([
        api('GET', `/conjunctions/${encodeURIComponent(key)}/geometry`).catch(() => null),
        api('GET', `/conjunctions/${encodeURIComponent(key)}`).catch(() => null),
      ]);
      const e = ev(key); // re-read: a scenario reload may have replaced the object while we waited
      if (g) {
        e.geometry = g;
        e.tcaMs = ms(g.tca_sim) ?? e.tcaMs;
        e.chaser_type = g.chaser_type ?? g.elements?.c_object_type ?? g.chaser?.object_type ?? e.chaser_type;
        g.encounter = g.encounter || g.encounter_plane || g.elements?.encounter_plane || [];
        g.residuals = g.residuals || (g.elements?.inc_res_deg != null ? { inc_res_deg: g.elements.inc_res_deg, sma_res_km: g.elements.sma_res_km } : null);
      }
      if (hist) this.#mergeHistory(e, hist);
      api('GET', `/futures/${encodeURIComponent(key)}`).then((f) => { ev(key).futures = f; emit('event', key); }).catch(() => {});
    } finally { this.geomLoading.delete(key); }
    if (key === S.focus) this.#applyFocal();
    this.#refreshMarkers();
    emit('event', key);
  }
  #mergeHistory(e, h) {
    const cdms = h.cdms || h.cdm_history || [];
    const risks = h.risks || h.assessments || h.pc_series || [];
    for (const c of cdms) {
      const i = c.cdm_index ?? e.cdms.length;
      e.cdms[i] = { ...e.cdms[i], ...c };
      if (c.pc || c.tier) e.risks[i] = { ...e.risks[i], cdm_id: c.cdm_id, pc: c.pc, tier: c.tier, reported_log10: c.reported_log10 };
      if (c.forecast) e.forecast = c.forecast;
      if (c.decision) e.decision = c.decision;
      if (c.verdict) e.verdict = c.verdict;
      if (c.explanation) e.explanation = c.explanation;
    }
    if (h.summary) { e.summary = { ...e.summary, ...h.summary }; e.summaryTier = h.summary.tier ?? e.summaryTier; e.mission = h.summary.mission_id ?? e.mission; }
    if (h.meta) { e.mission = h.meta.mission_id ?? e.mission; e.chaser_type = h.meta.c_object_type ?? h.meta.chaser_type ?? e.chaser_type; }
    for (const a of h.approvals || []) {
      if (a.status && a.status !== 'pending') continue;
      const ap = { ...a, event_key: e.key };
      S.approvals.set(a.approval_id, ap); e.approval = ap;
    }
    risks.forEach((r, k) => {
      const i = r.cdm_index ?? cdms[k]?.cdm_index ?? k;
      if (r.pc || r.tier) e.risks[i] = { ...e.risks[i], ...r };
    });
    if (h.forecast) e.forecast = h.forecast;
    if (h.decision || (h.decisions && h.decisions.length)) e.decision = h.decision || h.decisions.at(-1);
    if (h.explanation || (h.explanations && h.explanations.length)) e.explanation = h.explanation || h.explanations.at(-1);
    if (h.outcome) e.outcome = h.outcome;
    e.tcaMs = ms(h.tca_sim) ?? e.tcaMs;
  }

  setFocus(key, { fly = false } = {}) {
    if (!key) return;
    const changed = S.focus !== key;
    S.focus = key;
    if (changed) this.#applyFocal();
    this.#refreshMarkers();
    if (fly) call(() => this.world.focusFocal());
    emit('focus', key);
  }

  #elements(g, which) {
    if (!g) return null;
    const src = g.elements && typeof g.elements === 'object' ? g.elements : g;
    const o = src[which] || g[which] || g.objects?.[which];
    if (o) return o.elements || o;
    const p = which === 'target' ? 't_' : 'c_';
    if (src[p + 'a_km'] == null) return null;
    return { a_km: src[p + 'a_km'], e: src[p + 'e'] ?? 0, i_deg: src[p + 'i_deg'], raan_deg: src[p + 'raan_deg'] ?? 0, argp_deg: src[p + 'argp_deg'] ?? 0, nu_deg_at_tca: src[p + 'nu_deg_at_tca'] ?? 0 };
  }
  #applyFocal() {
    const e = ev(S.focus);
    const g = e?.geometry;
    if (!g) { call(() => this.world.clearFocal()); return; }
    const target = this.#elements(g, 'target'), chaser = this.#elements(g, 'chaser');
    if (!normElements(target) || !normElements(chaser)) { call(() => this.world.clearFocal()); return; }
    call(() => this.world.setFocal({ event_key: e.key, tca_ms: e.tcaMs ?? S.clock.simMs, target, chaser, chaser_type: e.chaser_type, tier: tierOf(e) || 'SAFE' }));
    emit('focal');
  }
  /** Marker position: the reconstructed conjunction point at TCA (inertial). */
  #markerPos(e) {
    const g = e.geometry;
    if (!g) return null;
    const t = normElements(this.#elements(g, 'target'));
    if (t) return scenePos(t, 0);
    const cj = g.conjunction || g;
    if (cj.lat_deg != null && cj.lon_deg != null) return latLonToScene(cj.lat_deg, cj.lon_deg, cj.alt_km ?? 700, e.tcaMs ?? S.clock.simMs);
    return null;
  }
  #refreshMarkers() {
    const list = [];
    for (const k of S.cast || []) {
      const e = S.events.get(k);
      const pos = e && this.#markerPos(e);
      if (pos) list.push({ key: k, pos, tier: tierOf(e) || 'SAFE', focal: k === S.focus && !!this.world.focal });
    }
    call(() => this.world.setMarkers(list));
  }

  // ─── event stream ────────────────────────────────────────────────────
  handle(evt) {
    const d = evt.data || {};
    const key = evt.event_key || d.event_key || null;
    if (evt.sim_time && evt.type !== 'state.snapshot') { /* sim_time on every envelope keeps the clock honest */ }
    switch (evt.type) {
      case 'state.snapshot': this.applyState(d); break;
      case 'clock.tick': this.#clock(d); break;
      case 'scenario.loaded': {
        S.events.clear(); S.approvals.clear(); S.focus = null;
        S.scenario = { ...(S.scenarios.find((s) => s.id === d.id) || {}), ...d };
        S.cast = (d.cast || []).map((c) => (typeof c === 'string' ? c : c.event_key));
        for (const c of d.cast || []) if (typeof c === 'object' && c.role === 'focal') S.focus = c.event_key;
        S.focus = S.focus || S.cast[0] || null;
        call(() => this.world.clearFocal());
        for (const k of S.cast) this.#loadEvent(k);
        emit('scenario', S.scenario);
        emit('approvals');
        break;
      }
      case 'cdm.received': {
        const e = ev(key);
        const i = d.cdm_index ?? e.cdms.length;
        e.cdms[i] = { ...d, at_sim: evt.sim_time };
        e.steps = { CDM: 'done', Pc: 'on' };
        e.decision = null; e.verdict = null; e.gated = null; e.explanation = null;
        if (e.tcaMs == null && d.ttca_d != null && evt.sim_time) e.tcaMs = ms(evt.sim_time) + d.ttca_d * 86400e3;
        emit('event', key); emit('cdm', key);
        break;
      }
      case 'risk.assessed': {
        const e = ev(key);
        const prevTier = tierOf(e);
        const i = d.cdm_index ?? (d.cdm_id ? +String(d.cdm_id).split(':').pop() : e.cdms.length - 1);
        e.risks[Number.isFinite(i) ? i : e.risks.length] = { ...d, at_sim: evt.sim_time };
        const cd = e.cdms[i] || latestCdm(e);
        e.steps = { ...e.steps, Pc: 'done', Tier: 'done', Forecast: cd && cd.ttca_d != null && cd.ttca_d < 2 ? 'skip' : (e.steps.Forecast === 'done' ? 'done' : 'on') };
        const t = tierOf(e);
        call(() => this.world.setMarkerTier(key, t));
        emit('event', key); emit('risk', key, prevTier, t);
        if (t === 'CRITICAL' && prevTier !== 'CRITICAL') {
          if (key !== S.focus) this.setFocus(key);
          call(() => this.world.burstFocal('#ff3448'));
          emit('alarm', key);
        }
        break;
      }
      case 'forecast.updated': { const e = ev(key); e.forecast = d; e.steps = { ...e.steps, Forecast: 'done' }; emit('event', key); break; }
      case 'policy.gated': { const e = ev(key); e.gated = d; e.steps = { ...e.steps, Forecast: e.steps.Forecast === 'on' ? 'skip' : e.steps.Forecast, Decision: 'on' }; emit('event', key); break; }
      case 'decision.requested': { const e = ev(key); e.decisionPending = true; e.steps = { ...e.steps, Decision: 'on' }; emit('event', key); emit('agent', key); break; }
      case 'decision.made': { const e = ev(key); e.decision = d; e.decisionPending = false; e.steps = { ...e.steps, Decision: 'done' }; emit('event', key); emit('agent', key); break; }
      case 'policy.verdict': {
        const e = ev(key); e.verdict = d;
        e.steps = { ...e.steps, Decision: 'done', You: d.verdict === 'AUTO' ? 'skip' : 'on' };
        emit('event', key); emit('agent', key);
        break;
      }
      case 'approval.required': {
        const e = ev(key);
        const a = { ...d, event_key: key };
        e.approval = a; S.approvals.set(d.approval_id, a);
        e.steps = { ...e.steps, You: 'on' };
        if (key !== S.focus) this.setFocus(key);
        emit('event', key); emit('approvals'); emit('approval.required', a);
        break;
      }
      case 'human.action': {
        const e = ev(key);
        e.human = d; e.steps = { ...e.steps, You: 'done' };
        S.approvals.delete(d.approval_id);
        if (e.approval?.approval_id === d.approval_id) e.approval = null;
        const t = tierOf(e);
        call(() => this.world.setMarkerTier(key, t));
        call(() => this.world.burstFocal(/PREPARE/.test(d.action) ? '#3fa4ff' : '#7dff8a'));
        emit('event', key); emit('approvals'); emit('human', key, d);
        break;
      }
      case 'explanation.ready': { const e = ev(key); e.explanation = d; emit('event', key); emit('agent', key); break; }
      case 'log.entry': {
        const row = { ...d, sim_time: evt.sim_time, event_key: key, seq: evt.seq, ledger_seq: evt.ledger_seq };
        S.log.push(row); if (S.log.length > 400) S.log.shift();
        emit('log', row);
        break;
      }
      case 'outcome.revealed': {
        const e = ev(key); e.outcome = d;
        call(() => this.world.burstFocal(d.verdict === 'FN' ? '#ff3448' : '#7dff8a'));
        emit('event', key); emit('outcome', key, d);
        break;
      }
      case 'policy.updated': {
        if (d.policy) S.policy = { ...S.policy, ...d.policy };
        else if (d.diff) for (const [k, v] of Object.entries(d.diff)) S.policy[k] = Array.isArray(v) ? v[1] : v;
        S.policy.version = d.version ?? S.policy.version;
        emit('policy', d);
        // W3: rules re-run server-side; pull the re-assessed tiers for the whole cast
        if (d.diff && ('tier_mode' in d.diff || 'thresholds' in d.diff)) this.#reloadTiers();
        break;
      }
      case 'ledger.appended': S.ledger = { seq: d.seq, n: d.seq, last: d }; emit('ledger', d); break;
      case 'system.degraded': emit('degraded', d); break;
      default: break;
    }
    // keep markers' tiers in sync after any tier-affecting event (e.g. tier-mode flip re-runs rules server-side)
    if (evt.type === 'risk.assessed' || evt.type === 'policy.updated') {
      for (const k of S.cast || []) call(() => this.world.setMarkerTier(k, tierOf(S.events.get(k))));
      emit('tiers');
    }
  }

  async #reloadTiers() {
    await Promise.all((S.cast || []).map(async (k) => {
      try { this.#mergeHistory(ev(k), await api('GET', `/conjunctions/${encodeURIComponent(k)}`)); } catch { /* keep */ }
    }));
    for (const k of S.cast || []) call(() => this.world.setMarkerTier(k, tierOf(S.events.get(k))));
    for (const k of S.cast || []) emit('event', k);
    emit('tiers');
  }

  async resync() {
    try { this.applyState(await api('GET', '/state')); } catch (e) { console.warn('[director] resync failed', e); }
  }
}

export { latestRisk };
