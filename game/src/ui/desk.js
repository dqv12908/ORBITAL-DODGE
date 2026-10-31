// Analyst desk: the analyst owns the policy. Every change → PUT /api/policy (Idempotency-Key) → policy.updated + ledger row.
import { S, on, esc } from '../store.js';
import { PlotCard } from './gg/card.js';

/** Find an operating curve and the persistence point in an /api/eval payload (shape-tolerant). */
function findCurve(ev) {
  if (!ev || typeof ev !== 'object') return { curve: null, base: null };
  let curve = null, base = null;
  const isCurve = (a) => Array.isArray(a) && a.length && typeof a[0] === 'object' && ('recall' in a[0] || 'crit_recall' in a[0]) && ('tau' in a[0] || 'threshold' in a[0]);
  // backend shape: metrics.tier_forecaster.by_event.{operating_curve_calibrated, tier_persistence{rec_crit,false_alarms}}
  const be = ev.metrics?.tier_forecaster?.by_event;
  if (be?.operating_curve_calibrated) {
    curve = be.operating_curve_calibrated.map((p) => ({ ...p, recall: p.recall ?? p.crit_recall }));
    const tp = be.tier_persistence;
    if (tp) base = { recall: tp.rec_crit ?? tp.recall, false_alarms: tp.false_alarms };
    return { curve, base };
  }
  const walk = (o, depth) => {
    if (!o || typeof o !== 'object' || depth > 4) return;
    for (const [k, v] of Object.entries(o)) {
      if (!curve && /operating/i.test(k) && isCurve(v)) curve = v;
      else if (!curve && isCurve(v)) curve = v;
      if (!base && /persistence/i.test(k) && v && typeof v === 'object' && 'recall' in v) base = v;
      if (v && typeof v === 'object' && !Array.isArray(v)) walk(v, depth + 1);
    }
  };
  walk(ev, 0);
  return { curve, base };
}
const fa = (p) => p.false_alarms ?? p.fa ?? p.n_false_alarms ?? p.false_positives ?? '—';
const num = (x, d = 3) => (typeof x === 'number' ? x.toFixed(d) : x ?? '—');

export class Desk {
  constructor(root, { api } = {}) {
    this.root = root;
    this.api = api;
    this.el = null;
    this.dragging = new Set();
    this.timers = {};
    this.curve = null; this.base = null;
    this.onKey = (e) => { if (e.key === 'Escape') this.close(); };
    on('policy', () => this.#sync());
  }

  get isOpen() { return !!this.el; }

  open() {
    if (this.el) return;
    const p = S.policy || {};
    this.el = document.createElement('div');
    this.el.className = 'mdl-back';
    this.el.innerHTML = `
      <div class="mdl mdl-desk" role="dialog" aria-label="Analyst desk">
        <div class="mdl-head"><div><h2>Analyst desk</h2><p>Your policy. The system shows its consequences with measured numbers.</p></div>
          <button class="pop-x" type="button" aria-label="Close">✕</button></div>
        <div class="mdl-body">
          <p class="dk-rule"></p>
          <div class="dk-field">
            <div class="dk-top"><span>TIER MODE</span><b class="dk-mode-v"></b></div>
            <div class="dk-seg" role="radiogroup">
              <button type="button" data-mode="BRIEF">BRIEF</button><button type="button" data-mode="CARA_PC">CARA_PC</button>
            </div>
            <small>BRIEF: critical if Pc ≥ 1e-4 <i>or</i> miss &lt; 1 km. CARA_PC: tiers from Pc only (miss is context).<br>
              Brief rule: 551 critical final events; Pc rule: 6 (archive-wide, measured).</small>
          </div>
          <div class="dk-field">
            <div class="dk-top"><span>WATCH THRESHOLD τ</span><b class="dk-tau-v"></b></div>
            <input type="range" class="dk-tau" min="0.05" max="0.50" step="0.01">
            <small class="dk-tau-read">Loading operating curve…</small>
            <div class="dk-gg"></div>
          </div>
          <div class="dk-field">
            <div class="dk-top"><span>CONFIDENCE NEEDED</span><b class="dk-conf-v"></b></div>
            <input type="range" class="dk-conf" min="0.50" max="0.95" step="0.05">
            <small>Below this, the decision model's choice always comes to you. Decision confidence is not a collision probability.</small>
          </div>
          <div class="dk-status"></div>
          <p class="dk-foot">Every change is a ledger row. The rules re-run on the server; the sky recolours.</p>
        </div>
      </div>`;
    this.root.appendChild(this.el);
    const q = (s) => this.el.querySelector(s);
    this.tau = q('.dk-tau'); this.conf = q('.dk-conf');
    this.tau.value = p.tau_watch ?? 0.1;
    this.conf.value = p.confidence_bar ?? 0.75;
    q('.pop-x').addEventListener('click', () => this.close());
    this.el.addEventListener('pointerdown', (e) => { if (e.target === this.el) this.close(); });
    document.addEventListener('keydown', this.onKey);
    this.el.querySelectorAll('.dk-seg button').forEach((b) => b.addEventListener('click', () => {
      if ((S.policy?.tier_mode || 'BRIEF') === b.dataset.mode) return;
      this.#setSeg(b.dataset.mode);
      this.#put({ tier_mode: b.dataset.mode });
    }));
    for (const [inp, key] of [[this.tau, 'tau_watch'], [this.conf, 'confidence_bar']]) {
      inp.addEventListener('pointerdown', () => this.dragging.add(key));
      inp.addEventListener('pointerup', () => this.dragging.delete(key));
      inp.addEventListener('input', () => this.#labels());
      inp.addEventListener('change', () => {
        this.dragging.delete(key);
        clearTimeout(this.timers[key]);
        this.timers[key] = setTimeout(() => this.#put({ [key]: +inp.value }), 250);
      });
    }
    this.#sync();
    this.#loadEval();
  }

  close() {
    if (!this.el) return;
    document.removeEventListener('keydown', this.onKey);
    this.el.remove();
    this.el = null;
    this.gg = null;
  }

  #setSeg(mode) {
    this.el?.querySelectorAll('.dk-seg button').forEach((b) => b.classList.toggle('is-on', b.dataset.mode === mode));
    const v = this.el?.querySelector('.dk-mode-v'); if (v) v.textContent = mode;
  }

  #sync() {
    if (!this.el) return;
    const p = S.policy || {};
    this.#setSeg(p.tier_mode || 'BRIEF');
    if (!this.dragging.has('tau_watch') && p.tau_watch != null && document.activeElement !== this.tau) this.tau.value = p.tau_watch;
    if (!this.dragging.has('confidence_bar') && p.confidence_bar != null && document.activeElement !== this.conf) this.conf.value = p.confidence_bar;
    this.#labels();
  }

  #labels() {
    if (!this.el) return;
    const tau = +this.tau.value, conf = +this.conf.value;
    this.el.querySelector('.dk-tau-v').textContent = tau.toFixed(2);
    this.el.querySelector('.dk-conf-v').textContent = `${Math.round(conf * 100)}%`;
    this.el.querySelector('.dk-rule').innerHTML =
      `The system logs and keeps monitoring on its own only when an event is <b>SAFE</b> or <b>ELEVATED</b>, the decision model is at least <b>${Math.round(conf * 100)}%</b> sure and agrees with the cost rule. ` +
      `Events with a forecast chance of ending critical ≥ <b>${tau.toFixed(2)}</b> are put on watch. Critical events, default covariances and disagreements always come to you. ` +
      `Tier mode: <b>${esc(S.policy?.tier_mode || 'BRIEF')}</b>.`;
    const read = this.el.querySelector('.dk-tau-read');
    if (this.curve) {
      const near = this.curve.reduce((b, pt) => (Math.abs((pt.tau ?? pt.threshold) - tau) < Math.abs((b.tau ?? b.threshold) - tau) ? pt : b));
      const bs = this.base ? ` <span class="dk-base">(persistence ${num(this.base.recall)} / ${fa(this.base)})</span>` : '';
      read.innerHTML = `At τ ${num(near.tau ?? near.threshold, 2)}: recall <b>${num(near.recall)}</b> · false alarms <b>${fa(near)}</b>${bs} <span class="dk-measured">measured, OOF</span>`;
      this.gg ||= new PlotCard(this.el.querySelector('.dk-gg'), { name: 'operating-curve', base: 10.5 });
      this.gg.draw('operatingCurve', [{ curve: this.curve.map((p) => ({ tau: p.tau ?? p.threshold, recall: p.recall, false_alarms: p.false_alarms ?? p.fa ?? p.n_false_alarms ?? p.false_positives })), tau, base: this.base }]);
    } else if (this.evalTried) {
      read.textContent = 'Operating curve not available (backend /api/eval).';
    }
  }

  async #loadEval() {
    try {
      const ev = await this.api('GET', '/eval');
      const { curve, base } = findCurve(ev);
      this.curve = curve; this.base = base;
    } catch { /* not available */ }
    this.evalTried = true;
    this.#labels();
  }

  async #put(patch) {
    const st = this.el?.querySelector('.dk-status');
    if (st) { st.className = 'dk-status'; st.textContent = 'Saving…'; }
    try {
      const res = await this.api('PUT', '/policy', patch, { idem: true });
      if (res && typeof res === 'object') S.policy = { ...S.policy, ...(res.policy || res) };
      if (st && this.el) { st.className = 'dk-status is-ok'; st.textContent = `Saved · policy ${S.policy.version ?? ''}`.trim(); }
      this.#sync();
    } catch (e) {
      if (st && this.el) { st.className = 'dk-status is-err'; st.textContent = `Not saved: ${e.message}`; }
      this.#sync();
    }
  }
}
