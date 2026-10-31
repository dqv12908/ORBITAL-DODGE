// ops.log terminal (reference plant.log): ALL · ALERTS · AGENT tabs, dodge@ops:~$ prompt.
import { S, on, emit, fmtSim, esc, h } from '../store.js';

const RANK = { INFO: 0, CDM: 0, AGENT: 1, WARN: 2, REFUSE: 3, ERROR: 3, CRIT: 4 };
const PREF = 'orbitaldodge.term.v1';

export class OpsLog {
  constructor(root) {
    let pref = {};
    try { pref = JSON.parse(localStorage.getItem(PREF) || '{}'); } catch { /* ignore */ }
    this.filter = pref.filter || 'ALL';
    this.size = pref.size || 'small';
    this.min = !!pref.min;
    this.unread = 0;
    const slot = root.appendChild(h('<div class="slot-log"></div>'));
    this.el = slot.appendChild(h(`<div class="term">
      <div class="term-head">
        <button class="term-title" data-a="min"><i class="term-led"></i>ops.log<span class="term-unread" hidden>0</span></button>
        <div class="term-tabs">${['ALL', 'ALERTS', 'AGENT'].map((t) => `<button class="term-tab" data-f="${t}">${t}</button>`).join('')}</div>
        <button class="term-btn" data-a="size" title="Size (L)"><i class="term-ico-size"></i></button>
        <button class="term-btn" data-a="min" title="Minimize"><i class="term-ico-min"></i></button>
      </div>
      <div class="term-body"><div class="term-rows"></div><div class="term-cursor"><span>dodge@ops:~$</span><i></i></div></div>
      <button class="term-jump" hidden>jump to latest</button>
    </div>`));
    this.body = this.el.querySelector('.term-body');
    this.rows = this.el.querySelector('.term-rows');
    this.jump = this.el.querySelector('.term-jump');
    this.el.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      const k = e.target.closest('.tl-m');
      if (k) { emit('inspect', k.dataset.k); return; }
      if (!b) return;
      if (b.dataset.f) { this.filter = b.dataset.f; this.#render(); }
      if (b.dataset.a === 'size') this.toggleSize();
      if (b.dataset.a === 'min') { this.min = !this.min; this.#apply(); }
      if (b === this.jump) this.#bottom();
      this.#save();
    });
    this.body.addEventListener('scroll', () => { if (this.#atBottom()) { this.jump.hidden = true; this.unread = 0; this.#unread(); } });
    on('log', (row) => this.#append(row, true));
    on('scenario', () => { this.rows.innerHTML = ''; });
    this.#apply();
    this.#render();
  }
  toggleSize() { this.size = this.size === 'large' ? 'small' : 'large'; this.min = false; this.#apply(); this.#save(); }
  #save() { try { localStorage.setItem(PREF, JSON.stringify({ filter: this.filter, size: this.size, min: this.min })); } catch { /* ignore */ } }
  #apply() {
    this.el.classList.toggle('is-large', this.size === 'large');
    this.el.classList.toggle('is-min', this.min);
    for (const t of this.el.querySelectorAll('.term-tab')) t.classList.toggle('is-on', t.dataset.f === this.filter);
  }
  #pass(r) {
    if (this.filter === 'ALERTS') return (RANK[r.level] ?? 0) >= 2;
    if (this.filter === 'AGENT') return r.level === 'AGENT' || ['DEC', 'EXPLAIN', 'ASK', 'HUMAN'].includes(r.code);
    return true;
  }
  #render() {
    this.#apply();
    this.rows.innerHTML = '';
    for (const r of S.log) this.#append(r, false);
    this.#bottom();
  }
  #rowEl(r) {
    const t = r.sim_time ? fmtSim(Date.parse(r.sim_time), false) : '--:--';
    let msg = esc(r.msg || '');
    if (r.event_key) msg = msg.replace(esc(r.event_key), `<span class="tl-m" data-k="${esc(r.event_key)}">${esc(r.event_key)}</span>`);
    const lv = r.level || 'INFO';
    const cls = lv === 'CRIT' ? 'lv-FAULT' : lv === 'REFUSE' ? 'lv-REFUSE' : `lv-${lv}`;
    return h(`<div class="term-row ${cls}" title="${r.ledger_seq ? 'ledger #' + r.ledger_seq : 'seq ' + (r.seq ?? '')}"><span class="tl-t">${t} </span><span class="tl-lv">${esc(lv)}</span> <span class="tl-code">${esc(r.code || '')}</span> <span class="tl-msg">${msg}</span></div>`);
  }
  #append(r, live) {
    if (!this.#pass(r)) return;
    const stick = this.#atBottom();
    const el = this.#rowEl(r);
    if (live) el.classList.add('is-new');
    this.rows.appendChild(el);
    while (this.rows.children.length > 400) this.rows.firstChild.remove();
    if (stick || !live) this.#bottom();
    else {
      this.unread++; this.jump.hidden = false;
      if ((RANK[r.level] ?? 0) >= 2) this.jump.classList.add('is-alert');
      this.#unread();
    }
    if (live && this.min) { this.unread++; this.#unread(); }
  }
  #unread() {
    const u = this.el.querySelector('.term-unread');
    u.hidden = !this.unread; u.textContent = this.unread;
    if (!this.unread) this.jump.classList.remove('is-alert');
  }
  #atBottom() { return this.body.scrollHeight - this.body.scrollTop - this.body.clientHeight < 24; }
  #bottom() { this.body.scrollTop = this.body.scrollHeight; this.jump.hidden = true; }
}
