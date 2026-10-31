// Full-screen scenario picker (reference card grid; each card = one idea).
import { S, esc } from '../store.js';

const TAG_COLOR = {
  LATE: 'var(--warn)', ESCALATE: 'var(--warn)', REFUSAL: 'var(--warn)',
  'FAILURE CASE': 'var(--bad)', RULES: 'var(--bad)',
  TRIAGE: 'var(--accent)', BLIND: 'var(--accent)',
  WAIT: 'var(--good)', FREE: 'var(--good)',
};
const STEPS = ['Read the CDM', 'Physics Pc', 'Tier rules', 'Forecast', 'Decision', 'You approve'];

/** 23x18 pixel-art satellite (body + two panels + dish), drawn at 5x. */
function drawSat(c) {
  const g = c.getContext('2d');
  const px = 5;
  c.width = 23 * px; c.height = 18 * px;
  const rows = [
    '..........r............',
    '..........r............',
    '.........www...........',
    '........wwwww..........',
    '...........y...........',
    'bbbbbb..yyyyyyy..bbbbbb',
    'bBbBbb..yyyyyyy..bBbBbb',
    'bbbbbbssyyyKyyyssbbbbbb',
    'bBbBbb..yyyyyyy..bBbBbb',
    'bbbbbb..yyyyyyy..bbbbbb',
    'bBbBbb..yyygyyy..bBbBbb',
    'bbbbbb..ooooooo..bbbbbb',
    '.........o...o.........',
    '.......................',
    '...............c.......',
    '..............ccc......',
    '...............c.......',
    '.......................',
  ];
  const col = { r: '#d06b75', w: '#eceff4', y: '#f5c445', o: '#d08770', b: '#4f7fc4', B: '#7cc7dc', s: '#a9b2c0', K: '#2e3440', g: '#5dff6e', c: '#ffb52e' };
  rows.forEach((row, y) => [...row].forEach((ch, x) => {
    if (ch === '.') return;
    g.fillStyle = col[ch]; g.fillRect(x * px, y * px, px, px);
  }));
}

function modeBadges() {
  const m = S.mode || {};
  const b = [];
  const dec = m.decisions || '?';
  b.push(`<span class="mode-badge ${dec === 'live' ? 'is-good' : dec === 'replay' ? 'is-accent' : 'is-warn'}">Decisions ${esc(dec)}</span>`);
  const llm = m.llm || '?';
  b.push(`<span class="mode-badge ${llm === 'live' ? 'is-good' : 'is-accent'}">LLM ${esc(llm)}</span>`);
  b.push('<span class="mode-badge is-accent">Orbits reconstructed</span>');
  if (m.fixture) b.push('<span class="mode-badge is-warn">FIXTURE (backend offline)</span>');
  return b.join('');
}

export class ScenarioPicker {
  constructor(root, { api, onStarted } = {}) {
    this.api = api;
    this.onStarted = onStarted;
    this.closable = true;
    this.busy = false;
    this.el = document.createElement('div');
    this.el.className = 'scn-overlay scn-hidden';
    this.el.setAttribute('role', 'dialog');
    this.el.setAttribute('aria-label', 'Choose a scenario');
    root.appendChild(this.el);
    this.el.innerHTML = `
      <div class="scn-inner">
        <div class="scn-head">
          <canvas class="scn-bot" aria-hidden="true"></canvas>
          <div>
            <h1 class="scn-title"><span class="scn-t1">ORBITAL DODGE</span><span class="scn-t2">CONJUNCTION OPS</span></h1>
            <p class="scn-sub">You are the <b>duty conjunction analyst</b> for one Earth-observation satellite. A CDM drop lands every
              <b>~8 h</b>: triage it, see why, decide what happens next.</p>
          </div>
        </div>
        <ol class="scn-steps">${STEPS.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>
        <div class="scn-label"><span>Choose a scenario</span><em>Each one starts a fresh SIM day</em></div>
        <div class="scn-grid"><div class="scn-loadmsg">Loading scenarios…</div></div>
        <div class="scn-error" hidden></div>
        <div class="scn-foot">
          <div class="hud-badges"></div>
          <span class="scn-keys">Arrows move · Enter starts</span>
          <button class="scn-close" type="button" hidden>Close</button>
        </div>
      </div>`;
    drawSat(this.el.querySelector('.scn-bot'));
    this.grid = this.el.querySelector('.scn-grid');
    this.err = this.el.querySelector('.scn-error');
    this.closeBtn = this.el.querySelector('.scn-close');
    this.closeBtn.addEventListener('click', () => this.hide());
    this.grid.addEventListener('click', (e) => {
      const card = e.target.closest('.scn-card');
      if (card) this.#start(card.dataset.id, card);
    });
    this.el.addEventListener('click', (e) => { if (e.target === this.el && this.closable) this.hide(); });
    this.onKey = (e) => this.#key(e);
  }

  get visible() { return !this.el.classList.contains('scn-hidden'); }

  async show({ closable = true } = {}) {
    this.closable = closable;
    this.closeBtn.hidden = !closable;
    this.el.querySelector('.hud-badges').innerHTML = modeBadges();
    this.err.hidden = true;
    this.el.classList.remove('scn-hidden');
    document.addEventListener('keydown', this.onKey, true);
    await this.#load();
    this.grid.querySelector('.scn-card:not(.is-off)')?.focus();
  }

  hide() {
    this.el.classList.add('scn-hidden');
    document.removeEventListener('keydown', this.onKey, true);
  }

  async #load() {
    let list = S.scenarios?.length ? S.scenarios : null;
    try {
      const r = await this.api('GET', '/scenarios');
      list = Array.isArray(r) ? r : r?.scenarios || list || [];
    } catch (e) {
      if (!list) { this.grid.innerHTML = ''; this.#error(`Could not load scenarios: ${e.message}`); return; }
    }
    list = [...list].sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
    S.scenarios = list;
    this.grid.innerHTML = list.map((s) => this.#card(s)).join('');
  }

  #card(s) {
    const tag = String(s.tag || '').toUpperCase();
    const c = TAG_COLOR[tag] || 'var(--accent)';
    const keys = s.event_keys || s.cast?.map((x) => (typeof x === 'string' ? x : x.event_key)) || [];
    const off = s.playable === false;
    const facts = off ? 'needs backend' : keys.length ? keys.slice(0, 4).join(' · ') + (keys.length > 4 ? ` +${keys.length - 4}` : '') : '';
    return `<button type="button" class="scn-card${off ? ' is-off' : ''}" data-id="${esc(s.id)}" style="--scn-c:${c}">
      <div class="scn-card-top"><span class="scn-name">${esc(s.title || s.id)}</span>${tag ? `<span class="scn-tag">${esc(tag)}</span>` : ''}</div>
      ${s.tagline ? `<div class="scn-tagline">${esc(s.tagline)}</div>` : ''}
      ${s.teaches ? `<div class="scn-teaches"><b>Teaches</b>${esc(s.teaches)}</div>` : ''}
      <div class="scn-card-foot"><span class="scn-facts">${esc(facts)}</span><span class="scn-go">Start <i></i></span></div>
    </button>`;
  }

  #error(msg) { this.err.textContent = msg; this.err.hidden = false; }

  async #start(id, card) {
    if (this.busy || !id) return;
    this.busy = true;
    this.err.hidden = true;
    card.classList.add('is-loading');
    this.grid.classList.add('is-busy');
    try {
      const res = await this.api('POST', '/sim/scenario', { id });
      this.hide();
      this.onStarted?.(id, res);
    } catch (e) {
      this.#error(e.message || 'Could not start the scenario');
    } finally {
      this.busy = false;
      card.classList.remove('is-loading');
      this.grid.classList.remove('is-busy');
    }
  }

  #key(e) {
    if (!this.visible) return;
    if (e.key === 'Escape') { if (this.closable) { e.preventDefault(); this.hide(); } return; }
    const cards = [...this.grid.querySelectorAll('.scn-card')];
    if (!cards.length) return;
    const i = cards.indexOf(document.activeElement);
    if (e.key === 'Enter' && i >= 0) { e.preventDefault(); this.#start(cards[i].dataset.id, cards[i]); return; }
    if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(e.key)) return;
    e.preventDefault();
    const cols = Math.max(1, Math.round(this.grid.clientWidth / (cards[0].offsetWidth + 14)));
    const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols }[e.key];
    const j = i < 0 ? 0 : Math.min(cards.length - 1, Math.max(0, i + step));
    cards[j].focus();
  }
}
