// Wiring: 3D world + DOM overlay + Director, fed by the server's event stream (or the fixture).
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/jetbrains-mono/600.css';
import '@fontsource/jetbrains-mono/700.css';
import './ui/ui.css';
import './ui/ui-shell.css';
import './ui/ui-agent.css';
import './ui/dashboard.css';
import './ui/modals.css';
import './ui/app.css';
import './ui/theme.css';
import { S, on, emit, simNow } from './store.js';
import { api, net, probeBackend, connectEvents } from './net.js';
import { Director } from './director.js';
import { Hud } from './ui/hud.js';
import { OpsLog } from './ui/opslog.js';
import { Inspector, AgentPanel, ApprovalCard, LiveSkyPanel } from './ui/panels.js';
import { ScenarioPicker } from './ui/scenarios.js';
import { Desk } from './ui/desk.js';
import { Dashboard } from './ui/dashboard.js';

const worldEl = document.getElementById('world');
const uiEl = document.getElementById('ui');
const boot = document.getElementById('boot');

async function createWorld() {
  try {
    const { World } = await import('./world/World.js');
    return new World(worldEl);
  } catch (err) {
    console.error('[main] 3D world failed to start', err);
    worldEl.innerHTML = '<div class="world-fallback">3D view unavailable.<br>Every panel still works.</div>';
    return new Proxy({}, { get: (_, key) => (key === 'then' ? undefined : () => null) });
  }
}

async function start() {
  const params = new URLSearchParams(location.search);
  let health = params.has('mock') ? null : await probeBackend();
  if (!health) {
    const { MockServer } = await import('./mock.js');
    net.mock = new MockServer();
    health = await api('GET', '/health');
    console.info('[main] FIXTURE mode (backend offline or ?mock=1)');
  }
  S.mode = { ...S.mode, ...(health.mode || {}), fixture: !!net.mock };

  const world = await createWorld();
  const director = new Director(world, null);
  let hud;
  const agent = new AgentPanel(uiEl);
  const live = new LiveSkyPanel(uiEl, { api, world, toast: (...a) => hud?.toast(...a) });
  const actions = {
    scenarios: () => picker.show({ closable: !!S.scenario }),
    desk: () => desk.open(),
    dashboard: () => dash.open(),
    agent: (k) => agent.toggle(k),
    liveSky: () => live.toggle(),
  };
  hud = new Hud(uiEl, { api, world, actions });
  new OpsLog(uiEl);
  const inspector = new Inspector(uiEl, { world });
  const approval = new ApprovalCard(uiEl, { api, toast: (...a) => hud.toast(...a) });
  const desk = new Desk(uiEl, { api });
  const dash = new Dashboard(uiEl, { api });
  const picker = new ScenarioPicker(uiEl, {
    api,
    onStarted: async (id, st) => {
      if (st && typeof st === 'object' && (st.clock || st.scenario)) director.applyState(st);
      hud.toast(`Scenario started: ${S.scenario?.title || id}`, 'good', 'SIM');
      if (S.clock.paused) api('POST', '/sim/resume').catch(() => {});
      setTimeout(() => world.flyHome?.(1.4), 50);
    },
  });

  // W1/W2: when the focal event turns CRITICAL, fly from the whole sky down to the conjunction
  on('alarm', (key) => { if (key === S.focus) setTimeout(() => world.focusFocal?.({ dist: 0.9, dur: 1.6 }), 250); });
  on('focal', () => { if (S.focus && !world.userMoved) setTimeout(() => world.focusFocal?.({ dist: 1.6, dur: 1.8 }), 1200); });
  on('inspect', (k) => { if (k && k !== S.focus) director.setFocus(k, { fly: true }); });

  // Approach: as the focal event enters TCA − 20 min, ease the warp to 1× and dolly in (time dilation)
  const dilated = new Set();
  setInterval(() => {
    const e = S.focus && S.events.get(S.focus);
    if (!e || e.tcaMs == null || S.clock.paused || dilated.has(e.key)) return;
    const left = e.tcaMs - simNow();
    if (left > 0 && left < 20 * 60e3) {
      dilated.add(e.key);
      if (S.clock.speed > 1) api('POST', '/sim/speed', { speed: 1 }).catch(() => {});
      hud.float('TIME DILATION · 1× · TCA APPROACH', 'info', 'tca');
      world.focusFocal?.({ dist: 0.35, dur: 2.0 });
    }
  }, 250);

  connectEvents((evt) => director.handle(evt), {
    onGap: () => director.resync(),
    onConn: (ok) => { S.connected = ok; hud.setConnected(ok); },
  });
  try { director.applyState(await api('GET', '/state')); } catch (e) { console.warn('[main] /state failed', e); }

  // keyboard (reference): Esc close, Space pause, 1-4 speed, N next CDM, F fly, H home, L log, D dashboard, P desk, S scenarios, A agent, V live sky
  window.addEventListener('keydown', (e) => {
    if (e.target.closest?.('input, textarea, select')) return;
    if (picker.visible) return;
    const k = e.key;
    if (k === 'Escape') { if (approval.isOpen) approval.close(); else if (dash.isOpen) dash.close(); else if (desk.isOpen) desk.close(); else if (agent.isOpen) agent.close(); else if (inspector.isOpen) inspector.close(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (k === ' ') { e.preventDefault(); hud.togglePause(); }
    else if ('1234'.includes(k) && k.length === 1) hud.setSpeed([1, 2, 4, 8][+k - 1]);
    else if (k === 'n' || k === 'N') api('POST', '/sim/next-cdm').catch((err) => hud.toast(err.message, 'bad'));
    else if (k === 'f' || k === 'F') world.focusFocal?.({ dist: 0.9 });
    else if (k === 'h' || k === 'H') world.flyHome?.();
    else if (k === 'l' || k === 'L') document.querySelector('.term [data-a=size]')?.click();
    else if (k === 'd' || k === 'D') dash.isOpen ? dash.close() : dash.open();
    else if (k === 'p' || k === 'P') desk.isOpen ? desk.close() : desk.open();
    else if (k === 's' || k === 'S') picker.show({ closable: true });
    else if (k === 'a' || k === 'A') agent.toggle();
    else if (k === 'v' || k === 'V') live.toggle();
  });

  boot.classList.add('out');
  if (!params.get('scenario') && !params.has('nopick')) picker.show({ closable: !!S.scenario });
  if (params.get('scenario')) api('POST', '/sim/scenario', { id: params.get('scenario') }).then((st) => { picker.hide(); director.applyState(st); if (S.clock.paused) api('POST', '/sim/resume').catch(() => {}); }).catch(() => {});
  Object.assign(window, { world, director, S, api, emit });
}

start().catch((e) => {
  console.error(e);
  boot.innerHTML = `<div>ORBITAL DODGE<small>Failed to start: ${String(e.message || e)}</small></div>`;
});
