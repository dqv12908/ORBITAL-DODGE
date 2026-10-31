// Dashboard "Statistics" tab: stat / probability plots from GET /api/eval/curves (out-of-fold, train events)
// plus the live miss-distance check from GET /api/dashboard. Idempotent: called on every dashboard render.
import { PlotCard } from './card.js';

let curves = null, curvesErr = null, loading = null;

function liveRows(dash) {
  const rows = Array.isArray(dash?.next) ? dash.next : [];
  return rows.map((r) => ({ key: r.event_key, miss_m: r.latest?.miss_m ?? r.latest_miss_m ?? null, tier: r.tier ?? r.latest?.tier ?? null }))
    .filter((r) => r.key);
}

/** pane: dashboard pane element; ctx: {api, dash} */
export function mountStats(pane, { api, dash }) {
  let root = pane.querySelector(':scope > .gg-stats');
  if (!root) {
    pane.innerHTML = '';
    root = document.createElement('div');
    root.className = 'gg-stats';
    root.innerHTML = `<p class="gg-stats-note">Statistics of the served forecaster, <b>measured out-of-fold</b> on training events (labels: final tier of each event). P(critical) is a forecast, <b>not a collision probability</b>. The last plot puts the live next-24 h events on the archive distribution.</p>
      <div class="gg-grid"></div>`;
    pane.appendChild(root);
    const grid = root.querySelector('.gg-grid');
    root._cards = Object.fromEntries(['roc', 'pr', 'reliability', 'cost', 'density', 'miss'].map((n) => {
      const host = document.createElement('div'); grid.appendChild(host);
      return [n, new PlotCard(host, { name: `stats-${n}`, base: 13 })];
    }));
  }
  const C = root._cards;
  if (!curves && !curvesErr) {
    if (!loading) {
      loading = api('GET', '/eval/curves').then((c) => { curves = c; }).catch((e) => { curvesErr = e.message || String(e); })
        .finally(() => { loading = null; if (root.isConnected) mountStats(pane, { api, dash }); });
    }
    return;
  }
  if (curvesErr) {
    root.querySelector('.gg-grid').innerHTML = `<p class="gg-stats-note">GET /api/eval/curves failed: ${curvesErr.replace(/</g, '&lt;')}</p>`;
    return;
  }
  C.roc.draw('rocPlot', [curves]);
  C.pr.draw('prPlot', [curves]);
  C.reliability.draw('reliabilityPlot', [curves]);
  C.cost.draw('costPlot', [curves]);
  C.density.draw('densityPlot', [curves]);
  C.miss.draw('missHistPlot', [curves, liveRows(dash)]);
}
