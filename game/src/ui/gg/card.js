// Plot host: draws a hand-made SVG chart (plots.js) into the console panel. No chrome beyond a small export button.
// Redraws only when the data or the width changes (JSON key).
import './gg.css';
import * as plots from './plots.js';
import { GG } from './theme.js';

const ICO = '<svg viewBox="0 0 16 16" width="11" height="11" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 2.5v7.5M4.8 7 8 10.2 11.2 7M3 13.2h10"/></svg>';

function download(url, name) {
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

/** Export an inline chart SVG with the console's solid background (SVG as-is, PNG at 2x). */
export function exportSvg(svgEl, name, fmt) {
  const w = +svgEl.getAttribute('width'), h = +svgEl.getAttribute('height');
  const src = svgEl.outerHTML.replace('<defs>', `<rect width="100%" height="100%" fill="${GG.bgExport}"/><defs>`);
  const url = URL.createObjectURL(new Blob([src], { type: 'image/svg+xml' }));
  if (fmt === 'svg') { download(url, `${name}.svg`); setTimeout(() => URL.revokeObjectURL(url), 2000); return; }
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = w * 2; c.height = h * 2;
    const g = c.getContext('2d'); g.scale(2, 2); g.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    download(c.toDataURL('image/png'), `${name}.png`);
  };
  img.src = url;
}

export class PlotCard {
  constructor(parent, { name = 'plot', base } = {}) {
    this.name = name; this.base = base; this.key = null;
    this.el = document.createElement('div');
    this.el.className = 'gg-card';
    this.el.innerHTML = `<div class="gg-tools"><button type="button" class="gg-exp" title="Export chart">${ICO}</button>
      <span class="gg-menu" hidden><button type="button" data-fmt="svg">SVG</button><button type="button" data-fmt="png">PNG</button></span></div><div class="gg-plot"></div>`;
    parent.appendChild(this.el);
    this.plotEl = this.el.querySelector('.gg-plot');
    const menu = this.el.querySelector('.gg-menu');
    this.el.querySelector('.gg-exp').addEventListener('click', (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; });
    menu.addEventListener('click', (e) => {
      const b = e.target.closest('[data-fmt]'); if (!b) return;
      menu.hidden = true;
      const svg = this.plotEl.querySelector('svg'); if (svg) exportSvg(svg, this.name, b.dataset.fmt);
    });
    document.addEventListener('pointerdown', (e) => { if (!this.el.contains(e.target)) menu.hidden = true; });
  }

  get width() { return Math.floor(this.el.clientWidth || this.el.parentElement?.clientWidth || 380); }

  /** draw('riskEvolution', [data]) */
  draw(fn, args, opts = {}) {
    const width = this.width;
    const key = JSON.stringify([fn, args, opts, width]);
    if (key === this.key) return;
    this.key = key;
    try {
      this.plotEl.innerHTML = plots[fn](...args, { width, base: this.base, ...opts });
    } catch (e) {
      console.error('[gg]', fn, e);
      this.plotEl.innerHTML = `<div class="gg-err">Chart error: ${String(e.message || e).replace(/</g, '&lt;')}</div>`;
    }
  }
}

export const preloadPlots = () => Promise.resolve(plots);
