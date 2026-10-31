// Chart tokens for the ORBITAL DODGE console. No charting library: plots are hand-written SVG (core.js).
// Structure borrowed from ggplot2 (panel, major + minor grid, facet strips, titled legend, caption);
// colours, fonts and line weights are the console's own.

export const GG = {
  bgExport: '#0a1019',                 // --surface-solid (PNG/SVG export background)
  panel: 'rgba(255,255,255,0.035)',    // plot panel, a touch lighter than the glass
  grid: 'rgba(255,255,255,0.09)',      // major grid
  gridMinor: 'rgba(255,255,255,0.04)', // minor grid
  strip: 'rgba(255,255,255,0.065)',    // facet strip
  stripText: '#aab6c6',
  axisText: '#7d8ca2',                 // tick labels
  tick: 'rgba(140,170,210,0.35)',
  title: '#e8eef6',                    // --text
  text: '#c3cedb',                     // --text-2
  muted: '#8796ab',                    // --muted
  dim: '#5f6f86',                      // --dim
  key: 'rgba(255,255,255,0.035)',
  ref: '#6f8099',                      // neutral reference lines
  ribbon: '#8ca8d2',                   // fan ribbons
  median: '#9fb3cc',
  bar: '#3b4c63',                      // histogram bars
  barLine: '#566a85',
  accent: '#35c9f0', warn: '#ffb020', bad: '#ff4d5e', good: '#3ddc97', agent: '#a594ff', info: '#5aa9ff',
  font: "'IBM Plex Sans', Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace",
};

/** Multi-series palette tuned to the dark UI (no saturated pure hues). */
const HUE = ['#35c9f0', '#e9a23b', '#a594ff', '#4fd1a0', '#f07a8a'];
export const hue = (n) => HUE.slice(0, Math.max(1, Math.min(5, n)));
export const TIER_COL = { SAFE: GG.good, ELEVATED: GG.warn, CRITICAL: GG.bad };

/** Type and line sizes from one base size (px). */
export function sizes(b = 11) {
  const k = b / 11;
  return {
    b, k,
    title: +(b * 1.14).toFixed(2),
    subtitle: +(b * 0.9).toFixed(2),
    axisText: +(b * 0.8).toFixed(2),
    axisTitle: +(b * 0.84).toFixed(2),
    strip: +(b * 0.72).toFixed(2),
    stripH: Math.round(b * 1.5),
    legendTitle: +(b * 0.8).toFixed(2),
    legendText: +(b * 0.8).toFixed(2),
    key: Math.round(12 * k),
    caption: +(b * 0.76).toFixed(2),
    annot: +(b * 0.74).toFixed(2),
    gridMajor: +(0.8 * k).toFixed(2),
    gridMinor: +(0.6 * k).toFixed(2),
    line: +(1.4 * k).toFixed(2),
    ref: +(0.9 * k).toFixed(2),
    r: +(2.4 * k).toFixed(2),          // point radius
    tickLen: +(3 * k).toFixed(2),
    spacing: Math.round(8 * k),
  };
}

// ─── continuous scales: 5 % expansion, nice breaks, minor breaks at midpoints ──
function niceStep(range, n) {
  const raw = range / Math.max(1, n);
  const p = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw * 0.999) return m * p;
  return 10 * p;
}
export const decimals = (step) => {
  for (let d = 0; d <= 6; d++) if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d;
  return 6;
};

/** {domain, majors, minors, step} */
export function cscale(lo, hi, { n = 5, expand = 0.05, step, limits } = {}) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  if (lo > hi) [lo, hi] = [hi, lo];
  const r = hi - lo;
  const domain = limits || [lo - expand * r, hi + expand * r];
  const st = step || niceStep(r, n);
  const span = domain[1] - domain[0];
  const majors = [];
  for (let v = Math.ceil((domain[0] - 1e-9 * span) / st) * st; v <= domain[1] + 1e-9 * span; v += st) majors.push(+v.toFixed(10));
  const minors = [];
  for (let v = Math.floor(domain[0] / st) * st + st / 2; v <= domain[1]; v += st) if (v >= domain[0]) minors.push(+v.toFixed(10));
  return { domain, majors, minors, step: st };
}
