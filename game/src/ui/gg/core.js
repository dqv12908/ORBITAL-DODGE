// Tiny hand-written SVG plotting engine (no library). Panels with major + minor grid, ticks, axis titles,
// facet strips, a titled legend, title/subtitle/caption; marks: line, area (ribbon), points, rules, bands,
// segments, arrows, polygons, text. Everything returns SVG strings; plots.js composes them.
import { GG, sizes, cscale, decimals, hue, TIER_COL } from './theme.js';

export { GG, sizes, cscale, hue, TIER_COL };

let uid = 0;
const f2 = (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : 0);
export const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const fin = (v) => v != null && Number.isFinite(+v);
/** Approximate text width (Plex ~0.53 em/char, JetBrains Mono 0.6). */
export const textW = (t, fs, mono = false) => String(t).length * fs * (mono ? 0.6 : 0.53);

const DY = { middle: '0.34em', top: '0.78em', bottom: '-0.12em' };
/** SVG text. o: {size, color, anchor:start|middle|end, base:middle|top|bottom, weight, mono, caps, rotate, opacity} */
export function text(x, y, t, o = {}) {
  const a = [`x="${f2(x)}"`, `y="${f2(y)}"`, `dy="${DY[o.base || 'bottom']}"`, `font-size="${o.size || 10}"`, `fill="${o.color || GG.text}"`];
  if (o.anchor && o.anchor !== 'start') a.push(`text-anchor="${o.anchor}"`);
  if (o.weight) a.push(`font-weight="${o.weight}"`);
  if (o.mono) a.push(`font-family="${esc(GG.mono)}"`);
  if (o.caps) a.push('letter-spacing="0.07em"');
  if (o.opacity != null) a.push(`fill-opacity="${o.opacity}"`);
  if (o.rotate) a.push(`transform="rotate(${o.rotate} ${f2(x)} ${f2(y)})"`);
  return `<text ${a.join(' ')}>${esc(o.caps ? String(t).toUpperCase() : t)}</text>`;
}
const stroke = (o, s) => `stroke="${o.color || GG.text}" stroke-width="${o.width ?? s.line}"${o.dash ? ` stroke-dasharray="${o.dash === true ? `${f2(3.5 * s.k)} ${f2(2.5 * s.k)}` : o.dash}"` : ''}${o.opacity != null ? ` stroke-opacity="${o.opacity}"` : ''}`;
export const fmtStep = (step) => { const d = decimals(step); return (v) => (+v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); };
/** labels for log10-valued axes, printed as plain numbers */
export const pow10 = (v) => (v >= 0 ? (10 ** v).toLocaleString('en-US', { maximumFractionDigits: 0 }) : (10 ** v).toFixed(Math.round(-v)));
export const sup = (n) => String(n).replace(/-/g, '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
export function sciC(log10) {
  if (!fin(log10)) return '—';
  const e = Math.floor(log10), m = 10 ** (log10 - e), r = m.toFixed(1) === '10.0';
  return `${r ? '1.0' : m.toFixed(1)}×10${sup(r ? e + 1 : e)}`;
}
export const fmtN = (n) => (n == null ? '—' : Number(n).toLocaleString('en-US'));
export const fmtM = (m) => (!fin(m) ? '—' : m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);

// ═══ Panel: a data rectangle with scales; marks are clipped to it ═════════════════════════════════════
export class Panel {
  /** {x, y, w, h, xs, ys (cscale results), xReverse, s} */
  constructor({ x, y, w, h, xs, ys, xReverse = false, s }) {
    Object.assign(this, { x, y, w, h, xs, ys, s, xReverse });
    this.id = `ggc${++uid}`;
    const [x0, x1] = xs.domain, [y0, y1] = ys.domain;
    this.X = (v) => (xReverse ? x + ((x1 - v) / (x1 - x0)) * w : x + ((v - x0) / (x1 - x0)) * w);
    this.Y = (v) => y + h - ((v - y0) / (y1 - y0)) * h;
    this.out = []; this.over = [];
  }
  clipDef() { return `<clipPath id="${this.id}"><rect x="${f2(this.x)}" y="${f2(this.y)}" width="${f2(this.w)}" height="${f2(this.h)}"/></clipPath>`; }
  svg() { return `<g clip-path="url(#${this.id})">${this.out.join('')}</g>${this.over.join('')}${this.#labels()}`; }
  /** Reference-line labels, laid out together so they never overlap each other or cross a vertical rule.
   *  vline labels sit inside the top of the panel (stacked into rows when they collide); hline labels sit
   *  just above their line at the requested side, slide away from vertical rules and placed labels, and
   *  drop below the line if there is no room above. Each gets a small background so lines don't cut it. */
  #labels() {
    const L = this.refLabels || [];
    if (!L.some((l) => l.label)) return '';
    const fs = this.s.annot, pad = 2.5, hgt = fs * 1.25, placed = [], out = [];
    const hit = (a) => placed.some((b) => a.x0 < b.x1 + 2 && a.x1 > b.x0 - 2 && a.y0 < b.y1 + 1 && a.y1 > b.y0 - 1);
    const inX = (x0, x1) => Math.max(this.x + 1, Math.min(x0, this.x + this.w - 1 - (x1 - x0)));
    const vx = L.filter((l) => l.kind === 'v').map((l) => l.x);
    const box = (l, x0, y0) => { const w = textW(l.label, fs) + 2 * pad; const X0 = inX(x0, x0 + w); return { x0: X0, x1: X0 + w, y0, y1: y0 + hgt }; };
    const emit = (l, b) => {
      placed.push(b);
      out.push(`<rect x="${f2(b.x0)}" y="${f2(b.y0)}" width="${f2(b.x1 - b.x0)}" height="${f2(b.y1 - b.y0)}" rx="2" fill="${GG.bgExport}" fill-opacity="0.78"/>`);
      out.push(text(b.x0 + pad, b.y0 + hgt / 2, l.label, { size: fs, color: l.color, base: 'middle' }));
    };
    for (const l of L.filter((q) => q.kind === 'v' && q.label)) {
      const w = textW(l.label, fs) + 2 * pad;
      const cands = [];
      for (let row = 0; row < 4; row++) {
        const y0 = this.y + 2 + row * (hgt + 1);
        const sides = l.side === 'left' ? [l.x - 2 - w, l.x + 2] : [l.x + 2, l.x - 2 - w];
        for (const x0 of sides) cands.push(box(l, x0, y0));
      }
      emit(l, cands.find((b) => !hit(b)) || cands[0]);
    }
    for (const l of L.filter((q) => q.kind === 'h')) {
      const w = textW(l.label, fs) + 2 * pad;
      const xs0 = [];
      // preferred anchor, then step away from each vertical rule the label would cross
      let x0 = l.side === 'left' ? this.x + 2 : this.x + this.w - 2 - w;
      for (let i = 0; i < 6; i++) {
        xs0.push(x0);
        const cross = vx.filter((v) => v > x0 - 2 && v < x0 + w + 2);
        if (!cross.length) break;
        x0 = l.side === 'left' ? Math.max(...cross) + 3 : Math.min(...cross) - 3 - w;
      }
      const clear = (b) => !vx.some((v) => v > b.x0 - 1 && v < b.x1 + 1);
      const cands = [];
      for (const ya of [l.y - 1.5 - hgt, l.y + 1.5]) for (const xx of xs0.slice().reverse()) cands.push(box(l, xx, ya));
      const inside = (b) => b.y0 >= this.y && b.y1 <= this.y + this.h;
      emit(l, cands.find((b) => inside(b) && clear(b) && !hit(b)) || cands.find((b) => inside(b) && !hit(b)) || cands[0]);
    }
    return out.join('');
  }
  #d(pts) {
    let d = '', pen = false;
    for (const p of pts) {
      if (!p || !fin(p[0]) || !fin(p[1])) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${f2(this.X(p[0]))},${f2(this.Y(p[1]))}`; pen = true;
    }
    return d;
  }
  /** pts: [[x, y], ...] (null breaks the line) */
  line(pts, o = {}) { const d = this.#d(pts); if (d) this.out.push(`<path d="${d}" fill="none" ${stroke({ color: GG.accent, ...o }, this.s)} stroke-linejoin="round" stroke-linecap="round"/>`); return this; }
  /** step-after line */
  step(pts, o = {}) { const q = []; pts.forEach((p, i) => { q.push(p); if (pts[i + 1]) q.push([pts[i + 1][0], p[1]]); }); return this.line(q, o); }
  /** ribbon: pts [[x, ylo, yhi], ...] */
  area(pts, o = {}) {
    const ok = pts.filter((p) => fin(p[0]) && fin(p[1]) && fin(p[2]));
    if (ok.length < 2) return this;
    const d = `M${ok.map((p) => `${f2(this.X(p[0]))},${f2(this.Y(p[2]))}`).join('L')}L${ok.slice().reverse().map((p) => `${f2(this.X(p[0]))},${f2(this.Y(p[1]))}`).join('L')}Z`;
    this.out.push(`<path d="${d}" fill="${o.fill || GG.ribbon}" fill-opacity="${o.opacity ?? 0.2}" stroke="none"/>`);
    return this;
  }
  /** pts: [{x, y, tip}] */
  points(pts, o = {}) {
    const r = o.r ?? this.s.r, c = o.color || GG.accent;
    for (const p of pts) {
      if (!fin(p.x) || !fin(p.y)) continue;
      const t = p.tip ? `<title>${esc(p.tip)}</title>` : '';
      const cx = f2(this.X(p.x)), cy = f2(this.Y(p.y) + (o.dy || 0));
      if (o.shape === 'tri') {
        const a = r * 1.5;
        this.out.push(`<path d="M${cx},${f2(cy - a)}L${f2(cx + a * 0.9)},${f2(cy + a * 0.6)}L${f2(cx - a * 0.9)},${f2(cy + a * 0.6)}Z" fill="${c}">${t}</path>`);
      } else if (o.shape === 'cross') {
        this.out.push(`<path d="M${f2(cx - r)},${f2(cy - r)}L${f2(cx + r)},${f2(cy + r)}M${f2(cx - r)},${f2(cy + r)}L${f2(cx + r)},${f2(cy - r)}" stroke="${c}" stroke-width="${this.s.ref}">${t}</path>`);
      } else {
        this.out.push(`<circle cx="${cx}" cy="${cy}" r="${r}" ${o.hollow ? `fill="none" stroke="${c}" stroke-width="${o.sw ?? 1.1 * this.s.k}"` : `fill="${c}"${o.ring ? ` stroke="${o.ring}" stroke-width="1"` : ''}`}>${t}</circle>`);
      }
    }
    return this;
  }
  /** horizontal reference line with optional small label */
  hline(v, o = {}) {
    if (!fin(v)) return this;
    const y = f2(this.Y(v));
    this.out.push(`<line x1="${f2(this.x)}" x2="${f2(this.x + this.w)}" y1="${y}" y2="${y}" ${stroke({ color: GG.ref, width: this.s.ref, dash: true, ...o }, this.s)}/>`);
    if (o.label && v >= this.ys.domain[0] && v <= this.ys.domain[1]) {
      (this.refLabels ||= []).push({ kind: 'h', y: +y, label: o.label, side: o.side === 'left' ? 'left' : 'right', color: o.labelColor || o.color || GG.muted });
    }
    return this;
  }
  /** vertical reference line; label at the top of the panel */
  vline(v, o = {}) {
    if (!fin(v) || v < this.xs.domain[0] || v > this.xs.domain[1]) return this;
    const x = f2(this.X(v));
    this.out.push(`<line x1="${x}" x2="${x}" y1="${f2(this.y)}" y2="${f2(this.y + this.h)}" ${stroke({ color: GG.ref, width: this.s.ref, dash: true, ...o }, this.s)}/>`);
    {  // unlabelled rules are registered too, so hline labels can steer clear of them
      (this.refLabels ||= []).push({ kind: 'v', x: +x, label: o.label || null, side: o.side === 'left' ? 'left' : 'right', color: o.labelColor || o.color || GG.muted });
    }
    return this;
  }
  /** vertical band between x0 and x1 (full height) */
  band(x0, x1, o = {}) {
    const a = this.X(x0), b = this.X(x1);
    this.out.push(`<rect x="${f2(Math.min(a, b))}" y="${f2(this.y)}" width="${f2(Math.abs(b - a))}" height="${f2(this.h)}" fill="${o.fill || GG.warn}" fill-opacity="${o.opacity ?? 0.08}"/>`);
    return this;
  }
  rect(x0, x1, y0, y1, o = {}) {
    const a = this.X(x0), b = this.X(x1), c = this.Y(y0), d = this.Y(y1);
    this.out.push(`<rect x="${f2(Math.min(a, b))}" y="${f2(Math.min(c, d))}" width="${f2(Math.abs(b - a))}" height="${f2(Math.abs(d - c))}" fill="${o.fill || GG.bar}"${o.opacity != null ? ` fill-opacity="${o.opacity}"` : ''}${o.stroke ? ` stroke="${o.stroke}" stroke-width="${o.sw ?? 0.6}"` : ''}>${o.tip ? `<title>${esc(o.tip)}</title>` : ''}</rect>`);
    return this;
  }
  seg(x0, y0, x1, y1, o = {}) {
    this.out.push(`<line x1="${f2(this.X(x0))}" y1="${f2(this.Y(y0))}" x2="${f2(this.X(x1))}" y2="${f2(this.Y(y1))}" ${stroke({ color: GG.text, width: this.s.line, ...o }, this.s)} stroke-linecap="round"/>`);
    return this;
  }
  /** segment with an arrowhead whose tip lands exactly on (x1, y1) */
  arrow(x0, y0, x1, y1, o = {}) {
    const ax = this.X(x0), ay = this.Y(y0), bx = this.X(x1), by = this.Y(y1);
    const L = Math.hypot(bx - ax, by - ay), hl = o.head ?? 7 * this.s.k, hw = hl * 0.42;
    const c = o.color || GG.title, sw = o.width ?? this.s.line;
    if (L < hl * 1.4) { this.out.push(`<line x1="${f2(ax)}" y1="${f2(ay)}" x2="${f2(bx)}" y2="${f2(by)}" stroke="${c}" stroke-width="${sw}"/>`); return this; }
    const ux = (bx - ax) / L, uy = (by - ay) / L, baseX = bx - ux * hl, baseY = by - uy * hl;
    this.out.push(`<line x1="${f2(ax)}" y1="${f2(ay)}" x2="${f2(baseX + ux * 0.5)}" y2="${f2(baseY + uy * 0.5)}" stroke="${c}" stroke-width="${sw}" stroke-linecap="round"/>`);
    this.out.push(`<path d="M${f2(bx)},${f2(by)}L${f2(baseX - uy * hw)},${f2(baseY + ux * hw)}L${f2(baseX + uy * hw)},${f2(baseY - ux * hw)}Z" fill="${c}"/>`);
    return this;
  }
  /** closed polygon in data coordinates */
  poly(pts, o = {}) {
    const d = this.#d(pts);
    if (d) this.out.push(`<path d="${d}Z" fill="${o.fill || 'none'}"${o.fillOpacity != null ? ` fill-opacity="${o.fillOpacity}"` : ''} ${o.color ? stroke(o, this.s) : 'stroke="none"'} stroke-linejoin="round"/>`);
    return this;
  }
  /** text in data coordinates; unclipped annotation layer unless o.clip */
  text(x, y, t, o = {}) {
    const sv = text(this.X(x) + (o.dx || 0), this.Y(y) + (o.dy || 0), t, { size: this.s.annot, color: GG.text, ...o });
    (o.clip ? this.out : this.over).push(sv);
    return this;
  }
}

/** Panel frame: background, minor + major grid, ticks and tick labels, axis titles, facet strip. */
export function frame(P, o = {}) {
  const { s, xs, ys } = P;
  const out = [`<rect x="${f2(P.x)}" y="${f2(P.y)}" width="${f2(P.w)}" height="${f2(P.h)}" fill="${GG.panel}"/>`];
  const vx = (v) => f2(P.X(v)), vy = (v) => f2(P.Y(v));
  const inX = (v) => v >= xs.domain[0] - 1e-9 && v <= xs.domain[1] + 1e-9, inY = (v) => v >= ys.domain[0] - 1e-9 && v <= ys.domain[1] + 1e-9;
  for (const v of xs.minors.filter(inX)) out.push(`<line x1="${vx(v)}" x2="${vx(v)}" y1="${f2(P.y)}" y2="${f2(P.y + P.h)}" stroke="${GG.gridMinor}" stroke-width="${s.gridMinor}"/>`);
  for (const v of ys.minors.filter(inY)) out.push(`<line y1="${vy(v)}" y2="${vy(v)}" x1="${f2(P.x)}" x2="${f2(P.x + P.w)}" stroke="${GG.gridMinor}" stroke-width="${s.gridMinor}"/>`);
  for (const v of xs.majors.filter(inX)) out.push(`<line x1="${vx(v)}" x2="${vx(v)}" y1="${f2(P.y)}" y2="${f2(P.y + P.h)}" stroke="${GG.grid}" stroke-width="${s.gridMajor}"/>`);
  for (const v of ys.majors.filter(inY)) out.push(`<line y1="${vy(v)}" y2="${vy(v)}" x1="${f2(P.x)}" x2="${f2(P.x + P.w)}" stroke="${GG.grid}" stroke-width="${s.gridMajor}"/>`);
  const xf = o.xFmt || fmtStep(xs.step), yf = o.yFmt || fmtStep(ys.step);
  const tk = s.tickLen;
  if (o.xLabels !== false) {
    for (const v of xs.majors.filter(inX)) {
      out.push(`<line x1="${vx(v)}" x2="${vx(v)}" y1="${f2(P.y + P.h)}" y2="${f2(P.y + P.h + tk)}" stroke="${GG.tick}" stroke-width="${0.8 * s.k}"/>`);
      out.push(text(P.X(v), P.y + P.h + tk + 2 * s.k, xf(v), { size: s.axisText, color: GG.axisText, anchor: 'middle', base: 'top', mono: true }));
    }
    if (o.xTitle) out.push(text(P.x + P.w / 2, P.y + P.h + tk + s.axisText + 6 * s.k, o.xTitle, { size: s.axisTitle, color: GG.muted, anchor: 'middle', base: 'top' }));
  }
  if (o.yLabels !== false) {
    for (const v of ys.majors.filter(inY)) {
      out.push(`<line y1="${vy(v)}" y2="${vy(v)}" x1="${f2(P.x - tk)}" x2="${f2(P.x)}" stroke="${GG.tick}" stroke-width="${0.8 * s.k}"/>`);
      out.push(text(P.x - tk - 2.5 * s.k, P.Y(v), yf(v), { size: s.axisText, color: GG.axisText, anchor: 'end', base: 'middle', mono: true }));
    }
  }
  if (o.yTitle) {
    const tx = P.x - (o.yLabelW ?? 30) - tk - 6 * s.k;
    out.push(text(tx, P.y + P.h / 2, o.yTitle, { size: s.axisTitle, color: GG.muted, anchor: 'middle', base: 'bottom', rotate: -90 }));
  }
  if (o.strip) {
    out.push(`<rect x="${f2(P.x)}" y="${f2(P.y - s.stripH)}" width="${f2(P.w)}" height="${f2(s.stripH - 1)}" fill="${GG.strip}"/>`);
    out.push(text(P.x + P.w / 2, P.y - s.stripH / 2 - 0.5, o.strip, { size: s.strip, color: GG.stripText, anchor: 'middle', base: 'middle', weight: 600, caps: true }));
  }
  return out.join('');
}
export const yLabelWidth = (ys, s, fmt) => Math.max(0, ...ys.majors.filter((v) => v >= ys.domain[0] - 1e-9 && v <= ys.domain[1] + 1e-9).map((v) => textW((fmt || fmtStep(ys.step))(v), s.axisText, true)));
export const xAxisH = (s, title) => s.tickLen + 2 * s.k + s.axisText * 1.15 + (title ? s.axisTitle * 1.2 + 5 * s.k : 0);

// ═══ Legend: titled groups of keys (glyph on a faint key box + label) ════════════════════════════════════
/** groups: [{title, items:[{label, glyph, color, opacity, stroke}]}]; glyphs: line dashed point hollow pointline errorbar fill circle ribbon bar tri */
export function legendBox(groups, s) {
  const G = groups.filter((g) => g && g.items?.length);
  if (!G.length) return null;
  const K = s.key, gap = 5 * s.k, lh = Math.max(K + 3 * s.k, s.legendText * 1.5);
  let w = 0, h = 0;
  for (const g of G) {
    if (h) h += 8 * s.k;
    if (g.title) { w = Math.max(w, textW(g.title, s.legendTitle)); h += s.legendTitle * 1.45; }
    for (const it of g.items) { w = Math.max(w, K + gap + textW(it.label, s.legendText)); h += lh; }
  }
  w = Math.ceil(w);
  const draw = (x0, y0) => {
    const o = [];
    let y = y0;
    for (const g of G) {
      if (y > y0) y += 8 * s.k;
      if (g.title) { o.push(text(x0, y, g.title, { size: s.legendTitle, color: GG.text, weight: 500, base: 'top' })); y += s.legendTitle * 1.45; }
      for (const it of g.items) {
        const ky = y + (lh - K) / 2, cx = x0 + K / 2, cy = ky + K / 2, c = it.color || GG.text, op = it.opacity ?? 1;
        o.push(`<rect x="${f2(x0)}" y="${f2(ky)}" width="${K}" height="${K}" fill="${GG.key}"/>`);
        const ln = (dash, wdt) => `<line x1="${f2(x0 + 1.5)}" x2="${f2(x0 + K - 1.5)}" y1="${f2(cy)}" y2="${f2(cy)}" stroke="${c}" stroke-opacity="${op}" stroke-width="${wdt ?? s.line}"${dash ? ` stroke-dasharray="${f2(2.5 * s.k)} ${f2(1.8 * s.k)}"` : ''}/>`;
        const g2 = it.glyph || 'point';
        if (g2 === 'line' || g2 === 'pointline') o.push(ln(false));
        if (g2 === 'dashed') o.push(ln(true, s.ref * 1.2));
        if (g2 === 'errorbar') o.push(`<line x1="${f2(cx)}" x2="${f2(cx)}" y1="${f2(ky + 2)}" y2="${f2(ky + K - 2)}" stroke="${c}" stroke-width="${s.ref}"/><line x1="${f2(cx - 2.5)}" x2="${f2(cx + 2.5)}" y1="${f2(ky + 2)}" y2="${f2(ky + 2)}" stroke="${c}" stroke-width="${s.ref}"/><line x1="${f2(cx - 2.5)}" x2="${f2(cx + 2.5)}" y1="${f2(ky + K - 2)}" y2="${f2(ky + K - 2)}" stroke="${c}" stroke-width="${s.ref}"/>`);
        if (g2 === 'point' || g2 === 'pointline' || g2 === 'errorbar') o.push(`<circle cx="${f2(cx)}" cy="${f2(cy)}" r="${s.r}" fill="${c}"/>`);
        if (g2 === 'hollow') o.push(`<circle cx="${f2(cx)}" cy="${f2(cy)}" r="${f2(s.r * 1.35)}" fill="none" stroke="${c}" stroke-width="${1.1 * s.k}"/>`);
        if (g2 === 'tri') o.push(`<path d="M${f2(cx)},${f2(cy - 3.6 * s.k)}L${f2(cx + 3.3 * s.k)},${f2(cy + 2.4 * s.k)}L${f2(cx - 3.3 * s.k)},${f2(cy + 2.4 * s.k)}Z" fill="${c}"/>`);
        if (g2 === 'ribbon' || g2 === 'bar') o.push(`<rect x="${f2(x0)}" y="${f2(ky)}" width="${K}" height="${K}" fill="${c}" fill-opacity="${op}"/>`);
        if (g2 === 'fill') o.push(`<rect x="${f2(x0 + 1.5)}" y="${f2(ky + 1.5)}" width="${f2(K - 3)}" height="${f2(K - 3)}" fill="${c}" fill-opacity="${op}" stroke="${it.stroke || c}" stroke-width="${0.9 * s.k}"/>`);
        if (g2 === 'circle') o.push(`<circle cx="${f2(cx)}" cy="${f2(cy)}" r="${f2(K / 2 - 2)}" fill="${c}" fill-opacity="${op}" stroke="${it.stroke || c}" stroke-width="${0.9 * s.k}"/>`);
        o.push(text(x0 + K + gap, y + lh / 2, it.label, { size: s.legendText, color: GG.muted, base: 'middle' }));
        y += lh;
      }
    }
    return o.join('');
  };
  return { w, h: Math.ceil(h), draw };
}

/** Title + subtitle block; returns {svg, h}. */
export function header(title, subtitle, x, s) {
  const o = []; let y = 0;
  if (title) { o.push(text(x, y, title, { size: s.title, color: GG.title, weight: 600, base: 'top' })); y += s.title * 1.3; }
  if (subtitle) { o.push(text(x, y, subtitle, { size: s.subtitle, color: GG.muted, base: 'top' })); y += s.subtitle * 1.35; }
  return { svg: o.join(''), h: y ? y + 7 * s.k : 0 };
}
/** Right-aligned caption lines ending at xr; returns {svg, h}. */
export function caption(lines, xr, y0, s, anchor = 'end', maxW = anchor === 'end' ? xr - 2 : Infinity) {
  // wrap any line wider than the room it has (greedy on spaces), so long captions never run off the card
  const L = [];
  for (const t of (Array.isArray(lines) ? lines : [lines]).filter(Boolean)) {
    let cur = '';
    for (const w of String(t).split(' ')) {
      const nxt = cur ? `${cur} ${w}` : w;
      if (cur && textW(nxt, s.caption) > maxW) { L.push(cur); cur = w; } else cur = nxt;
    }
    if (cur) L.push(cur);
  }
  return { svg: L.map((t, i) => text(xr, y0 + i * s.caption * 1.35, t, { size: s.caption, color: GG.dim, anchor, base: 'top' })).join(''), h: L.length * s.caption * 1.35 };
}
export function wrapSvg(w, h, inner, panels) {
  return `<svg xmlns="http://www.w3.org/2000/svg" class="gg-svg" width="${Math.ceil(w)}" height="${Math.ceil(h)}" viewBox="0 0 ${Math.ceil(w)} ${Math.ceil(h)}" font-family="${esc(GG.font)}"><defs>${panels.map((p) => p.clipDef()).join('')}</defs>${inner}</svg>`;
}

/**
 * Vertical stack of facet panels sharing a width, with a titled legend on the right.
 * rows: [{strip, h:(W)=>px, xs, ys, xReverse, xTitle, yTitle, xLabels, xFmt, yFmt, draw:(P)=>void}]
 */
export function figure({ width = 400, base = 11, title, subtitle, cap, legend, rows }) {
  const s = sizes(base);
  const L = legend ? legendBox(legend, s) : null;
  const ylw = Math.max(...rows.map((r) => yLabelWidth(r.ys, s, r.yFmt)));
  const hasYT = rows.some((r) => r.yTitle);
  const left = Math.ceil(ylw + s.tickLen + 2.5 * s.k + (hasYT ? s.axisTitle * 1.25 + 4 * s.k : 0) + 1);
  const right = L ? L.w + 14 * s.k : 4;
  const W = Math.max(120, Math.floor(width - left - right));
  const H = header(title, subtitle, 0, s);
  let y = H.h;
  const parts = [H.svg], panels = [];
  let legendY = null;
  rows.forEach((r, i) => {
    if (r.strip) y += s.stripH;
    const h = Math.round(typeof r.h === 'function' ? r.h(W) : r.h || W * 0.5);
    const P = new Panel({ x: left, y, w: W, h, xs: r.xs, ys: r.ys, xReverse: r.xReverse, s });
    if (legendY == null) legendY = y - (r.strip ? s.stripH : 0);
    panels.push(P);
    parts.push(frame(P, { ...r, yLabelW: ylw }));
    r.draw(P, s);
    parts.push(P.svg());
    y += h;
    if (r.xLabels !== false) y += xAxisH(s, r.xTitle);
    if (i < rows.length - 1) y += s.spacing;
  });
  if (L) parts.push(L.draw(left + W + 14 * s.k, legendY));
  y += 6 * s.k;
  const C = cap ? caption(cap, left + W, y, s) : { svg: '', h: 0 };
  parts.push(C.svg);
  const hgt = Math.max(y + C.h, L ? legendY + L.h : 0) + 2;
  return wrapSvg(width, hgt, parts.join(''), panels);
}

export function emptyFig(title, msg, o = {}) {
  const s = sizes(o.base || 11), w = o.width || 380;
  const H = header(title, null, 0, s);
  const h = 80 * s.k;
  return wrapSvg(w, H.h + h, H.svg + `<rect x="0" y="${H.h}" width="${w}" height="${h}" fill="${GG.panel}"/>` + text(w / 2, H.h + h / 2, msg, { size: s.subtitle, color: GG.muted, anchor: 'middle', base: 'middle' }), []);
}
