// Console charts, hand-drawn SVG (core.js). One function per plot: (data..., opts {width, base}) -> SVG string.
// Every value comes from server payloads (SSE store, /api/futures, /api/conjunctions/*/rtn, /api/eval/curves, /api/dashboard).
import {
  GG, sizes, cscale, hue, TIER_COL, Panel, frame, figure, legendBox, header, caption, wrapSvg, emptyFig,
  yLabelWidth, xAxisH, text, fin, fmtN, fmtM, sciC, pow10, textW,
} from './core.js';

const H3 = hue(3);
/** Greedy word wrap to a pixel width (approximate text metrics). */
function wrapLines(t, w, fs) {
  const out = []; let cur = '';
  for (const word of t.split(' ')) {
    const next = cur ? `${cur} ${word}` : word;
    if (textW(next, fs) > w && cur) { out.push(cur); cur = word; } else cur = next;
  }
  if (cur) out.push(cur);
  return out;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const PERS = '#c3cedb';   // persistence baseline (light grey)
const srcCap = (c) => `${c.source?.name || '?'}${c.source?.model ? ` (${c.source.model})` : ''} · out-of-fold, train events`;
const dayFmt = (step) => (step < 1 ? (v) => (+v).toFixed(1) : (v) => String(Math.round(v)));

// ═══ 1. Risk evolution: two facets over time to TCA (TCA at the right) ═══════════════════════════════════
/** d: {key, points:[{ttca, log10, reported, miss_m, status}], nowTtca, futures, tier} */
export function riskEvolution(d, o = {}) {
  const pts = (d.points || []).filter((p) => fin(p.ttca)).sort((a, z) => z.ttca - a.ttca);
  if (!pts.length) return emptyFig('Risk evolution', 'Waiting for a CDM…', o);
  const f = d.futures?.ttca_grid_d && d.futures.quantiles ? d.futures : null;
  const q = f?.quantiles || {}, mq = f?.miss_quantiles || {};
  const xMax = Math.max(3, ...pts.map((p) => p.ttca), ...(f ? f.ttca_grid_d : []), fin(d.nowTtca) ? d.nowTtca : 0);
  const xs = cscale(0, xMax, { step: xMax > 4 ? 1 : 0.5, expand: 0.03 });
  const pY = pts.flatMap((p) => [p.log10, p.reported]).filter(fin);
  const floor = Math.min(-8, Math.floor(pY.length ? Math.min(...pY) : -8) - 3);
  const fanY = f ? ['p05', 'p25', 'p50', 'p75', 'p95'].flatMap((k) => q[k] || []).filter(fin) : [];
  const yLo = Math.max(floor, Math.min(...pY, ...fanY, -6.5)), yHi = Math.min(0, Math.max(...pY, ...fanY, -3.5));
  const ys1 = cscale(yLo, yHi, { n: 4 });
  const clipped = fanY.some((v) => v < yLo - 0.01);
  const lm = (m) => (fin(m) && m > 0 ? Math.log10(m / 1000) : null);
  const mY = pts.map((p) => lm(p.miss_m)).filter(fin);
  const fanM = f ? ['p05', 'p25', 'p50', 'p75', 'p95'].flatMap((k) => (mq[k] || []).map(lm)).filter(fin) : [];
  const ys2 = cscale(Math.min(...mY, ...fanM, -0.3), Math.max(...mY, ...fanM, Math.log10(5)), { step: 1 });
  const nowOk = fin(d.nowTtca) && d.nowTtca >= 0 && d.nowTtca <= xs.domain[1];

  const common = (P, label) => {
    P.band(1, 0.5, { fill: GG.warn, opacity: 0.07 });
    P.vline(2, { label: label ? '2 d cut' : null });
    if (nowOk) P.vline(d.nowTtca, { color: GG.text, dash: false, opacity: 0.55, label: label ? 'now' : null, side: 'left', labelColor: GG.text });
  };
  const fan = (P, qq, tf) => {
    if (!f) return;
    const [lo, hi] = P.ys.domain;
    for (const [a, z, op] of [['p05', 'p95', 0.13], ['p25', 'p75', 0.24]]) {
      if (qq[a] && qq[z]) P.area(f.ttca_grid_d.map((t, i) => [t, clamp(tf(qq[a][i]), lo, hi), clamp(tf(qq[z][i]), lo, hi)]), { fill: GG.ribbon, opacity: op });
    }
    if (qq.p50) P.line(f.ttca_grid_d.map((t, i) => [t, tf(qq.p50[i])]), { color: GG.median, width: P.s.ref, opacity: 0.8, dash: true });
  };
  const pcVals = pts.filter((p) => fin(p.log10));
  const rows = [
    {
      strip: 'Collision probability (physics)', xs, ys: ys1, xReverse: true, xLabels: false, yTitle: 'log₁₀ Pc', yFmt: (v) => String(Math.round(v)),
      h: (W) => W * 0.5,
      draw: (P) => {
        common(P, true); fan(P, q, (v) => v);
        P.hline(-4, { color: GG.bad, label: '10⁻⁴ critical', opacity: 0.85 });
        P.hline(-6, { color: GG.warn, label: '10⁻⁶ elevated', opacity: 0.85 });
        P.line(pcVals.map((p) => [p.ttca, p.log10]), { color: GG.accent });
        P.points(pts.filter((p) => fin(p.reported)).map((p) => ({ x: p.ttca, y: p.reported, tip: `ESA reported ${sciC(p.reported)} at T−${p.ttca.toFixed(2)} d` })), { hollow: true, color: GG.muted, r: P.s.r * 1.7 });
        P.points(pcVals.map((p) => ({ x: p.ttca, y: p.log10, tip: `Pc ${sciC(p.log10)} at T−${p.ttca.toFixed(2)} d` })), { color: GG.accent, ring: '#0a1019' });
        P.points(pts.filter((p) => p.status === 'COV_UNKNOWN').map((p) => ({ x: p.ttca, y: ys1.domain[0] + 0.3, tip: 'no covariance: Pc refused' })), { shape: 'cross', color: GG.muted });
      },
    },
    {
      strip: 'Miss distance', xs, ys: ys2, xReverse: true, xTitle: 'Time to TCA (days)', xFmt: dayFmt(xs.step), yTitle: 'km (log)', yFmt: pow10,
      h: (W) => W * 0.36,
      draw: (P) => {
        common(P, false); fan(P, mq, lm);
        P.hline(0, { color: GG.bad, label: '1 km', opacity: 0.85 });
        P.hline(Math.log10(5), { color: GG.warn, label: '5 km', opacity: 0.85 });
        const mv = pts.filter((p) => fin(lm(p.miss_m)));
        P.line(mv.map((p) => [p.ttca, lm(p.miss_m)]), { color: GG.accent });
        P.points(mv.map((p) => ({ x: p.ttca, y: lm(p.miss_m), tip: `miss ${fmtM(p.miss_m)} at T−${p.ttca.toFixed(2)} d` })), { color: GG.accent, ring: '#0a1019' });
      },
    },
  ];
  const legend = [
    { title: 'Series', items: [{ label: 'physics', glyph: 'pointline', color: GG.accent }, { label: 'ESA reported', glyph: 'hollow', color: GG.muted }] },
    f && { title: 'Forecast', items: [{ label: 'p05–p95', glyph: 'ribbon', color: GG.ribbon, opacity: 0.2 }, { label: 'p25–p75', glyph: 'ribbon', color: GG.ribbon, opacity: 0.4 }, { label: 'median', glyph: 'dashed', color: GG.median }] },
    { title: 'Window', items: [{ label: 'commit 0.5–1 d', glyph: 'ribbon', color: GG.warn, opacity: 0.18 }] },
  ];
  const n = pts.length;
  return figure({
    ...o, title: 'Risk evolution', subtitle: `${n} CDM${n === 1 ? '' : 's'} released · latest Pc ${sciC(pcVals.at(-1)?.log10)}`, legend, rows,
    cap: f ? `fan: futures model, ${fmtN(f.n_paths)} paths from T−${(+f.start_ttca_d || f.ttca_grid_d[0]).toFixed(2)} d (statistics)${clipped ? ', clipped' : ''}` : 'no forecast fan for this event',
  });
}

// ═══ 2. RTN relative position vs time to TCA: three facets, ±1σ ═════════════════════════════════════════
/** d: {key, cdms:[{ttca_d, rtn_m:[r,t,n], sigma_comb_m:[r,t,n]|null}]} */
export function rtnPlot(d, o = {}) {
  const rows0 = (d.cdms || []).filter((c) => fin(c.ttca_d) && Array.isArray(c.rtn_m));
  if (!rows0.length) return emptyFig('Relative position (RTN)', 'Waiting for a CDM…', o);
  const xMax = Math.max(3, ...rows0.map((c) => c.ttca_d));
  const xs = cscale(0, xMax, { step: xMax > 4 ? 1 : 0.5, expand: 0.03 });
  const anySigma = rows0.some((c) => (c.sigma_comb_m || []).some(fin));
  const names = ['R · radial', 'T · along-track', 'N · cross-track'];
  const rows = names.map((nm, k) => {
    const vals = rows0.filter((c) => fin(c.rtn_m[k])).map((c) => ({ x: c.ttca_d, y: c.rtn_m[k], sg: fin(c.sigma_comb_m?.[k]) ? +c.sigma_comb_m[k] : null }));
    const ext = vals.flatMap((v) => [v.y, v.sg != null ? v.y - v.sg : null, v.sg != null ? v.y + v.sg : null]).filter(fin);
    const ys = cscale(Math.min(...ext, 0), Math.max(...ext, 0), { n: 3 });
    const big = Math.max(...ext.map(Math.abs)) >= 1e4;
    return {
      strip: nm, xs, ys, xReverse: true, xLabels: k === 2, xTitle: 'Time to TCA (days)', xFmt: dayFmt(xs.step), yTitle: k === 1 ? 'Position (m)' : null,
      yFmt: big ? (v) => (v === 0 ? '0' : `${(v / 1000).toLocaleString('en-US', { maximumFractionDigits: 1 })}k`) : undefined,
      h: (W) => W * 0.26,
      draw: (P) => {
        const c = H3[k], cw = 3.2 * P.s.k / P.w * (xs.domain[1] - xs.domain[0]);
        P.hline(0, { opacity: 0.7 }); P.vline(2, { opacity: 0.7 });
        const sv = vals.filter((v) => v.sg != null).sort((a, b) => b.x - a.x);
        if (sv.length >= 2) P.area(sv.map((v) => [v.x, v.y - v.sg, v.y + v.sg]), { fill: c, opacity: 0.16 });  // ±1σ band
        for (const v of vals.filter((v) => v.sg != null)) {
          P.seg(v.x, v.y - v.sg, v.x, v.y + v.sg, { color: c, width: P.s.ref, opacity: 0.75 });
          P.seg(v.x - cw, v.y - v.sg, v.x + cw, v.y - v.sg, { color: c, width: P.s.ref, opacity: 0.75 });
          P.seg(v.x - cw, v.y + v.sg, v.x + cw, v.y + v.sg, { color: c, width: P.s.ref, opacity: 0.75 });
        }
        P.line(vals.map((v) => [v.x, v.y]), { color: c, width: P.s.line * 0.9 });
        P.points(vals.map((v) => ({ x: v.x, y: v.y, tip: `${nm}: ${Math.round(v.y).toLocaleString()} m${v.sg != null ? ` ± ${Math.round(v.sg).toLocaleString()} m` : ''} at T−${v.x.toFixed(2)} d` })), { color: c, ring: '#0a1019' });
      },
    };
  });
  const legend = [{ title: 'Component', items: ['R', 'T', 'N'].map((l, i) => ({ label: l, glyph: anySigma ? 'errorbar' : 'pointline', color: H3[i] })) }];
  return figure({
    ...o, title: 'Relative position (RTN)', subtitle: 'chaser relative to target, per released CDM', legend, rows,
    cap: anySigma ? 'shaded band and bars: ±1σ, target + chaser combined √(σt² + σc²) from the CDM · dashed: 2 d cut' : 'σ not usable for these CDMs (default covariance) · dashed: 2 d cut',
  });
}

// ═══ 3. Encounter plane + true-scale hard-body facet ═══════════════════════════════════════════════════
const NICE = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
const niceUp = (v) => { const p = 10 ** Math.floor(Math.log10(v)); return (NICE.find((m) => m * p >= v - 1e-9) || 10) * p; };

/** p: planeParams() {sx, sy, th(rad), mx, my, hbr, log10, status, k}; d: {key, tier} */
export function encounterPlane(p, d = {}, o = {}) {
  if (!p || !p.sx) return emptyFig('Encounter plane', p?.status === 'COV_UNKNOWN' ? 'No covariance: Pc not computable (refused)' : 'Waiting for a CDM…', o);
  const s = sizes(o.base || 11), width = o.width || 400;
  const hbr = fin(p.hbr) && p.hbr > 0 ? +p.hbr : 5;
  const missM = Math.hypot(p.mx, p.my);
  const OP = [0.95, 0.58, 0.32];
  const ell = (k) => Array.from({ length: 145 }, (_, i) => {
    const a = (i / 144) * 2 * Math.PI, u = k * p.sx * Math.cos(a), v = k * p.sy * Math.sin(a);
    return [p.mx + u * Math.cos(p.th) - v * Math.sin(p.th), p.my + u * Math.sin(p.th) + v * Math.cos(p.th)];
  });
  const E = [1, 2, 3].map(ell);
  const all = [...E.flat(), [0, 0]];
  const xsAll = all.map((q) => q[0]), ysAll = all.map((q) => q[1]);
  const cx = (Math.min(...xsAll) + Math.max(...xsAll)) / 2, cy = (Math.min(...ysAll) + Math.max(...ysAll)) / 2;
  const half = Math.max(Math.max(...xsAll) - Math.min(...xsAll), Math.max(...ysAll) - Math.min(...ysAll)) / 2;
  const mxs = cscale(cx - half, cx + half, { n: 4 }), mys = cscale(cy - half, cy + half, { n: 4 });
  const span = mxs.domain[1] - mxs.domain[0];
  const zr = niceUp(Math.max(2.4 * hbr, 4));
  const zxs = cscale(p.mx - zr, p.mx + zr, { n: 2, expand: 0 }), zys = cscale(p.my - zr, p.my + zr, { n: 2, expand: 0 });
  const N = Math.max(1, Math.round(span / (2 * zr)));
  const kfmt = (sc) => (Math.max(...sc.majors.map(Math.abs)) >= 1e4 ? (v) => `${(v / 1000).toLocaleString('en-US', { maximumFractionDigits: 1 })}k` : (v) => Math.round(v).toLocaleString('en-US'));
  const iFmt = (v) => Math.round(v).toLocaleString('en-US');

  const L = legendBox([
    { title: 'Position uncertainty', items: ['1σ', '2σ', '3σ'].map((l, i) => ({ label: `${l} · ${['39', '86', '99'][i]} % inside`, glyph: 'line', color: GG.accent, opacity: OP[i] })) },
    { title: 'Geometry', items: [{ label: 'primary', glyph: 'point', color: GG.title }, { label: 'miss vector', glyph: 'line', color: GG.title }, { label: `hard body ${hbr} m`, glyph: 'circle', color: GG.title, opacity: 0.16, stroke: GG.title }] },
  ], s);
  const H = header('Encounter plane', `${p.status && p.status !== 'OK' && p.status !== 'NPD_CLIPPED' ? p.status : `Pc ${sciC(p.log10)}`} · miss ${fmtM(missM)} · HBR ${hbr} m`, 0, s);
  const ylwM = yLabelWidth(mys, s, kfmt(mys)), ylwZ = yLabelWidth(zys, s, iFmt);
  const yt = s.axisTitle * 1.25 + 4 * s.k;
  const leftM = Math.ceil(ylwM + s.tickLen + 2.5 * s.k + yt + 1);
  const Z = Math.max(Math.ceil(L.w + 2), Math.round(width * 0.25));
  const leftZ = Math.ceil(ylwZ + s.tickLen + 2.5 * s.k + 2);
  const gap = 12 * s.k;
  const Pw = Math.max(150, Math.floor(width - leftM - gap - leftZ - Z - 2));
  const top = H.h + s.stripH;
  const M = new Panel({ x: leftM, y: top, w: Pw, h: Pw, xs: mxs, ys: mys, s });
  const zx = leftM + Pw + gap + leftZ;
  const Zp = new Panel({ x: zx, y: top, w: Z, h: Z, xs: zxs, ys: zys, s });
  const parts = [H.svg];
  parts.push(frame(M, { strip: 'Encounter plane · metres', xTitle: 'x, along miss (m)', yTitle: 'y, across (m)', xFmt: kfmt(mxs), yFmt: kfmt(mys), yLabelW: ylwM }));
  parts.push(frame(Zp, { strip: `Hard body (zoom ×${fmtN(N)})`, xTitle: 'x (m)', xFmt: iFmt, yFmt: iFmt, yLabelW: ylwZ }));

  // main panel: graded contours, labels at each contour's top, miss vector with arrowhead, primary at the origin
  // faint density cloud under the contours: stacked translucent ellipses, darker towards the centre
  for (let k = 3; k > 0.2; k -= 0.375) M.poly(ell(k), { fill: GG.accent, fillOpacity: 0.03 });
  E.forEach((pts, i) => M.line(pts, { color: GG.accent, width: s.line * 0.85, opacity: OP[i] }));
  E.forEach((pts, i) => {
    const topPt = pts.reduce((a, b) => (b[1] > a[1] ? b : a));
    if (topPt[1] < mys.domain[1]) M.text(topPt[0], topPt[1], `${i + 1}σ`, { color: GG.accent, opacity: OP[i], anchor: 'middle', base: 'bottom', dy: -2, size: s.annot * 0.95 });
  });
  M.arrow(0, 0, p.mx, p.my, { color: GG.title, width: s.line * 0.95 });
  const ax = M.X(0), ay = M.Y(0), bx = M.X(p.mx), by = M.Y(p.my), Lp = Math.hypot(bx - ax, by - ay) || 1;
  let nx = -(by - ay) / Lp, ny = (bx - ax) / Lp; if (ny > 0) { nx = -nx; ny = -ny; }
  M.over.push(text((ax + bx) / 2 + nx * 8 * s.k, (ay + by) / 2 + ny * 8 * s.k, `miss ${fmtM(missM)}`, { size: s.annot, color: GG.title, weight: 600, anchor: Math.abs(nx) > 0.5 ? (nx < 0 ? 'end' : 'start') : 'middle', base: 'middle' }));
  M.points([{ x: 0, y: 0, tip: 'primary (origin)' }], { color: GG.title, r: s.r * 1.1, ring: '#0a1019' });
  M.text(0, 0, 'primary', { color: GG.muted, dx: -5 * s.k, dy: 5 * s.k, anchor: 'end', base: 'top' });

  // zoom panel: true scale around the chaser's predicted position (the miss point), combined hard body as an outline
  const circ = Array.from({ length: 97 }, (_, i) => [p.mx + hbr * Math.cos((i / 96) * 2 * Math.PI), p.my + hbr * Math.sin((i / 96) * 2 * Math.PI)]);
  Zp.seg(0, 0, p.mx, p.my, { color: GG.title, width: s.line * 0.9, opacity: 0.45 });
  Zp.poly(circ, { fill: GG.title, fillOpacity: 0.12, color: GG.title, width: s.line * 0.9 });
  Zp.points([{ x: p.mx, y: p.my, tip: 'chaser predicted position' }], { color: GG.title, r: s.r * 0.8 });
  Zp.text(p.mx + hbr * 0.72, p.my + hbr * 0.72, `r ${hbr} m`, { color: GG.text, dx: 2, dy: -1, base: 'bottom' });
  const bar = zxs.step >= zr ? zxs.step / 2 : zxs.step;
  const bx0 = zxs.domain[0] + 0.1 * 2 * zr, byy = zys.domain[0] + 0.1 * 2 * zr, tk = 0.03 * 2 * zr;
  Zp.seg(bx0, byy, bx0 + bar, byy, { color: GG.text, width: s.ref });
  Zp.seg(bx0, byy - tk, bx0, byy + tk, { color: GG.text, width: s.ref });
  Zp.seg(bx0 + bar, byy - tk, bx0 + bar, byy + tk, { color: GG.text, width: s.ref });
  Zp.text(bx0 + bar / 2, byy + tk, `${bar} m`, { color: GG.text, anchor: 'middle', base: 'bottom', dy: -1 });
  parts.push(M.svg(), Zp.svg());

  const zBottom = top + Z + xAxisH(s, 'x (m)') + 12 * s.k;
  parts.push(L.draw(zx - leftZ + 4, zBottom));
  let y = top + Pw + xAxisH(s, true) + 6 * s.k;
  const capW = width - 4;
  const C = caption([
    ...wrapLines('Share of the relative-position uncertainty enclosed by each contour (2-D Gaussian: 1 − e^(−k²/2)); density falls off outward. Pc = the share that lands inside the hard-body circle.', capW, s.caption),
    'from CDM covariance (physics), not reconstructed',
    `σ ${Math.round(p.sx)} × ${Math.round(p.sy)} m${p.k != null ? ` · k* ${(+p.k).toFixed(2)}` : ''} · hard body drawn at the chaser's predicted position`,
  ], 0, Math.max(y, zBottom + L.h + 8 * s.k), s, 'start');
  parts.push(C.svg);
  const hgt = Math.max(y, zBottom + L.h + 8 * s.k) + C.h + 2;
  return wrapSvg(width, hgt, parts.join(''), [M, Zp]);
}

// ═══ 4. Statistics (GET /api/eval/curves) ═════════════════════════════════════════════════════════════════
const unit = () => cscale(0, 1, { n: 4, expand: 0.03 });
const f2d = (v) => (+v).toFixed(2);

export function rocPlot(c, o = {}) {
  const pe = c.persistence;
  return figure({
    ...o, title: 'ROC curve', subtitle: `AUC ${c.auc.toFixed(3)} · n = ${fmtN(c.n)}, ${fmtN(c.n_critical)} critical`, cap: srcCap(c),
    legend: [{ title: 'Predictor', items: [{ label: 'model P(critical)', glyph: 'line', color: GG.accent }, { label: 'persistence', glyph: 'point', color: PERS }] }],
    rows: [{
      xs: unit(), ys: unit(), xTitle: 'False positive rate', yTitle: 'True positive rate', xFmt: f2d, yFmt: f2d, h: (W) => W * 0.7,
      draw: (P) => {
        P.seg(0, 0, 1, 1, { color: GG.ref, width: P.s.ref, dash: true });
        P.text(0.72, 0.72, 'chance', { color: GG.dim, rotate: 0, dx: 4, dy: 4, base: 'top' });
        P.step(c.roc.map((r) => [r.fpr, r.tpr]), { color: GG.accent, width: P.s.line });
        P.points([{ x: pe.fpr, y: pe.tpr, tip: `persistence: FPR ${pe.fpr.toFixed(3)}, TPR ${pe.tpr.toFixed(3)}` }], { color: PERS, r: P.s.r * 1.3, ring: '#0a1019' });
        P.text(pe.fpr, pe.tpr, 'persistence', { color: PERS, dx: 7, dy: 5, base: 'top' });
      },
    }],
  });
}

export function prPlot(c, o = {}) {
  const pe = c.persistence;
  return figure({
    ...o, title: 'Precision–recall', subtitle: `AP ${c.ap.toFixed(3)} · base rate ${(100 * c.base_rate).toFixed(1)} %`, cap: srcCap(c),
    legend: [{ title: 'Predictor', items: [{ label: 'model P(critical)', glyph: 'line', color: GG.accent }, { label: 'persistence', glyph: 'point', color: PERS }] }, { title: 'Reference', items: [{ label: 'base rate', glyph: 'dashed', color: GG.ref }] }],
    rows: [{
      xs: unit(), ys: unit(), xTitle: 'Recall', yTitle: 'Precision', xFmt: f2d, yFmt: f2d, h: (W) => W * 0.7,
      draw: (P) => {
        P.hline(c.base_rate, { label: `base rate ${c.base_rate.toFixed(3)}`, side: 'left' });
        P.line(c.pr.map((r) => [r.recall, r.precision]), { color: GG.accent, width: P.s.line });
        P.points([{ x: pe.recall, y: pe.precision, tip: `persistence: recall ${pe.recall.toFixed(3)}, precision ${pe.precision.toFixed(3)}` }], { color: PERS, r: P.s.r * 1.3, ring: '#0a1019' });
        P.text(pe.recall, pe.precision, 'persistence', { color: PERS, dx: -7, dy: 5, anchor: 'end', base: 'top' });
      },
    }],
  });
}

export function reliabilityPlot(c, o = {}) {
  const rel = c.reliability.filter((r) => r.n > 0);
  const hist = c.pred_hist.filter((h) => h.n > 0);
  const maxL = Math.ceil(Math.log10(Math.max(...hist.map((h) => h.n), 10)));
  return figure({
    ...o, title: 'Reliability', subtitle: `Brier ${c.brier.toFixed(4)} · ECE ${c.ece.toFixed(4)} · 10 bins`, cap: ['95 % CI: exact binomial (Clopper–Pearson)', srcCap(c)],
    legend: [{ title: 'Bins', items: [{ label: 'observed ± 95 % CI', glyph: 'errorbar', color: GG.accent }, { label: 'perfect calibration', glyph: 'dashed', color: GG.ref }] }, { title: 'Count', items: [{ label: 'predictions', glyph: 'bar', color: GG.barLine }] }],
    rows: [
      {
        strip: 'Observed vs predicted', xs: unit(), ys: unit(), xLabels: false, yTitle: 'Observed fraction', yFmt: f2d, h: (W) => W * 0.62,
        draw: (P) => {
          P.seg(0, 0, 1, 1, { color: GG.ref, width: P.s.ref, dash: true });
          const cw = 3 * P.s.k / P.w;
          for (const r of rel) {
            P.seg(r.mean_pred, r.ci_lo, r.mean_pred, r.ci_hi, { color: GG.accent, width: P.s.ref, opacity: 0.8 });
            P.seg(r.mean_pred - cw, r.ci_lo, r.mean_pred + cw, r.ci_lo, { color: GG.accent, width: P.s.ref, opacity: 0.8 });
            P.seg(r.mean_pred - cw, r.ci_hi, r.mean_pred + cw, r.ci_hi, { color: GG.accent, width: P.s.ref, opacity: 0.8 });
          }
          P.line(rel.map((r) => [r.mean_pred, r.obs]), { color: GG.accent, width: P.s.ref, opacity: 0.55 });
          P.points(rel.map((r) => ({ x: r.mean_pred, y: r.obs, tip: `bin ${r.lo.toFixed(1)}–${r.hi.toFixed(1)}: ${r.k}/${r.n} critical (${(100 * r.obs).toFixed(1)} %), CI ${(100 * r.ci_lo).toFixed(1)}–${(100 * r.ci_hi).toFixed(1)} %` })), { color: GG.accent, ring: '#0a1019' });
        },
      },
      {
        strip: 'Predictions per bin', xs: unit(), ys: cscale(0, maxL, { step: 1, expand: 0.04 }), xTitle: 'Predicted P(critical)', xFmt: f2d, yTitle: 'Events', yFmt: pow10, h: (W) => W * 0.24,
        draw: (P) => { for (const h of hist) P.rect(h.lo + 0.006, h.hi - 0.006, 0, Math.log10(h.n), { fill: GG.bar, stroke: GG.barLine, tip: `${fmtN(h.n)} predictions in ${h.lo.toFixed(1)}–${h.hi.toFixed(1)}` }); },
      },
    ],
  });
}

export function costPlot(c, o = {}) {
  const cc = c.cost, ch = cc.chosen, pe = c.persistence;
  const pts = cc.curve.filter((r) => r.threshold >= 1e-3).map((r) => [Math.log10(r.threshold), r.cost]);
  const ys = cscale(0, Math.max(...pts.map((p) => p[1]), pe.cost), { n: 4 });
  const fx = cc.floor_max_threshold ? Math.log10(cc.floor_max_threshold) : null;
  return figure({
    ...o, title: `Expected cost vs threshold · R = ${cc.R}`, cap: [ch?.source ? `t*: ${ch.source}` : `t*: lowest cost with recall ≥ ${Math.round(cc.recall_floor * 100)} %`, srcCap(c)],
    subtitle: ch ? `t* ${ch.threshold.toFixed(3)}: cost ${fmtN(ch.cost)} (${ch.missed} missed, ${fmtN(ch.false_alarms)} false alarms) · persistence ${fmtN(pe.cost)}` : '',
    legend: [
      { title: 'Policy', items: [{ label: 'model, alert if P ≥ t', glyph: 'line', color: GG.accent }, { label: 'persistence', glyph: 'dashed', color: PERS }] },
      { title: 'Constraint', items: [{ label: `recall ≥ ${Math.round(cc.recall_floor * 100)} %`, glyph: 'ribbon', color: GG.good, opacity: 0.2 }] },
    ],
    rows: [{
      xs: cscale(-3, 0, { step: 1, expand: 0.03 }), ys, xTitle: 'Alert threshold t on P(critical), log scale', xFmt: pow10, yTitle: `${cc.R}·missed + false alarms`, yFmt: (v) => Math.round(v).toLocaleString('en-US'), h: (W) => W * 0.6,
      draw: (P) => {
        if (fx != null) P.band(P.xs.domain[0], fx, { fill: GG.good, opacity: 0.07 });
        P.hline(pe.cost, { color: PERS, label: `persistence ${fmtN(pe.cost)}`, opacity: 0.8, side: 'left' });
        P.step(pts, { color: GG.accent, width: P.s.line });
        if (ch) {
          P.vline(Math.log10(ch.threshold), { color: GG.text, label: `t* = ${ch.threshold.toFixed(3)}`, labelColor: GG.text });
          P.points([{ x: Math.log10(ch.threshold), y: ch.cost, tip: `t* ${ch.threshold.toFixed(3)}: ${ch.missed} missed, ${ch.false_alarms} false alarms` }], { color: GG.title, r: P.s.r * 1.2, ring: '#0a1019' });
        }
      },
    }],
  });
}

export function densityPlot(c, o = {}) {
  const D = c.density, tiers = ['SAFE', 'ELEVATED', 'CRITICAL'];
  const ymax = Math.max(...D.points.map((p) => p.density));
  return figure({
    ...o, title: 'P(critical) by true tier', subtitle: 'kernel density on log₁₀ P (Gaussian, bw.nrd0)', cap: [`${fmtN(D.n_below_floor)} predictions < 10⁻⁴ drawn at the floor`, srcCap(c)],
    legend: [{ title: 'True final tier', items: tiers.map((t) => ({ label: `${t.toLowerCase()} (${fmtN(D.counts[t])})`, glyph: 'fill', color: TIER_COL[t], opacity: 0.25, stroke: TIER_COL[t] })) }],
    rows: [{
      xs: cscale(Math.log10(D.floor), 0, { step: 1, expand: 0.03 }), ys: cscale(0, ymax, { n: 4 }), xTitle: 'Predicted P(critical), log scale', xFmt: pow10, yTitle: 'Density', h: (W) => W * 0.56,
      draw: (P) => {
        for (const t of tiers) {
          const v = D.points.filter((q) => q.tier === t).map((q) => [q.log10_p, 0, q.density]);
          P.area(v, { fill: TIER_COL[t], opacity: 0.14 });
          P.line(v.map((q) => [q[0], q[2]]), { color: TIER_COL[t], width: P.s.line * 0.85 });
        }
      },
    }],
  });
}

/** Archive miss histogram (final CDM ≤ 1 d before TCA) + live events with TCA in the next 24 h. */
export function missHistPlot(c, live = [], o = {}) {
  const H = c.miss_hist;
  const lv = live.filter((r) => fin(r.miss_m) && r.miss_m > 0);
  return figure({
    ...o, title: 'Miss distance, TCA within 24 h', subtitle: `archive: final CDM ≤ 1 d before TCA · ${fmtN(H.n)} events, ${fmtN(H.final_tiers?.CRITICAL)} critical`,
    cap: ['archive: train events (events.parquet)', 'live: latest CDM of cast events with TCA in the next 24 h'],
    legend: [
      { title: 'Source', items: [{ label: 'archive', glyph: 'bar', color: GG.barLine }, { label: `live (${lv.length})`, glyph: 'tri', color: GG.accent }] },
      { title: 'Screen', items: [{ label: '1 km critical', glyph: 'dashed', color: GG.bad }, { label: '5 km elevated', glyph: 'dashed', color: GG.warn }] },
    ],
    rows: [{
      xs: cscale(-2, 2, { step: 1, expand: 0.02 }), ys: cscale(0, Math.max(...H.bins.map((h) => h.n)), { n: 4 }), xTitle: 'Miss distance (km), log scale', xFmt: pow10, yTitle: 'Events',
      yFmt: (v) => Math.round(v).toLocaleString('en-US'), h: (W) => W * 0.56,
      draw: (P) => {
        for (const h of H.bins) if (h.n) P.rect(h.lo_log10_km + 0.006, h.hi_log10_km - 0.006, 0, h.n, { fill: GG.bar, stroke: GG.barLine, tip: `${fmtN(h.n)} events, ${pow10(h.lo_log10_km)}–${pow10(h.hi_log10_km)} km` });
        P.vline(0, { color: GG.bad, label: '1 km', opacity: 0.85 });
        P.vline(Math.log10(5), { color: GG.warn, label: '5 km', opacity: 0.85 });
        P.points(lv.map((r) => ({ x: clamp(Math.log10(r.miss_m / 1000), -2, 2), y: 0, tip: `${r.key}: ${fmtM(r.miss_m)}` })), { shape: 'tri', color: GG.accent, r: P.s.r * 1.4, dy: -5 * P.s.k });
      },
    }],
  });
}

// ═══ 5. Desk operating curve, dashboard verdict bar ═════════════════════════════════════════════════════
/** d: {curve:[{tau|threshold, recall, false_alarms}], tau, base {recall, false_alarms}} */
export function operatingCurve(d, o = {}) {
  const pts = (d.curve || []).map((p) => ({ x: +(p.tau ?? p.threshold), r: p.recall ?? p.crit_recall, fa: p.false_alarms ?? p.fa })).filter((p) => fin(p.x)).sort((a, z) => a.x - z.x);
  if (!pts.length) return emptyFig('Operating curve', 'Operating curve not available (/api/eval)', o);
  const xs = cscale(Math.min(...pts.map((p) => p.x)), Math.max(...pts.map((p) => p.x)), { n: 4, expand: 0.03 });
  const row = (key, strip, title, base, last, fmt) => {
    const vals = pts.filter((p) => fin(p[key])).map((p) => [p.x, p[key]]);
    const ys = vals.map((v) => v[1]).concat(fin(base) ? [base] : []);
    return {
      strip, xs, ys: cscale(Math.min(...ys), Math.max(...ys), { n: 3 }), xLabels: last, xTitle: 'Watch threshold τ on P(critical)', xFmt: f2d, yTitle: title, yFmt: fmt, h: (W) => W * 0.28,
      draw: (P) => {
        if (fin(base)) P.hline(base, { color: PERS, opacity: 0.8, label: 'persistence', side: 'left' });
        if (fin(d.tau)) P.vline(+d.tau, { color: GG.warn, dash: false, opacity: 0.9 });
        P.line(vals, { color: GG.accent, width: P.s.line * 0.9 });
        P.points(vals.map((v) => ({ x: v[0], y: v[1], tip: `τ ${v[0].toFixed(2)}: ${title.toLowerCase()} ${fmt ? fmt(v[1]) : v[1]}` })), { color: GG.accent, r: P.s.r * 0.8 });
      },
    };
  };
  return figure({
    ...o, title: 'Operating curve', subtitle: fin(d.tau) ? `your τ = ${(+d.tau).toFixed(2)}` : '', cap: 'measured out-of-fold (GET /api/eval)',
    legend: [{ title: 'Policy', items: [{ label: 'model', glyph: 'pointline', color: GG.accent }, ...(d.base ? [{ label: 'persistence', glyph: 'dashed', color: PERS }] : []), { label: 'your τ', glyph: 'line', color: GG.warn }] }],
    rows: [row('r', 'Recall of final critical', 'Recall', d.base?.recall, false, f2d), row('fa', 'False alarms', 'Events', d.base?.false_alarms, true, (v) => Math.round(v).toLocaleString('en-US'))],
  });
}

/** Dashboard verdicts as one stacked horizontal bar with an inline key. */
export function verdictBar(split, o = {}) {
  const s = sizes(o.base || 10), width = o.width || 300;
  const cats = [['AUTO', GG.good], ['ESCALATE', GG.warn], ['BLOCK', GG.bad]];
  const tot = cats.reduce((a, [c]) => a + (+(split?.[c] || 0)), 0);
  const xs = cscale(0, Math.max(1, tot), { n: 4, step: tot <= 5 ? 1 : undefined, expand: 0 });
  const h = Math.round(12 * s.k);
  const P = new Panel({ x: 2, y: 0, w: width - 6, h, xs, ys: cscale(0, 1, { expand: 0 }), s });
  let acc = 0;
  for (const [c, col] of cats) { const n = +(split?.[c] || 0); if (n) P.rect(acc, acc + n, 0, 1, { fill: col, opacity: 0.8, tip: `${c}: ${n}` }); acc += n; }
  const fr = frame(P, { xFmt: (v) => String(Math.round(v)), yLabels: false });
  let x = 2;
  const key = cats.map(([c, col]) => {
    const t = `${c} ${+(split?.[c] || 0)}`;
    const sv = `<rect x="${x}" y="${h + xAxisH(s) + 4}" width="${s.key - 3}" height="${s.key - 3}" fill="${col}" fill-opacity="0.8"/>` + text(x + s.key + 1, h + xAxisH(s) + 4 + (s.key - 3) / 2, t, { size: s.legendText, color: GG.muted, base: 'middle' });
    x += s.key + 14 * s.k + textW(t, s.legendText);
    return sv;
  }).join('');
  return wrapSvg(width, h + xAxisH(s) + s.key + 8, fr + P.svg() + key, [P]);
}
