// Encounter-plane view and risk-evolution chart (2D canvas). Fully data-driven from server payloads.
import { fmtPc, fmtMiss } from '../store.js';

const COL = {
  bg: '#060a11', grid: 'rgba(140,170,210,0.08)', axis: 'rgba(140,170,210,0.3)', text: '#8796ab', dim: '#5f6f86',
  good: '#3ddc97', warn: '#ffb020', bad: '#ff4d5e', accent: '#35c9f0', info: '#5aa9ff', agent: '#a594ff', white: '#e8eef6',
};
const TIER = { SAFE: COL.good, ELEVATED: COL.warn, CRITICAL: COL.bad, PLANNED: COL.info };
const FONT = (px, w = 600) => `${Math.min(w, 600)} ${px}px 'JetBrains Mono', ui-monospace, monospace`;

function setup(canvas, w, hgt) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = hgt * dpr; canvas.style.width = w + 'px'; canvas.style.height = hgt + 'px'; }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, hgt);
  return g;
}
const lerp = (a, b, k) => a + (b - a) * k;
const easeOut = (k) => 1 - Math.pow(1 - k, 3);

/** Plane params {sx, sy, theta(rad), mx, my, hbr, log10, status, k} from geometry.encounter[i] or risk.pc */
export function planeParams(enc, risk, cdm) {
  const pc = risk?.pc || {};
  let sx, sy, th;
  if (enc && enc.c2_xx != null && enc.cov_m2 == null) enc = { ...enc, cov_m2: [[enc.c2_xx, enc.c2_xy ?? 0], [enc.c2_xy ?? 0, enc.c2_yy]] };
  if (enc?.cov_m2 || enc?.cov) {
    const [[a, b], [, c]] = enc.cov_m2 || enc.cov;
    const tr = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    sx = Math.sqrt(tr + d); sy = Math.sqrt(Math.max(tr - d, 0)); th = 0.5 * Math.atan2(2 * b, a - c);
  } else if (enc?.sigma_x_m || enc?.sigma_x) {
    sx = enc.sigma_x_m ?? enc.sigma_x; sy = enc.sigma_y_m ?? enc.sigma_y; th = ((enc.theta_deg ?? 0) * Math.PI) / 180;
  } else if (enc?.sigma_plane_m) {
    [sx, sy] = enc.sigma_plane_m; th = ((enc.theta_deg ?? 0) * Math.PI) / 180;
  } else if (pc.sigma_plane_m) {
    [sx, sy] = pc.sigma_plane_m; th = ((pc.theta_deg ?? 0) * Math.PI) / 180;
  }
  const miss = enc?.miss_xy_m || (enc?.miss_x != null ? [enc.miss_x, enc.miss_y] : null) || [cdm?.miss_m ?? 0, 0];
  return { sx, sy, th, mx: miss[0], my: miss[1], hbr: enc?.hbr_m ?? pc.hbr_m ?? 5, log10: pc.log10 ?? enc?.log10_pc, status: pc.status || enc?.pc_status || (sx ? 'OK' : '—'), k: pc.k_star ?? enc?.k_star, dilution: pc.dilution };
}

export class EncounterPlane {
  constructor(canvas, w = 348, hgt = 290) {
    this.c = canvas; this.w = w; this.h = hgt;
    this.cur = null; this.from = null; this.to = null; this.t0 = 0;
    this.tier = 'SAFE';
    this.dirty = true;
    const loop = () => { if (this.dirty || this.from) { this.dirty = false; this.#draw(); } requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
  set(params, tier) {
    this.tier = tier || 'SAFE';
    this.dirty = true;
    if (!params || !params.sx) { this.to = params; this.cur = params; this.from = null; return; }
    if (this.cur && this.cur.sx && (params.sx !== this.to?.sx || params.mx !== this.to?.mx)) { this.from = { ...this.cur }; this.t0 = performance.now(); }
    else if (!this.cur || !this.cur.sx) { this.cur = { ...params }; this.from = null; }
    this.to = params;
  }
  #draw() {
    const g = setup(this.c, this.w, this.h);
    const W = this.w, H = this.h;
    g.fillStyle = COL.bg; g.fillRect(0, 0, W, H);
    let p = this.to;
    if (this.from && p) {
      const k = Math.min(1, (performance.now() - this.t0) / 600), e = easeOut(k);
      p = { ...p };
      for (const f of ['sx', 'sy', 'mx', 'my']) p[f] = lerp(this.from[f], this.to[f], e);
      p.th = lerp(this.from.th, this.to.th, e);
      if (k >= 1) this.from = null;
    }
    this.cur = p;
    if (!p || !p.sx) {
      g.fillStyle = COL.dim; g.font = FONT(13, 500); g.textAlign = 'center';
      g.fillText(p?.status === 'COV_UNKNOWN' ? 'No covariance: Pc not computable (refused)' : 'Waiting for a CDM…', W / 2, H / 2);
      return;
    }
    const cx = W / 2, cy = H / 2 + 6;
    const ext = Math.max(3.2 * Math.max(p.sx, p.sy) * 0.55, Math.hypot(p.mx, p.my) * 1.35, 3 * p.hbr);
    const scale = (Math.min(W, H) / 2 - 18) / ext;
    // grid + scale bar
    const nice = (v) => { const e = 10 ** Math.floor(Math.log10(v)); return [1, 2, 5, 10].map((m) => m * e).find((x) => x >= v) || v; };
    const step = nice(ext / 3);
    g.strokeStyle = COL.grid; g.lineWidth = 1;
    for (let k = -6; k <= 6; k++) {
      const x = cx + k * step * scale, y = cy + k * step * scale;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
      g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
    }
    g.strokeStyle = COL.axis; g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, H); g.moveTo(0, cy); g.lineTo(W, cy); g.stroke();
    // ellipses 3,2,1 sigma around the chaser's relative position
    const col = TIER[this.tier] || COL.accent;
    const ex = cx + p.mx * scale, ey = cy - p.my * scale;
    [3, 2, 1].forEach((n, k) => {
      g.save(); g.translate(ex, ey); g.rotate(-p.th);
      g.beginPath(); g.ellipse(0, 0, Math.max(0.5, n * p.sx * scale), Math.max(0.5, n * p.sy * scale), 0, 0, Math.PI * 2);
      g.fillStyle = hexA(col, [0.08, 0.12, 0.2][k]); g.fill();
      g.strokeStyle = hexA(col, [0.45, 0.65, 0.95][k]); g.lineWidth = k === 2 ? 2 : 1.2; g.setLineDash(k === 0 ? [4, 4] : []); g.stroke();
      g.restore();
      g.fillStyle = hexA(col, 0.9); g.font = FONT(10, 700); g.textAlign = 'left';
    });
    // miss vector
    g.strokeStyle = COL.white; g.fillStyle = COL.white; g.lineWidth = 2;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(ex, ey); g.stroke();
    const a = Math.atan2(ey - cy, ex - cx);
    g.beginPath(); g.moveTo(ex, ey); g.lineTo(ex - 9 * Math.cos(a - 0.4), ey - 9 * Math.sin(a - 0.4)); g.lineTo(ex - 9 * Math.cos(a + 0.4), ey - 9 * Math.sin(a + 0.4)); g.closePath(); g.fill();
    g.font = FONT(12, 700); g.textAlign = 'center';
    g.fillText(`miss ${fmtMiss(Math.hypot(p.mx, p.my))}`, (cx + ex) / 2, (cy + ey) / 2 - 8);
    // hard-body circle at true scale (usually sub-pixel)
    g.strokeStyle = '#ffffff'; g.lineWidth = 1.5;
    g.beginPath(); g.arc(cx, cy, Math.max(1.5, p.hbr * scale), 0, Math.PI * 2); g.stroke();
    // magnifier inset around the origin
    const iw = 92, ix = 10, iy = H - iw - 10;
    const mag = Math.max(1, Math.round((iw * 0.22) / (p.hbr * scale)));
    g.save();
    g.fillStyle = 'rgba(8,10,13,0.92)'; g.strokeStyle = 'rgba(216,222,233,0.3)'; g.lineWidth = 1;
    roundRect(g, ix, iy, iw, iw, 8); g.fill(); g.stroke();
    g.beginPath(); roundRect(g, ix, iy, iw, iw, 8); g.clip();
    const icx = ix + iw / 2, icy = iy + iw / 2, is = scale * mag;
    g.save(); g.translate(icx + p.mx * is, icy - p.my * is); g.rotate(-p.th);
    g.beginPath(); g.ellipse(0, 0, p.sx * is, p.sy * is, 0, 0, Math.PI * 2); g.fillStyle = hexA(col, 0.18); g.fill(); g.strokeStyle = hexA(col, 0.9); g.stroke();
    g.restore();
    g.fillStyle = 'rgba(255,255,255,0.9)'; g.beginPath(); g.arc(icx, icy, p.hbr * is, 0, Math.PI * 2); g.fill();
    g.restore();
    g.fillStyle = COL.text; g.font = FONT(10, 700); g.textAlign = 'left';
    g.fillText(`HARD BODY ×${mag.toLocaleString()}`, ix + 6, iy - 5);
    g.fillStyle = COL.dim; g.font = FONT(10, 500);
    g.fillText(`r = ${p.hbr} m`, ix + 6, iy + iw - 6);
    // scale bar
    const sb = step * scale;
    g.strokeStyle = COL.text; g.lineWidth = 2; g.beginPath(); g.moveTo(W - 14 - sb, H - 14); g.lineTo(W - 14, H - 14); g.stroke();
    g.fillStyle = COL.text; g.font = FONT(11, 600); g.textAlign = 'right';
    g.fillText(fmtMiss(step), W - 14, H - 20);
    // Pc badge
    const label = `Pc ${fmtPc(p.log10, { status: p.status })}`;
    g.font = FONT(14, 800);
    const tw = g.measureText(label).width + 20;
    g.fillStyle = col; roundRect(g, 10, 10, tw, 26, 6); g.fill();
    g.fillStyle = this.tier === 'CRITICAL' ? '#fff' : '#1a1f2b'; g.textAlign = 'left'; g.fillText(label, 20, 28);
    g.fillStyle = COL.dim; g.font = FONT(10, 700);
    g.fillText('PHYSICS', 10 + tw + 8, 27);
    g.textAlign = 'right'; g.fillStyle = COL.text; g.font = FONT(11, 600);
    g.fillText(`σ ${fmtMiss(p.sx)} × ${fmtMiss(p.sy)}`, W - 10, 24);
    if (p.k != null) g.fillText(`k* ${(+p.k).toFixed(2)}${p.dilution ? ' · dilution' : ''}`, W - 10, 38);
  }
}

export class RiskChart {
  constructor(canvas, w = 348, hgt = 250) { this.c = canvas; this.w = w; this.h = hgt; this.data = null; }
  /** data: {points:[{ttca, log10, reported, miss_m, status}], nowTtca, futures, tier} */
  set(data) { this.data = data; this.#draw(); }
  redraw() { this.#draw(); }
  #draw() {
    const g = setup(this.c, this.w, this.h);
    const W = this.w, H = this.h, d = this.data;
    g.fillStyle = COL.bg; g.fillRect(0, 0, W, H);
    const L = 34, R = 10, T = 12, mainH = H * 0.66, B2 = H - 18;
    const xMax = Math.max(7, ...(d?.points || []).map((p) => p.ttca + 0.2));
    const X = (t) => L + ((xMax - t) / xMax) * (W - L - R);
    const lows = (d?.points || []).map((p) => p.log10).filter((v) => v != null && Number.isFinite(v));
    if (d?.futures?.quantiles?.p25) lows.push(...d.futures.quantiles.p25.map((v) => Math.max(v, -20)));
    const yMin = Math.max(-30, Math.min(-12, Math.floor(Math.min(...lows, -8) - 1)));
    const Y = (v) => T + ((0 - Math.max(yMin, Math.min(0, v))) / (0 - yMin)) * (mainH - T);
    // MCP band 0.5-1 d, Kelvins cut at 2 d
    g.fillStyle = 'rgba(255,176,32,0.08)'; g.fillRect(X(1), T, X(0.5) - X(1), B2 - T);
    g.fillStyle = COL.warn; g.font = FONT(9, 700); g.textAlign = 'center'; g.fillText('MCP', (X(1) + X(0.5)) / 2, T + 9);
    g.strokeStyle = 'rgba(53,201,240,0.6)'; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(2), T); g.lineTo(X(2), B2); g.stroke(); g.setLineDash([]);
    g.fillStyle = COL.accent; g.fillText('2 d cut', X(2), T + 9);
    // grid
    g.strokeStyle = COL.grid; g.lineWidth = 1; g.fillStyle = COL.dim; g.font = FONT(9, 600); g.textAlign = 'right';
    for (let v = 0; v >= yMin; v -= yMin < -16 ? 5 : 2) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke(); g.fillText(String(v), L - 4, Y(v) + 3); }
    g.textAlign = 'center';
    for (let t = 0; t <= xMax; t++) g.fillText(`${t}d`, X(t), H - 4);
    // thresholds
    const thr = (v, c, lbl) => { g.strokeStyle = c; g.lineWidth = 1.2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke(); g.setLineDash([]); g.fillStyle = c; g.textAlign = 'left'; g.font = FONT(9, 700); g.fillText(lbl, L + 3, Y(v) - 3); };
    thr(-4, COL.bad, '1e-4'); thr(-6, COL.warn, '1e-6');
    g.save(); g.translate(10, (T + mainH) / 2); g.rotate(-Math.PI / 2); g.fillStyle = COL.dim; g.font = FONT(9, 700); g.textAlign = 'center'; g.fillText('log10 Pc', 0, 0); g.restore();
    // forecast fan (statistics), drawn from the server's futures file only
    const f = d?.futures;
    if (f?.ttca_grid_d && f.quantiles) {
      const q = f.quantiles, xs = f.ttca_grid_d;
      const band = (lo, hi, a) => {
        if (!q[lo] || !q[hi]) return;
        g.beginPath();
        xs.forEach((t, k) => (k ? g.lineTo(X(t), Y(q[hi][k])) : g.moveTo(X(t), Y(q[hi][k]))));
        for (let k = xs.length - 1; k >= 0; k--) g.lineTo(X(xs[k]), Y(q[lo][k]));
        g.closePath(); g.fillStyle = `rgba(53,201,240,${a * 0.8})`; g.fill();
      };
      band('p05', 'p95', 0.18); band('p25', 'p75', 0.32);
      if (q.p50) { g.strokeStyle = '#7fd6ff'; g.lineWidth = 1.6; g.setLineDash([5, 3]); g.beginPath(); xs.forEach((t, k) => (k ? g.lineTo(X(t), Y(q.p50[k])) : g.moveTo(X(t), Y(q.p50[k])))); g.stroke(); g.setLineDash([]); }
      g.fillStyle = '#7fd6ff'; g.font = FONT(9, 700); g.textAlign = 'right';
      g.fillText(`FORECAST FAN p05–p95 (${f.n_paths || '?'} paths · statistics)`, W - R - 2, mainH - 4);
    }
    // physics line + reported dots
    const pts = (d?.points || []).slice().sort((a, b) => b.ttca - a.ttca);
    const col = TIER[d?.tier] || COL.accent;
    g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath();
    let first = true;
    for (const p of pts) { if (p.log10 == null || !Number.isFinite(p.log10)) continue; first ? g.moveTo(X(p.ttca), Y(p.log10)) : g.lineTo(X(p.ttca), Y(p.log10)); first = false; }
    g.stroke();
    for (const p of pts) {
      if (p.status === 'COV_UNKNOWN') { g.strokeStyle = COL.dim; g.lineWidth = 1.5; g.strokeRect(X(p.ttca) - 4, Y(yMin + 1) - 4, 8, 8); continue; }
      if (p.log10 != null) { g.fillStyle = col; g.beginPath(); g.arc(X(p.ttca), Y(p.log10), 3.5, 0, Math.PI * 2); g.fill(); }
      if (p.reported != null) { g.strokeStyle = COL.white; g.lineWidth = 1.2; g.beginPath(); g.arc(X(p.ttca), Y(p.reported), 5.5, 0, Math.PI * 2); g.stroke(); }
    }
    // miss sub-panel (log km): 1 km and 5 km lines
    const mT = mainH + 8, mB = B2;
    const MY = (m) => { const v = Math.log10(Math.max(0.01, m / 1000)); return mB - ((v + 2) / 3) * (mB - mT); }; // 10 m .. 10 km
    g.strokeStyle = COL.grid; g.strokeRect(L, mT, W - L - R, mB - mT);
    for (const [km, c] of [[1, COL.bad], [5, COL.warn]]) { g.strokeStyle = c; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(L, MY(km * 1000)); g.lineTo(W - R, MY(km * 1000)); g.stroke(); g.setLineDash([]); g.fillStyle = c; g.font = FONT(9, 700); g.textAlign = 'right'; g.fillText(`${km} km`, L - 3, MY(km * 1000) + 3); }
    g.strokeStyle = COL.white; g.lineWidth = 1.6; g.beginPath(); first = true;
    for (const p of pts) { if (p.miss_m == null) continue; first ? g.moveTo(X(p.ttca), MY(p.miss_m)) : g.lineTo(X(p.ttca), MY(p.miss_m)); first = false; }
    g.stroke();
    for (const p of pts) if (p.miss_m != null) { g.fillStyle = COL.white; g.beginPath(); g.arc(X(p.ttca), MY(p.miss_m), 2.5, 0, Math.PI * 2); g.fill(); }
    g.fillStyle = COL.dim; g.font = FONT(9, 700); g.textAlign = 'left'; g.fillText('miss', L + 3, mT + 10);
    // now cursor + future greyed (unreleased CDMs are never sent)
    if (d?.nowTtca != null && d.nowTtca < xMax) {
      const nx = X(Math.max(0, d.nowTtca));
      g.fillStyle = 'rgba(8,10,13,0.55)'; g.fillRect(nx, T, W - R - nx, B2 - T);
      g.strokeStyle = COL.accent; g.lineWidth = 1.5; g.beginPath(); g.moveTo(nx, T); g.lineTo(nx, B2); g.stroke();
      g.fillStyle = COL.accent; g.font = FONT(9, 800); g.textAlign = 'center'; g.fillText('NOW', nx, B2 + 9);
      if (W - R - nx > 40) { g.fillStyle = COL.dim; g.font = FONT(9, 600); g.fillText('future: not released', (nx + W - R) / 2, mT - 4); }
    }
  }
}

function roundRect(g, x, y, w, hh, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + hh, r); g.arcTo(x + w, y + hh, x, y + hh, r); g.arcTo(x, y + hh, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }
