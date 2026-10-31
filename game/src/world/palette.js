// Nord-based palette, lifted and saturated so objects read clearly under ACES (copied from the reference).
import * as THREE from 'three';

export const C = {
  n0: '#2f3542', n1: '#3d4556', n2: '#4a5367', n3: '#5b667d',
  s0: '#dfe4ec', s1: '#eaeef4', s2: '#f4f6f9',
  f0: '#8fcfc8', f1: '#7cc7dc', f2: '#7ea6d6', f3: '#4f7fc4',
  red: '#e0525f', orange: '#f0874e', yellow: '#f5c445', green: '#78c25f', purple: '#a77fd0',
  bg: '#1a1f2b', ink: '#242933', steel: '#a9b2c0', glass: '#a8d4ec', sky: '#c6dbe6',
};
export const LAMP = { green: '#5dff6e', amber: '#ffb52e', red: '#ff3448', blue: '#3fa4ff', off: '#20242d' };
export const TIER_COLOR = { SAFE: '#3ddc97', ELEVATED: '#ffb020', CRITICAL: '#ff3b4e', PLANNED: '#3fb6ff', UNKNOWN: '#9aa4b5', REFUSED: '#a98bd6' };

export function canvasTex(w, h, draw, { srgb = true, mip = true } = {}) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (!mip) { t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; }
  return t;
}

/** Thin crisp ring with a faint inner disc (event marker). */
export function ringTexture() {
  return canvasTex(256, 256, (g, w) => {
    const r = w / 2;
    g.strokeStyle = 'rgba(255,255,255,1)'; g.lineWidth = 7;
    g.beginPath(); g.arc(r, r, r * 0.78, 0, Math.PI * 2); g.stroke();
    g.strokeStyle = 'rgba(255,255,255,0.35)'; g.lineWidth = 3;
    g.beginPath(); g.arc(r, r, r * 0.92, 0, Math.PI * 2); g.stroke();
    const grd = g.createRadialGradient(r, r, 0, r, r, r * 0.78);
    grd.addColorStop(0, 'rgba(255,255,255,0.10)'); grd.addColorStop(1, 'rgba(255,255,255,0.02)');
    g.fillStyle = grd; g.beginPath(); g.arc(r, r, r * 0.76, 0, Math.PI * 2); g.fill();
    for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2; g.fillStyle = 'rgba(255,255,255,0.9)'; g.fillRect(r + Math.cos(a) * r * 0.86 - 4, r + Math.sin(a) * r * 0.86 - 4, 8, 8); }
  });
}

/** Soft radial glow sprite: a cheap stand-in for bloom. */
export function glowTexture() {
  return canvasTex(128, 128, (g, w) => {
    const r = w / 2;
    const grd = g.createRadialGradient(r, r, 0, r, r, r);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, w, w);
  });
}

export const matte = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.68, metalness: 0.02, ...extra });
export const metal = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.35, ...extra });
export const glow = (color, extra = {}) => new THREE.MeshBasicMaterial({ color, toneMapped: false, ...extra });
