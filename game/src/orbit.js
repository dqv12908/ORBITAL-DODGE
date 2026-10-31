// Display-only orbital helpers (float64 two-body). The browser never computes risk.
// Frames: ECI (x, y, z) maps to scene (x, z, -y): scene +Y is the Earth's north pole.
// 1 scene unit = 1 Earth radius (6378.137 km).

export const RE_KM = 6378.137;
export const MU = 398600.4418; // km^3/s^2
const D2R = Math.PI / 180;
const TAU = Math.PI * 2;

/** Greenwich mean sidereal time (rad) for a JS time in ms. */
export function gmst(ms) {
  const jd = ms / 86400000 + 2440587.5;
  const T = (jd - 2451545.0) / 36525;
  let g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T;
  g = ((g % 360) + 360) % 360;
  return g * D2R;
}

/** Unit vector to the Sun in ECI (low precision, fine for lighting). */
export function sunEci(ms) {
  const n = ms / 86400000 + 2440587.5 - 2451545.0;
  const L = (280.46 + 0.9856474 * n) * D2R;
  const g = (357.528 + 0.9856003 * n) * D2R;
  const lam = L + (1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * D2R;
  const eps = 23.439 * D2R;
  return [Math.cos(lam), Math.cos(eps) * Math.sin(lam), Math.sin(eps) * Math.sin(lam)];
}

export const eciToScene = (v) => [v[0], v[2], -v[1]];

function solveKepler(M, e) {
  let E = e < 0.8 ? M : Math.PI;
  for (let k = 0; k < 12; k++) {
    const d = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= d;
    if (Math.abs(d) < 1e-12) break;
  }
  return E;
}
const trueToMean = (nu, e) => {
  const E = 2 * Math.atan2(Math.sqrt(1 - e) * Math.sin(nu / 2), Math.sqrt(1 + e) * Math.cos(nu / 2));
  return E - e * Math.sin(E);
};

/**
 * Normalise an element set from the server (or the fixture) into radians/km.
 * Accepts {a_km, e, i_deg, raan_deg, argp_deg, nu_deg_at_tca} (contract) plus a few aliases.
 */
export function normElements(o) {
  if (!o) return null;
  const a = o.a_km ?? o.sma_km ?? o.a ?? null;
  if (!a) return null;
  const e = o.e ?? o.ecc ?? 0;
  const i = (o.i_deg ?? o.inc_deg ?? 0) * D2R;
  const raan = (o.raan_deg ?? 0) * D2R;
  const argp = (o.argp_deg ?? 0) * D2R;
  const nu = (o.nu_deg_at_tca ?? o.nu_deg ?? 0) * D2R;
  return { a, e, i, raan, argp, M0: trueToMean(nu, e), n: Math.sqrt(MU / (a * a * a)) };
}

/** ECI position (km) at dt seconds after the element epoch (TCA). */
export function propagate(el, dtSec) {
  const M = el.M0 + el.n * dtSec;
  const E = solveKepler(((M % TAU) + TAU) % TAU, el.e);
  const cosE = Math.cos(E), sinE = Math.sin(E);
  const xp = el.a * (cosE - el.e);
  const yp = el.a * Math.sqrt(1 - el.e * el.e) * sinE;
  const cO = Math.cos(el.raan), sO = Math.sin(el.raan);
  const cw = Math.cos(el.argp), sw = Math.sin(el.argp);
  const ci = Math.cos(el.i), si = Math.sin(el.i);
  const x1 = cw * xp - sw * yp, y1 = sw * xp + cw * yp;
  return [cO * x1 - sO * ci * y1, sO * x1 + cO * ci * y1, si * y1];
}

/** Scene-unit position (Earth radii) at dtSec from TCA. */
export function scenePos(el, dtSec) {
  const p = propagate(el, dtSec);
  return [p[0] / RE_KM, p[2] / RE_KM, -p[1] / RE_KM];
}
export const period = (el) => TAU / el.n;

/**
 * Circular-ish orbit that passes through (lat, right ascension) at TCA.
 * branch 0 = ascending pass, 1 = descending pass. Display-only reconstruction (11 §E style).
 */
export function elementsThroughPoint({ a_km, e = 0, i_deg, lat_deg, ra_deg, branch = 0 }) {
  const i = i_deg * D2R, lat = lat_deg * D2R;
  const s = Math.max(-1, Math.min(1, Math.sin(lat) / Math.max(1e-6, Math.sin(i))));
  let u = Math.asin(s);
  if (branch) u = Math.PI - u;
  const dra = Math.atan2(Math.cos(i) * Math.sin(u), Math.cos(u));
  const raan = ra_deg * D2R - dra;
  return { a_km, e, i_deg, raan_deg: raan / D2R, argp_deg: 0, nu_deg_at_tca: u / D2R };
}

/** Scene position of a lat/lon/alt point fixed to the Earth at sim time ms. */
export function latLonToScene(latDeg, lonDeg, altKm, ms) {
  const r = 1 + (altKm || 0) / RE_KM;
  const lat = latDeg * D2R, lon = lonDeg * D2R + gmst(ms);
  const eci = [r * Math.cos(lat) * Math.cos(lon), r * Math.cos(lat) * Math.sin(lon), r * Math.sin(lat)];
  return eciToScene(eci);
}
