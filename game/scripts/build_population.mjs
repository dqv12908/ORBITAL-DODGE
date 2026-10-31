#!/usr/bin/env node
// Build the background population for the globe from CelesTrak GP (OMM JSON).
// Output: public/population.f32.bin (N x 10 float32) + public/population.meta.json
//   per object: [a (Earth radii), e, i, raan, argp, M0 (rad at ref epoch), n (rad/min), raanDot, argpDot (rad/min), type]
//   type: 0 payload, 1 starlink, 2 debris, 3 rocket body, 4 station
// Display only. Kepler + J2 secular. NOT used for risk.
// Usage: node scripts/build_population.mjs [--from <dir with <group>.json>]
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const CACHE = path.join(HERE, '.cache');
const GROUPS = ['active', 'fengyun-1c-debris', 'cosmos-2251-debris', 'iridium-33-debris'];
const MU = 398600.4418, RE = 6378.137, J2 = 1.08262668e-3;

const fromIdx = process.argv.indexOf('--from');
const fromDir = fromIdx > 0 ? process.argv[fromIdx + 1] : process.env.CELESTRAK_DIR;
fs.mkdirSync(CACHE, { recursive: true });

function load(group) {
  const cached = path.join(CACHE, `${group}.json`);
  if (fromDir && fs.existsSync(path.join(fromDir, `${group}.json`))) {
    fs.copyFileSync(path.join(fromDir, `${group}.json`), cached);
  } else if (!fs.existsSync(cached) || process.argv.includes('--refresh')) {
    const url = `https://celestrak.org/NORAD/elements/gp.php?GROUP=${group}&FORMAT=json`;
    console.log('curl', url);
    execFileSync('curl', ['-sSfL', '--max-time', '120', '-o', cached, url], { stdio: 'inherit' });
  }
  return JSON.parse(fs.readFileSync(cached, 'utf8'));
}

const seen = new Set();
const rows = [];
let snapshotEpoch = 0;
const counts = { payload: 0, starlink: 0, debris: 0, rocket_body: 0, station: 0 };
for (const g of GROUPS) {
  let list;
  try { list = load(g); } catch (e) { console.warn('skip', g, e.message); continue; }
  for (const o of list) {
    if (seen.has(o.NORAD_CAT_ID)) continue;
    seen.add(o.NORAD_CAT_ID);
    const nRevDay = o.MEAN_MOTION;
    if (!(nRevDay > 0.5)) continue;
    const n = nRevDay * 2 * Math.PI / 1440; // rad/min
    const nSec = n / 60;
    const aKm = Math.cbrt(MU / (nSec * nSec));
    const e = o.ECCENTRICITY, i = o.INCLINATION * Math.PI / 180;
    const p = aKm * (1 - e * e);
    const k = 1.5 * J2 * (RE / p) ** 2 * n;
    const raanDot = -k * Math.cos(i);
    const argpDot = k * (2 - 2.5 * Math.sin(i) ** 2);
    const name = o.OBJECT_NAME || '';
    let type = 0;
    if (g.includes('debris') || / DEB/.test(name)) type = 2;
    else if (/R\/B/.test(name)) type = 3;
    else if (/STARLINK/.test(name)) type = 1;
    else if (/ISS \(ZARYA\)|CSS \(TIANHE\)|TIANGONG/.test(name)) type = 4;
    counts[['payload', 'starlink', 'debris', 'rocket_body', 'station'][type]]++;
    const ep = Date.parse(o.EPOCH + 'Z');
    snapshotEpoch = Math.max(snapshotEpoch, ep);
    rows.push({ a: aKm / RE, e, i, raan: o.RA_OF_ASC_NODE * Math.PI / 180, argp: o.ARG_OF_PERICENTER * Math.PI / 180,
      M: o.MEAN_ANOMALY * Math.PI / 180, n, raanDot, argpDot, type, ep });
  }
}
// Reference epoch = latest epoch rounded down to the hour. Propagate every object to it.
const ref = Math.floor(snapshotEpoch / 3600e3) * 3600e3;
const TAU = Math.PI * 2;
const buf = new Float32Array(rows.length * 10);
rows.forEach((r, k) => {
  const dt = (ref - r.ep) / 60000;
  const wrap = (x) => ((x % TAU) + TAU) % TAU;
  buf.set([r.a, r.e, r.i, wrap(r.raan + r.raanDot * dt), wrap(r.argp + r.argpDot * dt), wrap(r.M + r.n * dt),
    r.n, r.raanDot, r.argpDot, r.type], k * 10);
});
fs.writeFileSync(path.join(ROOT, 'public', 'population.f32.bin'), Buffer.from(buf.buffer));
const meta = {
  source: `CelesTrak GP groups ${GROUPS.join(', ')}`, built_utc: new Date().toISOString(),
  ref_epoch_utc: new Date(ref).toISOString(), n: rows.length, stride: 10,
  fields: ['a_re', 'e', 'i', 'raan', 'argp', 'M0', 'n_rad_min', 'raan_dot', 'argp_dot', 'type'],
  types: ['payload', 'starlink', 'debris', 'rocket_body', 'station'], counts,
  note: 'Display only: Kepler + J2 secular from mean elements. Not used for risk.',
};
fs.writeFileSync(path.join(ROOT, 'public', 'population.meta.json'), JSON.stringify(meta, null, 1));
console.log(meta);
