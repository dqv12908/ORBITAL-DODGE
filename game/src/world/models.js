// Procedural spacecraft and debris (PBR). Local frame: +Z nadir, +Y along-track, ±X solar-array wings.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';

let cache = null;
function textures() {
  if (cache) return cache;
  // crinkled MLI foil: noisy height -> normal map
  const N = 256, c = document.createElement('canvas'); c.width = c.height = N;
  const g = c.getContext('2d');
  const hgt = new Float32Array(N * N);
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  for (let k = 0; k < 90; k++) { // creases
    const x0 = rnd() * N, y0 = rnd() * N, a = rnd() * Math.PI, len = 30 + rnd() * 120, w = 1 + rnd() * 3, amp = (rnd() - 0.5) * 2;
    for (let t = 0; t < len; t++) {
      const x = Math.floor(x0 + Math.cos(a) * t) & (N - 1), y = Math.floor(y0 + Math.sin(a) * t) & (N - 1);
      for (let d = -w; d <= w; d++) hgt[((y + d) & (N - 1)) * N + x] += amp * (1 - Math.abs(d) / (w + 1));
    }
  }
  const img = g.createImageData(N, N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const hx = hgt[y * N + ((x + 1) & (N - 1))] - hgt[y * N + ((x - 1) & (N - 1))];
    const hy = hgt[((y + 1) & (N - 1)) * N + x] - hgt[((y - 1) & (N - 1)) * N + x];
    const nx = -hx * 0.6, ny = -hy * 0.6, nz = 1, l = Math.hypot(nx, ny, nz);
    const i = (y * N + x) * 4;
    img.data[i] = (nx / l * 0.5 + 0.5) * 255; img.data[i + 1] = (ny / l * 0.5 + 0.5) * 255; img.data[i + 2] = (nz / l * 0.5 + 0.5) * 255; img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const foilN = new THREE.CanvasTexture(c);
  foilN.wrapS = foilN.wrapT = THREE.RepeatWrapping;
  // solar array: dark blue cells, silver interconnects
  const a = document.createElement('canvas'); a.width = 512; a.height = 256;
  const q = a.getContext('2d');
  q.fillStyle = '#0b1a3a'; q.fillRect(0, 0, 512, 256);
  const cw = 512 / 16, ch = 256 / 6;
  for (let i = 0; i < 16; i++) for (let j = 0; j < 6; j++) {
    const grd = q.createLinearGradient(i * cw, j * ch, (i + 1) * cw, (j + 1) * ch);
    grd.addColorStop(0, '#16336e'); grd.addColorStop(1, '#0d2150');
    q.fillStyle = grd; q.fillRect(i * cw + 1.5, j * ch + 1.5, cw - 3, ch - 3);
    q.strokeStyle = 'rgba(160,180,220,0.18)'; q.lineWidth = 0.6;
    for (let k = 1; k < 4; k++) { q.beginPath(); q.moveTo(i * cw + 2, j * ch + (k * ch) / 4); q.lineTo((i + 1) * cw - 2, j * ch + (k * ch) / 4); q.stroke(); }
  }
  q.strokeStyle = '#b8c2d0'; q.lineWidth = 2;
  q.strokeRect(1, 1, 510, 254);
  const panel = new THREE.CanvasTexture(a);
  panel.colorSpace = THREE.SRGBColorSpace; panel.anisotropy = 8;
  cache = { foilN, panel };
  return cache;
}

export function createSpacecraft(glowTex) {
  const { foilN, panel } = textures();
  const g = new THREE.Group();
  const gold = new THREE.MeshStandardMaterial({ color: '#d4a64a', metalness: 1.0, roughness: 0.32, normalMap: foilN, normalScale: new THREE.Vector2(0.9, 0.9) });
  const silver = new THREE.MeshStandardMaterial({ color: '#c9ced6', metalness: 0.9, roughness: 0.28 });
  const white = new THREE.MeshStandardMaterial({ color: '#e9ecef', metalness: 0.05, roughness: 0.55 });
  const dark = new THREE.MeshStandardMaterial({ color: '#2a2f38', metalness: 0.4, roughness: 0.5 });
  const cells = new THREE.MeshStandardMaterial({ map: panel, metalness: 0.35, roughness: 0.28, color: '#ffffff' });
  const back = new THREE.MeshStandardMaterial({ color: '#c8ccd2', metalness: 0.2, roughness: 0.7 });
  const bus = new THREE.Mesh(new RoundedBoxGeometry(1.0, 1.0, 1.4, 2, 0.04), gold);
  g.add(bus);
  const top = new THREE.Mesh(new THREE.BoxGeometry(1.02, 1.02, 0.08), white); top.position.z = -0.72; g.add(top);
  // instrument on the nadir face
  const inst = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.26, 0.35, 24), dark);
  inst.rotation.x = Math.PI / 2; inst.position.set(0.18, 0.1, 0.86); g.add(inst);
  const lens = new THREE.Mesh(new THREE.CircleGeometry(0.17, 24), new THREE.MeshStandardMaterial({ color: '#10141c', metalness: 0.9, roughness: 0.08 }));
  lens.position.set(0.18, 0.1, 1.04); g.add(lens);
  // high-gain antenna (parabolic dish) on a boom
  const pts = []; for (let k = 0; k <= 12; k++) { const r = (k / 12) * 0.38; pts.push(new THREE.Vector2(r, r * r * 1.4)); }
  const dish = new THREE.Mesh(new THREE.LatheGeometry(pts, 32), new THREE.MeshStandardMaterial({ color: '#f1f3f5', metalness: 0.1, roughness: 0.4, side: THREE.DoubleSide }));
  dish.rotation.x = -Math.PI / 2; dish.position.set(-0.25, -0.45, 1.25); g.add(dish);
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.5, 8), silver);
  boom.rotation.x = Math.PI / 2; boom.position.set(-0.25, -0.45, 0.95); g.add(boom);
  // solar array wings along X
  for (const sgn of [-1, 1]) {
    const yoke = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.55, 8), silver);
    yoke.rotation.z = Math.PI / 2; yoke.position.x = sgn * 0.77; g.add(yoke);
    for (let k = 0; k < 2; k++) {
      const p = new THREE.Group();
      const front = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.95), cells);
      const rear = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.95), back);
      front.position.z = -0.006; rear.rotation.y = Math.PI; rear.position.z = 0.006;
      p.add(front, rear);
      p.rotation.x = Math.PI / 2; // face the panels away from nadir (towards -Z side lit by the sun most of the time)
      p.position.x = sgn * (1.1 + 0.77 + k * 1.53);
      g.add(p);
    }
  }
  const glint = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: new THREE.Color(1.6, 1.5, 1.3), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
  glint.scale.setScalar(2.2); g.add(glint);
  g.scale.setScalar(0.012);
  return g;
}

export function createDebris(type, glowTex) {
  const g = new THREE.Group();
  const t = String(type || 'DEBRIS').toUpperCase();
  const alu = new THREE.MeshStandardMaterial({ color: '#9aa1aa', metalness: 0.85, roughness: 0.42, flatShading: true });
  const burnt = new THREE.MeshStandardMaterial({ color: '#4a4440', metalness: 0.6, roughness: 0.6, flatShading: true });
  const foil = new THREE.MeshStandardMaterial({ color: '#c79a45', metalness: 1, roughness: 0.38, flatShading: true, side: THREE.DoubleSide });
  if (t.includes('ROCKET')) {
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 2.4, 28, 1, true), new THREE.MeshStandardMaterial({ color: '#d9dcdf', metalness: 0.5, roughness: 0.5, side: THREE.DoubleSide }));
    g.add(body);
    const noz = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.4, 0.5, 24, 1, true), burnt); noz.position.y = -1.45; g.add(noz);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.04, 8, 32), alu); ring.rotation.x = Math.PI / 2; ring.position.y = 1.2; g.add(ring);
  } else {
    // irregular fragment: jagged plate + twisted shard + foil scrap
    let s = 3;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const geo = new THREE.IcosahedronGeometry(0.8, 1);
    const p = geo.attributes.position;
    for (let k = 0; k < p.count; k++) {
      const v = new THREE.Vector3().fromBufferAttribute(p, k);
      const n = 0.55 + rnd() * 0.75;
      v.multiply(new THREE.Vector3(1.4 * n, 0.35 * n, 0.9 * n));
      p.setXYZ(k, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    g.add(new THREE.Mesh(geo, t.includes('PAYLOAD') ? alu : alu));
    const shard = new THREE.Mesh(new THREE.TetrahedronGeometry(0.6, 0), burnt);
    shard.scale.set(1.6, 0.4, 0.7); shard.position.set(0.9, 0.15, 0.2); shard.rotation.set(0.4, 0.8, 0.2); g.add(shard);
    const scrap = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.5, 4, 2), foil);
    const sp = scrap.geometry.attributes.position;
    for (let k = 0; k < sp.count; k++) sp.setZ(k, (rnd() - 0.5) * 0.25);
    scrap.geometry.computeVertexNormals();
    scrap.position.set(-0.8, 0.2, -0.3); scrap.rotation.set(0.6, 0.3, 1.1); g.add(scrap);
  }
  const glint = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: new THREE.Color(1.6, 1.15, 0.8), transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending }));
  glint.scale.setScalar(2.0); g.add(glint);
  g.scale.setScalar(0.012);
  return g;
}
