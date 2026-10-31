// Background population: ~19.5k CelesTrak objects propagated on the GPU (Kepler + J2 secular).
// Display only, labelled "not used for risk" in the HUD.
import * as THREE from 'three';

export const POP_COLORS = ['#7fd6ff', '#5c7fae', '#ffab5c', '#ffd36b', '#ffffff'];

export async function createPopulation() {
  const [meta, buf] = await Promise.all([
    fetch('/population.meta.json').then((r) => r.json()),
    fetch('/population.f32.bin').then((r) => r.arrayBuffer()),
  ]);
  const data = new Float32Array(buf);
  const n = meta.n, stride = meta.stride;
  const ib = new THREE.InterleavedBuffer(data, stride);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0)); // dummy, shader ignores it
  g.setAttribute('aEl0', new THREE.InterleavedBufferAttribute(ib, 4, 0)); // a, e, i, raan
  g.setAttribute('aEl1', new THREE.InterleavedBufferAttribute(ib, 4, 4)); // argp, M0, n, raanDot
  g.setAttribute('aEl2', new THREE.InterleavedBufferAttribute(ib, 2, 8)); // argpDot, type
  g.setDrawRange(0, n);
  const uniforms = {
    uT: { value: 0 }, // minutes since ref epoch
    uRot: { value: 0 },
    uPx: { value: window.devicePixelRatio || 1 },
    uDim: { value: 1 },
    uColors: { value: POP_COLORS.map((c) => new THREE.Color(c)) },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: /* glsl */`
      attribute vec4 aEl0; attribute vec4 aEl1; attribute vec2 aEl2;
      uniform float uT; uniform float uPx; uniform vec3 uColors[5];
      varying vec3 vCol; varying float vA;
      const float TAU = 6.28318530718;
      void main() {
        float a = aEl0.x, e = aEl0.y, inc = aEl0.z;
        float raan = aEl0.w + aEl1.w * uT;
        float argp = aEl1.x + aEl2.x * uT;
        float M = mod(aEl1.y + aEl1.z * uT, TAU);
        float E = M + e * sin(M);
        for (int k = 0; k < 4; k++) E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
        float xp = a * (cos(E) - e);
        float yp = a * sqrt(max(1.0 - e * e, 0.0)) * sin(E);
        float cO = cos(raan), sO = sin(raan), cw = cos(argp), sw = sin(argp), ci = cos(inc), si = sin(inc);
        float x1 = cw * xp - sw * yp, y1 = sw * xp + cw * yp;
        vec3 eci = vec3(cO * x1 - sO * ci * y1, sO * x1 + cO * ci * y1, si * y1);
        vec3 p = vec3(eci.x, eci.z, -eci.y);
        int t = int(aEl2.y + 0.5);
        vCol = uColors[t];
        float size = t == 1 ? 1.3 : (t == 2 ? 1.8 : (t == 4 ? 3.5 : 1.7));
        vA = t == 1 ? 0.45 : 0.85;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = size * uPx * clamp(6.0 / -mv.z, 0.7, 2.0);
        // fade points behind the Earth's limb a little (depth test hides them; this softens the edge)
      }`,
    fragmentShader: /* glsl */`
      uniform float uDim; varying vec3 vCol; varying float vA;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float r = length(c);
        if (r > 0.5) discard;
        float core = smoothstep(0.5, 0.0, r);
        float a = core * core * vA * uDim;
        gl_FragColor = vec4(vCol * 1.4, a);
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(g, mat);
  points.frustumCulled = false;
  points.renderOrder = 2;
  return { points, uniforms, meta, refEpochMs: Date.parse(meta.ref_epoch_utc) };
}
