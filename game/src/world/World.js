// The 3D world: stylised Earth, today's sky, the focal conjunction pair, event markers.
// Narrow API used by the Director only (reference pattern). No risk logic here.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { createEarth, createSky } from './earth.js';
import { createSpacecraft, createDebris } from './models.js';
import { createPopulation } from './population.js';
import { TIER_COLOR, ringTexture, glowTexture } from './palette.js';
import { gmst, sunEci, eciToScene, normElements, scenePos, period, latLonToScene, RE_KM } from '../orbit.js';

const FOV = 32;
const HOME_DIST = 7.2;
const ease = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

export class World {
  constructor(el) {
    this.el = el;
    this.simMs = Date.parse('2026-10-31T06:00:00Z');
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    el.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.imageRendering = 'auto';

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color('#010205');
    try {
      const pm = new THREE.PMREMGenerator(this.renderer);
      scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
      scene.environmentIntensity = 0.35;
    } catch (e) { /* optional */ }

    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 600);
    this.camera.position.set(0, 1.6, HOME_DIST);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, rotateSpeed: 0.45, zoomSpeed: 0.9, panSpeed: 0.8,
      minDistance: 1.12, maxDistance: 30, enablePan: false, autoRotate: true, autoRotateSpeed: 0.25 });
    this.userMoved = false;
    this.controls.addEventListener('start', () => { this.userMoved = true; this.fly = null; this.controls.autoRotate = false; });

    // Lights: the Sun + faint earthshine; metals also reflect a dim environment
    this.sun = new THREE.DirectionalLight('#fff6ea', 3.2);
    scene.add(this.sun, this.sun.target);
    scene.add(new THREE.AmbientLight('#8fa6c8', 0.06));
    const sky = createSky();
    this.sky = sky;
    scene.add(sky.group);
    // post: bloom on the Sun, city lights and alerts only (high threshold)
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.5, 0.4, 1.0);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.inertial = new THREE.Group(); // ECI-fixed things (orbits, focal pair, markers)
    scene.add(this.inertial);
    this.ringTex = ringTexture();
    this.glowTex = glowTexture();
    this.markers = new Map();
    this.bursts = [];
    this.live = [];
    this.liveGroup = new THREE.Group();
    this.liveGroup.visible = false;
    scene.add(this.liveGroup);
    this.focal = null;
    this.anchors = {};
    this.lineMats = [];
    this.clock = new THREE.Clock();
    this._t = 0;
    this._frames = []; this._fpsAt = 0; this.fps = 60;

    this.ready = this.#init();
    this.#bindPick();
    window.addEventListener('resize', () => this.#resize());
    this.#resize();
    this.renderer.setAnimationLoop(() => this.#tick());
  }

  async #init() {
    const e = await createEarth(this.renderer);
    this.earthApi = e;
    this.scene.add(e.group);
    this.#intro();
    try {
      this.pop = await createPopulation();
      this.scene.add(this.pop.points);
    } catch (err) { console.warn('[world] population unavailable', err); }
  }

  #intro() {
    // swoop from dist x1.55, yaw +0.45 (reference) over 2.2 s
    const end = new THREE.Vector3(1.6, 1.9, 6.6).setLength(HOME_DIST);
    const start = end.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.45).multiplyScalar(1.55);
    start.y += 2.2;
    this.camera.position.copy(start);
    this.flyTo({ pos: end, target: new THREE.Vector3(0, 0, 0) }, 2.2);
  }

  #resize() {
    const w = this.el.clientWidth || window.innerWidth, h = this.el.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.composer?.setPixelRatio(this.pixelRatio);
    this.composer?.setSize(w, h);
    this.bloom?.setSize(Math.round(w * this.pixelRatio / 2), Math.round(h * this.pixelRatio / 2)); // half-res bloom
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const m of this.lineMats) m.resolution.set(w, h);
    this._rect = this.renderer.domElement.getBoundingClientRect();
  }

  // ─── camera ──────────────────────────────────────────────────────────
  /** Fly camera to {pos, target} with easeInOutCubic; position interpolated on a sphere about the origin. */
  flyTo({ pos, target }, dur = 1.6) {
    this.fly = {
      t: 0, dur,
      p0: this.camera.position.clone(), p1: pos.clone(),
      t0: this.controls.target.clone(), t1: target.clone(),
    };
  }
  flyHome(dur = 1.6) {
    const dir = this.camera.position.clone().normalize();
    if (Math.abs(dir.y) > 0.8) dir.set(0.3, 0.35, 0.88).normalize();
    this.flyTo({ pos: dir.multiplyScalar(HOME_DIST), target: new THREE.Vector3() }, dur);
    this.userMoved = false;
  }
  /** Frame a conjunction point: camera above P, off to the side, Earth limb in view. */
  flyToPoint(p, { dist = 1.1, dur = 1.6 } = {}) {
    const P = p.clone();
    const up = P.clone().normalize();
    // look from the sunlit side when we can (cinematic, readable)
    const sunDir = this.sun.position.clone().normalize();
    let side = sunDir.clone().sub(up.clone().multiplyScalar(sunDir.dot(up)));
    if (side.lengthSq() < 1e-3) side = new THREE.Vector3().crossVectors(up, new THREE.Vector3(0, 1, 0));
    if (side.lengthSq() < 1e-4) side.set(1, 0, 0);
    side.normalize();
    const camPos = P.clone().add(up.clone().multiplyScalar(dist * 0.85)).add(side.multiplyScalar(dist * 0.55));
    this.flyTo({ pos: camPos, target: P.clone().multiplyScalar(0.94) }, dur);
  }
  focusFocal(opts) { if (this.focal) this.flyToPoint(this.focal.P, opts); }

  // ─── time ────────────────────────────────────────────────────────────
  setSimTime(ms) { this.simMs = ms; }

  // ─── focal conjunction pair ──────────────────────────────────────────
  /**
   * geom: {event_key, tca_ms, target: elements, chaser: elements, chaser_type, tier}
   * Elements are reconstructed (RECONSTRUCTED GEOMETRY): display only.
   */
  setFocal(geom) {
    this.clearFocal();
    const t = normElements(geom.target), c = normElements(geom.chaser);
    if (!t || !c) return;
    const g = new THREE.Group();
    const mkTrail = (el, color, dashed, width) => {
      const T = period(el), N = 360, pts = [];
      for (let k = 0; k <= N; k++) pts.push(...scenePos(el, -T / 2 + (k / N) * T));
      const geo = new LineGeometry(); geo.setPositions(pts);
      const m = new LineMaterial({ color, linewidth: width, dashed, dashSize: 0.02, gapSize: 0.012, transparent: true, opacity: 0.85 });
      m.resolution.set(this.el.clientWidth, this.el.clientHeight);
      this.lineMats.push(m);
      const line = new Line2(geo, m); line.computeLineDistances();
      return line;
    };
    g.add(mkTrail(t, '#bfe9ff', false, 1.4));
    g.add(mkTrail(c, '#ffb070', true, 1.4));
    // red arc +-2 min around TCA on the focal (target) orbit
    const arcPts = [];
    for (let k = 0; k <= 40; k++) arcPts.push(...scenePos(t, -120 + k * 6));
    const ag = new LineGeometry(); ag.setPositions(arcPts);
    this.arcMat = new LineMaterial({ color: new THREE.Color(TIER_COLOR[geom.tier] || TIER_COLOR.SAFE).multiplyScalar(1.6), linewidth: 3, transparent: true, opacity: 1 });
    this.arcMat.resolution.set(this.el.clientWidth, this.el.clientHeight);
    this.lineMats.push(this.arcMat);
    g.add(new Line2(ag, this.arcMat));

    const sat = createSpacecraft(this.glowTex);
    const deb = createDebris(geom.chaser_type, this.glowTex);
    g.add(sat, deb);
    const P = V(scenePos(t, 0));
    const ring = this.#ringSprite(geom.tier, 0.07);
    ring.position.copy(P);
    g.add(ring);
    // translucent covariance ellipsoid at TCA (exaggerated; the 2D encounter-plane panel is the precise view)
    const vt = V(scenePos(t, 1)).sub(V(scenePos(t, -1)));
    const vc = V(scenePos(c, 1)).sub(V(scenePos(c, -1)));
    const vrel = vt.clone().sub(vc).normalize();
    const radial = P.clone().normalize();
    const ex = radial.clone().sub(vrel.clone().multiplyScalar(radial.dot(vrel))).normalize();
    const ey = new THREE.Vector3().crossVectors(vrel, ex);
    const cov = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.16, depthWrite: false }));
    const covWire = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.12, wireframe: true, depthWrite: false }));
    cov.add(covWire);
    cov.position.copy(P);
    cov.userData.basis = new THREE.Matrix4().makeBasis(ex, ey, vrel);
    cov.visible = false;
    g.add(cov);
    this.inertial.add(g);
    this.focal = { key: geom.event_key, t, c, g, sat, deb, ring, P, cov, tcaMs: geom.tca_ms, tier: geom.tier };
    this.setFocalTier(geom.tier);
  }
  /** Covariance ellipsoid from encounter-plane sigmas (m) and angle (rad). Exaggerated to be visible; returns the factor. */
  setCovariance(p) {
    const f = this.focal; if (!f) return null;
    if (!p || !p.sx) { f.cov.visible = false; this.covInfo = null; return null; }
    const target = 0.03; // scene units for the 3-sigma major axis
    const k = target / (3 * Math.max(p.sx, p.sy) / (RE_KM * 1000));
    const toU = (m) => (3 * m * k) / (RE_KM * 1000);
    const rot = new THREE.Matrix4().makeRotationZ(p.th || 0);
    const m = f.cov.userData.basis.clone().multiply(rot);
    f.cov.quaternion.setFromRotationMatrix(m);
    f.cov.scale.set(toU(p.sx), Math.max(toU(p.sy), 0.0015), Math.max(toU(Math.min(p.sx, p.sy)) * 0.35, 0.001));
    f.cov.visible = true;
    this.covInfo = { factor: k };
    return k;
  }
  clearFocal() {
    if (!this.focal) return;
    this.inertial.remove(this.focal.g);
    this.focal.g.traverse((o) => { o.geometry?.dispose?.(); });
    this.lineMats = this.lineMats.filter((m) => m !== this.arcMat && this.focal.g.children.every((ch) => ch.material !== m));
    this.focal = null;
  }
  setFocalTier(tier) {
    if (!this.focal) return;
    this.focal.tier = tier;
    const col = new THREE.Color(TIER_COLOR[tier] || TIER_COLOR.SAFE);
    this.focal.ring.material.color.copy(col).multiplyScalar(1.5);
    this.arcMat?.color.copy(col).multiplyScalar(1.6);
    this.focal.cov?.material.color.copy(col);
    this.focal.cov?.children[0]?.material.color.copy(col);
  }

  #ringSprite(tier, size) {
    const m = new THREE.SpriteMaterial({ map: this.ringTex, color: TIER_COLOR[tier] || TIER_COLOR.SAFE, transparent: true, depthWrite: false, depthTest: true, blending: THREE.AdditiveBlending });
    const s = new THREE.Sprite(m);
    s.scale.setScalar(size);
    s.userData.base = size;
    s.renderOrder = 5;
    return s;
  }

  // ─── event markers (cast) ────────────────────────────────────────────
  /** list: [{key, pos:[x,y,z] scene units, tier, focal}] */
  setMarkers(list) {
    const keep = new Set(list.map((m) => m.key));
    for (const [k, m] of this.markers) if (!keep.has(k)) { this.inertial.remove(m.sprite); this.markers.delete(k); }
    for (const it of list) {
      let m = this.markers.get(it.key);
      if (!m) {
        const sprite = this.#ringSprite(it.tier, 0.05);
        const dot = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glowTex, color: '#ffffff', transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
        dot.scale.setScalar(0.18); sprite.add(dot);
        sprite.userData.key = it.key;
        this.inertial.add(sprite);
        m = { sprite, dot, tier: it.tier, phase: Math.random() * 6 };
        this.markers.set(it.key, m);
      }
      m.sprite.position.set(...it.pos);
      this.setMarkerTier(it.key, it.tier);
      m.sprite.visible = !it.focal;
    }
  }
  setMarkerTier(key, tier) {
    const m = this.markers.get(key);
    if (m) { m.tier = tier; m.sprite.material.color.set(TIER_COLOR[tier] || TIER_COLOR.SAFE); m.dot.material.color.set(TIER_COLOR[tier] || '#fff'); }
    if (this.focal && this.focal.key === key) this.setFocalTier(tier);
  }
  markerPos(key) { return this.markers.get(key)?.sprite.position.clone() || null; }

  /** Expanding ring burst at a world position (reference event rings). */
  burst(pos, color = '#7dff8a', { from = 0.05, to = 0.5, dur = 0.9 } = {}) {
    const s = this.#ringSprite('SAFE', from);
    s.material.color.set(color);
    s.position.copy(pos);
    this.inertial.add(s);
    this.bursts.push({ s, t: 0, from, to, dur });
  }
  burstFocal(color) { if (this.focal) this.burst(this.focal.P, color, { from: 0.08, to: 0.9, dur: 1.1 }); }

  // ─── live sky ────────────────────────────────────────────────────────
  setLiveSky(list) {
    this.liveGroup.clear();
    this.live = (list || []).map((c, k) => {
      // colour by screened miss distance (max Pc is a worst case without covariance, so it does not drive colour)
      const r = c.min_range_km ?? 99;
      const col = r < 0.05 ? TIER_COLOR.CRITICAL : r < 0.2 ? TIER_COLOR.ELEVATED : '#35c9f0';
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.ringTex, color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
      s.scale.setScalar(0.035);
      const d = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glowTex, color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
      d.scale.setScalar(0.3); s.add(d);
      s.userData.key = 'live:' + k;
      this.liveGroup.add(s);
      return { s, c };
    });
  }
  setLiveSkyVisible(v) { this.liveGroup.visible = !!v; }
  livePos(k) { return this.live[k]?.s.position.clone(); }
  setPopulationDim(v) { if (this.pop) this.pop.uniforms.uDim.value = v; }

  // ─── picking + screen anchoring ──────────────────────────────────────
  onPick(cb) { this._onPick = cb; }
  #bindPick() {
    const ray = new THREE.Raycaster();
    let down = null;
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    dom.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
      const r = dom.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      // markers are small: pick by screen distance instead of raycasting sprites
      let best = null, bd = 22;
      const test = (key, pos) => {
        const sp = this.project(pos); if (!sp) return;
        const d = Math.hypot(sp.x - (e.clientX - r.left), sp.y - (e.clientY - r.top));
        if (d < bd) { bd = d; best = key; }
      };
      for (const [k, m] of this.markers) if (m.sprite.visible) test(k, m.sprite.getWorldPosition(new THREE.Vector3()));
      if (this.focal) test(this.focal.key, this.focal.P);
      if (this.liveGroup.visible) this.live.forEach((l, k) => test('live:' + k, l.s.getWorldPosition(new THREE.Vector3())));
      if (best) this._onPick?.(best);
      void ray; void ndc;
    });
    dom.addEventListener('dblclick', () => this.flyHome());
  }
  /** World position -> CSS px within the canvas (null if behind the camera or the Earth). */
  project(pos, occlude = true) {
    const v = pos.clone().project(this.camera);
    if (v.z > 1 || v.z < -1) return null;
    if (occlude) {
      // hidden behind the globe?
      const cam = this.camera.position;
      const d = pos.clone().sub(cam);
      const len = d.length(); d.normalize();
      const b = cam.dot(d), c = cam.lengthSq() - 1;
      const disc = b * b - c;
      if (disc > 0) { const tHit = -b - Math.sqrt(disc); if (tHit > 0 && tHit < len) return null; }
    }
    const r = this._rect || this.renderer.domElement.getBoundingClientRect();
    return { x: (v.x * 0.5 + 0.5) * r.width + r.left, y: (-v.y * 0.5 + 0.5) * r.height + r.top };
  }
  /** Named anchors: 'target', 'chaser', 'tca', or a marker key. */
  screenPos(name) {
    if (this.focal) {
      if (name === 'target') return this.project(this.focal.sat.position);
      if (name === 'chaser') return this.project(this.focal.deb.position);
      if (name === 'tca' || name === this.focal.key) return this.project(this.focal.P);
    }
    const m = this.markers.get(name);
    if (m && m.sprite.visible) return this.project(m.sprite.position);
    if (String(name).startsWith('live:')) { const p = this.livePos(+name.slice(5)); return p ? this.project(p) : null; }
    return null;
  }

  // ─── frame loop ──────────────────────────────────────────────────────
  #tick() {
    try {
      const dt = Math.min(this.clock.getDelta(), 0.05);
      this._t += dt;
      const t = this._t;
      const ms = this.simMs;
      const rot = gmst(ms);
      if (this.earthApi) {
        this.earthApi.group.rotation.y = rot;
        const s = eciToScene(sunEci(ms));
        this.earthApi.uniforms.uSun.value.set(s[0], s[1], s[2]);
        this.sun.position.set(s[0] * 20, s[1] * 20, s[2] * 20);
        this.sky.sun.position.set(s[0] * 260, s[1] * 260, s[2] * 260);
        this.earthApi.uniforms.uCloudRot.value = (t * 0.0008) % 1;
      }
      if (this.pop) this.pop.uniforms.uT.value = (ms - this.pop.refEpochMs) / 60000;
      if (this.focal) {
        const f = this.focal, dts = (ms - f.tcaMs) / 1000;
        f.sat.position.set(...scenePos(f.t, dts));
        f.deb.position.set(...scenePos(f.c, dts));
        // nadir-pointing bus, arrays along the orbit normal
        f.sat.up.copy(V(scenePos(f.t, dts + 1)).sub(f.sat.position).normalize());
        f.sat.lookAt(0, 0, 0);
        f.deb.rotation.x += dt * 0.7; f.deb.rotation.y += dt * 0.45; f.deb.rotation.z += dt * 0.2;
        // keep the tiny models legible at any zoom (not to scale, labelled)
        const camD = this.camera.position.distanceTo(f.sat.position);
        const sc = THREE.MathUtils.clamp(camD * 0.012, 0.0025, 0.02);
        f.sat.scale.setScalar(sc); f.deb.scale.setScalar(sc * 0.8);
        const pulse = 0.5 + 0.5 * Math.sin(t * 4);
        const near = Math.abs(dts) < 900 ? 1.35 : 1;
        f.ring.scale.setScalar(f.ring.userData.base * near * (1 + 0.12 * pulse));
        f.ring.material.opacity = f.tier === 'CRITICAL' ? 0.55 + 0.45 * (Math.sin(t * 2.2 * Math.PI) > -0.1 ? 1 : 0.4) : 0.5 + 0.3 * pulse;
      }
      for (const m of this.markers.values()) {
        const p = 0.5 + 0.5 * Math.sin(t * 3 + m.phase);
        m.sprite.scale.setScalar(m.sprite.userData.base * (1 + 0.15 * p) * (m.tier === 'CRITICAL' ? 1.35 : 1));
        m.sprite.material.opacity = m.tier === 'CRITICAL' ? 0.6 + 0.4 * p : 0.55 + 0.3 * p;
      }
      for (let k = this.bursts.length - 1; k >= 0; k--) {
        const b = this.bursts[k]; b.t += dt;
        const q = Math.min(1, b.t / b.dur), e = 1 - Math.pow(1 - q, 3);
        b.s.scale.setScalar(b.from + (b.to - b.from) * e);
        b.s.material.opacity = (1 - q) * 0.9;
        if (q >= 1) { this.inertial.remove(b.s); this.bursts.splice(k, 1); }
      }
      if (this.liveGroup.visible) {
        this.live.forEach(({ s, c }, k) => {
          s.position.set(...latLonToScene(c.lat ?? 0, c.lon ?? 0, c.alt_km ?? 550, ms));
          s.material.opacity = 0.6 + 0.35 * Math.sin(t * 3 + k);
        });
      }
      if (this.fly) {
        const f = this.fly;
        f.t += Math.min(dt, 0.2);
        const k = ease(Math.min(1, f.t / f.dur));
        const r0 = f.p0.length(), r1 = f.p1.length();
        const d0 = f.p0.clone().normalize(), d1 = f.p1.clone().normalize();
        const q = new THREE.Quaternion().setFromUnitVectors(d0, d1);
        const qk = new THREE.Quaternion().slerp(q, k);
        this.camera.position.copy(d0.applyQuaternion(qk).multiplyScalar(r0 + (r1 - r0) * k));
        this.controls.target.lerpVectors(f.t0, f.t1, k);
        if (k >= 1) this.fly = null;
      }
      this.controls.update();
      this.composer.render(dt);
      this.#adapt(dt);
    } catch (err) {
      this._errs = (this._errs || 0) + 1;
      if (this._errs <= 3) console.error('[world] frame failed', err);
    }
  }
  // adaptive pixel ratio (reference): after 4 s warm-up, average 90 frames; if slow step down by 0.25 (never up)
  #adapt(dt) {
    this._frames.push(dt);
    if (this._frames.length > 90) this._frames.shift();
    const avg = this._frames.reduce((a, b) => a + b, 0) / this._frames.length;
    this.fps = 1 / Math.max(avg, 1e-3);
    if (this._t < 4 || this._frames.length < 90) return;
    if (avg > 1 / 40 && this.pixelRatio > 1) {
      this.pixelRatio = Math.max(1, this.pixelRatio - 0.25);
      this.renderer.setPixelRatio(this.pixelRatio);
      if (this.pop) this.pop.uniforms.uPx.value = this.pixelRatio;
      this.composer.setPixelRatio(this.pixelRatio);
      this._frames = [];
      this.#resize();
    }
  }
}

export { RE_KM };
