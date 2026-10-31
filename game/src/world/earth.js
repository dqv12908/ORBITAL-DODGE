// Physically-inspired Earth: day albedo, night lights, ocean specular, normal map, drifting clouds,
// atmospheric scattering rim (blue limb, warm terminator), starfield with a faint Milky Way, the Sun.
// Textures: three.js examples/textures/planets (MIT; derived from NASA Blue Marble / Black Marble), see game/CREDITS.md.
import * as THREE from 'three';

const loader = new THREE.TextureLoader();
function tex(url, srgb = true) {
  return new Promise((res) => loader.load(url, (t) => {
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    res(t);
  }, undefined, () => res(null)));
}

export async function createEarth(renderer) {
  const [day, night, spec, normal, clouds] = await Promise.all([
    tex('/textures/earth_atmos_2048.jpg'), tex('/textures/earth_lights_2048.png'), tex('/textures/earth_specular_2048.jpg', false),
    tex('/textures/earth_normal_2048.jpg', false), tex('/textures/earth_clouds_1024.png'),
  ]);
  const aniso = renderer?.capabilities.getMaxAnisotropy?.() || 8;
  for (const t of [day, night, spec, normal, clouds]) if (t) t.anisotropy = aniso;
  const group = new THREE.Group();
  const uniforms = {
    uDay: { value: day }, uNight: { value: night }, uSpec: { value: spec }, uNormal: { value: normal }, uClouds: { value: clouds },
    uSun: { value: new THREE.Vector3(1, 0, 0) }, uCloudRot: { value: 0 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: /* glsl */`
      varying vec2 vUv; varying vec3 vN; varying vec3 vPosW; varying vec3 vObjN;
      void main() {
        vUv = uv;
        vObjN = normalize(normal);
        vN = normalize(mat3(modelMatrix) * normal);
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vPosW = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uDay, uNight, uSpec, uNormal, uClouds; uniform vec3 uSun; uniform float uCloudRot;
      varying vec2 vUv; varying vec3 vN; varying vec3 vPosW; varying vec3 vObjN;
      void main() {
        vec3 N = normalize(vN);
        vec3 V = normalize(cameraPosition - vPosW);
        vec3 L = normalize(uSun);
        // tangent frame on the sphere (world space)
        vec3 T = normalize(cross(vec3(0.0, 1.0, 0.0), N) + vec3(1e-5));
        vec3 B = cross(N, T);
        vec3 nm = texture2D(uNormal, vUv).xyz * 2.0 - 1.0;
        vec3 Nb = normalize(T * nm.x * 0.9 + B * nm.y * 0.9 + N * nm.z);
        float mu = dot(N, L);
        float dayMix = smoothstep(-0.06, 0.16, mu);
        vec3 albedo = texture2D(uDay, vUv).rgb;
        float diff = max(dot(Nb, L), 0.0);
        vec3 sunCol = vec3(1.0, 0.97, 0.92) * 1.3;
        // warm light near the terminator
        vec3 warm = mix(vec3(1.0, 0.55, 0.3), vec3(1.0), smoothstep(0.0, 0.35, mu));
        vec3 col = albedo * diff * sunCol * warm;
        // ocean specular (spec map: white = water)
        float ocean = texture2D(uSpec, vUv).r;
        vec3 H = normalize(L + V);
        float sp = pow(max(dot(N, H), 0.0), 60.0) * ocean * 1.1 + pow(max(dot(N, H), 0.0), 8.0) * ocean * 0.08;
        col += sunCol * warm * sp * dayMix;
        // cloud shadow (approx: sample clouds slightly offset toward the sun)
        vec2 cuv = vec2(fract(vUv.x - uCloudRot), vUv.y);
        float cl = texture2D(uClouds, cuv + vec2(0.0015, 0.0)).a;
        col *= 1.0 - 0.35 * cl * dayMix;
        // night side: city lights (bloom picks them up)
        vec3 lights = texture2D(uNight, vUv).rgb;
        col += lights * vec3(1.0, 0.78, 0.48) * 2.2 * (1.0 - smoothstep(-0.12, 0.08, mu));
        // faint earthshine / airglow ambient
        col += albedo * 0.012;
        // atmospheric rim on the disc: blue on the day side, warm at the terminator
        float f = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        vec3 rimCol = mix(vec3(1.0, 0.5, 0.25), vec3(0.32, 0.6, 1.0), smoothstep(-0.1, 0.35, mu));
        col += rimCol * f * smoothstep(-0.25, 0.2, mu) * 0.75;
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 192, 128), mat);
  group.add(earth);

  // drifting cloud layer
  const cloudMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: earth.material.vertexShader,
    fragmentShader: /* glsl */`
      uniform sampler2D uClouds; uniform vec3 uSun; uniform float uCloudRot;
      varying vec2 vUv; varying vec3 vN; varying vec3 vPosW;
      void main() {
        vec3 N = normalize(vN); vec3 L = normalize(uSun);
        float a = texture2D(uClouds, vec2(fract(vUv.x - uCloudRot), vUv.y)).a;
        float mu = dot(N, L);
        float lit = smoothstep(-0.12, 0.25, mu);
        vec3 warm = mix(vec3(1.0, 0.6, 0.4), vec3(1.0), smoothstep(0.0, 0.3, mu));
        vec3 col = vec3(0.96, 0.97, 1.0) * (0.03 + 0.92 * max(mu, 0.0)) * warm;
        gl_FragColor = vec4(col, a * (0.15 + 0.8 * lit));
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false,
  });
  const cloud = new THREE.Mesh(new THREE.SphereGeometry(1.006, 160, 96), cloudMat);
  group.add(cloud);

  // atmospheric scattering shell (limb glow)
  const atmo = new THREE.Mesh(
    new THREE.SphereGeometry(1.035, 128, 64),
    new THREE.ShaderMaterial({
      uniforms: { uSun: uniforms.uSun },
      vertexShader: /* glsl */`
        varying vec3 vN; varying vec3 vPosW;
        void main() {
          vN = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vPosW = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 uSun; varying vec3 vN; varying vec3 vPosW;
        void main() {
          vec3 N = normalize(vN);
          vec3 V = normalize(cameraPosition - vPosW);
          float rim = 1.0 - abs(dot(N, V));
          float a = pow(rim, 5.0) * 2.2;
          float mu = dot(N, normalize(uSun));
          vec3 blue = vec3(0.30, 0.58, 1.0);
          vec3 warm = vec3(1.0, 0.52, 0.25);
          vec3 c = mix(warm, blue, smoothstep(-0.15, 0.3, mu));
          float lit = smoothstep(-0.35, 0.15, mu);
          // forward scattering toward the sun
          float fwd = pow(max(dot(-V, normalize(uSun)), 0.0), 10.0) * 0.35;
          gl_FragColor = vec4(c * (lit + fwd) * 0.9, a * (lit + fwd * 0.5));
        }`,
      transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
    }),
  );
  group.add(atmo);
  return { group, earth, cloud, uniforms };
}

/** Starfield with a faint Milky Way band (procedural), plus the Sun sprite. */
export function createSky() {
  const g = new THREE.Group();
  const n = 9000;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), size = new Float32Array(n);
  let s = 1729;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  // galactic plane tilted ~62 deg to the equator
  const tilt = new THREE.Matrix4().makeRotationX(THREE.MathUtils.degToRad(62)).multiply(new THREE.Matrix4().makeRotationY(1.1));
  const v = new THREE.Vector3();
  for (let k = 0; k < n; k++) {
    const band = k < n * 0.45;
    const lon = rnd() * Math.PI * 2;
    const lat = band ? gauss() * 0.13 : Math.asin(rnd() * 2 - 1);
    v.set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)).applyMatrix4(tilt).multiplyScalar(300);
    pos.set([v.x, v.y, v.z], k * 3);
    const tcol = rnd();
    const c = tcol < 0.15 ? [1.0, 0.82, 0.65] : tcol < 0.35 ? [0.75, 0.85, 1.0] : [1, 1, 1];
    const b = Math.pow(rnd(), 3.2) * 1.4 + 0.12;
    col.set([c[0] * b, c[1] * b, c[2] * b], k * 3);
    size[k] = 0.8 + Math.pow(rnd(), 6) * 2.8;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(size, 1));
  const stars = new THREE.Points(geo, new THREE.ShaderMaterial({
    uniforms: { uPx: { value: Math.min(2, window.devicePixelRatio || 1) } },
    vertexShader: /* glsl */`attribute float size; attribute vec3 color; varying vec3 vC; uniform float uPx;
      void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position,1.0); gl_Position = projectionMatrix * mv; gl_PointSize = size * uPx; }`,
    fragmentShader: /* glsl */`varying vec3 vC; void main(){ float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard; gl_FragColor = vec4(vC, smoothstep(0.5, 0.0, r)); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  stars.frustumCulled = false;
  g.add(stars);
  // Milky Way haze: procedural fbm band on a big inverted sphere
  const mw = new THREE.Mesh(new THREE.SphereGeometry(320, 64, 32), new THREE.ShaderMaterial({
    uniforms: { uTilt: { value: new THREE.Matrix3().setFromMatrix4(tilt).invert() } },
    vertexShader: /* glsl */`varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: /* glsl */`uniform mat3 uTilt; varying vec3 vP;
      float h(vec3 p){ return fract(sin(dot(p, vec3(127.1,311.7,74.7))) * 43758.5453); }
      float n(vec3 p){ vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
        return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),
                   mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z); }
      void main(){ vec3 g = uTilt * vP; float b = exp(-pow(g.y/0.16,2.0));
        float f = n(vP*4.0)*0.5 + n(vP*9.0)*0.3 + n(vP*21.0)*0.2;
        float dust = smoothstep(0.55, 0.8, n(vP*6.0+3.1)) * exp(-pow(g.y/0.05,2.0));
        float a = b * (0.35 + 0.9*f) * (1.0 - 0.7*dust) * 0.085;
        gl_FragColor = vec4(vec3(0.78,0.82,1.0)*a + vec3(1.0,0.86,0.7)*a*0.4*f, 1.0); }`,
    side: THREE.BackSide, depthWrite: false, blending: THREE.AdditiveBlending, transparent: true,
  }));
  g.add(mw);
  // the Sun: HDR sprite far away (bloom makes the flare)
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const x = c.getContext('2d');
  const grd = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.08, 'rgba(255,250,235,1)');
  grd.addColorStop(0.2, 'rgba(255,220,170,0.35)'); grd.addColorStop(1, 'rgba(255,200,150,0)');
  x.fillStyle = grd; x.fillRect(0, 0, 256, 256);
  const st = new THREE.CanvasTexture(c); st.colorSpace = THREE.SRGBColorSpace;
  const sun = new THREE.Sprite(new THREE.SpriteMaterial({ map: st, color: new THREE.Color(6, 5.6, 5), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  sun.scale.setScalar(14);
  g.add(sun);
  return { group: g, sun, stars };
}
