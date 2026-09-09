import { THREE, api, latestOnly, createViewer, blochSphere, makeArrow, makeTriad, makeTrail, panel,
         makePlot, drawPhasors, fmt, fmtC, deg, matrixHTML } from './common.js';

const TWO_PI = Math.PI * 2;
const MAX_ANGLE = 2 * TWO_PI;          // sweep 0..720°
const SAMPLES = 721;                   // one sample per degree

// ---------------------------------------------------------------- state
const state = {
  mode: 'single',
  axisTilt: 0, axisAz: 0,              // rotation axis in spherical angles (deg)
  stateTilt: 90, stateAz: 0,           // initial Bloch vector (deg)
  angle: 0, playing: true, speed: 60,  // degrees per second
  sweep: null, cloud: null,
};
const axisVec = () => sph(state.axisTilt, state.axisAz);
const stateVec = () => sph(state.stateTilt, state.stateAz);
function sph(tilt, az) { const t = tilt * Math.PI / 180, p = az * Math.PI / 180; return [Math.sin(t) * Math.cos(p), Math.sin(t) * Math.sin(p), Math.cos(t)]; }

// ---------------------------------------------------------------- scene
const viewer = createViewer(document.getElementById('view'), { cameraPos: [2.7, -2.9, 1.6] });
const { scene } = viewer;
scene.add(blochSphere());

const axisArrow = makeArrow(0xffd166, { radius: 0.014, headLength: 0.14, headRadius: 0.05 });
const axisLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: 0xffd166, dashSize: 0.06, gapSize: 0.04 }));
const stateArrow = makeArrow(0x5ec8ff, { radius: 0.03, headLength: 0.2, headRadius: 0.075 });
const startArrow = makeArrow(0x5ec8ff, { radius: 0.012, headLength: 0.12, headRadius: 0.04, opacity: 0.35 });
const trail = makeTrail(SAMPLES, 0x5ec8ff, 0.7);
const tip = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x335577 }));
scene.add(axisArrow, axisLine, stateArrow, startArrow, trail, tip);

// A little rigid body off to the side, driven by the SO(3) matrix of the same
// SU(2) element, so the 360° periodicity of ordinary rotations is visible next
// to the 720° periodicity of the spinor.
const body = new THREE.Group();
const cube = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.3),
  [0xff6b6b, 0xaa3333, 0x6bff8a, 0x2f8a47, 0x6b9bff, 0x2f4f9a].map((c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.5 })));
const triad = makeTriad(0.36);
body.add(cube, triad);
body.position.set(0, 1.9, -0.9);
scene.add(body);

// Precession cloud
const cloud = new THREE.Group();
const cloudPoints = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0x9ad8ff, size: 0.045, sizeAttenuation: true }));
const cloudTrails = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.35 }));
const meanArrow = makeArrow(0xff6bd6, { radius: 0.025, headLength: 0.16, headRadius: 0.06 });
cloud.add(cloudPoints, cloudTrails, meanArrow);
cloud.visible = false;
scene.add(cloud);

// ------------------------------------------------------------------ UI
const ui = panel(document.getElementById('panel'));
ui.para('A spin-½ state |ψ⟩ is rotated by <b>R(θ) = exp(−iθ n·σ/2)</b>. The Bloch arrow ⟨ψ|σ|ψ⟩ is an ordinary ' +
  'vector and returns after 360°. The spinor is not: at 360° it has become −|ψ⟩, and only at 720° is it back. ' +
  'The chart shows both periods at once.');

ui.section('Mode');
ui.select({ label: 'show', value: 'single', options: [['single', 'one state + its trail'], ['cloud', 'precessing cloud (150 spins)']],
  onChange: (v) => { state.mode = v; applyMode(); } });

ui.section('Rotation axis n', 'Also the magnetic field direction in cloud mode.');
const sAxisTilt = ui.slider({ label: 'tilt from z', min: 0, max: 180, step: 1, value: 0, format: (v) => `${v}°`, onInput: (v) => { state.axisTilt = v; refetch(); } });
const sAxisAz = ui.slider({ label: 'azimuth', min: 0, max: 360, step: 1, value: 0, format: (v) => `${v}°`, onInput: (v) => { state.axisAz = v; refetch(); } });
ui.buttons([
  ['z', () => setAxis(0, 0)], ['x', () => setAxis(90, 0)], ['y', () => setAxis(90, 90)], ['(1,1,1)', () => setAxis(54.7356, 45)],
]);

ui.section('Initial state |ψ(0)⟩', 'Given as a point on the Bloch sphere; the library builds the spinor with from_bloch_vector.');
const sStateTilt = ui.slider({ label: 'polar θ', min: 0, max: 180, step: 1, value: 90, format: (v) => `${v}°`, onInput: (v) => { state.stateTilt = v; refetch(); } });
const sStateAz = ui.slider({ label: 'azimuth φ', min: 0, max: 360, step: 1, value: 0, format: (v) => `${v}°`, onInput: (v) => { state.stateAz = v; refetch(); } });
ui.buttons([
  ['|0⟩', () => setState(0, 0)], ['|+x⟩', () => setState(90, 0)], ['|+y⟩', () => setState(90, 90)], ['tilted', () => setState(50, 30)],
]);

ui.section('Rotation angle θ');
const sAngle = ui.slider({ label: 'θ', min: 0, max: 720, step: 0.5, value: 0, format: (v) => `${(+v).toFixed(0)}°`, onInput: (v) => { state.angle = v; state.playing = false; playBtn.textContent = 'play'; } });
const sSpeed = ui.slider({ label: 'speed', min: 5, max: 240, step: 5, value: 60, format: (v) => `${v}°/s`, onInput: (v) => { state.speed = v; } });
const [playBtn] = ui.buttons([
  ['pause', () => { state.playing = !state.playing; playBtn.textContent = state.playing ? 'pause' : 'play'; }, 'primary'],
  ['→ 360°', () => jump(360)], ['→ 720°', () => jump(720)], ['reset', () => jump(0)],
]);

ui.section('Overlap with the start state');
const chart = ui.canvas(140);
ui.html('<div class="legend"><span><i style="background:#5ec8ff"></i>Re ⟨ψ(0)|ψ(θ)⟩ (spinor, period 720°)</span>' +
  '<span><i style="background:#ffb347"></i>b(0)·b(θ) (Bloch arrow, period 360°)</span></div>');
const plot = makePlot(chart, { xmin: 0, xmax: 720, ymin: -1.05, ymax: 1.05,
  xticks: [[180, '180°'], [360, '360°'], [540, '540°'], [720, '720°']], yticks: [[1, '+1'], [0, '0'], [-1, '−1']] });

ui.section('Spinor components');
const phasors = ui.canvas(110);

ui.section('Numbers');
const out = ui.readouts([['angle', 'θ'], ['bloch', 'Bloch vector'], ['overlap', '⟨ψ(0)|ψ(θ)⟩'], ['quat', 'quaternion (w,x,y,z)'], ['mat', 'R(θ) ∈ SU(2)']]);
const overlay = document.getElementById('overlay');

// ----------------------------------------------------------------- data
const fetchSweep = latestOnly('rotation_sweep', (data) => { state.sweep = data; trail.setPoints(data.bloch); });
const fetchCloud = latestOnly('precession_sweep', (data) => { state.cloud = data; buildCloud(data); });

function refetch() {
  fetchSweep({ axis: axisVec(), bloch: stateVec(), max_angle: MAX_ANGLE, samples: SAMPLES });
  if (state.mode === 'cloud') fetchCloud({ axis: axisVec(), n_spins: 150, samples: 361, max_angle: MAX_ANGLE });
}
function setAxis(t, a) { state.axisTilt = t; state.axisAz = a; sAxisTilt.set(t); sAxisAz.set(a); refetch(); }
function setState(t, a) { state.stateTilt = t; state.stateAz = a; sStateTilt.set(t); sStateAz.set(a); refetch(); }
function jump(a) { state.angle = a; state.playing = false; playBtn.textContent = 'play'; sAngle.set(a); }
function applyMode() {
  const single = state.mode === 'single';
  for (const o of [stateArrow, startArrow, trail, tip]) o.visible = single;
  cloud.visible = !single;
  refetch();
}

function buildCloud(data) {
  const n = data.bloch[0].length;
  cloudPoints.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  const TRAIL = 14;
  cloudTrails.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * TRAIL * 2 * 3), 3));
  cloudTrails.userData.TRAIL = TRAIL;
}

// -------------------------------------------------------------- animate
let lastIdx = -1;
viewer.onFrame((dt) => {
  if (state.playing) { state.angle = (state.angle + state.speed * dt) % 720; sAngle.set(state.angle); }
  const a = axisVec();
  axisArrow.setVector(a[0] * 1.45, a[1] * 1.45, a[2] * 1.45);
  axisLine.geometry.setFromPoints([new THREE.Vector3(-a[0] * 1.45, -a[1] * 1.45, -a[2] * 1.45), new THREE.Vector3(0, 0, 0)]);
  axisLine.computeLineDistances();

  const sw = state.sweep;
  if (!sw) return;
  const i = Math.max(0, Math.min(SAMPLES - 1, Math.round(state.angle / 720 * (SAMPLES - 1))));
  const b = sw.bloch[i], b0 = sw.bloch[0];
  if (state.mode === 'single') { stateArrow.setVector(...b); tip.position.set(...b); startArrow.setVector(...b0); }
  triad.setMatrix3(sw.so3[i]);
  cube.setRotationFromMatrix(new THREE.Matrix4().set(...sw.so3[i][0], 0, ...sw.so3[i][1], 0, ...sw.so3[i][2], 0, 0, 0, 0, 1));

  if (state.mode === 'cloud' && state.cloud) {
    const c = state.cloud, m = c.bloch.length, j = Math.round(state.angle / 720 * (m - 1));
    const pos = cloudPoints.geometry.attributes.position, tr = cloudTrails.geometry.attributes.position, TRAIL = cloudTrails.userData.TRAIL;
    const n = c.bloch[j].length;
    for (let k = 0; k < n; k++) {
      pos.setXYZ(k, ...c.bloch[j][k]);
      for (let q = 0; q < TRAIL; q++) {
        const j1 = Math.max(0, j - q), j2 = Math.max(0, j - q - 1);
        tr.setXYZ((k * TRAIL + q) * 2, ...c.bloch[j1][k]); tr.setXYZ((k * TRAIL + q) * 2 + 1, ...c.bloch[j2][k]);
      }
    }
    pos.needsUpdate = true; tr.needsUpdate = true;
    meanArrow.setVector(...c.mean_spin[j]);
  }

  if (i === lastIdx) return;   // HUD only when the sample changes
  lastIdx = i;
  const ov = sw.overlap[i], dot = b[0] * b0[0] + b[1] * b0[1] + b[2] * b0[2];
  plot.draw([
    { points: sw.overlap.map((o, k) => [k, o[0]]), color: '#5ec8ff' },
    { points: sw.bloch.map((v, k) => [k, v[0] * b0[0] + v[1] * b0[1] + v[2] * b0[2]]), color: '#ffb347', dash: [4, 3] },
  ], state.angle);
  drawPhasors(phasors, sw.components[i]);
  out.angle(`${fmt(state.angle, 1)}° &nbsp; (${fmt(state.angle / 180, 3)} π)`);
  out.bloch(`(${b.map((v) => fmt(v)).join(', ')})`);
  out.overlap(`${fmtC(ov)} &nbsp; |·|² = ${fmt(ov[0] ** 2 + ov[1] ** 2)}`);
  out.quat(`(${sw.quaternion[i].map((v) => fmt(v)).join(', ')})`);
  out.mat(matrixHTML(sw.matrix[i], { complex: true }));
  const near = (x, y) => Math.abs(x - y) < 2;
  overlay.innerHTML = near(state.angle, 360)
    ? `<b>θ = 360°:</b> the Bloch arrow and the cube are exactly back where they started, but ⟨ψ(0)|ψ(θ)⟩ = ${fmt(ov[0], 2)}: the spinor has changed sign.`
    : near(state.angle, 720) || near(state.angle, 0)
      ? `<b>θ = 720°:</b> now the spinor itself is back: ⟨ψ(0)|ψ(θ)⟩ = ${fmt(ov[0], 2)}. SU(2) double-covers SO(3); the cube cannot tell 360° from 720°, the spinor can.`
      : `θ = ${fmt(state.angle, 0)}° &nbsp;·&nbsp; Bloch arrow: b(0)·b(θ) = ${fmt(dot, 2)} &nbsp;·&nbsp; spinor: Re⟨ψ(0)|ψ(θ)⟩ = cos(θ/2) = ${fmt(ov[0], 2)}`;
});

refetch();
