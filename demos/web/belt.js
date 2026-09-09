import { THREE, latestOnly, createViewer, makeArrow, makeTriad, textSprite, circle, panel, fmt, deg } from './common.js';

const SEGMENTS = 120;
const state = { turns: 2, t: 0, playing: false, dir: 1, width: 0.36, data: null };

// ---------------------------------------------------------------- scene
const viewer = createViewer(document.getElementById('view'), { cameraPos: [4.2, -5.2, 3.2], target: [0, 0, 1.4] });
const { scene } = viewer;

// The clamp at the near end: a plate the belt emerges from.
const clamp = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 0.08), new THREE.MeshStandardMaterial({ color: 0x2a3547, roughness: 0.8 }));
clamp.position.z = -0.04;
scene.add(clamp);
const grid = new THREE.GridHelper(8, 16, 0x1f2838, 0x1a2130);
grid.rotation.x = Math.PI / 2; grid.position.z = -0.08;
scene.add(grid);

// The belt: a ribbon with vertex colours (one edge orange, the other cyan) so twists are obvious.
const ribbonGeo = new THREE.BufferGeometry();
const ribbonPos = new Float32Array(SEGMENTS * 2 * 3), ribbonCol = new Float32Array(SEGMENTS * 2 * 3);
const ribbonIdx = [];
for (let i = 0; i < SEGMENTS - 1; i++) { const a = 2 * i; ribbonIdx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
ribbonGeo.setAttribute('position', new THREE.BufferAttribute(ribbonPos, 3));
ribbonGeo.setAttribute('color', new THREE.BufferAttribute(ribbonCol, 3));
ribbonGeo.setIndex(ribbonIdx);
const ribbon = new THREE.Mesh(ribbonGeo, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.55, metalness: 0.05 }));
ribbon.frustumCulled = false;
scene.add(ribbon);
const edgeA = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffb347 }));
const edgeB = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5ec8ff }));
scene.add(edgeA, edgeB);

// The buckle at the far end, plus its frame.
const buckle = new THREE.Group();
buckle.add(new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.12, 0.3), new THREE.MeshStandardMaterial({ color: 0xd4a017, metalness: 0.6, roughness: 0.35 })));
const buckleTriad = makeTriad(0.45);
buckle.add(buckleTriad);
const buckleLabel = textSprite('buckle', { size: 0.2, color: '#ffd166' }); buckleLabel.position.set(0, 0, 0.45); buckle.add(buckleLabel);
scene.add(buckle);
const refTriad = makeTriad(0.45, { opacity: 0.25 }); refTriad.position.set(0, 0, 0); // the identity frame at the clamp
scene.add(refTriad);

// Side view: the lifted path in SU(2). Every quaternion here has y = 0, so the
// path lives exactly on the 2-sphere in (w, x, z) coordinates.
const s3 = new THREE.Group();
s3.position.set(3.6, 2.2, 1.2);
s3.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), new THREE.MeshPhysicalMaterial({ color: 0x7fa8ff, transparent: true, opacity: 0.08, depthWrite: false })));
const gridMat = new THREE.LineBasicMaterial({ color: 0x35435c });
for (const ax of ['x', 'y', 'z']) s3.add(circle(1, ax, 0, gridMat));
const mark = (p, txt, color) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 8), new THREE.MeshBasicMaterial({ color })); m.position.set(...p); s3.add(m);
  const l = textSprite(txt, { size: 0.22, color }); l.position.set(p[0] * 1.3, p[1] * 1.3, p[2] * 1.3 + 0.1); s3.add(l); };
mark([0, 0, 1], '+1 (identity)', '#7ee787');   // we draw (x, z, w): w up
mark([0, 0, -1], '−1', '#ff7b72');
mark([1, 0, 0], 'R_x(π)', '#ffd166');
const s3title = textSprite('path in SU(2)', { size: 0.3, color: '#8a94a6' }); s3title.position.set(0, 0, 1.6); s3.add(s3title);
const s3path = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff }));
const s3start = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color: 0x5ec8ff }));
const s3end = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffb347 }));
s3.add(s3path, s3start, s3end);
scene.add(s3);
const q2p = (q) => [q[1], q[3], q[0]];  // (x, z, w) → draw w upwards

// ------------------------------------------------------------------ UI
const ui = panel(document.getElementById('panel'));
ui.para('The belt\'s orientation along its length is a path in SO(3). Lift it to SU(2) and a 360° twist is an <b>open arc</b> ' +
  'from +1 to −1, while a 720° twist is a <b>closed loop</b>. Closed loops in SU(2) can be shrunk to a point; the arc cannot ' +
  'without moving its ends. Moving an end means rotating the buckle.');
ui.section('Twist');
ui.select({ label: 'twist', value: '2', options: [['1', '360°  (one full twist)'], ['2', '720°  (two full twists)']],
  onChange: (v) => { state.turns = +v; refetch(); } });
const sT = ui.slider({ label: 'untwist', min: 0, max: 1, step: 0.002, value: 0, format: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { state.t = v; state.playing = false; playBtn.textContent = 'play'; refetch(); } });
const sSpeed = ui.slider({ label: 'speed', min: 0.05, max: 1, step: 0.05, value: 0.3, format: (v) => `${(+v).toFixed(2)}/s` });
ui.slider({ label: 'belt width', min: 0.1, max: 0.7, step: 0.02, value: 0.36, onInput: (v) => { state.width = v; if (state.data) buildRibbon(state.data); } });
const [playBtn] = ui.buttons([
  ['play', () => { state.playing = !state.playing; playBtn.textContent = state.playing ? 'pause' : 'play'; }, 'primary'],
  ['twisted', () => jump(0)], ['halfway', () => jump(0.5)], ['untwisted', () => jump(1)],
]);
ui.section('What to watch', 'The <b>buckle frame</b> (bright arrows at the far end) against the identity frame (faint arrows at the clamp). ' +
  'The homotopy is the same in both cases: slide every point of the lifted path along a great circle towards R<sub>x</sub>(π), ' +
  'then re-anchor the clamp end at the identity with P(0)<sup>−1</sup>P(s).');
const out = ui.readouts([['t', 'untwist progress'], ['end', 'buckle rotation'], ['verdict', 'far end held fixed?']]);
const overlay = document.getElementById('overlay');

// ----------------------------------------------------------------- data
const fetchBelt = latestOnly('belt', (data) => { state.data = data; buildRibbon(data); updateHUD(data); });
function refetch() { fetchBelt({ turns: state.turns, t: state.t, segments: SEGMENTS, length: 3.2 }); }
function jump(t) { state.t = t; sT.set(t); state.playing = false; playBtn.textContent = 'play'; refetch(); }

function buildRibbon(d) {
  const hw = state.width / 2, n = d.centre.length;
  const A = [], B = [];
  for (let i = 0; i < n; i++) {
    const c = d.centre[i], M = d.so3[i];
    const wx = M[0][0] * hw, wy = M[1][0] * hw, wz = M[2][0] * hw;   // M·x̂ is the across-belt direction
    const pa = [c[0] - wx, c[1] - wy, c[2] - wz], pb = [c[0] + wx, c[1] + wy, c[2] + wz];
    ribbonPos.set(pa, 6 * i); ribbonPos.set(pb, 6 * i + 3);
    const stripe = Math.floor(i / (n / 12)) % 2 ? 0.75 : 1.0;      // faint stripes show motion along the belt
    ribbonCol.set([1.0 * stripe, 0.7 * stripe, 0.28 * stripe], 6 * i);
    ribbonCol.set([0.37 * stripe, 0.78 * stripe, 1.0 * stripe], 6 * i + 3);
    A.push(new THREE.Vector3(...pa)); B.push(new THREE.Vector3(...pb));
  }
  ribbonGeo.attributes.position.needsUpdate = true; ribbonGeo.attributes.color.needsUpdate = true;
  ribbonGeo.computeVertexNormals();
  edgeA.geometry.setFromPoints(A); edgeB.geometry.setFromPoints(B);

  const end = d.centre[n - 1], M = d.end_so3;
  buckle.position.set(...end);
  buckle.setRotationFromMatrix(new THREE.Matrix4().set(...M[0], 0, ...M[1], 0, ...M[2], 0, 0, 0, 0, 1));
  buckleTriad.setMatrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]); // triad is a child: inherits the buckle's rotation

  s3path.geometry.setFromPoints(d.lifted_path.map((q) => new THREE.Vector3(...q2p(q))));
  s3start.position.set(...q2p(d.lifted_path[0]));
  s3end.position.set(...q2p(d.lifted_path[n - 1]));
}

function updateHUD(d) {
  const a = deg(d.end_angle);
  const fixed = a < 0.5;
  out.t(`${Math.round(state.t * 100)}%`);
  out.end(`${fmt(a, 1)}°`);
  out.verdict(fixed ? '<span class="badge good">yes, identity</span>' : `<span class="badge bad">no, rotated ${fmt(a, 0)}°</span>`);
  overlay.innerHTML = state.turns === 2
    ? `<b>720° twist.</b> The lifted path is a closed loop through +1 and −1. As it shrinks to a point the belt untwists while the buckle's frame stays exactly the identity (rotation ${fmt(a, 1)}°). That is the belt trick.`
    : `<b>360° twist.</b> The lifted path is an open arc from +1 to −1. To shrink it the ends must move, and a moving end is a rotating buckle: right now it is turned by <b>${fmt(a, 1)}°</b>. One twist can only be removed by flipping the buckle over.`;
}

viewer.onFrame((dt) => {
  if (!state.playing) return;
  state.t += state.dir * sSpeed.valueAsNumber * dt;
  if (state.t >= 1) { state.t = 1; state.dir = -1; } else if (state.t <= 0) { state.t = 0; state.dir = 1; }
  sT.set(state.t);
  refetch();
});

refetch();
