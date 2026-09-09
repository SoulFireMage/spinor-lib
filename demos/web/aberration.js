import { THREE, latestOnly, createViewer, makeArrow, circle, textSprite, panel, fmt, matrixHTML } from './common.js';

const N_STARS = 1500;
const state = { beta: 0, view: 'outside', showRest: true, showGrid: true, flying: false, data: null };

// ---------------------------------------------------------------- scene
const viewer = createViewer(document.getElementById('view'), { cameraPos: [2.4, -2.6, 1.3] });
const { scene, camera, controls } = viewer;
const sky = new THREE.Group();
scene.add(sky);

sky.add(new THREE.Mesh(new THREE.SphereGeometry(1, 64, 40), new THREE.MeshPhysicalMaterial({ color: 0x7fa8ff, transparent: true, opacity: 0.05, depthWrite: false, side: THREE.DoubleSide })));
const forwardArrow = makeArrow(0xffd166, { radius: 0.02, headLength: 0.16, headRadius: 0.06 });
forwardArrow.setVector(0, 0, 0.9);
sky.add(forwardArrow);
const fwdLabel = textSprite('velocity', { size: 0.2, color: '#ffd166' }); fwdLabel.position.set(0, 0, 1.12); sky.add(fwdLabel);

// Stars: colour and size from the Doppler factor computed by the library.
const starGeo = new THREE.BufferGeometry();
starGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N_STARS * 3), 3));
starGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N_STARS * 3), 3));
starGeo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(N_STARS), 1));
const starMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute float size; varying vec3 vColor; void main(){ vColor=color; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_PointSize=size*(${''}1.0); gl_Position=projectionMatrix*mv; }`,
  fragmentShader: `varying vec3 vColor; void main(){ float d=length(gl_PointCoord-0.5); if(d>0.5) discard; float a=smoothstep(0.5,0.1,d); gl_FragColor=vec4(vColor*a,a); }`,
  vertexColors: true,
});
const stars = new THREE.Points(starGeo, starMat);
stars.frustumCulled = false;
sky.add(stars);

const restStars = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0x556677, size: 3, sizeAttenuation: false, transparent: true, opacity: 0.55 }));
restStars.frustumCulled = false;
sky.add(restStars);

const gridLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.55 }));
const restGrid = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x35435c, transparent: true, opacity: 0.5 }));
gridLines.frustumCulled = restGrid.frustumCulled = false;
sky.add(gridLines, restGrid);

// ------------------------------------------------------------------ UI
const ui = panel(document.getElementById('panel'));
ui.para('A photon arriving from sky direction n̂ has null wave-vector k = (1, −n̂), and that null vector <b>is</b> a Weyl spinor: ' +
  'ψ = from_bloch_vector(−n̂) has weyl_current(ψ) = k. A boost is an SL(2,C) matrix; multiply the spinor by it and read the ' +
  'current back to get where the star appears and how much it is Doppler shifted. Because SL(2,C) acts on the sky by Möbius ' +
  'transformations, circles remain circles.');
ui.section('Motion');
const sBeta = ui.slider({ label: 'speed β', min: 0, max: 0.99, step: 0.005, value: 0, format: (v) => `${(+v).toFixed(3)} c`, onInput: (v) => { state.beta = v; state.flying = false; flyBtn.textContent = 'fly to 0.99c'; refetch(); } });
const [flyBtn] = ui.buttons([
  ['fly to 0.99c', () => { state.flying = !state.flying; flyBtn.textContent = state.flying ? 'stop' : 'fly to 0.99c'; }, 'primary'],
  ['rest', () => { state.beta = 0; sBeta.set(0); refetch(); }], ['0.5c', () => { state.beta = 0.5; sBeta.set(0.5); refetch(); }], ['0.9c', () => { state.beta = 0.9; sBeta.set(0.9); refetch(); }],
]);
ui.section('View');
ui.select({ label: 'camera', value: 'outside', options: [['outside', 'outside the celestial sphere'], ['cockpit', 'from the pilot\'s seat']], onChange: (v) => { state.view = v; applyView(); } });
ui.toggle({ label: 'show rest-frame sky (grey)', value: true, onChange: (v) => { state.showRest = v; restStars.visible = restGrid.visible = v; } });
ui.toggle({ label: 'show sky grid', value: true, onChange: (v) => { state.showGrid = v; gridLines.visible = v; restGrid.visible = v && state.showRest; } });
ui.html('<div class="legend"><span><i style="background:#8fb8ff"></i>blueshifted (ahead)</span><span><i style="background:#ffffff"></i>unshifted</span><span><i style="background:#ff8a70"></i>redshifted (behind)</span></div>');
ui.section('Numbers');
const out = ui.readouts([['gamma', 'γ = cosh η'], ['eta', 'rapidity η'], ['dahead', 'Doppler factor ahead'], ['dbehind', 'Doppler factor behind'], ['half', 'half the sky is within'], ['A', 'A ∈ SL(2,C), right-handed'], ['L', 'Λ (4×4)']]);
const overlay = document.getElementById('overlay');

// ----------------------------------------------------------------- data
const fetchSky = latestOnly('aberration', (d) => { state.data = d; render(d); });
function refetch() { fetchSky({ beta: state.beta, direction: [0, 0, 1], n_stars: N_STARS }); }

function dopplerColor(D, mag) {
  // white at D=1, towards blue for D>1 and red for D<1; brightness grows with D
  const x = Math.max(-1, Math.min(1, Math.log(D) / Math.log(4)));
  const r = x > 0 ? 1 - 0.45 * x : 1, g = 1 - 0.35 * Math.abs(x), b = x < 0 ? 1 + 0.55 * x : 1;
  const bright = (0.35 + 0.65 * mag) * Math.min(2.2, Math.pow(D, 1.2));
  return [r * bright, g * bright, b * bright];
}

function render(d) {
  const pos = starGeo.attributes.position, col = starGeo.attributes.color, size = starGeo.attributes.size;
  const dpr = Math.min(window.devicePixelRatio, 2);
  for (let i = 0; i < d.aberrated.length; i++) {
    pos.setXYZ(i, ...d.aberrated[i]);
    col.setXYZ(i, ...dopplerColor(d.doppler[i], d.magnitude[i]));
    size.setX(i, (2.2 + 4 * d.magnitude[i]) * Math.pow(d.doppler[i], 0.5) * dpr);
  }
  pos.needsUpdate = col.needsUpdate = size.needsUpdate = true;
  if (!restStars.geometry.attributes.position) restStars.geometry.setAttribute('position', new THREE.Float32BufferAttribute(d.stars.flat(), 3));

  const segs = (lines) => { const a = []; for (const line of lines) for (let i = 0; i < line.length - 1; i++) a.push(...line[i], ...line[i + 1]); return new THREE.Float32BufferAttribute(a, 3); };
  gridLines.geometry.setAttribute('position', segs(d.grid_aberrated));
  if (!restGrid.geometry.attributes.position) restGrid.geometry.setAttribute('position', segs(d.grid));

  const ahead = Math.sqrt((1 + d.beta) / (1 - d.beta));
  const halfAngle = Math.acos(d.beta) * 180 / Math.PI;   // stars from the rest-frame front hemisphere end up within this cone
  out.gamma(fmt(d.gamma, 3)); out.eta(fmt(d.rapidity, 3));
  out.dahead(fmt(ahead, 3)); out.dbehind(fmt(1 / ahead, 3));
  out.half(`${fmt(halfAngle, 1)}° of forward`);
  out.A(matrixHTML(d.sl2c, { complex: true })); out.L(matrixHTML(d.lorentz, { d: 2 }));
  overlay.innerHTML = d.beta < 0.01
    ? '<b>At rest.</b> The grey rest-frame sky and the coloured moving-frame sky coincide. Drag the speed slider or press fly.'
    : `<b>β = ${fmt(d.beta, 3)} c</b> (γ = ${fmt(d.gamma, 2)}). The rest-frame front hemisphere is squeezed into a cone of half-angle ${fmt(halfAngle, 0)}° ahead of you and blueshifted by up to ×${fmt(ahead, 2)}. Blue grid = boosted, grey grid = at rest: every circle is still a circle.`;
}

function applyView() {
  if (state.view === 'cockpit') {
    sky.scale.setScalar(20);
    controls.minDistance = controls.maxDistance = 0.05; controls.enablePan = false; controls.enableZoom = false;
    // Sit at the centre looking along the velocity (+z); a slight tilt keeps
    // OrbitControls away from the singular "looking straight up" pose.
    camera.position.set(0.0, -0.012, -0.0485); controls.target.set(0, 0, 0); camera.near = 0.01;
    forwardArrow.visible = fwdLabel.visible = false;
  } else {
    sky.scale.setScalar(1);
    controls.minDistance = 1.2; controls.maxDistance = 20; controls.enablePan = true; controls.enableZoom = true;
    camera.position.set(2.4, -2.6, 1.3); controls.target.set(0, 0, 0);
    forwardArrow.visible = fwdLabel.visible = true;
  }
  camera.updateProjectionMatrix();
}

viewer.onFrame((dt) => {
  if (!state.flying) return;
  state.beta = Math.min(0.99, state.beta + 0.12 * dt * (1 - state.beta * 0.5));
  sBeta.set(state.beta);
  refetch();
  if (state.beta >= 0.99) { state.flying = false; flyBtn.textContent = 'fly to 0.99c'; }
});

refetch();
