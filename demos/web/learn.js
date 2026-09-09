import { THREE, api, createViewer, blochSphere, makeTriad, panel, makePlot, fmt, showError } from './common.js';

const state = { session: null, running: false, stepsPerTick: 2, busy: false, history: [], n: 64, noise: 0.03, lr: 0.05, seed: 0, targets: null, inputs: null };

// ---------------------------------------------------------------- scene
const viewer = createViewer(document.getElementById('view'), { cameraPos: [2.6, -2.8, 1.7] });
const { scene } = viewer;
scene.add(blochSphere({ opacity: 0.08 }));

const mkPoints = (color, size) => { const p = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color, size, sizeAttenuation: true })); p.frustumCulled = false; return p; };
const inputPts = mkPoints(0x8a94a6, 0.035), targetPts = mkPoints(0x7ee787, 0.06), predPts = mkPoints(0xffb347, 0.06);
const links = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffb347, transparent: true, opacity: 0.6 }));
links.frustumCulled = false;
scene.add(inputPts, targetPts, predPts, links);

const trueTriad = makeTriad(1.35, { opacity: 0.28 });   // the hidden rotation's frame (faint)
const learnedTriad = makeTriad(1.2);                    // the model's frame (solid)
scene.add(trueTriad, learnedTriad);

// ------------------------------------------------------------------ UI
const ui = panel(document.getElementById('panel'));
ui.para('A hidden rotation R* turns input spinors x (grey) into targets y = R*x plus noise (green). ' +
  '<b>QuaternionRotation</b> holds an unconstrained 4-vector that is normalised to a unit quaternion on every forward pass, ' +
  'so the gradient of the loss <b>1 − fidelity(R x, y)</b> flows through quaternion_to_su2 and rotate. Each tick asks the ' +
  'server for a few Adam steps and redraws the model\'s predictions (orange) with lines to their targets.');
ui.section('Problem');
ui.slider({ label: 'points', min: 8, max: 256, step: 8, value: 64, format: (v) => `${v}`, onInput: (v) => { state.n = v; } });
ui.slider({ label: 'noise', min: 0, max: 0.3, step: 0.01, value: 0.03, onInput: (v) => { state.noise = v; } });
ui.slider({ label: 'learning rate', min: 0.005, max: 0.3, step: 0.005, value: 0.05, format: (v) => (+v).toFixed(3), onInput: (v) => { state.lr = v; } });
ui.slider({ label: 'seed', min: 0, max: 99, step: 1, value: 0, format: (v) => `${v}`, onInput: (v) => { state.seed = v; } });
ui.buttons([['new problem', () => reset(), 'primary']]);
ui.section('Training');
ui.slider({ label: 'steps / tick', min: 1, max: 20, step: 1, value: 2, format: (v) => `${v}`, onInput: (v) => { state.stepsPerTick = v; } });
const [runBtn] = ui.buttons([
  ['run', () => { state.running = !state.running; runBtn.textContent = state.running ? 'pause' : 'run'; if (state.running) runLoop(); }, 'primary'],
  ['single step', () => tick(1)],
]);
ui.section('Loss  (log scale)');
const lossCanvas = ui.canvas(140);
const plot = makePlot(lossCanvas, { xmin: 0, xmax: 200, ymin: 1e-4, ymax: 1, ylog: true, yticks: [[1, '1'], [1e-1, '0.1'], [1e-2, '0.01'], [1e-3, '10⁻³'], [1e-4, '10⁻⁴']] });
ui.section('Numbers');
const out = ui.readouts([['step', 'step'], ['loss', 'loss (1 − fidelity)'], ['q', 'learned quaternion'], ['qt', 'hidden quaternion'], ['err', 'max |SO(3) error|'], ['grad', '|∇ loss|']]);
ui.para('The learned and hidden quaternions may differ by an overall sign: q and −q are the same rotation, which is the double cover again.');
const overlay = document.getElementById('overlay');

// ----------------------------------------------------------------- data
function setPositions(points, arr) { points.geometry.setAttribute('position', new THREE.Float32BufferAttribute(arr.flat(), 3)); }

function show(d) {
  setPositions(predPts, d.predicted);
  const seg = [];
  for (let i = 0; i < d.predicted.length; i++) seg.push(...d.predicted[i], ...state.targets[i]);
  links.geometry.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3));
  learnedTriad.setMatrix3(d.so3);
  state.history.push([d.step, Math.max(d.loss, 1e-6)]);
  plot.state.xmax = Math.max(200, d.step);
  plot.draw([{ points: state.history, color: '#ffb347', width: 2 }]);
  out.step(d.step); out.loss(d.loss.toExponential(3));
  out.q(`(${d.quaternion.map((v) => fmt(v)).join(', ')})`);
  out.err(d.so3_error.toExponential(2)); out.grad(d.grad_norm.toExponential(2));
  overlay.innerHTML = d.step === 0
    ? '<b>Step 0.</b> The model starts at the identity: orange predictions sit on the grey inputs. Press run.'
    : d.so3_error < 0.02
      ? `<b>Converged</b> after ${d.step} steps: the solid frame lies on the faint hidden frame (max SO(3) error ${d.so3_error.toExponential(1)}). The residual loss is the noise floor.`
      : `<b>Step ${d.step}.</b> Adam is moving the solid frame onto the faint one; the lines shrink as predictions reach their targets.`;
}

async function reset() {
  state.running = false; runBtn.textContent = 'run'; state.history = [];
  try {
    const d = await api('learn_reset', { n_points: state.n, noise: state.noise, lr: state.lr, seed: state.seed });
    state.session = d.session; state.targets = d.targets; state.inputs = d.inputs;
    setPositions(inputPts, d.inputs); setPositions(targetPts, d.targets);
    trueTriad.setMatrix3(d.target_so3);
    out.qt(`(${d.target_quaternion.map((v) => fmt(v)).join(', ')})`);
    show(d);
  } catch (e) { showError(e); }
}

async function tick(steps) {
  if (state.busy || !state.session) return;
  state.busy = true;
  try { show(await api('learn_step', { session: state.session, steps })); }
  catch (e) { showError(e); state.running = false; }
  state.busy = false;
}

// Training is driven by a timer rather than the render loop so it keeps
// going when the tab is in the background; the scene picks up the latest
// state whenever it next draws.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function runLoop() {
  while (state.running) { await tick(state.stepsPerTick); await sleep(16); }
}
reset();
