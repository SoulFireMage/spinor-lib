import { THREE, latestOnly, createViewer, blochSphere, makeArrow, textSprite, panel, fmt, fmtC } from './common.js';

const AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const state = { theta: 0, l0: { axis: 'z', angle: 0 }, l1: { axis: 'z', angle: 0 }, spin: false, data: null };

// ---------------------------------------------------------------- scene
const viewer = createViewer(document.getElementById('view'), { cameraPos: [0.6, -5.4, 2.2] });
const { scene } = viewer;
const balls = [-1.45, 1.45].map((x, i) => {
  const g = blochSphere({ opacity: 0.12, poleLabels: ['|0⟩', '|1⟩'] });
  g.position.x = x;
  const title = textSprite(i ? 'qubit B' : 'qubit A', { size: 0.34, color: i ? '#ffb347' : '#5ec8ff' });
  title.position.set(0, 0, 1.75); g.add(title);
  scene.add(g);
  return g;
});
const arrows = [makeArrow(0x5ec8ff, { radius: 0.03, headLength: 0.2, headRadius: 0.075 }), makeArrow(0xffb347, { radius: 0.03, headLength: 0.2, headRadius: 0.075 })];
const ghosts = arrows.map((a, i) => makeArrow(i ? 0xffb347 : 0x5ec8ff, { radius: 0.01, headLength: 0.12, headRadius: 0.04, opacity: 0.25 }));
arrows.forEach((a, i) => { balls[i].add(a, ghosts[i]); });
const tips = arrows.map((a, i) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: i ? 0x664411 : 0x224466 })); balls[i].add(m); return m; });

// ------------------------------------------------------------------ UI
const ui = panel(document.getElementById('panel'));
ui.para('Start from |00⟩, rotate qubit A by R<sub>y</sub>(2θ), then apply a CNOT: <b>cos θ|00⟩ + sin θ|11⟩</b>. ' +
  'Each qubit on its own is described by the reduced density matrix from a partial trace. Its Bloch vector has length ' +
  '√(1 − C²) where C is the concurrence, so entangled qubits have arrows that sink inside the ball: a maximally entangled ' +
  'qubit points nowhere at all.');
ui.section('Entangling angle');
const sTheta = ui.slider({ label: 'θ', min: 0, max: 90, step: 0.5, value: 0, format: (v) => `${(+v).toFixed(1)}°`, onInput: (v) => { state.theta = v; refetch(); } });
ui.buttons([['product (0°)', () => setTheta(0)], ['partial (22.5°)', () => setTheta(22.5)], ['Bell (45°)', () => setTheta(45)]]);
ui.section('Local rotation of qubit A', 'Applied with embed_operator(R, dims=(2,2), target=0). Moves arrow A only; entanglement measures do not change.');
ui.select({ label: 'axis', value: 'z', options: [['x', 'x'], ['y', 'y'], ['z', 'z']], onChange: (v) => { state.l0.axis = v; refetch(); } });
ui.slider({ label: 'angle', min: 0, max: 360, step: 1, value: 0, format: (v) => `${v}°`, onInput: (v) => { state.l0.angle = v; refetch(); } });
ui.section('Local rotation of qubit B');
ui.select({ label: 'axis', value: 'z', options: [['x', 'x'], ['y', 'y'], ['z', 'z']], onChange: (v) => { state.l1.axis = v; refetch(); } });
ui.slider({ label: 'angle', min: 0, max: 360, step: 1, value: 0, format: (v) => `${v}°`, onInput: (v) => { state.l1.angle = v; refetch(); } });

ui.section('Entanglement');
const out = ui.readouts([['C', 'concurrence C'], ['S', 'entanglement entropy'], ['pa', 'purity Tr ρ_A²'], ['la', '|Bloch A|, |Bloch B|']]);
ui.section('State  (|00⟩, |01⟩, |10⟩, |11⟩)');
const amps = ui.html('');
ui.section('Correlations ⟨σᵢ ⊗ σⱼ⟩', 'Rows: qubit A axis; columns: qubit B axis. For a Bell state the arrows vanish but the correlations are ±1.');
const corr = ui.html('');
const overlay = document.getElementById('overlay');

// ----------------------------------------------------------------- data
const fetchState = latestOnly('two_qubit', (d) => { state.data = d; render(d); });
function refetch() {
  fetchState({ theta: state.theta * Math.PI / 180,
    local0: { axis: AXES[state.l0.axis], angle: state.l0.angle * Math.PI / 180 },
    local1: { axis: AXES[state.l1.axis], angle: state.l1.angle * Math.PI / 180 } });
}
function setTheta(v) { state.theta = v; sTheta.set(v); refetch(); }

function render(d) {
  const b = [d.bloch0, d.bloch1];
  b.forEach((v, i) => {
    arrows[i].setVector(...v); tips[i].position.set(...v);
    const len = Math.hypot(...v);
    ghosts[i].setVector(...(len > 1e-6 ? v.map((c) => c / len) : [0, 0, 1]));   // where the arrow would be if it were pure
    ghosts[i].visible = len > 1e-6 && len < 0.999;
  });
  out.C(fmt(d.concurrence)); out.S(`${fmt(d.entropy_bits)} bits`); out.pa(fmt(d.purity0));
  out.la(`${fmt(Math.hypot(...d.bloch0))}, ${fmt(Math.hypot(...d.bloch1))}`);
  const labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩'];
  amps.innerHTML = '<div class="readouts">' + d.amplitudes.map((a, i) =>
    `<div class="kv"><span>${labels[i]}</span><span>${fmtC(a)} &nbsp;<span style="color:#8a94a6">p=</span>${fmt(d.probabilities[i])}</span></div>`).join('') + '</div>';
  const cell = (v) => { const t = Math.abs(v); const bg = v > 0 ? `rgba(94,200,255,${0.15 + 0.6 * t})` : `rgba(255,123,114,${0.15 + 0.6 * t})`;
    return `<td style="background:${t < 1e-3 ? 'transparent' : bg}">${fmt(v, 2)}</td>`; };
  corr.innerHTML = `<table class="mat"><tr><td></td><td>x</td><td>y</td><td>z</td></tr>${['x', 'y', 'z'].map((r, i) => `<tr><td>${r}</td>${d.correlation[i].map(cell).join('')}</tr>`).join('')}</table>`;
  overlay.innerHTML = d.concurrence < 0.02
    ? '<b>Product state.</b> Both arrows reach the surface: each qubit is in a pure state of its own and the correlation matrix is just the outer product of the two arrows.'
    : d.concurrence > 0.98
      ? '<b>Maximally entangled.</b> Both reduced states are the maximally mixed I/2, the arrows have length 0, yet ⟨σᵢ⊗σⱼ⟩ is ±1 along three axes: all the information is in the correlations.'
      : `<b>Partially entangled</b>: C = ${fmt(d.concurrence, 2)}, arrows of length √(1−C²) = ${fmt(Math.sqrt(1 - d.concurrence ** 2), 2)}. Faint arrows show where the pure state would point. Local rotations turn the arrows but leave C and the entropy alone.`;
}

refetch();
