// Shared helpers for the spinor demos: API calls, a three.js viewer with
// physics-style axes (z up), Bloch-sphere scenery, fat arrows, small UI
// builders and 2D plots. Every number drawn here comes from the Python
// server, which runs spinor_lib; the browser only renders.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export { THREE, OrbitControls };

// ------------------------------------------------------------------ API

export async function api(name, params = {}) {
  const res = await fetch(`/api/${name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  let data;
  try { data = await res.json(); } catch { throw new Error(`${name}: bad response (${res.status})`); }
  if (!res.ok || data.error) throw new Error(data.error || `${name}: ${res.status}`);
  return data;
}

// A "latest wins" requester for slider-driven demos: while one call is in
// flight, only the newest parameters are kept, so the server never falls
// behind a fast-moving slider.
export function latestOnly(name, onData, onError = showError) {
  let inflight = false, pending = null;
  async function pump() {
    if (inflight || !pending) return;
    const params = pending; pending = null; inflight = true;
    try { onData(await api(name, params), params); }
    catch (e) { onError(e); }
    inflight = false;
    pump();
  }
  return (params) => { pending = params; pump(); };
}

export function showError(e) {
  console.error(e);
  let box = document.getElementById('errbox');
  if (!box) {
    box = document.createElement('div');
    box.id = 'errbox'; box.className = 'error';
    document.querySelector('.panel')?.prepend(box);
  }
  box.textContent = `Error: ${e.message}\nIs demos/server.py running?`;
}

// --------------------------------------------------------------- viewer

export function createViewer(container, opts = {}) {
  const { cameraPos = [2.8, -2.6, 1.9], target = [0, 0, 0], background = 0x0b0e14, fov = 42 } = opts;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(background);
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.01, 500);
  camera.up.set(0, 0, 1);                         // physics convention: z is up
  camera.position.set(...cameraPos);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(3, -4, 6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.5);
  fill.position.set(-4, 3, -2);
  scene.add(fill);

  let lastW = 0, lastH = 0;
  function resize() {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    if (w === lastW && h === lastH) return;   // ignore observer echoes of our own resize
    lastW = w; lastH = h;
    renderer.setSize(w, h, false);            // buffer size only; CSS pins the canvas to the container
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  const callbacks = [];
  const clock = new THREE.Clock();
  (function loop() {
    const dt = Math.min(clock.getDelta(), 0.1);
    for (const cb of callbacks) cb(dt, clock.elapsedTime);
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();

  return { scene, camera, renderer, controls, onFrame: (cb) => callbacks.push(cb) };
}

// ------------------------------------------------------------- scenery

const Z = new THREE.Vector3(0, 0, 1);

export function circle(radius, axis = 'z', offset = 0, material, segments = 96) {
  const pts = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2, c = radius * Math.cos(a), s = radius * Math.sin(a);
    pts.push(axis === 'z' ? new THREE.Vector3(c, s, offset)
      : axis === 'x' ? new THREE.Vector3(offset, c, s) : new THREE.Vector3(c, offset, s));
  }
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), material);
}

export function textSprite(text, { color = '#d7dde8', size = 0.22, font = 'bold 44px system-ui' } = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.font = font; ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 64);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sprite.scale.set(size * 2, size, 1);
  return sprite;
}

export function blochSphere({ radius = 1, opacity = 0.10, color = 0x7fa8ff, labels = true, poleLabels = ['|0⟩', '|1⟩'] } = {}) {
  const g = new THREE.Group();
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 64, 40),
    new THREE.MeshPhysicalMaterial({ color, transparent: true, opacity, roughness: 0.35, depthWrite: false, side: THREE.DoubleSide }));
  g.add(ball);
  const grid = new THREE.LineBasicMaterial({ color: 0x35435c, transparent: true, opacity: 0.85 });
  const eq = new THREE.LineBasicMaterial({ color: 0x5e6f92 });
  for (const lat of [-60, -30, 30, 60]) {
    const z = radius * Math.sin(lat * Math.PI / 180), r = radius * Math.cos(lat * Math.PI / 180);
    g.add(circle(r, 'z', z, grid));
  }
  g.add(circle(radius, 'z', 0, eq));
  for (let lon = 0; lon < 180; lon += 30) {
    const line = circle(radius, 'x', 0, lon % 90 === 0 ? eq : grid);
    line.rotation.z = lon * Math.PI / 180;
    g.add(line);
  }
  const axisMat = (c) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.6 });
  const L = radius * 1.25;
  const seg = (a, b, m) => new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), m);
  g.add(seg(new THREE.Vector3(-L, 0, 0), new THREE.Vector3(L, 0, 0), axisMat(0xff6b6b)));
  g.add(seg(new THREE.Vector3(0, -L, 0), new THREE.Vector3(0, L, 0), axisMat(0x6bff8a)));
  g.add(seg(new THREE.Vector3(0, 0, -L), new THREE.Vector3(0, 0, L), axisMat(0x6b9bff)));
  if (labels) {
    const put = (t, x, y, z, color) => { const s = textSprite(t, { color }); s.position.set(x, y, z); g.add(s); };
    put('x', L * 1.08, 0, 0, '#ff9b9b');
    put('y', 0, L * 1.08, 0, '#9bffb0');
    put(poleLabels[0], 0, 0, L * 1.1, '#a9c3ff');
    put(poleLabels[1], 0, 0, -L * 1.1, '#a9c3ff');
  }
  return g;
}

// A solid arrow (cylinder + cone) that can be re-pointed with setVector().
export function makeArrow(color, { radius = 0.022, headLength = 0.16, headRadius = 0.06, opacity = 1 } = {}) {
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.1, transparent: opacity < 1, opacity });
  const shaftGeo = new THREE.CylinderGeometry(radius, radius, 1, 16); shaftGeo.rotateX(Math.PI / 2); shaftGeo.translate(0, 0, 0.5);
  const headGeo = new THREE.ConeGeometry(headRadius, headLength, 20); headGeo.rotateX(Math.PI / 2); headGeo.translate(0, 0, headLength / 2);
  const shaft = new THREE.Mesh(shaftGeo, mat), head = new THREE.Mesh(headGeo, mat);
  const g = new THREE.Group();
  g.add(shaft, head);
  g.material = mat;
  g.setVector = (x, y, z) => {
    const v = Array.isArray(x) ? new THREE.Vector3(...x) : new THREE.Vector3(x, y, z);
    const len = v.length();
    if (len < 1e-6) { g.visible = false; return g; }
    g.visible = true;
    g.quaternion.setFromUnitVectors(Z, v.clone().divideScalar(len));
    const hl = Math.min(headLength, len * 0.6);
    shaft.scale.z = Math.max(len - hl, 1e-4);
    head.position.z = len - hl;
    head.scale.set(1, 1, hl / headLength);
    return g;
  };
  g.setVector(0, 0, 1);
  return g;
}

// Three arrows (x red, y green, z blue) showing a rotated frame; feed it a 3×3 matrix.
export function makeTriad(size = 0.5, opts = {}) {
  const g = new THREE.Group();
  const cols = [0xff5c5c, 0x5cff7a, 0x5c8cff];
  g.arrows = cols.map((c) => { const a = makeArrow(c, { radius: 0.018, headLength: 0.12, headRadius: 0.045, ...opts }); g.add(a); return a; });
  g.setMatrix3 = (m) => {                 // m: row-major nested [[..],[..],[..]]
    for (let j = 0; j < 3; j++) g.arrows[j].setVector(m[0][j] * size, m[1][j] * size, m[2][j] * size);
    return g;
  };
  g.setMatrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  return g;
}

export function polyline(points, color, { opacity = 1, loop = false, width = 1 } = {}) {
  const geo = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  const mat = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, linewidth: width });
  return loop ? new THREE.LineLoop(geo, mat) : new THREE.Line(geo, mat);
}

// A growable trail line with a fixed capacity.
export function makeTrail(capacity, color, opacity = 0.9) {
  const positions = new Float32Array(capacity * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, 0);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  line.frustumCulled = false;
  let n = 0;
  line.setPoints = (pts) => {                     // pts: array of [x,y,z]
    n = Math.min(pts.length, capacity);
    for (let i = 0; i < n; i++) { positions[3 * i] = pts[i][0]; positions[3 * i + 1] = pts[i][1]; positions[3 * i + 2] = pts[i][2]; }
    geo.attributes.position.needsUpdate = true;
    geo.setDrawRange(0, n);
  };
  line.clear = () => { n = 0; geo.setDrawRange(0, 0); };
  return line;
}

export function setMatrix3(obj, m) {
  const M = new THREE.Matrix4().set(m[0][0], m[0][1], m[0][2], 0, m[1][0], m[1][1], m[1][2], 0, m[2][0], m[2][1], m[2][2], 0, 0, 0, 0, 1);
  obj.setRotationFromMatrix(M);
}

// ------------------------------------------------------------------ UI

export function panel(root) {
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const api = {
    root,
    section(title, text) { root.append(el('h2', null, title)); if (text) root.append(el('p', null, text)); return api; },
    para(text) { root.append(el('p', null, text)); return api; },
    html(html) { const d = el('div', null, html); root.append(d); return d; },
    slider({ label, min, max, step = 0.01, value, format = (v) => (+v).toFixed(2), onInput }) {
      const wrap = el('label', 'ctl');
      const span = el('span', null, label), input = el('input'), out = el('output');
      Object.assign(input, { type: 'range', min, max, step, value });
      out.textContent = format(value);
      input.addEventListener('input', () => { out.textContent = format(input.valueAsNumber); onInput?.(input.valueAsNumber); });
      wrap.append(span, input, out); root.append(wrap);
      input.set = (v) => { input.value = v; out.textContent = format(v); };
      return input;
    },
    select({ label, options, value, onChange }) {
      const wrap = el('label', 'ctl'), span = el('span', null, label), sel = el('select');
      for (const [v, t] of options) { const o = el('option', null, t); o.value = v; sel.append(o); }
      sel.value = value;
      sel.addEventListener('change', () => onChange?.(sel.value));
      wrap.append(span, sel, el('span')); root.append(wrap);
      return sel;
    },
    toggle({ label, value = false, onChange }) {
      const wrap = el('label', 'toggle'), input = el('input'); input.type = 'checkbox'; input.checked = value;
      input.addEventListener('change', () => onChange?.(input.checked));
      wrap.append(input, document.createTextNode(label)); root.append(wrap);
      return input;
    },
    buttons(list) {
      const row = el('div', 'row');
      const out = list.map(([label, fn, cls]) => { const b = el('button', cls, label); b.addEventListener('click', fn); row.append(b); return b; });
      root.append(row);
      return out;
    },
    readouts(keys) {
      const box = el('div', 'readouts'); root.append(box);
      const setters = {};
      for (const [key, label] of keys) {
        const kv = el('div', 'kv'); const v = el('span', null, '–');
        kv.append(el('span', null, label), v); box.append(kv);
        setters[key] = (html) => { v.innerHTML = html; };
      }
      return setters;
    },
    canvas(height = 130) { const c = el('canvas', 'plot'); c.style.height = `${height}px`; root.append(c); return c; },
  };
  return api;
}

// ------------------------------------------------------------ formatting

export const fmt = (x, d = 3) => (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d);
export const fmtC = ([re, im], d = 3) => `${fmt(re, d)}${im < 0 ? '−' : '+'}${fmt(Math.abs(im), d)}i`;
export const deg = (rad) => (rad * 180 / Math.PI);

export function matrixHTML(rows, { complex = false, d = 3 } = {}) {
  const cell = (v) => {
    const s = complex ? fmtC(v, d) : fmt(v, d);
    const neg = complex ? v[0] < -1e-9 : v < -1e-9;
    return `<td class="${neg ? 'neg' : ''}">${s}</td>`;
  };
  return `<table class="mat">${rows.map((r) => `<tr>${r.map(cell).join('')}</tr>`).join('')}</table>`;
}

// ---------------------------------------------------------------- plots

// A tiny 2D line plot. series: [{ points: [[x,y],...], color, dash }], marker: x value
export function makePlot(canvas, { xmin = 0, xmax = 1, ymin = -1, ymax = 1, xlabel = '', ylog = false, xticks = [], yticks = [] } = {}) {
  const ctx = canvas.getContext('2d');
  const state = { xmin, xmax, ymin, ymax, xlabel, ylog, xticks, yticks };
  function draw(series, marker) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 34, r: 8, t: 8, b: 18 };
    const ty = (y) => state.ylog ? Math.log10(Math.max(y, 1e-12)) : y;
    const y0 = ty(state.ymin), y1 = ty(state.ymax);
    const X = (x) => pad.l + (x - state.xmin) / (state.xmax - state.xmin) * (W - pad.l - pad.r);
    const Y = (y) => H - pad.b - (ty(y) - y0) / (y1 - y0) * (H - pad.t - pad.b);
    ctx.strokeStyle = '#26314a'; ctx.lineWidth = 1; ctx.fillStyle = '#8a94a6'; ctx.font = '10px system-ui';
    for (const [v, label] of state.xticks) { ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), H - pad.b); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(label, X(v), H - 5); }
    for (const [v, label] of state.yticks) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(W - pad.r, Y(v)); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(label, pad.l - 4, Y(v) + 3); }
    ctx.strokeStyle = '#3a4a66'; ctx.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      s.points.forEach(([x, y], i) => { const px = X(x), py = Math.max(pad.t, Math.min(H - pad.b, Y(y))); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (marker != null) { ctx.strokeStyle = '#ffb347'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(marker), pad.t); ctx.lineTo(X(marker), H - pad.b); ctx.stroke(); }
  }
  return { draw, state };
}

// Two phasor dials for the components of a 2-spinor.
export function drawPhasors(canvas, comps, labels = ['ψ₀', 'ψ₁']) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const r = Math.min(H * 0.38, W * 0.2);
  comps.forEach(([re, im], i) => {
    const cx = W * (0.27 + 0.46 * i), cy = H * 0.52;
    ctx.strokeStyle = '#3a4a66'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy); ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); ctx.stroke();
    ctx.strokeStyle = i ? '#ffb347' : '#5ec8ff'; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + re * r, cy - im * r); ctx.stroke();
    ctx.fillStyle = i ? '#ffb347' : '#5ec8ff';
    ctx.beginPath(); ctx.arc(cx + re * r, cy - im * r, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#8a94a6'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(`${labels[i]} = ${fmtC([re, im], 2)}`, cx, H - 4);
  });
}
