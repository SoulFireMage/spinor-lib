# three.js demos for spinor-lib

Five interactive scenes that put the library on screen. The browser never
re-implements any maths: a tiny Python server runs `spinor_lib` and streams
the results as JSON, and three.js only draws them.

```
python demos/server.py          # then open http://localhost:8765
```

Nothing beyond the library's own dependencies is required (torch, numpy and
the standard library on the Python side; three.js is loaded from a CDN in the
browser, so the first load needs internet access).

## The demos

| page | what you see | library calls doing the work |
| --- | --- | --- |
| `bloch.html` | A state rotated about any axis. Its Bloch arrow returns after 360° while the spinor comes back negated and needs 720°. The chart shows both periods; a cube driven by the same SU(2) element shows what SO(3) sees. Cloud mode precesses 150 spins in a field. | `from_bloch_vector`, `generate_su2_rotation` (batched over angles), `rotate`, `bloch_vector`, `inner_product`, `su2_to_quaternion`, `su2_to_so3` |
| `belt.html` | Dirac's belt trick. The belt's orientation along its length is a path in SO(3); lifted to SU(2) a 360° twist is an open arc from +1 to −1 and a 720° twist is a closed loop. A homotopy on the unit quaternions untwists the belt; the buckle's frame stays fixed for 720° and is forced to flip for 360°. The lifted path is drawn on the sphere alongside. | `generate_su2_rotation`, `su2_to_quaternion`, `quaternion_to_su2`, `su2_to_so3` |
| `aberration.html` | The night sky at up to 0.99 c. Each star direction becomes a Weyl spinor whose current is the photon's null wave-vector; a boost from `generate_boost` acts on the spinor and the current is read back. Stars bunch ahead and blueshift; the sky grid shows circles staying circles (Möbius action of SL(2,C)). | `rapidity_from_velocity`, `generate_boost`, `from_bloch_vector`, `rotate`, `weyl_current`, `sl2c_to_lorentz` |
| `entangle.html` | cos θ\|00⟩ + sin θ\|11⟩ built from a rotation and a CNOT. Partial traces give each qubit's reduced state; the Bloch arrows shrink to length √(1 − C²) and vanish for a Bell state while the ⟨σᵢ⊗σⱼ⟩ correlations stay ±1. Local rotations move arrows without changing entanglement. | `tensor_product`, `embed_operator`, `rotate`, `reduced_density_matrix`, `bloch_vector` (on a density matrix), `purity`, `concurrence`, `entanglement_entropy`, `expectation`, `kron`, `pauli` |
| `learn.html` | A `QuaternionRotation` module recovering a hidden rotation from noisy before/after pairs, one Adam step at a time. Predictions march onto their targets and the learned frame settles on the hidden one; loss curve and SO(3) error are live. | `QuaternionRotation`, `fidelity`, autograd, `su2_to_so3`, `su2_to_quaternion` |

## Layout

- `server.py` — `ThreadingHTTPServer`; serves `web/` and routes `POST /api/<name>` to `api.py`.
- `api.py` — one handler per demo. Each takes a dict of JSON values and returns
  a dict of JSON values (complex numbers as `[re, im]` pairs). The physics of
  each demo is documented in the handler docstrings.
- `web/common.js` — the three.js viewer (z up, orbit controls), Bloch sphere
  scenery, solid arrows, frame triads, small UI builders and 2D plots.
- `web/<demo>.html` + `web/<demo>.js` — one pair per demo.

Slider-driven demos use a "latest wins" request queue, so the server never
falls behind a fast-moving slider. The Bloch demo fetches a whole sweep of
angles at once and scrubs through it locally.

## Adding a demo

Register a handler in `api.py`:

```python
@handler("my_demo")
def my_demo(p: dict) -> dict:
    ...
    return {"bloch": r2j(bloch_vector(psi)), "matrix": c2j(R)}
```

and call it from a page with `api('my_demo', {...})` or `latestOnly('my_demo', onData)`.
