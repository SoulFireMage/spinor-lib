"""JSON-friendly handlers that drive the three.js demos with ``spinor_lib``.

Every handler takes a dict of plain JSON values and returns a dict of plain
JSON values. Real tensors become nested lists; complex tensors become nested
lists of ``[re, im]`` pairs. The demos never re-implement any of the maths:
all rotations, boosts, Bloch vectors, partial traces and gradient steps are
computed here by the library and shipped to the browser as numbers.
"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from typing import Callable, Dict

import torch

from spinor_lib import (
    QuaternionRotation,
    Spinor,
    bloch_vector,
    concurrence,
    embed_operator,
    entanglement_entropy,
    expectation,
    fidelity,
    from_bloch_vector,
    generate_boost,
    generate_su2_rotation,
    inner_product,
    kron,
    pauli,
    purity,
    quaternion_to_su2,
    rapidity_from_velocity,
    reduced_density_matrix,
    rotate,
    sl2c_to_lorentz,
    su2_to_quaternion,
    su2_to_so3,
    tensor_product,
    weyl_current,
)

REAL = torch.float64
COMPLEX = torch.complex128

HANDLERS: Dict[str, Callable[[dict], dict]] = {}


def handler(name: str):
    def register(fn):
        HANDLERS[name] = fn
        return fn

    return register


# ------------------------------------------------------------ serialisation


def r2j(t: torch.Tensor):
    """Real tensor -> nested lists."""
    return t.detach().cpu().tolist()


def c2j(t: torch.Tensor):
    """Complex tensor -> nested lists of [re, im]."""
    return torch.stack([t.real, t.imag], dim=-1).detach().cpu().tolist()


def _int(p: dict, key: str, default: int, lo: int, hi: int) -> int:
    """An integer parameter clamped to [lo, hi], so a public server cannot be
    asked for a million stars or a hundred-thousand-sample sweep."""
    try:
        v = int(p.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _vec3(v, default):
    v = torch.as_tensor(v if v is not None else default, dtype=REAL)
    if v.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {tuple(v.shape)}")
    if float(v.norm()) < 1e-9:
        raise ValueError("vector must be non-zero")
    return v


def _state_from_params(p: dict) -> Spinor:
    """A single normalised spinor from either Bloch coordinates or components."""
    if p.get("spinor") is not None:
        comps = torch.tensor(p["spinor"], dtype=REAL)  # [2, 2] as [re, im]
        return Spinor(torch.complex(comps[:, 0], comps[:, 1])).normalize()
    return from_bloch_vector(_vec3(p.get("bloch"), [0, 0, 1]), dtype=COMPLEX)


# ------------------------------------------- demo 1: Bloch sphere & double cover


@handler("rotation_sweep")
def rotation_sweep(p: dict) -> dict:
    """Rotate one state through angles 0..max_angle about an axis.

    Returns everything the browser needs to scrub the angle without further
    requests: the Bloch vector, the raw components, the overlap with the
    starting state (which reaches -1 at 2π and +1 at 4π), the unit quaternion
    and the SO(3) matrix of every sample.
    """
    axis = _vec3(p.get("axis"), [0, 0, 1])
    max_angle = float(p.get("max_angle", 4 * math.pi))
    n = _int(p, "samples", 721, 2, 2001)
    psi0 = _state_from_params(p)

    angles = torch.linspace(0, max_angle, n, dtype=REAL)
    R = generate_su2_rotation(axis, angles)  # [n, 2, 2]
    psi = rotate(psi0, R)  # [n, 2]
    return {
        "angles": r2j(angles),
        "axis": r2j(axis / axis.norm()),
        "bloch": r2j(bloch_vector(psi)),
        "components": c2j(psi.data),
        "overlap": c2j(inner_product(psi0, psi)),
        "quaternion": r2j(su2_to_quaternion(R)),
        "so3": r2j(su2_to_so3(R)),
        "matrix": c2j(R),
    }


@handler("precession_sweep")
def precession_sweep(p: dict) -> dict:
    """Larmor precession of a cloud of random spins about a field axis.

    The time evolution exp(-i B·σ t/2) of a spin-1/2 in a field is exactly an
    SU(2) rotation about B by angle ∝ t, so a whole batch is one broadcast
    call to ``generate_su2_rotation``.
    """
    axis = _vec3(p.get("axis"), [0, 0, 1])
    n_spins = _int(p, "n_spins", 150, 1, 500)
    n = _int(p, "samples", 241, 2, 721)
    max_angle = float(p.get("max_angle", 4 * math.pi))
    g = torch.Generator().manual_seed(int(p.get("seed", 0)))
    psi0 = Spinor(torch.randn(n_spins, 2, dtype=COMPLEX, generator=g)).normalize()

    angles = torch.linspace(0, max_angle, n, dtype=REAL)
    R = generate_su2_rotation(axis, angles)[:, None]  # [n, 1, 2, 2]
    psi = rotate(psi0, R)  # [n, n_spins, 2]
    b = bloch_vector(psi)
    return {
        "angles": r2j(angles),
        "axis": r2j(axis / axis.norm()),
        "bloch": r2j(b),
        "mean_spin": r2j(b.mean(dim=1)),
    }


# ------------------------------------------------------ demo 2: the belt trick


@handler("belt")
def belt(p: dict) -> dict:
    """A belt twisted by ``turns`` full turns, deformed by a homotopy parameter.

    The belt's orientation along its length is a path R(s) in SO(3). Lifted to
    SU(2) the 2π twist is a path from I to -I (an open arc) while the 4π twist
    is a closed loop. Sliding the lifted path along great circles of S³
    towards a fixed rotation gives a family of paths P_t(s); left-multiplying
    by P_t(0)⁻¹ pins the near end of the belt to the identity. For 4π the far
    end stays put for every t and the twist disappears; for 2π the far end is
    forced to tumble, which is exactly why one twist cannot be undone.

    ``slerp_su2`` is deliberately *not* used: it takes the shortest arc in
    SO(3) and so discards the sign of the SU(2) element, which here is the
    whole point. The interpolation is done on unit quaternions instead,
    through the library's ``su2_to_quaternion`` / ``quaternion_to_su2``.
    """
    turns = float(p.get("turns", 2))
    t = float(p.get("t", 0.0))
    n = _int(p, "segments", 96, 4, 400)

    s = torch.linspace(0, 1, n, dtype=REAL)
    loop = generate_su2_rotation([0, 0, 1], 2 * math.pi * turns * s)  # [n, 2, 2]
    anchor = generate_su2_rotation([1, 0, 0], math.pi, dtype=COMPLEX)  # contract towards this
    q_loop, q_anchor = su2_to_quaternion(loop), su2_to_quaternion(anchor)
    lam = t * math.pi / 2  # the two are orthogonal in R⁴, so this is an exact slerp
    P = quaternion_to_su2(math.cos(lam) * q_loop + math.sin(lam) * q_anchor)  # [n, 2, 2]
    R = P[0].conj().mT @ P  # pin the near end: R(0) = I
    M = su2_to_so3(R)  # [n, 3, 3]

    # Centreline: integrate the belt's local "along" direction (its z axis).
    tangent = M[:, :, 2]  # M @ ẑ
    length = float(p.get("length", 3.0))
    centre = torch.cumsum(tangent, dim=0) * (length / n)
    centre = centre - centre[0]

    end_q = su2_to_quaternion(R[-1])
    return {
        "s": r2j(s),
        "so3": r2j(M),
        "centre": r2j(centre),
        "quaternion": r2j(su2_to_quaternion(R)),
        "lifted_path": r2j(su2_to_quaternion(P)),  # the path in S³ before pinning
        "end_so3": r2j(M[-1]),
        "end_angle": float(2 * torch.arccos(end_q[0].abs().clamp(max=1))),
    }


# --------------------------------------- demo 3: relativistic aberration


def _star_field(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(n, 3, dtype=REAL, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    mag = torch.rand(n, dtype=REAL, generator=g)
    return d, mag


def _sky_grid(n_lat=5, n_lon=12, samples=73):
    """Latitude and longitude circles of the sky as polylines, shape [k, samples, 3]."""
    u = torch.linspace(0, 2 * math.pi, samples, dtype=REAL)
    lines = []
    for i in range(1, n_lat + 1):
        th = math.pi * i / (n_lat + 1)
        lines.append(torch.stack([math.sin(th) * torch.cos(u), math.sin(th) * torch.sin(u),
                                  torch.full_like(u, math.cos(th))], -1))
    v = torch.linspace(0, math.pi, samples, dtype=REAL)
    for j in range(n_lon):
        ph = 2 * math.pi * j / n_lon
        lines.append(torch.stack([torch.sin(v) * math.cos(ph), torch.sin(v) * math.sin(ph),
                                  torch.cos(v)], -1))
    return torch.stack(lines)


def _aberrate(directions: torch.Tensor, A: torch.Tensor):
    """Transform sky directions (unit vectors pointing *at* the sources).

    A photon arriving from direction n̂ has null wave-vector k = (1, -n̂). The
    right-handed spinor with Bloch vector -n̂ has Weyl current exactly k, so
    boosting the spinor and reading its current back gives Λk, from which the
    new direction and the Doppler factor follow.
    """
    psi = from_bloch_vector(-directions, dtype=COMPLEX)
    k = weyl_current(rotate(psi, A), "right")  # [..., 4] = (D, -D n̂')
    doppler = k[..., 0]
    new_dir = -k[..., 1:] / doppler[..., None]
    return new_dir, doppler


@handler("aberration")
def aberration(p: dict) -> dict:
    """The celestial sphere seen by an observer moving at speed β.

    Boosts are elements of SL(2,C) acting on Weyl spinors, and SL(2,C) acts
    on the sky as a Möbius transformation: circles stay circles, stars bunch
    up ahead of you and blueshift there.
    """
    beta = float(p.get("beta", 0.0))
    beta = max(-0.999, min(0.999, beta))
    direction = _vec3(p.get("direction"), [0, 0, 1])
    n_stars = _int(p, "n_stars", 1500, 1, 5000)
    seed = int(p.get("seed", 1))

    eta = rapidity_from_velocity(torch.tensor(beta, dtype=REAL))
    # Λ(η) carries rest -> velocity β·n, i.e. maps the moving frame's
    # coordinates into ours; the observer's own view needs the inverse boost.
    A = generate_boost(direction, -eta, handedness="right", dtype=COMPLEX)

    stars, mag = _star_field(n_stars, seed)
    new_dir, doppler = _aberrate(stars, A)
    grid = _sky_grid()
    grid_new, _ = _aberrate(grid, A)

    return {
        "beta": beta,
        "gamma": float(torch.cosh(eta)),
        "rapidity": float(eta),
        "direction": r2j(direction / direction.norm()),
        "stars": r2j(stars),
        "magnitude": r2j(mag),
        "aberrated": r2j(new_dir),
        "doppler": r2j(doppler),
        "grid": r2j(grid),
        "grid_aberrated": r2j(grid_new),
        "sl2c": c2j(A),
        "lorentz": r2j(sl2c_to_lorentz(A, "right")),
    }


# ---------------------------------------------- demo 4: two-qubit entanglement


CNOT = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=COMPLEX
)


@handler("two_qubit")
def two_qubit(p: dict) -> dict:
    """cos θ|00⟩ + sin θ|11⟩ followed by local rotations on each qubit.

    The reduced density matrices come from a partial trace; their Bloch
    vectors shrink inside the ball as the concurrence grows. Local rotations
    move the arrows but cannot change concurrence, entropy or purity.
    """
    theta = float(p.get("theta", 0.0))
    up = Spinor(torch.tensor([1, 0], dtype=COMPLEX))
    psi = tensor_product(up, up)

    ry = generate_su2_rotation([0, 1, 0], 2 * theta, dtype=COMPLEX)
    psi = rotate(psi, embed_operator(ry, (2, 2), target=0))
    psi = rotate(psi, CNOT)

    for target in (0, 1):
        loc = p.get(f"local{target}") or {}
        angle = float(loc.get("angle", 0.0))
        if abs(angle) > 1e-12:
            R = generate_su2_rotation(_vec3(loc.get("axis"), [0, 0, 1]), angle, dtype=COMPLEX)
            psi = rotate(psi, embed_operator(R, (2, 2), target=target))

    rho0 = reduced_density_matrix(psi, (2, 2), keep=0)
    rho1 = reduced_density_matrix(psi, (2, 2), keep=1)
    sig = pauli(dtype=COMPLEX)
    corr = torch.stack(
        [torch.stack([expectation(psi, kron(sig[i], sig[j])) for j in range(3)]) for i in range(3)]
    )
    return {
        "amplitudes": c2j(psi.data),
        "probabilities": r2j(psi.data.abs() ** 2),
        "bloch0": r2j(bloch_vector(rho0)),
        "bloch1": r2j(bloch_vector(rho1)),
        "purity0": float(purity(rho0)),
        "purity1": float(purity(rho1)),
        "concurrence": float(concurrence(psi)),
        "entropy_bits": float(entanglement_entropy(psi, (2, 2), keep=0, base=2)),
        "correlation": r2j(corr),
        "rho0": c2j(rho0),
    }


# ------------------------------------------ demo 5: learning a rotation live


class _Session:
    def __init__(self, p: dict):
        seed = int(p.get("seed", 0))
        g = torch.Generator().manual_seed(seed)
        n = _int(p, "n_points", 64, 4, 512)
        noise = float(p.get("noise", 0.02))
        lr = float(p.get("lr", 0.05))

        axis = torch.randn(3, generator=g)
        angle = torch.rand(1, generator=g).item() * 1.6 * math.pi + 0.4 * math.pi
        self.target = generate_su2_rotation(axis, torch.tensor(angle))
        self.x = Spinor(torch.randn(n, 2, dtype=torch.complex64, generator=g)).normalize()
        y = rotate(self.x, self.target)
        y = Spinor(y.data + noise * torch.randn(y.data.shape, dtype=y.dtype, generator=g))
        self.y = y.normalize()
        self.model = QuaternionRotation()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.step = 0
        with torch.no_grad():
            self.loss = (1 - fidelity(self.model(self.x), self.y)).mean().item()

    def train(self, steps: int):
        for _ in range(steps):
            self.opt.zero_grad()
            loss = (1 - fidelity(self.model(self.x), self.y)).mean()
            loss.backward()
            self.opt.step()
            self.step += 1
            self.loss = loss.item()

    def snapshot(self) -> dict:
        with torch.no_grad():
            R = self.model.matrix
            pred = self.model(self.x)
            err = (su2_to_so3(R) - su2_to_so3(self.target)).abs().max().item()
            grad = self.model.quaternion.grad
            return {
                "step": self.step,
                "loss": self.loss,
                "quaternion": r2j(su2_to_quaternion(R)),
                "so3": r2j(su2_to_so3(R)),
                "predicted": r2j(bloch_vector(pred)),
                "so3_error": err,
                "grad_norm": float(grad.norm()) if grad is not None else 0.0,
            }


_SESSIONS: "OrderedDict[str, _Session]" = OrderedDict()
_MAX_SESSIONS = 16


@handler("learn_reset")
def learn_reset(p: dict) -> dict:
    """Start a fresh ``QuaternionRotation`` fit against a hidden rotation."""
    sid = uuid.uuid4().hex
    sess = _Session(p)
    _SESSIONS[sid] = sess
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    with torch.no_grad():
        return {
            "session": sid,
            "inputs": r2j(bloch_vector(sess.x)),
            "targets": r2j(bloch_vector(sess.y)),
            "target_quaternion": r2j(su2_to_quaternion(sess.target)),
            "target_so3": r2j(su2_to_so3(sess.target)),
            **sess.snapshot(),
        }


@handler("learn_step")
def learn_step(p: dict) -> dict:
    """Run a few optimiser steps and report the new state."""
    sess = _SESSIONS.get(str(p.get("session", "")))
    if sess is None:
        raise KeyError("unknown or expired session; call learn_reset")
    sess.train(_int(p, "steps", 1, 1, 200))
    return sess.snapshot()
