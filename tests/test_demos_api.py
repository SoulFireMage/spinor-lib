"""Tests for the JSON handlers behind the three.js demos (demos/api.py)."""

import json
import math

import pytest
import torch

from demos.api import HANDLERS, _aberrate, COMPLEX, REAL
from spinor_lib import generate_boost, rapidity_from_velocity


def _json_roundtrip(obj):
    # allow_nan=False is what the server uses; NaN/inf would break the browser
    return json.loads(json.dumps(obj, allow_nan=False))


def test_rotation_sweep_double_cover():
    r = _json_roundtrip(HANDLERS["rotation_sweep"]({"axis": [0, 0, 1], "bloch": [1, 0, 0], "samples": 5}))
    overlaps = [o[0] for o in r["overlap"]]  # real parts at 0, π, 2π, 3π, 4π
    assert overlaps == pytest.approx([1, 0, -1, 0, 1], abs=1e-9)
    assert r["bloch"][2] == pytest.approx([1, 0, 0], abs=1e-9)  # Bloch vector back at 2π
    assert r["bloch"][1] == pytest.approx([-1, 0, 0], abs=1e-9)


def test_precession_sweep_shapes():
    r = HANDLERS["precession_sweep"]({"n_spins": 7, "samples": 4})
    assert len(r["bloch"]) == 4 and len(r["bloch"][0]) == 7 and len(r["bloch"][0][0]) == 3
    assert len(r["mean_spin"]) == 4


@pytest.mark.parametrize("turns, expected_mid_tumble", [(1, math.pi), (2, 0.0)])
def test_belt_far_end(turns, expected_mid_tumble):
    """One twist forces the buckle to flip halfway through; two twists never move it."""
    for t in (0.0, 0.5, 1.0):
        b = HANDLERS["belt"]({"turns": turns, "t": t, "segments": 40})
        expected = expected_mid_tumble if t == 0.5 else 0.0
        assert b["end_angle"] == pytest.approx(expected, abs=1e-6)
        assert b["centre"][0] == pytest.approx([0, 0, 0], abs=1e-12)  # clamped at the origin


def test_aberration_matches_textbook_formula():
    beta = 0.6
    eta = rapidity_from_velocity(torch.tensor(beta, dtype=REAL))
    A = generate_boost([0, 0, 1], -eta, handedness="right", dtype=COMPLEX)
    theta = torch.linspace(0, math.pi, 25, dtype=REAL)
    dirs = torch.stack([torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], -1)
    new_dir, doppler = _aberrate(dirs, A)
    # θ measured from the direction of motion in the rest frame:
    # cos θ' = (cos θ + β) / (1 + β cos θ);  Doppler factor D = γ (1 + β cos θ)
    cos_expected = (torch.cos(theta) + beta) / (1 + beta * torch.cos(theta))
    gamma = 1 / math.sqrt(1 - beta**2)
    d_expected = gamma * (1 + beta * torch.cos(theta))
    torch.testing.assert_close(new_dir[:, 2], cos_expected)
    torch.testing.assert_close(new_dir.norm(dim=-1), torch.ones_like(theta))
    torch.testing.assert_close(doppler, d_expected)


def test_aberration_handler_output():
    r = _json_roundtrip(HANDLERS["aberration"]({"beta": 0.9, "n_stars": 50}))
    assert r["gamma"] == pytest.approx(1 / math.sqrt(1 - 0.81))
    assert len(r["aberrated"]) == 50 and len(r["doppler"]) == 50
    # stars straight ahead blueshift, stars behind redshift
    assert max(r["doppler"]) > 1 > min(r["doppler"])
    assert len(r["lorentz"]) == 4 and len(r["sl2c"]) == 2


@pytest.mark.parametrize("theta, C", [(0.0, 0.0), (math.pi / 8, math.sqrt(0.5)), (math.pi / 4, 1.0)])
def test_two_qubit_entanglement(theta, C):
    r = HANDLERS["two_qubit"]({"theta": theta, "local0": {"axis": [1, 0, 0], "angle": 0.7}})
    assert r["concurrence"] == pytest.approx(C, abs=1e-9)
    for key in ("bloch0", "bloch1"):
        assert math.hypot(*r[key]) == pytest.approx(math.sqrt(1 - C**2), abs=1e-9)
    assert r["purity0"] == pytest.approx((1 + (1 - C**2)) / 2, abs=1e-9)
    assert sum(r["probabilities"]) == pytest.approx(1.0)


def test_learning_session_converges():
    s = HANDLERS["learn_reset"]({"n_points": 64, "noise": 0.0, "seed": 3})
    _json_roundtrip(s)  # must be valid JSON, including the step-0 loss
    assert s["step"] == 0 and math.isfinite(s["loss"])
    r = None
    for _ in range(6):
        r = HANDLERS["learn_step"]({"session": s["session"], "steps": 50})
    assert r["step"] == 300
    assert r["so3_error"] < 1e-2
    with pytest.raises(KeyError):
        HANDLERS["learn_step"]({"session": "nope"})
