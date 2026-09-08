"""Tests for the equivariant building blocks and nn layers."""

import math

import pytest
import torch
from torch import nn

from spinor_lib import (
    EquivariantLinear,
    InvariantReadout,
    NormGate,
    QuaternionRotation,
    Spinor,
    apply_vector,
    bloch_vector,
    fidelity,
    generate_su2_rotation,
    global_phase,
    gram,
    inner_product,
    invariant_features,
    overlap,
    rotate,
    singlet,
    su2_to_so3,
    vector,
)

C128 = torch.complex128
F64 = torch.float64


def random_spinor(*batch, generator=None):
    return Spinor(torch.randn(*batch, 2, dtype=C128, generator=generator))


def random_su2(*batch, generator=None):
    axis = torch.randn(*batch, 3, dtype=F64, generator=generator)
    angle = torch.rand(tuple(batch), dtype=F64, generator=generator) * 2 * math.pi
    return generate_su2_rotation(axis, angle)


class TestClebschGordan:
    def test_singlet_is_antisymmetric_and_invariant(self):
        g = torch.Generator().manual_seed(0)
        a, b = random_spinor(7, generator=g), random_spinor(7, generator=g)
        R = random_su2(7, generator=g)
        assert torch.allclose(singlet(a, b), -singlet(b, a))
        assert torch.allclose(singlet(rotate(a, R), rotate(b, R)), singlet(a, b), atol=1e-12)
        assert torch.allclose(singlet(a, a), torch.zeros(7, dtype=C128))

    def test_overlap_is_inner_product(self):
        g = torch.Generator().manual_seed(1)
        a, b = random_spinor(3, generator=g), random_spinor(3, generator=g)
        assert torch.allclose(overlap(a, b), inner_product(a, b))

    def test_vector_rotates_as_so3_vector(self):
        g = torch.Generator().manual_seed(2)
        a, b = random_spinor(5, generator=g), random_spinor(5, generator=g)
        R = random_su2(5, generator=g)
        M = su2_to_so3(R).to(C128)
        lhs = vector(rotate(a, R), rotate(b, R))
        rhs = torch.einsum("bij,bj->bi", M, vector(a, b))
        assert torch.allclose(lhs, rhs, atol=1e-12)

    def test_self_vector_is_unnormalised_bloch_vector(self):
        g = torch.Generator().manual_seed(3)
        a = random_spinor(4, generator=g)
        v = vector(a)
        assert v.dtype == F64
        assert torch.allclose(v, bloch_vector(a) * (a.norm() ** 2)[:, None])

    def test_apply_vector_is_equivariant(self):
        g = torch.Generator().manual_seed(4)
        s = random_spinor(6, generator=g)
        v = torch.randn(6, 3, dtype=F64, generator=g)
        R = random_su2(6, generator=g)
        M = su2_to_so3(R)
        lhs = apply_vector(torch.einsum("bij,bj->bi", M, v), rotate(s, R))
        rhs = rotate(apply_vector(v, s), R)
        assert torch.allclose(lhs.data, rhs.data, atol=1e-12)

    def test_apply_vector_along_z_is_pauli_z(self):
        s = random_spinor()
        out = apply_vector(torch.tensor([0.0, 0.0, 1.0]), s)
        assert torch.allclose(out.data, s.data * torch.tensor([1, -1], dtype=C128))

    def test_gram_and_invariant_features(self):
        g = torch.Generator().manual_seed(5)
        f = random_spinor(3, 4, generator=g)  # 3 batches, 4 channels
        G = gram(f)
        assert G.shape == (3, 4, 4)
        assert torch.allclose(G, G.conj().mT)
        inv = invariant_features(f)
        assert inv.shape == (3, 16) and inv.dtype == F64
        R = random_su2(3, 1, generator=g)  # one rotation per batch, shared across channels
        assert torch.allclose(invariant_features(rotate(f, R)), inv, atol=1e-12)
        assert torch.allclose(invariant_features(global_phase(f, 0.7)), inv, atol=1e-12)


class TestQuaternionRotation:
    def test_starts_at_identity(self):
        m = QuaternionRotation()
        assert torch.allclose(m.matrix, torch.eye(2, dtype=m.matrix.dtype))

    def test_stays_in_su2_and_learns(self):
        torch.manual_seed(0)
        target = generate_su2_rotation(torch.randn(3), torch.tensor(2.0))
        x = Spinor(torch.randn(128, 2, dtype=torch.complex64))
        y = rotate(x, target)
        m = QuaternionRotation()
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            loss = (1 - fidelity(m(x), y)).mean()
            loss.backward()
            opt.step()
        assert loss.item() < 1e-4
        R = m.matrix
        assert torch.allclose(R @ R.conj().mT, torch.eye(2, dtype=R.dtype), atol=1e-5)
        assert torch.allclose(su2_to_so3(R), su2_to_so3(target), atol=1e-2)

    def test_gradient_at_identity_is_nonzero(self):
        m = QuaternionRotation()
        s = Spinor(torch.tensor([1.0, 0.0], dtype=torch.complex64))
        target = Spinor(torch.tensor([1.0, 1.0], dtype=torch.complex64))
        (1 - fidelity(m(s), target)).backward()
        assert m.quaternion.grad.abs().sum() > 0

    def test_bad_init(self):
        with pytest.raises(ValueError):
            QuaternionRotation(torch.zeros(3))


class TestEquivariantLayers:
    @staticmethod
    def network(channels=6):
        return nn.Sequential(
            EquivariantLinear(4, channels, dtype=C128),
            NormGate(channels),
            EquivariantLinear(channels, channels, dtype=C128),
            NormGate(channels, act=torch.nn.functional.silu),
            EquivariantLinear(channels, 3, dtype=C128),
        )

    def test_layers_are_equivariant(self):
        g = torch.Generator().manual_seed(6)
        torch.manual_seed(6)
        net = self.network()
        f = random_spinor(5, 4, generator=g)
        R = random_su2(5, 1, generator=g)
        lhs = net(rotate(f, R))
        rhs = rotate(net(f), R)
        assert lhs.shape == (5, 3, 2)
        assert torch.allclose(lhs.data, rhs.data, atol=1e-10)
        # Global phase equivariance as well.
        assert torch.allclose(net(global_phase(f, 1.1)).data, global_phase(net(f), 1.1).data, atol=1e-10)

    def test_readout_is_invariant(self):
        g = torch.Generator().manual_seed(7)
        torch.manual_seed(7)
        net = nn.Sequential(self.network(), InvariantReadout(3, 2))
        net = net.double()
        f = random_spinor(5, 4, generator=g)
        R = random_su2(5, 1, generator=g)
        out = net(f)
        assert out.shape == (5, 2) and out.dtype == F64
        assert torch.allclose(net(rotate(f, R)), out, atol=1e-10)

    def test_readout_hidden_layer_is_invariant(self):
        g = torch.Generator().manual_seed(10)
        torch.manual_seed(10)
        head = InvariantReadout(4, 3, hidden=8).double()
        f = random_spinor(5, 4, generator=g)
        R = random_su2(5, 1, generator=g)
        out = head(f)
        assert out.shape == (5, 3)
        assert torch.allclose(head(rotate(f, R)), out, atol=1e-10)

    def test_equivariant_linear_matches_einsum(self):
        torch.manual_seed(8)
        layer = EquivariantLinear(3, 2, dtype=C128)
        f = random_spinor(4, 3)
        expected = torch.einsum("oc,bci->boi", layer.weight, f.data)
        assert torch.allclose(layer(f).data, expected)

    def test_norm_gate_scales_by_invariant(self):
        gate = NormGate(2)
        with torch.no_grad():
            gate.scale.fill_(0.0)
            gate.shift.fill_(0.0)
        f = random_spinor(3, 2)
        assert torch.allclose(gate(f).data, 0.5 * f.data)  # sigmoid(0) = 1/2

    def test_shape_checks(self):
        f = random_spinor(2, 3)
        with pytest.raises(ValueError):
            EquivariantLinear(4, 4, dtype=C128)(f)
        with pytest.raises(ValueError):
            NormGate(4)(f)
        with pytest.raises(ValueError):
            InvariantReadout(4, 1)(f)
        with pytest.raises(ValueError):
            EquivariantLinear(2, 2, dtype=torch.float32)

    def test_trains_end_to_end(self):
        # Regress an invariant target (the singlet magnitude of channels 0 and 1).
        torch.manual_seed(9)
        net = nn.Sequential(self.network(8), InvariantReadout(3, 1)).double()
        x = random_spinor(256, 4)
        target = singlet(x[:, 0], x[:, 1]).abs()[:, None]
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        first = None
        for _ in range(150):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), target)
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
        assert loss.item() < 0.5 * first
        for p in net.parameters():
            assert torch.isfinite(p.grad).all()
