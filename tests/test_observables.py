"""Tests for observables, Bloch-sphere geometry, density matrices and the
SU(2) <-> SO(3) <-> quaternion bridge."""

import math

import pytest
import torch

from spinor_lib import (
    Spinor,
    bloch_vector,
    density_from_bloch,
    density_matrix,
    expectation,
    fidelity,
    from_bloch_vector,
    fubini_study_distance,
    generate_su2_rotation,
    global_phase,
    identity,
    normalize,
    pauli,
    purity,
    quaternion_to_su2,
    rotate,
    slerp,
    slerp_su2,
    so3_to_quaternion,
    so3_to_su2,
    su2_to_quaternion,
    su2_to_so3,
)

C128 = torch.complex128
F64 = torch.float64


def cs(*vals):
    return Spinor(torch.tensor(vals, dtype=C128))


def random_spinor(*batch, dim=2, generator=None):
    return Spinor(torch.randn(*batch, dim, dtype=C128, generator=generator))


def random_su2(*batch, generator=None):
    axis = torch.randn(*batch, 3, dtype=F64, generator=generator)
    angle = torch.rand(tuple(batch), dtype=F64, generator=generator) * 2 * math.pi
    return generate_su2_rotation(axis, angle)


def rodrigues(axis, angle):
    """SO(3) rotation about a unit axis, independent of the library."""
    n = torch.as_tensor(axis, dtype=F64)
    n = n / n.norm()
    K = torch.tensor(
        [[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]], dtype=F64
    )
    return torch.eye(3, dtype=F64) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


UP, DOWN = cs(1, 0), cs(0, 1)
PLUS_X = cs(1, 1)
PLUS_Y = cs(1, 1j)


# ---------------------------------------------------------------- operators


class TestOperators:
    def test_pauli_algebra(self):
        sig = pauli(dtype=C128)
        eye = identity(2, dtype=C128)
        for i in range(3):
            assert torch.allclose(sig[i] @ sig[i], eye)
            assert torch.allclose(sig[i], sig[i].conj().mT)  # Hermitian
        # σx σy = i σz and cyclic permutations
        assert torch.allclose(sig[0] @ sig[1], 1j * sig[2])
        assert torch.allclose(sig[1] @ sig[2], 1j * sig[0])
        assert torch.allclose(sig[2] @ sig[0], 1j * sig[1])

    def test_pauli_dtype_and_device(self):
        sig = pauli(dtype=torch.complex64)
        assert sig.dtype == torch.complex64 and sig.shape == (3, 2, 2)

    def test_expectation_of_pauli_z(self):
        sz = pauli(dtype=C128)[2]
        assert expectation(UP, sz).item() == pytest.approx(1.0)
        assert expectation(DOWN, sz).item() == pytest.approx(-1.0)
        assert expectation(PLUS_X, sz).item() == pytest.approx(0.0)

    def test_expectation_is_not_normalised(self):
        sz = pauli(dtype=C128)[2]
        assert expectation(cs(2, 0), sz).item() == pytest.approx(4.0)

    def test_expectation_complex_and_batched(self):
        g = torch.Generator().manual_seed(0)
        s = random_spinor(5, generator=g)
        A = torch.randn(5, 2, 2, dtype=C128, generator=g)  # not Hermitian
        got = expectation(s, A, hermitian=False)
        ref = torch.einsum("bi,bij,bj->b", s.data.conj(), A, s.data)
        assert got.shape == (5,) and got.dtype == C128
        assert torch.allclose(got, ref)

    def test_expectation_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            expectation(UP, torch.eye(3, dtype=C128))

    def test_expectation_gradient(self):
        data = torch.tensor([1.0, 0.5], dtype=C128, requires_grad=True)
        s = Spinor(data)
        expectation(s, pauli(dtype=C128)[0]).backward()
        assert data.grad is not None and torch.isfinite(data.grad).all()


# ----------------------------------------------------------- density matrix


class TestDensityMatrix:
    def test_pure_state_projector(self):
        rho = density_matrix(cs(3, 0))  # unnormalised input
        assert torch.allclose(rho, torch.tensor([[1, 0], [0, 0]], dtype=C128))
        assert purity(rho).item() == pytest.approx(1.0)

    def test_properties_random(self):
        g = torch.Generator().manual_seed(1)
        rho = density_matrix(random_spinor(7, generator=g))
        assert rho.shape == (7, 2, 2)
        trace = torch.einsum("bii->b", rho)
        assert torch.allclose(trace, torch.ones(7, dtype=C128))
        assert torch.allclose(rho, rho.conj().mT)
        assert torch.allclose(rho @ rho, rho)

    def test_unnormalised_option(self):
        rho = density_matrix(cs(2, 0), normalize=False)
        assert rho[0, 0].item() == pytest.approx(4.0)

    def test_density_from_bloch_mixed(self):
        rho = density_from_bloch([0.0, 0.0, 0.5])
        assert torch.allclose(rho, torch.tensor([[0.75, 0], [0, 0.25]], dtype=rho.dtype))
        assert purity(rho).item() == pytest.approx(0.625)

    def test_density_from_bloch_pure_matches_spinor(self):
        n = torch.tensor([0.6, 0.0, 0.8], dtype=F64)
        assert torch.allclose(density_from_bloch(n), density_matrix(from_bloch_vector(n)))

    def test_maximally_mixed_purity(self):
        rho = density_from_bloch(torch.zeros(3, dtype=F64))
        assert purity(rho).item() == pytest.approx(0.5)


# ------------------------------------------------------------ Bloch sphere


class TestBlochSphere:
    @pytest.mark.parametrize(
        "spinor, expected",
        [
            (UP, [0, 0, 1]),
            (DOWN, [0, 0, -1]),
            (PLUS_X, [1, 0, 0]),
            (PLUS_Y, [0, 1, 0]),
            (cs(1, -1), [-1, 0, 0]),
        ],
    )
    def test_named_states(self, spinor, expected):
        assert torch.allclose(bloch_vector(spinor), torch.tensor(expected, dtype=F64))

    def test_ignores_norm_and_global_phase(self):
        s = cs(0.3, 0.7 + 0.2j)
        assert torch.allclose(bloch_vector(s), bloch_vector(s * 5))
        assert torch.allclose(bloch_vector(s), bloch_vector(global_phase(s, 1.3)))

    def test_pure_states_lie_on_unit_sphere(self):
        g = torch.Generator().manual_seed(2)
        n = bloch_vector(random_spinor(4, 3, generator=g))
        assert n.shape == (4, 3, 3)
        assert torch.allclose(n.norm(dim=-1), torch.ones(4, 3, dtype=F64))

    def test_from_density_matrix(self):
        n = torch.tensor([0.1, -0.2, 0.3], dtype=F64)
        assert torch.allclose(bloch_vector(density_from_bloch(n)), n)

    def test_round_trip_spinor_bloch_spinor(self):
        g = torch.Generator().manual_seed(3)
        s = random_spinor(20, generator=g)
        back = from_bloch_vector(bloch_vector(s))
        assert torch.allclose(fidelity(s, back), torch.ones(20, dtype=F64))
        # Phase convention: first component real and non-negative.
        assert torch.allclose(back.data[:, 0].imag, torch.zeros(20, dtype=F64))
        assert (back.data[:, 0].real >= 0).all()

    def test_from_bloch_poles(self):
        assert torch.allclose(from_bloch_vector([0, 0, 1]).data, UP.data.to(torch.complex64))
        assert torch.allclose(from_bloch_vector([0, 0, -1]).data, DOWN.data.to(torch.complex64))

    def test_from_bloch_normalises_input(self):
        a = from_bloch_vector([2.0, 0.0, 0.0])
        assert torch.allclose(bloch_vector(a), torch.tensor([1.0, 0, 0]))

    def test_from_bloch_gradient(self):
        n = torch.tensor([0.3, 0.4, 0.5], dtype=F64, requires_grad=True)
        s = from_bloch_vector(n)
        expectation(s, pauli(dtype=C128)[0]).backward()
        assert torch.isfinite(n.grad).all()

    def test_bloch_rejects_non_qubit(self):
        with pytest.raises(ValueError):
            bloch_vector(Spinor(torch.zeros(3, dtype=C128), dim=3))


# ------------------------------------------------------ fidelity / distance


class TestFidelityAndDistance:
    def test_pure_fidelity_values(self):
        assert fidelity(UP, UP).item() == pytest.approx(1.0)
        assert fidelity(UP, DOWN).item() == pytest.approx(0.0)
        assert fidelity(UP, PLUS_X).item() == pytest.approx(0.5)
        assert fidelity(UP * 3, PLUS_X * 1j).item() == pytest.approx(0.5)

    def test_mixed_fidelity_matches_pure_case(self):
        g = torch.Generator().manual_seed(4)
        a, b = random_spinor(6, generator=g), random_spinor(6, generator=g)
        mixed = fidelity(density_matrix(a), density_matrix(b))
        assert torch.allclose(mixed, fidelity(a, b), atol=1e-8)

    def test_mixed_fidelity_with_maximally_mixed(self):
        rho = density_from_bloch(torch.zeros(3, dtype=F64))
        assert fidelity(rho, UP).item() == pytest.approx(0.5)

    def test_mixed_fidelity_rejects_larger_states(self):
        rho = torch.eye(3, dtype=C128) / 3
        with pytest.raises(NotImplementedError):
            fidelity(rho, rho)

    def test_fubini_study_distance(self):
        assert fubini_study_distance(UP, UP).item() == pytest.approx(0.0)
        assert fubini_study_distance(UP, DOWN).item() == pytest.approx(math.pi / 2)
        assert fubini_study_distance(UP, PLUS_X).item() == pytest.approx(math.pi / 4)

    def test_fubini_study_is_half_the_bloch_angle(self):
        g = torch.Generator().manual_seed(5)
        a, b = random_spinor(10, generator=g), random_spinor(10, generator=g)
        cos_angle = (bloch_vector(a) * bloch_vector(b)).sum(-1).clamp(-1, 1)
        assert torch.allclose(fubini_study_distance(a, b), torch.arccos(cos_angle) / 2)

    def test_fidelity_gradient(self):
        data = torch.tensor([1.0, 0.3j], dtype=C128, requires_grad=True)
        fidelity(Spinor(data), PLUS_X).backward()
        assert torch.isfinite(data.grad).all()


class TestStateSlerp:
    def test_endpoints(self):
        a, b = cs(1, 0.2j), cs(0.4, 1)
        assert fidelity(slerp(a, b, 0.0), a).item() == pytest.approx(1.0)
        assert fidelity(slerp(a, b, 1.0), b).item() == pytest.approx(1.0)

    def test_midpoint_is_equidistant_and_normalised(self):
        g = torch.Generator().manual_seed(6)
        a, b = random_spinor(8, generator=g), random_spinor(8, generator=g)
        mid = slerp(a, b, 0.5)
        assert torch.allclose(mid.norm(), torch.ones(8, dtype=F64))
        assert torch.allclose(fubini_study_distance(a, mid), fubini_study_distance(mid, b))
        assert torch.allclose(
            fubini_study_distance(a, mid) + fubini_study_distance(mid, b),
            fubini_study_distance(a, b),
        )

    def test_bloch_vector_follows_great_circle(self):
        # UP -> +X along the geodesic passes through the point half-way on the sphere.
        mid = slerp(UP, PLUS_X, 0.5)
        expected = torch.tensor([1, 0, 1], dtype=F64) / math.sqrt(2)
        assert torch.allclose(bloch_vector(mid), expected)

    def test_per_spinor_t(self):
        t = torch.linspace(0, 1, 5, dtype=F64)
        path = slerp(UP, DOWN, t)
        assert path.shape == (5, 2)
        z = bloch_vector(path)[:, 2]
        assert torch.allclose(z, torch.cos(math.pi * t))

    def test_identical_endpoints(self):
        assert torch.allclose(slerp(UP, UP * 2, 0.3).data, UP.data)

    def test_gradient(self):
        data = torch.tensor([1.0, 0.5j], dtype=C128, requires_grad=True)
        s = slerp(Spinor(data), PLUS_Y, 0.3)
        expectation(s, pauli(dtype=C128)[2]).backward()
        assert torch.isfinite(data.grad).all()


# ---------------------------------------------------- SU(2) <-> SO(3) bridge


class TestSO3Bridge:
    @pytest.mark.parametrize(
        "axis, angle",
        [([0, 0, 1], 0.7), ([1, 0, 0], math.pi / 2), ([1, 1, 1], 2.0), ([0, 1, 0], math.pi)],
    )
    def test_su2_to_so3_matches_rodrigues(self, axis, angle):
        M = su2_to_so3(generate_su2_rotation(axis, torch.tensor(angle, dtype=F64)))
        assert torch.allclose(M, rodrigues(axis, angle), atol=1e-12)

    def test_so3_is_orthogonal_with_unit_det(self):
        g = torch.Generator().manual_seed(7)
        M = su2_to_so3(random_su2(9, generator=g))
        assert M.shape == (9, 3, 3)
        assert torch.allclose(M @ M.mT, torch.eye(3, dtype=F64).expand(9, 3, 3), atol=1e-12)
        assert torch.allclose(torch.linalg.det(M), torch.ones(9, dtype=F64))

    def test_rotating_spinor_rotates_bloch_vector(self):
        g = torch.Generator().manual_seed(8)
        s, R = random_spinor(9, generator=g), random_su2(9, generator=g)
        lhs = bloch_vector(rotate(s, R))
        rhs = torch.einsum("bij,bj->bi", su2_to_so3(R), bloch_vector(s))
        assert torch.allclose(lhs, rhs, atol=1e-12)

    def test_double_cover(self):
        R = generate_su2_rotation([0, 0, 1], torch.tensor(2 * math.pi, dtype=F64))
        assert torch.allclose(R, -torch.eye(2, dtype=C128), atol=1e-12)
        assert torch.allclose(su2_to_so3(R), torch.eye(3, dtype=F64), atol=1e-12)
        assert torch.allclose(su2_to_so3(R), su2_to_so3(-R))

    def test_homomorphism(self):
        g = torch.Generator().manual_seed(9)
        A, B = random_su2(generator=g), random_su2(generator=g)
        assert torch.allclose(su2_to_so3(A @ B), su2_to_so3(A) @ su2_to_so3(B), atol=1e-12)

    def test_so3_to_su2_round_trip(self):
        g = torch.Generator().manual_seed(10)
        R = random_su2(50, generator=g)
        M = su2_to_so3(R)
        R_back = so3_to_su2(M)
        assert torch.allclose(su2_to_so3(R_back), M, atol=1e-10)
        # The preimage is R up to the double-cover sign.
        same = torch.allclose(R_back, R, atol=1e-10)
        flipped = torch.allclose(R_back, -R, atol=1e-10)
        sign = torch.where(
            (R_back[:, 0, 0] * R[:, 0, 0].conj()).real[:, None, None] < 0, -1.0, 1.0
        )
        assert same or flipped or torch.allclose(R_back * sign, R, atol=1e-10)

    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]])
    def test_so3_to_quaternion_at_pi_is_stable(self, axis):
        # Angle π makes 1 + trace vanish; Shepperd's method must pick another branch.
        M = rodrigues(axis, math.pi)
        q = so3_to_quaternion(M)
        assert torch.isfinite(q).all()
        assert torch.allclose(su2_to_so3(quaternion_to_su2(q)), M, atol=1e-10)

    def test_so3_to_quaternion_identity(self):
        q = so3_to_quaternion(torch.eye(3, dtype=F64))
        assert torch.allclose(q.abs(), torch.tensor([1.0, 0, 0, 0], dtype=F64))

    def test_su2_to_so3_gradient(self):
        angle = torch.tensor(0.4, dtype=F64, requires_grad=True)
        M = su2_to_so3(generate_su2_rotation([0, 0, 1], angle))
        M[0, 0].backward()  # d cos(θ)/dθ = −sin θ
        assert angle.grad.item() == pytest.approx(-math.sin(0.4))


class TestQuaternions:
    def test_axis_angle_quaternion(self):
        R = generate_su2_rotation([0, 1, 0], torch.tensor(0.8, dtype=F64))
        q = su2_to_quaternion(R)
        expected = torch.tensor([math.cos(0.4), 0, math.sin(0.4), 0], dtype=F64)
        assert torch.allclose(q, expected)

    def test_round_trip(self):
        g = torch.Generator().manual_seed(11)
        R = random_su2(30, generator=g)
        assert torch.allclose(quaternion_to_su2(su2_to_quaternion(R)), R, atol=1e-12)
        q = torch.randn(30, 4, dtype=F64, generator=g)
        q = q / q.norm(dim=-1, keepdim=True)
        assert torch.allclose(su2_to_quaternion(quaternion_to_su2(q)), q, atol=1e-12)

    def test_quaternion_to_su2_is_unitary_after_normalising(self):
        q = torch.tensor([3.0, -1.0, 2.0, 0.5], dtype=F64)  # not unit
        R = quaternion_to_su2(q)
        assert torch.allclose(R @ R.conj().mT, torch.eye(2, dtype=C128), atol=1e-12)
        assert torch.linalg.det(R).real.item() == pytest.approx(1.0)

    def test_quaternion_product_matches_matrix_product(self):
        g = torch.Generator().manual_seed(12)
        A, B = random_su2(generator=g), random_su2(generator=g)
        a, b = su2_to_quaternion(A), su2_to_quaternion(B)
        w = a[0] * b[0] - (a[1:] * b[1:]).sum()
        v = a[0] * b[1:] + b[0] * a[1:] + torch.cross(a[1:], b[1:], dim=0)
        assert torch.allclose(quaternion_to_su2(torch.cat([w[None], v])), A @ B, atol=1e-12)

    def test_gradient_through_unconstrained_quaternion(self):
        q = torch.tensor([1.0, 0.1, 0.2, 0.3], dtype=F64, requires_grad=True)
        s = rotate(UP, quaternion_to_su2(q))
        expectation(s, pauli(dtype=C128)[0]).backward()
        assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


class TestRotationSlerp:
    def test_endpoints_and_midpoint_about_fixed_axis(self):
        R0 = generate_su2_rotation([0, 0, 1], torch.tensor(0.2, dtype=F64))
        R1 = generate_su2_rotation([0, 0, 1], torch.tensor(1.4, dtype=F64))
        assert torch.allclose(slerp_su2(R0, R1, 0.0), R0, atol=1e-12)
        assert torch.allclose(slerp_su2(R0, R1, 1.0), R1, atol=1e-12)
        mid = generate_su2_rotation([0, 0, 1], torch.tensor(0.8, dtype=F64))
        assert torch.allclose(slerp_su2(R0, R1, 0.5), mid, atol=1e-12)

    def test_takes_shorter_arc(self):
        # R1 = -R0 represents the same rotation; slerp should not move.
        R0 = generate_su2_rotation([1, 0, 0], torch.tensor(0.5, dtype=F64))
        mid = slerp_su2(R0, -R0, 0.5)
        assert torch.allclose(su2_to_so3(mid), su2_to_so3(R0), atol=1e-12)

    def test_stays_in_su2_and_batched_t(self):
        g = torch.Generator().manual_seed(13)
        R0, R1 = random_su2(6, generator=g), random_su2(6, generator=g)
        t = torch.rand(6, dtype=F64, generator=g)
        R = slerp_su2(R0, R1, t)
        assert R.shape == (6, 2, 2)
        assert torch.allclose(R @ R.conj().mT, torch.eye(2, dtype=C128).expand(6, 2, 2), atol=1e-12)
        assert torch.allclose(torch.linalg.det(R).real, torch.ones(6, dtype=F64))

    def test_identical_rotations(self):
        R = generate_su2_rotation([1, 2, 3], torch.tensor(1.0, dtype=F64))
        assert torch.allclose(slerp_su2(R, R, 0.7), R, atol=1e-12)
