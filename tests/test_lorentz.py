"""Tests for SL(2, C) Lorentz transformations, Weyl currents and Dirac spinors."""

import math

import pytest
import torch

from spinor_lib import (
    Spinor,
    dirac_adjoint,
    dirac_bilinear,
    dirac_current,
    dirac_from_weyl,
    gamma5,
    gamma_matrices,
    generate_boost,
    generate_dirac,
    generate_lorentz,
    generate_su2_rotation,
    matrix_to_vector,
    minkowski_inner,
    minkowski_metric,
    other_handedness,
    rapidity_from_velocity,
    rotate,
    singlet,
    sl2c_to_lorentz,
    su2_to_so3,
    vector_to_matrix,
    velocity_from_rapidity,
    weyl_current,
    weyl_from_dirac,
)

C128 = torch.complex128
F64 = torch.float64
ETA = minkowski_metric(dtype=F64)


def random_spinor(*batch, dim=2, generator=None):
    return Spinor(torch.randn(*batch, dim, dtype=C128, generator=generator), dim=dim)


def random_lorentz(*batch, handedness="left", generator=None):
    theta = torch.randn(*batch, 3, dtype=F64, generator=generator)
    eta = torch.randn(*batch, 3, dtype=F64, generator=generator) * 0.7
    return generate_lorentz(theta, eta, handedness=handedness)


def assert_lorentz(L):
    """Proper orthochronous: Λᵀ η Λ = η, det Λ = 1, Λ⁰₀ ≥ 1."""
    eye = ETA.expand(L.shape)
    assert torch.allclose(L.mT @ ETA @ L, eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(L), torch.ones(L.shape[:-2], dtype=F64), atol=1e-10)
    assert (L[..., 0, 0] >= 1 - 1e-12).all()


class TestVectorMatrixBridge:
    def test_round_trip(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(5, 4, dtype=F64, generator=g)
        for h in ("left", "right"):
            X = vector_to_matrix(x, h)
            assert torch.allclose(X, X.conj().mT)  # Hermitian
            assert torch.allclose(matrix_to_vector(X, h), x)

    def test_determinant_is_minkowski_norm(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(7, 4, dtype=F64, generator=g)
        det = torch.real(torch.linalg.det(vector_to_matrix(x)))
        assert torch.allclose(det, minkowski_inner(x, x))

    def test_velocity_rapidity(self):
        beta = torch.tensor([0.0, 0.5, 0.99], dtype=F64)
        assert torch.allclose(velocity_from_rapidity(rapidity_from_velocity(beta)), beta)


class TestBoosts:
    def test_boost_along_z_matches_textbook(self):
        eta = torch.tensor(0.8, dtype=F64)
        for h in ("left", "right"):
            L = sl2c_to_lorentz(generate_boost([0, 0, 1], eta, handedness=h), h)
            ch, sh = math.cosh(0.8), math.sinh(0.8)
            expected = torch.tensor(
                [[ch, 0, 0, sh], [0, 1, 0, 0], [0, 0, 1, 0], [sh, 0, 0, ch]], dtype=F64
            )
            assert torch.allclose(L, expected, atol=1e-12)

    def test_boost_carries_rest_frame_to_velocity(self):
        eta = torch.tensor(1.1, dtype=F64)
        direction = torch.tensor([1.0, 2.0, -2.0], dtype=F64)
        L = sl2c_to_lorentz(generate_boost(direction, eta), "left")
        p = L @ torch.tensor([1.0, 0, 0, 0], dtype=F64)  # rest-frame 4-velocity
        beta = p[1:] / p[0]
        assert torch.allclose(beta, math.tanh(1.1) * direction / direction.norm())
        assert minkowski_inner(p, p).item() == pytest.approx(1.0)

    def test_boost_is_hermitian_unimodular(self):
        g = torch.Generator().manual_seed(2)
        A = generate_boost(torch.randn(6, 3, dtype=F64, generator=g), torch.rand(6, dtype=F64, generator=g))
        assert torch.allclose(A, A.conj().mT)
        assert torch.allclose(torch.linalg.det(A), torch.ones(6, dtype=C128))

    def test_handedness_relation(self):
        A_L = generate_boost([0.3, -1, 0.5], torch.tensor(0.6, dtype=F64), handedness="left")
        A_R = generate_boost([0.3, -1, 0.5], torch.tensor(0.6, dtype=F64), handedness="right")
        assert torch.allclose(other_handedness(A_L), A_R, atol=1e-12)
        assert torch.allclose(A_L @ A_R, torch.eye(2, dtype=C128), atol=1e-12)

    def test_both_handednesses_give_same_lorentz_matrix(self):
        g = torch.Generator().manual_seed(3)
        A_L = random_lorentz(8, handedness="left", generator=g)
        A_R = other_handedness(A_L)
        assert torch.allclose(sl2c_to_lorentz(A_L, "left"), sl2c_to_lorentz(A_R, "right"), atol=1e-10)

    def test_boost_composition_along_same_axis(self):
        a, b = torch.tensor(0.4, dtype=F64), torch.tensor(0.9, dtype=F64)
        assert torch.allclose(
            generate_boost([1, 0, 0], a) @ generate_boost([1, 0, 0], b),
            generate_boost([1, 0, 0], a + b),
            atol=1e-12,
        )

    def test_bad_handedness(self):
        with pytest.raises(ValueError):
            generate_boost([0, 0, 1], 0.1, handedness="up")


class TestGeneralLorentz:
    def test_pure_rotation_matches_su2(self):
        theta = torch.tensor([0.2, -0.5, 0.9], dtype=F64)
        A = generate_lorentz(rotation=theta)
        R = generate_su2_rotation(theta, theta.norm())
        assert torch.allclose(A, R, atol=1e-12)
        L = sl2c_to_lorentz(A)
        assert torch.allclose(L[1:, 1:], su2_to_so3(R), atol=1e-12)
        assert torch.allclose(L[0], torch.tensor([1.0, 0, 0, 0], dtype=F64), atol=1e-12)

    def test_pure_boost_matches_generate_boost(self):
        eta = torch.tensor([0.3, 0.4, 0.0], dtype=F64)
        for h in ("left", "right"):
            assert torch.allclose(
                generate_lorentz(boost=eta, handedness=h),
                generate_boost(eta, eta.norm(), handedness=h),
                atol=1e-12,
            )

    def test_random_elements_are_proper_orthochronous(self):
        g = torch.Generator().manual_seed(4)
        A = random_lorentz(10, generator=g)
        assert torch.allclose(torch.linalg.det(A), torch.ones(10, dtype=C128), atol=1e-10)
        assert_lorentz(sl2c_to_lorentz(A))

    def test_homomorphism_and_sign(self):
        g = torch.Generator().manual_seed(5)
        A, B = random_lorentz(generator=g), random_lorentz(generator=g)
        assert torch.allclose(sl2c_to_lorentz(A @ B), sl2c_to_lorentz(A) @ sl2c_to_lorentz(B), atol=1e-10)
        assert torch.allclose(sl2c_to_lorentz(-A), sl2c_to_lorentz(A))

    def test_defining_property(self):
        g = torch.Generator().manual_seed(6)
        A = random_lorentz(4, generator=g)
        x = torch.randn(4, 4, dtype=F64, generator=g)
        L = sl2c_to_lorentz(A, "left")
        lhs = vector_to_matrix(torch.einsum("bij,bj->bi", L, x), "left")
        rhs = A @ vector_to_matrix(x, "left") @ A.conj().mT
        assert torch.allclose(lhs, rhs, atol=1e-10)

    def test_requires_an_argument(self):
        with pytest.raises(TypeError):
            generate_lorentz()

    def test_gradient_through_rapidity(self):
        eta = torch.tensor([0.0, 0.0, 0.5], dtype=F64, requires_grad=True)
        L = sl2c_to_lorentz(generate_lorentz(boost=eta))
        L[0, 0].backward()  # d cosh(η)/dη = sinh(η)
        assert eta.grad[2].item() == pytest.approx(math.sinh(0.5))


class TestWeylSpinors:
    def test_current_is_null_and_future_pointing(self):
        g = torch.Generator().manual_seed(7)
        psi = random_spinor(9, generator=g)
        for h in ("left", "right"):
            j = weyl_current(psi, h)
            assert torch.allclose(minkowski_inner(j, j), torch.zeros(9, dtype=F64), atol=1e-10)
            assert (j[:, 0] > 0).all()
            assert torch.allclose(j[:, 0], psi.norm() ** 2)

    def test_current_transforms_as_vector(self):
        g = torch.Generator().manual_seed(8)
        psi = random_spinor(6, generator=g)
        for h in ("left", "right"):
            A = random_lorentz(6, handedness=h, generator=g)
            lhs = weyl_current(rotate(psi, A), h)
            rhs = torch.einsum("bij,bj->bi", sl2c_to_lorentz(A, h), weyl_current(psi, h))
            assert torch.allclose(lhs, rhs, atol=1e-10)

    def test_singlet_is_lorentz_invariant(self):
        g = torch.Generator().manual_seed(9)
        a, b = random_spinor(5, generator=g), random_spinor(5, generator=g)
        A = random_lorentz(5, generator=g)
        assert torch.allclose(singlet(rotate(a, A), rotate(b, A)), singlet(a, b), atol=1e-10)

    def test_left_right_product_is_invariant(self):
        # ψ_L† ψ_R is a Lorentz scalar because A_L† A_R = I.
        g = torch.Generator().manual_seed(10)
        psi_L, psi_R = random_spinor(5, generator=g), random_spinor(5, generator=g)
        A_L = random_lorentz(5, generator=g)
        A_R = other_handedness(A_L)
        lhs = torch.sum(torch.conj(rotate(psi_L, A_L).data) * rotate(psi_R, A_R).data, -1)
        rhs = torch.sum(torch.conj(psi_L.data) * psi_R.data, -1)
        assert torch.allclose(lhs, rhs, atol=1e-10)


class TestDiracSpinors:
    def test_clifford_algebra(self):
        gam = gamma_matrices(dtype=C128)
        eye = torch.eye(4, dtype=C128)
        for mu in range(4):
            for nu in range(4):
                anti = gam[mu] @ gam[nu] + gam[nu] @ gam[mu]
                assert torch.allclose(anti, 2 * ETA[mu, nu] * eye)

    def test_gamma5(self):
        gam = gamma_matrices(dtype=C128)
        g5 = gamma5(dtype=C128)
        assert torch.allclose(1j * gam[0] @ gam[1] @ gam[2] @ gam[3], g5)
        for mu in range(4):
            assert torch.allclose(g5 @ gam[mu], -gam[mu] @ g5)

    def test_gamma0_hermitian_gammai_antihermitian(self):
        gam = gamma_matrices(dtype=C128)
        assert torch.allclose(gam[0], gam[0].conj().mT)
        for i in range(1, 4):
            assert torch.allclose(gam[i], -gam[i].conj().mT)

    def test_spinor_representation_of_lorentz_group(self):
        g = torch.Generator().manual_seed(11)
        theta = torch.randn(3, dtype=F64, generator=g)
        eta = torch.randn(3, dtype=F64, generator=g) * 0.5
        S = generate_dirac(theta, eta)
        L = sl2c_to_lorentz(generate_lorentz(theta, eta))
        gam = gamma_matrices(dtype=C128)
        S_inv = torch.linalg.inv(S)
        for mu in range(4):
            lhs = S_inv @ gam[mu] @ S
            rhs = torch.einsum("n,nab->ab", L[mu].to(C128), gam)
            assert torch.allclose(lhs, rhs, atol=1e-10)

    def test_weyl_round_trip_and_batching(self):
        g = torch.Generator().manual_seed(12)
        l, r = random_spinor(3, 1, generator=g), random_spinor(1, 4, generator=g)
        psi = dirac_from_weyl(l, r)
        assert psi.shape == (3, 4, 4) and psi.dim == 4
        l2, r2 = weyl_from_dirac(psi)
        assert torch.allclose(l2.data, l.data.expand(3, 4, 2))
        assert torch.allclose(r2.data, r.data.expand(3, 4, 2))

    def test_scalar_bilinear_is_invariant_and_current_is_vector(self):
        g = torch.Generator().manual_seed(13)
        psi = random_spinor(6, dim=4, generator=g)
        theta = torch.randn(6, 3, dtype=F64, generator=g)
        eta = torch.randn(6, 3, dtype=F64, generator=g) * 0.5
        S = generate_dirac(theta, eta)
        L = sl2c_to_lorentz(generate_lorentz(theta, eta))
        transformed = rotate(psi, S)
        assert torch.allclose(dirac_bilinear(transformed, transformed), dirac_bilinear(psi, psi), atol=1e-10)
        assert torch.allclose(
            dirac_current(transformed), torch.einsum("bij,bj->bi", L, dirac_current(psi)), atol=1e-10
        )

    def test_pseudoscalar_flips_under_parity(self):
        g = torch.Generator().manual_seed(14)
        psi = random_spinor(generator=g, dim=4)
        g0, g5 = gamma_matrices(dtype=C128)[0], gamma5(dtype=C128)
        parity = rotate(psi, g0)
        assert dirac_bilinear(parity, parity, g5).item() == pytest.approx(-dirac_bilinear(psi, psi, g5).item())
        assert dirac_bilinear(parity, parity).item() == pytest.approx(dirac_bilinear(psi, psi).item())

    def test_current_is_sum_of_weyl_currents(self):
        g = torch.Generator().manual_seed(15)
        l, r = random_spinor(4, generator=g), random_spinor(4, generator=g)
        j = dirac_current(dirac_from_weyl(l, r))
        assert torch.allclose(j, weyl_current(l, "left") + weyl_current(r, "right"))

    def test_adjoint_and_bilinear_shapes(self):
        psi = random_spinor(2, 3, dim=4)
        assert dirac_adjoint(psi).shape == (2, 3, 4)
        assert dirac_bilinear(psi, psi).shape == (2, 3)
        assert dirac_current(psi).shape == (2, 3, 4)
        with pytest.raises(ValueError):
            dirac_adjoint(random_spinor())

    def test_gradient_through_boost_of_dirac_spinor(self):
        eta = torch.tensor([0.1, 0.2, 0.3], dtype=F64, requires_grad=True)
        psi = Spinor(torch.tensor([1, 0, 0, 1], dtype=C128), dim=4)
        j = dirac_current(rotate(psi, generate_dirac(boost=eta)))
        j[0].backward()
        assert torch.isfinite(eta.grad).all() and eta.grad.abs().sum() > 0
