"""Tests for multi-spinor (tensor-product) states."""

import math

import pytest
import torch

from spinor_lib import (
    Spinor,
    bloch_vector,
    concurrence,
    density_matrix,
    embed_operator,
    entanglement_entropy,
    expectation,
    fidelity,
    generate_su2_rotation,
    identity,
    inner_product,
    kron,
    partial_trace,
    pauli,
    purity,
    reduced_density_matrix,
    rotate,
    tensor_product,
    von_neumann_entropy,
)

C128 = torch.complex128
F64 = torch.float64


def cs(*vals, dim=2):
    return Spinor(torch.tensor(vals, dtype=C128), dim=dim)


def random_spinor(*batch, dim=2, generator=None):
    return Spinor(torch.randn(*batch, dim, dtype=C128, generator=generator), dim=dim)


UP, DOWN = cs(1, 0), cs(0, 1)
BELL = cs(1, 0, 0, 1, dim=4) * (1 / math.sqrt(2))  # (|00> + |11>) / sqrt 2


class TestTensorProduct:
    def test_basis_states(self):
        assert torch.allclose(tensor_product(UP, DOWN).data, torch.tensor([0, 1, 0, 0], dtype=C128))
        assert torch.allclose(tensor_product(DOWN, UP).data, torch.tensor([0, 0, 1, 0], dtype=C128))

    def test_matches_torch_kron(self):
        g = torch.Generator().manual_seed(0)
        a, b, c = (random_spinor(generator=g) for _ in range(3))
        out = tensor_product(a, b, c)
        assert out.dim == 8 and out.shape == (8,)
        assert torch.allclose(out.data, torch.kron(torch.kron(a.data, b.data), c.data))

    def test_batch_broadcast(self):
        g = torch.Generator().manual_seed(1)
        a = random_spinor(4, 1, generator=g)
        b = random_spinor(3, generator=g)
        out = tensor_product(a, b)
        assert out.shape == (4, 3, 4)
        assert torch.allclose(out.data[2, 1], torch.kron(a.data[2, 0], b.data[1]))

    def test_norm_is_product_of_norms(self):
        g = torch.Generator().manual_seed(2)
        a, b = random_spinor(5, generator=g), random_spinor(5, generator=g)
        assert torch.allclose(tensor_product(a, b).norm(), a.norm() * b.norm())

    def test_mixed_dims_and_dtype_promotion(self):
        a = Spinor(torch.tensor([1.0, 2.0, 3.0]), dim=3)  # real float32
        b = cs(1j, 1)
        out = tensor_product(a, b)
        assert out.dim == 6 and out.dtype == C128

    def test_gradient(self):
        data = torch.tensor([1.0, 0.5j], dtype=C128, requires_grad=True)
        out = tensor_product(Spinor(data), DOWN)
        torch.real(inner_product(out, out)).backward()
        assert torch.allclose(data.grad, 2 * data.detach())


class TestKronAndEmbed:
    def test_kron_matches_torch(self):
        g = torch.Generator().manual_seed(3)
        A = torch.randn(2, 2, dtype=C128, generator=g)
        B = torch.randn(3, 3, dtype=C128, generator=g)
        assert torch.allclose(kron(A, B), torch.kron(A, B))

    def test_kron_batched(self):
        g = torch.Generator().manual_seed(4)
        A = torch.randn(5, 2, 2, dtype=C128, generator=g)
        B = torch.randn(5, 2, 2, dtype=C128, generator=g)
        out = kron(A, B)
        assert out.shape == (5, 4, 4)
        assert torch.allclose(out[3], torch.kron(A[3], B[3]))

    def test_embed_operator_positions(self):
        sz = pauli(dtype=C128)[2]
        eye = identity(2, dtype=C128)
        assert torch.allclose(embed_operator(sz, (2, 2), 0), torch.kron(sz, eye))
        assert torch.allclose(embed_operator(sz, (2, 2), 1), torch.kron(eye, sz))
        assert torch.allclose(embed_operator(sz, (2, 2, 2), 1), torch.kron(torch.kron(eye, sz), eye))

    def test_embed_rejects_wrong_sizes(self):
        with pytest.raises(ValueError):
            embed_operator(torch.eye(3, dtype=C128), (2, 2), 0)
        with pytest.raises(IndexError):
            embed_operator(torch.eye(2, dtype=C128), (2, 2), 2)

    def test_rotating_one_factor_of_a_product_state(self):
        g = torch.Generator().manual_seed(5)
        a, b = random_spinor(generator=g), random_spinor(generator=g)
        R = generate_su2_rotation([1, 2, 0], torch.tensor(0.9, dtype=F64))
        lhs = rotate(tensor_product(a, b), embed_operator(R, (2, 2), 1))
        rhs = tensor_product(a, rotate(b, R))
        assert torch.allclose(lhs.data, rhs.data)

    def test_local_expectation_matches_single_spinor(self):
        g = torch.Generator().manual_seed(6)
        a, b = random_spinor(generator=g), random_spinor(generator=g)
        sx = pauli(dtype=C128)[0]
        ab = tensor_product(a, b).normalize()
        got = expectation(ab, embed_operator(sx, (2, 2), 0))
        assert got.item() == pytest.approx(expectation(a.normalize(), sx).item())


class TestPartialTrace:
    def test_product_state_reduces_to_factors(self):
        g = torch.Generator().manual_seed(7)
        a, b = random_spinor(generator=g), random_spinor(generator=g)
        ab = tensor_product(a, b)
        assert torch.allclose(reduced_density_matrix(ab, (2, 2), 0), density_matrix(a))
        assert torch.allclose(reduced_density_matrix(ab, (2, 2), 1), density_matrix(b))

    def test_bell_state_reduces_to_maximally_mixed(self):
        rho_a = reduced_density_matrix(BELL, (2, 2), 0)
        assert torch.allclose(rho_a, torch.eye(2, dtype=C128) / 2)
        assert purity(rho_a).item() == pytest.approx(0.5)
        assert torch.allclose(bloch_vector(rho_a), torch.zeros(3, dtype=F64), atol=1e-12)

    def test_keep_everything_is_identity(self):
        rho = density_matrix(BELL)
        assert torch.allclose(partial_trace(rho, (2, 2), [0, 1]), rho)

    def test_three_subsystems_keep_two(self):
        g = torch.Generator().manual_seed(8)
        a, b, c = (random_spinor(generator=g) for _ in range(3))
        abc = tensor_product(a, b, c)
        rho_ac = partial_trace(density_matrix(abc), (2, 2, 2), [0, 2])
        assert rho_ac.shape == (4, 4)
        assert torch.allclose(rho_ac, density_matrix(tensor_product(a, c)))

    def test_uneven_dims_and_batch(self):
        g = torch.Generator().manual_seed(9)
        a = random_spinor(6, dim=2, generator=g)
        b = random_spinor(6, dim=3, generator=g)
        rho = density_matrix(tensor_product(a, b))
        rho_b = partial_trace(rho, (2, 3), 1)
        assert rho_b.shape == (6, 3, 3)
        assert torch.allclose(rho_b, density_matrix(b))

    def test_trace_is_preserved(self):
        g = torch.Generator().manual_seed(10)
        rho = density_matrix(random_spinor(4, dim=8, generator=g))
        reduced = partial_trace(rho, (2, 2, 2), 1)
        assert torch.allclose(torch.einsum("bii->b", reduced), torch.ones(4, dtype=C128))

    def test_rejects_bad_inputs(self):
        rho = density_matrix(BELL)
        with pytest.raises(ValueError):
            partial_trace(rho, (2, 3), 0)
        with pytest.raises(ValueError):
            partial_trace(rho, (2, 2), [0, 0])
        with pytest.raises(IndexError):
            partial_trace(rho, (2, 2), 5)


class TestEntanglement:
    def test_von_neumann_entropy_values(self):
        eye = torch.eye(2, dtype=C128)
        assert von_neumann_entropy(eye / 2).item() == pytest.approx(math.log(2))
        assert von_neumann_entropy(eye / 2, base=2).item() == pytest.approx(1.0)
        assert von_neumann_entropy(density_matrix(UP)).item() == pytest.approx(0.0, abs=1e-12)

    def test_entanglement_entropy_bell_vs_product(self):
        assert entanglement_entropy(BELL, (2, 2), 0, base=2).item() == pytest.approx(1.0)
        product = tensor_product(cs(1, 1j), cs(2, 0.5))
        assert entanglement_entropy(product, (2, 2), 0).item() == pytest.approx(0.0, abs=1e-12)

    def test_entanglement_entropy_symmetric_in_bipartition(self):
        g = torch.Generator().manual_seed(11)
        s = random_spinor(7, dim=4, generator=g)
        assert torch.allclose(
            entanglement_entropy(s, (2, 2), 0), entanglement_entropy(s, (2, 2), 1)
        )

    def test_concurrence_values(self):
        assert concurrence(BELL).item() == pytest.approx(1.0)
        assert concurrence(BELL * 3).item() == pytest.approx(1.0)  # normalisation-free
        assert concurrence(tensor_product(cs(1, 2j), cs(0.3, 1))).item() == pytest.approx(0.0, abs=1e-12)

    def test_concurrence_agrees_with_entropy(self):
        # For two-qubit pure states S = h((1 + sqrt(1 - C^2)) / 2) in bits.
        g = torch.Generator().manual_seed(12)
        s = random_spinor(20, dim=4, generator=g)
        C = concurrence(s)
        lam = (1 + torch.sqrt(1 - C**2)) / 2
        h = -(lam * torch.log2(lam) + (1 - lam) * torch.log2(1 - lam))
        assert torch.allclose(entanglement_entropy(s, (2, 2), 0, base=2), h, atol=1e-8)

    def test_concurrence_rejects_non_two_qubit(self):
        with pytest.raises(ValueError):
            concurrence(UP)

    def test_gradients_through_entanglement_measures(self):
        data = torch.tensor([1.0, 0.2, 0.3j, 0.7], dtype=C128, requires_grad=True)
        s = Spinor(data, dim=4)
        (concurrence(s) + entanglement_entropy(s, (2, 2), 0)).backward()
        assert torch.isfinite(data.grad).all()

    def test_entanglement_is_invariant_under_local_rotations(self):
        g = torch.Generator().manual_seed(13)
        s = random_spinor(5, dim=4, generator=g)
        R = generate_su2_rotation(torch.randn(5, 3, dtype=F64, generator=g), torch.rand(5, dtype=F64, generator=g))
        local = rotate(s, embed_operator(R, (2, 2), 0))
        assert torch.allclose(concurrence(local), concurrence(s))
        assert torch.allclose(entanglement_entropy(local, (2, 2), 1), entanglement_entropy(s, (2, 2), 1))

    def test_bell_state_via_fidelity_is_far_from_all_products(self):
        g = torch.Generator().manual_seed(14)
        products = tensor_product(random_spinor(200, generator=g), random_spinor(200, generator=g))
        assert (fidelity(BELL, products) <= 0.5 + 1e-12).all()
