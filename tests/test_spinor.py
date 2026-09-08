"""Unit tests for the Spinor library."""

import numpy as np
import pytest
import torch

from spinor_lib import (
    Spinor,
    SpinorLinear,
    SU2Rotation,
    add,
    conjugate,
    contract,
    generate_su2_rotation,
    generate_su2_rotation_from_axis_angle,
    global_phase,
    inner_product,
    mul,
    norm,
    normalize,
    outer_product,
    rotate,
    sub,
)

C128 = torch.complex128

SX = torch.tensor([[0, 1], [1, 0]], dtype=C128)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=C128)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=C128)


def reference_su2(axis, angle):
    """exp(-i θ/2 n·σ) via the matrix exponential, independent of the library."""
    n = torch.as_tensor(axis, dtype=torch.float64)
    n = n / n.norm()
    return torch.linalg.matrix_exp(-1j * (angle / 2) * (n[0] * SX + n[1] * SY + n[2] * SZ))


def cs(*vals):
    return Spinor(torch.tensor(vals, dtype=C128))


# --------------------------------------------------------------------- core


class TestSpinor:
    def test_creation_from_tensor_shares_storage(self):
        data = torch.randn(2, dtype=C128)
        s = Spinor(data)
        assert s.data is data

    def test_creation_from_list(self):
        s = Spinor([1 + 2j, 3 + 4j])
        assert s.shape == (2,)
        assert s.dtype.is_complex

    def test_creation_from_numpy(self):
        s = Spinor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert s.shape == (2, 2)
        assert s.dtype == torch.float64

    def test_creation_from_shape(self):
        s = Spinor(shape=(4, 2))
        assert s.shape == (4, 2)
        assert s.dtype == torch.complex64
        assert torch.count_nonzero(s.data) == 0

    def test_creation_requires_exactly_one_of_data_or_shape(self):
        with pytest.raises(TypeError):
            Spinor()
        with pytest.raises(TypeError):
            Spinor([1, 2], shape=(2,))

    def test_dtype_and_device_kwargs(self):
        s = Spinor([1.0, 2.0], dtype=C128, device="cpu")
        assert s.dtype == C128
        assert s.device.type == "cpu"

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            Spinor(torch.randn(3))
        with pytest.raises(ValueError):
            Spinor(torch.tensor(1.0))

    def test_n_component_spinors(self):
        s = Spinor(torch.randn(5, 4), dim=4)
        assert s.dim == 4
        assert norm(s).shape == (5,)
        assert add(s, s).dim == 4

    def test_to_device(self):
        s = Spinor(torch.randn(2))
        assert s.to("cpu").device.type == "cpu"
        assert s.to(dtype=C128).dtype == C128

    def test_len_and_getitem(self):
        s = Spinor(torch.randn(32, 2))
        assert len(s) == 32
        assert isinstance(s[0], Spinor)
        assert s[0].shape == (2,)
        assert s[3:7].shape == (4, 2)

    def test_getitem_dropping_component_axis_returns_tensor(self):
        s = Spinor(torch.randn(32, 2))
        assert isinstance(s[..., 0], torch.Tensor)
        assert s[..., 0].shape == (32,)
        assert isinstance(Spinor(torch.randn(2))[0], torch.Tensor)

    def test_len_of_unbatched_raises(self):
        with pytest.raises(TypeError):
            len(Spinor(torch.randn(2)))

    def test_repr_and_str(self):
        s = Spinor(torch.randn(3, 2))
        assert "Spinor" in repr(s)
        assert "shape=(3, 2)" in str(s)

    def test_reshape(self):
        s = Spinor(torch.randn(6, 2))
        assert s.reshape(2, 3, 2).shape == (2, 3, 2)
        assert s.batch_shape == (6,)

    def test_operators(self):
        a, b = cs(1, 2), cs(3, 4)
        assert torch.allclose((a + b).data, cs(4, 6).data)
        assert torch.allclose((a - b).data, cs(-2, -2).data)
        assert torch.allclose((2 * a).data, cs(2, 4).data)
        assert torch.allclose((a * 2).data, cs(2, 4).data)
        assert torch.allclose((a / 2).data, cs(0.5, 1).data)
        assert torch.allclose((-a).data, cs(-1, -2).data)

    def test_properties_track_data(self):
        s = Spinor(torch.randn(2))
        s.data = s.data.to(C128)
        assert s.dtype == C128


# ---------------------------------------------------------------------- ops


class TestOperations:
    def test_add(self):
        result = add(cs(1 + 2j, 3 + 4j), cs(5 + 6j, 7 + 8j))
        assert torch.allclose(result.data, cs(6 + 8j, 10 + 12j).data)

    def test_sub(self):
        result = sub(cs(5 + 6j, 7 + 8j), cs(1 + 2j, 3 + 4j))
        assert torch.allclose(result.data, cs(4 + 4j, 4 + 4j).data)

    def test_add_broadcasts_over_batch(self):
        batch = Spinor(torch.randn(32, 2, dtype=C128))
        single = Spinor(torch.randn(2, dtype=C128))
        result = add(batch, single)
        assert result.shape == (32, 2)
        assert torch.allclose(result.data, batch.data + single.data)

    def test_add_rejects_incompatible_shapes(self):
        with pytest.raises(ValueError):
            add(Spinor(torch.randn(3, 2)), Spinor(torch.randn(4, 2)))
        with pytest.raises(ValueError):
            add(Spinor(torch.randn(3, 2)), Spinor(torch.randn(3, 4), dim=4))

    def test_mul_python_scalar(self):
        assert torch.allclose(mul(cs(1, 2), 2j).data, cs(2j, 4j).data)

    def test_mul_per_spinor_tensor_scalar(self):
        s = Spinor(torch.ones(4, 2))
        scale = torch.arange(4.0)
        result = mul(s, scale)
        assert torch.allclose(result.data, scale.unsqueeze(-1).expand(4, 2))

    def test_inner_product_is_hermitian(self):
        a, b = cs(1 + 1j, 2), cs(3, 1j)
        expected = torch.sum(torch.conj(a.data) * b.data)
        assert torch.allclose(inner_product(a, b), expected)
        assert torch.allclose(inner_product(b, a), expected.conj())

    def test_inner_product_scalar_value(self):
        assert torch.allclose(inner_product(cs(1, 1), cs(1, 1)), torch.tensor(2.0, dtype=C128))

    def test_inner_product_broadcasts(self):
        batch = Spinor(torch.randn(8, 2, dtype=C128))
        single = Spinor(torch.randn(2, dtype=C128))
        assert inner_product(batch, single).shape == (8,)

    def test_outer_product_is_projector_for_unit_spinor(self):
        s = normalize(Spinor(torch.randn(2, dtype=C128)))
        P = outer_product(s, s)
        assert P.shape == (2, 2)
        assert torch.allclose(P @ P, P)
        assert torch.allclose(P, P.conj().T)

    def test_outer_product_without_conjugate(self):
        a, b = cs(1, 2j), cs(3j, 4)
        P = outer_product(a, b, conjugate=False)
        assert torch.allclose(P, torch.outer(a.data, b.data))

    def test_norm_is_real(self):
        result = norm(cs(3, 4))
        assert result.dtype == torch.float64
        assert torch.allclose(result, torch.tensor(5.0, dtype=torch.float64))

    def test_normalize(self):
        s_n = normalize(cs(3, 4))
        assert torch.allclose(norm(s_n), torch.tensor(1.0, dtype=torch.float64))

    def test_normalize_zero_vector_is_finite(self):
        result = normalize(cs(0, 0))
        assert torch.isfinite(result.data).all()
        assert torch.count_nonzero(result.data) == 0

    def test_conjugate(self):
        assert torch.allclose(conjugate(cs(1 + 2j, 3 + 4j)).data, cs(1 - 2j, 3 - 4j).data)


# ------------------------------------------------------------ transformations


class TestSU2Generation:
    @pytest.mark.parametrize(
        "axis, angle",
        [
            ([0, 0, 1], torch.pi / 3),
            ([1, 0, 0], 0.7),
            ([0, 1, 0], 1.9),  # pure-y axis catches sign errors in the off-diagonals
            ([0.3, 0.7, 0.2], 1.1),
            ([2.0, -1.0, 0.5], -2.4),  # unnormalised axis, negative angle
        ],
    )
    def test_matches_matrix_exponential(self, axis, angle):
        R = generate_su2_rotation(axis, angle, dtype=C128)
        assert torch.allclose(R, reference_su2(axis, angle), atol=1e-12)

    def test_unitarity_and_unit_determinant(self):
        R = generate_su2_rotation([0.3, -0.5, 0.8], 2.2, dtype=C128)
        assert torch.allclose(R.conj().T @ R, torch.eye(2, dtype=C128), atol=1e-12)
        assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0, dtype=C128), atol=1e-12)

    def test_composition_about_same_axis(self):
        axis = [0.1, 0.9, 0.4]
        R1 = generate_su2_rotation(axis, 0.6, dtype=C128)
        R2 = generate_su2_rotation(axis, 1.1, dtype=C128)
        R12 = generate_su2_rotation(axis, 1.7, dtype=C128)
        assert torch.allclose(R1 @ R2, R12, atol=1e-12)

    def test_two_pi_is_minus_identity(self):
        R = generate_su2_rotation([0, 1, 0], 2 * torch.pi, dtype=C128)
        assert torch.allclose(R, -torch.eye(2, dtype=C128), atol=1e-12)

    def test_batched_angles(self):
        angles = torch.linspace(0, 3, 5, dtype=torch.float64)
        R = generate_su2_rotation([0, 1, 0], angles)
        assert R.shape == (5, 2, 2)
        for i, a in enumerate(angles):
            assert torch.allclose(R[i], reference_su2([0, 1, 0], a.item()), atol=1e-12)

    def test_batched_axes_and_angles_broadcast(self):
        axes = torch.randn(4, 1, 3, dtype=torch.float64)
        angles = torch.randn(3, dtype=torch.float64)
        R = generate_su2_rotation(axes, angles)
        assert R.shape == (4, 3, 2, 2)
        assert torch.allclose(R[2, 1], reference_su2(axes[2, 0], angles[1].item()), atol=1e-12)

    def test_default_dtype_follows_inputs(self):
        assert generate_su2_rotation([0, 0, 1], 0.5).dtype == torch.complex64
        assert generate_su2_rotation(torch.tensor([0.0, 0, 1], dtype=torch.float64), 0.5).dtype == C128
        assert generate_su2_rotation([0, 0, 1], 0.5, dtype=C128).dtype == C128

    def test_zero_axis_rejected(self):
        with pytest.raises(ValueError):
            generate_su2_rotation([0, 0, 0], 1.0)
        with pytest.raises(ValueError):
            generate_su2_rotation([0, 1], 1.0)

    def test_axis_angle_vector(self):
        v = torch.tensor([0.3, 0.7, 0.2], dtype=torch.float64)
        R = generate_su2_rotation_from_axis_angle(v)
        assert torch.allclose(R, reference_su2(v, v.norm().item()), atol=1e-12)

    def test_axis_angle_zero_is_identity(self):
        R = generate_su2_rotation_from_axis_angle([0.0, 0.0, 0.0], dtype=C128)
        assert torch.allclose(R, torch.eye(2, dtype=C128))

    def test_axis_angle_batched_with_zero_row(self):
        v = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=torch.float64)
        R = generate_su2_rotation_from_axis_angle(v)
        assert R.shape == (2, 2, 2)
        assert torch.allclose(R[0], torch.eye(2, dtype=C128))
        assert torch.allclose(R[1], reference_su2([0, 1, 0], 1.2), atol=1e-12)


class TestRotate:
    def test_rotate_values(self):
        # Rotating spin-up by π about y gives spin-down (up to the SU(2) phase).
        up = cs(1, 0)
        R = generate_su2_rotation([0, 1, 0], torch.pi, dtype=C128)
        down = rotate(up, R)
        assert torch.allclose(down.data, cs(0, 1).data, atol=1e-12)

    def test_rotate_matches_matmul(self):
        s = Spinor(torch.randn(16, 2, dtype=C128))
        R = generate_su2_rotation([0.2, 0.5, 0.1], 0.9, dtype=C128)
        assert torch.allclose(rotate(s, R).data, (R @ s.data.T).T)

    def test_rotation_preserves_norm(self):
        s = Spinor(torch.randn(32, 2, dtype=C128))
        R = generate_su2_rotation([0.4, 0.4, 0.8], torch.pi / 4, dtype=C128)
        assert torch.allclose(norm(s), norm(rotate(s, R)))

    def test_rotate_promotes_dtypes(self):
        s64 = Spinor(torch.randn(4, 2, dtype=torch.complex64))
        R128 = generate_su2_rotation([0, 0, 1], 0.5, dtype=C128)
        assert rotate(s64, R128).dtype == C128

        real = Spinor(torch.randn(4, 2))
        assert rotate(real, R128).dtype == C128
        assert rotate(real, generate_su2_rotation([0, 0, 1], 0.5)).dtype == torch.complex64

    def test_rotate_batched_matrices(self):
        s = Spinor(torch.randn(5, 2, dtype=C128))
        R = generate_su2_rotation([1, 0, 0], torch.linspace(0, 1, 5, dtype=torch.float64))
        out = rotate(s, R)
        assert out.shape == (5, 2)
        for i in range(5):
            assert torch.allclose(out.data[i], R[i] @ s.data[i])

    def test_rotate_rejects_bad_matrix(self):
        with pytest.raises(ValueError):
            rotate(cs(1, 0), torch.eye(3, dtype=C128))


class TestGlobalPhase:
    def test_phase_value(self):
        s = cs(1, 1j)
        out = global_phase(s, torch.pi / 2)
        assert torch.allclose(out.data, 1j * s.data, atol=1e-12)
        assert torch.allclose(norm(out), norm(s))

    def test_per_spinor_phases(self):
        s = Spinor(torch.ones(3, 2, dtype=C128))
        angles = torch.tensor([0.0, torch.pi / 2, torch.pi], dtype=torch.float64)
        out = global_phase(s, angles)
        expected = torch.polar(torch.ones_like(angles), angles).unsqueeze(-1) * s.data
        assert torch.allclose(out.data, expected, atol=1e-12)

    def test_real_spinor_becomes_complex(self):
        assert global_phase(Spinor(torch.randn(2)), 0.3).dtype == torch.complex64


class TestContract:
    def test_default_is_inner_product(self):
        a = Spinor(torch.randn(8, 2, dtype=C128))
        b = Spinor(torch.randn(8, 2, dtype=C128))
        assert torch.allclose(contract(a, b), inner_product(a, b))

    def test_explicit_indices_use_tensordot(self):
        a = Spinor(torch.randn(3, 2, dtype=C128))
        b = Spinor(torch.randn(3, 2, dtype=C128))
        gram = contract(a, b, indices=([1], [1]))
        assert gram.shape == (3, 3)
        assert torch.allclose(gram, a.data.conj() @ b.data.T)
        full = contract(a, b, indices=2)
        assert torch.allclose(full, torch.sum(a.data.conj() * b.data))

    def test_no_conjugate(self):
        a, b = cs(1j, 2), cs(3, 4j)
        assert torch.allclose(contract(a, b, conjugate=False), torch.sum(a.data * b.data))


# ----------------------------------------------------------------- autograd


class TestGradientPropagation:
    def test_gradient_through_inner_product(self):
        data = torch.randn(2, requires_grad=True)
        s = Spinor(data)
        inner_product(s, s).sum().backward()
        assert data.grad is not None
        assert torch.allclose(data.grad, 2 * data)

    def test_gradient_through_rotate_and_normalize(self):
        data = torch.randn(8, 2, dtype=C128, requires_grad=True)
        R = generate_su2_rotation([0.3, 0.7, 0.2], 1.1, dtype=C128)
        loss = (norm(normalize(rotate(Spinor(data), R))) ** 2).sum()
        loss.backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()

    def test_gradient_wrt_rotation_angle(self):
        angle = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        s = cs(1, 0)
        R = generate_su2_rotation([0, 1, 0], angle)
        # <up| R |up> = cos(θ/2); minimise its real part.
        out = inner_product(s, rotate(s, R)).real
        out.backward()
        assert torch.allclose(angle.grad, -torch.sin(angle.detach() / 2) / 2)

    def test_gradient_wrt_axis_angle_vector(self):
        v = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
        s = Spinor(torch.randn(6, 2, dtype=C128))
        R = generate_su2_rotation_from_axis_angle(v)
        (inner_product(s, rotate(s, R)).real.sum()).backward()
        assert v.grad is not None
        assert torch.isfinite(v.grad).all()

    def test_gradcheck_su2_generator(self):
        axis = torch.randn(3, dtype=torch.float64, requires_grad=True)
        angle = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)

        def f(a, t):
            return torch.view_as_real(generate_su2_rotation(a, t))

        assert torch.autograd.gradcheck(f, (axis, angle))

    def test_spec_example(self):
        data = torch.randn(32, 2, dtype=torch.complex64, requires_grad=True)
        s = Spinor(data)
        R = generate_su2_rotation(axis=[0, 0, 1], angle=torch.pi / 4)
        loss = (rotate(s, R).norm() ** 2).sum()
        loss.backward()
        assert data.grad is not None


# ----------------------------------------------------------------- batching


class TestBatching:
    def test_batch_norm_shape(self):
        s = Spinor(torch.randn(32, 2, dtype=C128))
        assert norm(s).shape == (32,)

    def test_multi_dim_batch(self):
        s = Spinor(torch.randn(4, 5, 2, dtype=C128))
        R = generate_su2_rotation([0, 1, 0], 0.3, dtype=C128)
        assert rotate(s, R).shape == (4, 5, 2)
        assert inner_product(s, s).shape == (4, 5)
        assert outer_product(s, s).shape == (4, 5, 2, 2)

    def test_batch_rotation(self):
        s = Spinor(torch.randn(16, 2, dtype=C128))
        R = generate_su2_rotation([0, 0, 1], torch.pi / 4, dtype=C128)
        assert rotate(s, R).shape == (16, 2)


# ----------------------------------------------------------------------- nn


class TestNNModules:
    def test_spinor_linear_shapes_and_grad(self):
        layer = SpinorLinear(2, 3, bias=True)
        s = Spinor(torch.randn(10, 2, dtype=torch.complex64))
        out = layer(s)
        assert out.shape == (10, 3)
        assert out.dim == 3
        (norm(out) ** 2).sum().backward()
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

    def test_spinor_linear_matches_matmul(self):
        layer = SpinorLinear(2, 2)
        s = Spinor(torch.randn(7, 2, dtype=torch.complex64))
        expected = s.data @ layer.weight.T
        assert torch.allclose(layer(s).data, expected)

    def test_spinor_linear_rejects_wrong_dim(self):
        with pytest.raises(ValueError):
            SpinorLinear(2, 2)(Spinor(torch.randn(3, 4), dim=4))

    def test_su2_rotation_is_unitary_and_trainable(self):
        torch.manual_seed(0)
        layer = SU2Rotation()
        R = layer.matrix
        assert torch.allclose(R.conj().T @ R, torch.eye(2, dtype=R.dtype), atol=1e-6)

        # Learn to map spin-up to spin-down.
        up = Spinor(torch.tensor([1, 0], dtype=torch.complex64))
        down = torch.tensor([0, 1], dtype=torch.complex64)
        opt = torch.optim.Adam(layer.parameters(), lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss = 1 - torch.abs(torch.vdot(down, layer(up).data)) ** 2
            loss.backward()
            opt.step()
        assert loss.item() < 1e-3

    def test_su2_rotation_explicit_init(self):
        layer = SU2Rotation(init=torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(layer.matrix, torch.eye(2, dtype=torch.complex64))
        with pytest.raises(ValueError):
            SU2Rotation(init=torch.zeros(2))

    def test_modules_compose_in_sequential(self):
        model = torch.nn.Sequential(SU2Rotation(), SpinorLinear(2, 2))
        s = Spinor(torch.randn(5, 2, dtype=torch.complex64))
        assert model(s).shape == (5, 2)
