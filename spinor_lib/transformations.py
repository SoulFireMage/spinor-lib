"""Rotation / SU(2) operations for spinors.

Rotation generators accept scalar or batched inputs and broadcast: an axis of
shape ``[..., 3]`` with an angle of shape ``[...]`` gives matrices of shape
``[..., 2, 2]``. Everything is differentiable.
"""

from typing import Optional, Sequence, Union
import torch

from .core import Spinor
from .ops import inner_product
from .utils import (
    TensorLike,
    complex_dtype_of,
    real_dtype_of,
    resolve_dtype_and_device,
)

DeviceLike = Optional[Union[str, torch.device]]


def generate_su2_rotation(
    axis: TensorLike,
    angle: Union[float, torch.Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    SU(2) rotation matrix ``R = cos(θ/2) I − i sin(θ/2) (n·σ)`` for axis ``n``.

    Args:
        axis: Rotation axis, shape ``[..., 3]``. Need not be normalised.
        angle: Rotation angle in radians, broadcastable against ``axis[..., 0]``.
        dtype: Output complex dtype (``complex64``/``complex128``). Defaults to
            the precision of the floating inputs, else the torch default.
        device: Output device. Defaults to the inputs' device, else CPU.

    Returns:
        Complex tensor of shape ``[..., 2, 2]`` where ``[...]`` is the broadcast
        of the axis and angle batch shapes.
    """
    real_dtype, device = resolve_dtype_and_device([axis, angle], dtype, device)
    axis = torch.as_tensor(axis, dtype=real_dtype, device=device)
    angle = torch.as_tensor(angle, dtype=real_dtype, device=device)

    if axis.ndim == 0 or axis.shape[-1] != 3:
        raise ValueError(f"axis must have shape [..., 3], got {tuple(axis.shape)}")

    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if bool((axis_norm == 0).any()):
        raise ValueError("Rotation axis must be non-zero")
    n = axis / axis_norm

    cos_half = torch.cos(angle / 2)
    sin_half = torch.sin(angle / 2)
    nx, ny, nz = n.unbind(-1)
    c, sx, sy, sz = torch.broadcast_tensors(
        cos_half, sin_half * nx, sin_half * ny, sin_half * nz
    )

    # R = c·I − i·s·(n·σ) with n·σ = [[nz, nx − i·ny], [nx + i·ny, −nz]]:
    #   R00 =  c − i·s·nz      R01 = −s·ny − i·s·nx
    #   R10 =  s·ny − i·s·nx   R11 =  c + i·s·nz
    real = torch.stack([torch.stack([c, -sy], -1), torch.stack([sy, c], -1)], -2)
    imag = torch.stack([torch.stack([-sz, -sx], -1), torch.stack([-sx, sz], -1)], -2)
    return torch.complex(real, imag)


def generate_su2_rotation_from_axis_angle(
    axis_angle: TensorLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    SU(2) rotation matrix from an axis-angle vector of shape ``[..., 3]``.

    The vector's direction is the axis and its magnitude the angle; a zero
    vector gives the identity. Differentiable w.r.t. ``axis_angle``.
    """
    real_dtype, device = resolve_dtype_and_device([axis_angle], dtype, device)
    v = torch.as_tensor(axis_angle, dtype=real_dtype, device=device)
    if v.ndim == 0 or v.shape[-1] != 3:
        raise ValueError(f"axis_angle must have shape [..., 3], got {tuple(v.shape)}")

    angle = torch.linalg.vector_norm(v, dim=-1)
    is_zero = angle == 0
    safe_angle = torch.where(is_zero, torch.ones_like(angle), angle)
    axis = v / safe_angle.unsqueeze(-1)

    # Any axis works for a zero angle; substitute z to avoid a zero axis.
    z_axis = torch.zeros_like(axis)
    z_axis[..., 2] = 1
    axis = torch.where(is_zero.unsqueeze(-1), z_axis, axis)

    return generate_su2_rotation(axis, angle, dtype=dtype, device=device)


def rotate(spinor: Spinor, matrix: torch.Tensor) -> Spinor:
    """
    Apply a rotation matrix: ``R @ s`` over the component axis.

    ``matrix`` has shape ``[..., N, N]`` and broadcasts against the spinor's
    batch shape. dtypes are promoted, so a ``complex128`` matrix can act on a
    ``complex64`` (or real) spinor.
    """
    n = spinor.dim
    if matrix.ndim < 2 or matrix.shape[-2:] != (n, n):
        raise ValueError(
            f"Rotation matrix must have shape [..., {n}, {n}], got {tuple(matrix.shape)}"
        )
    out_dtype = torch.promote_types(spinor.dtype, matrix.dtype)
    m = matrix.to(dtype=out_dtype, device=spinor.device)
    s = spinor.data.to(out_dtype)
    result = torch.matmul(m, s.unsqueeze(-1)).squeeze(-1)
    return spinor._wrap(result)


def global_phase(spinor: Spinor, angle: Union[float, torch.Tensor]) -> Spinor:
    """
    Multiply by the global phase ``exp(i·angle)``.

    ``angle`` may be a scalar or a tensor of batch shape ``[...]`` giving one
    phase per spinor. The result is complex.
    """
    angle = torch.as_tensor(angle, device=spinor.device)
    if not angle.is_floating_point():
        angle = angle.to(real_dtype_of(spinor.dtype))
    phase = torch.polar(torch.ones_like(angle), angle)
    if phase.ndim > 0:
        phase = phase.unsqueeze(-1)
    out_dtype = torch.promote_types(complex_dtype_of(spinor.dtype), phase.dtype)
    return spinor._wrap(spinor.data.to(out_dtype) * phase)


def contract(
    s1: Spinor,
    s2: Spinor,
    indices: Optional[Union[int, Sequence[Sequence[int]]]] = None,
    conjugate: bool = True,
) -> torch.Tensor:
    """
    Contract ``s1†`` (or ``s1`` if ``conjugate=False``) with ``s2``.

    With ``indices=None`` this is the batched ``inner_product`` over the
    component axis. Otherwise ``indices`` is passed to ``torch.tensordot`` as
    its ``dims`` argument — an int ``k`` contracts the last ``k`` axes of ``s1``
    with the first ``k`` of ``s2``, or a pair of axis lists contracts those
    axes pairwise — and the remaining axes of both operands are kept in order.
    """
    if indices is None:
        return inner_product(s1, s2) if conjugate else torch.sum(s1.data * s2.data, -1)
    out_dtype = torch.promote_types(s1.dtype, s2.dtype)
    left = torch.conj(s1.data) if conjugate else s1.data
    return torch.tensordot(left.to(out_dtype), s2.data.to(out_dtype), dims=indices)
