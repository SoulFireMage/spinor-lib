"""Mathematical operations for spinors.

All operations broadcast over batch dimensions and preserve autograd.
"""

from typing import Union
import torch

from .core import Spinor
from .utils import DEFAULT_EPS, check_broadcastable

Scalar = Union[float, complex, torch.Tensor]


def _check_compatible(s1: Spinor, s2: Spinor) -> None:
    if s1.dim != s2.dim:
        raise ValueError(f"Component mismatch: {s1.dim} vs {s2.dim}")
    check_broadcastable(s1.data, s2.data)


def _as_per_spinor_scalar(scalar: Scalar) -> Scalar:
    """
    Tensors with batch shape ``[...]`` are treated as one scalar per spinor and
    aligned against data of shape ``[..., N]``.
    """
    if isinstance(scalar, torch.Tensor) and scalar.ndim > 0:
        return scalar.unsqueeze(-1)
    return scalar


def add(s1: Spinor, s2: Spinor) -> Spinor:
    """Element-wise addition (broadcasts over batch dimensions)."""
    _check_compatible(s1, s2)
    return s1._wrap(s1.data + s2.data)


def sub(s1: Spinor, s2: Spinor) -> Spinor:
    """Element-wise subtraction (broadcasts over batch dimensions)."""
    _check_compatible(s1, s2)
    return s1._wrap(s1.data - s2.data)


def mul(s: Spinor, scalar: Scalar) -> Spinor:
    """
    Scalar multiplication.

    ``scalar`` may be a Python number or a tensor of batch shape ``[...]``
    holding one scalar per spinor.
    """
    return s._wrap(s.data * _as_per_spinor_scalar(scalar))


def inner_product(s1: Spinor, s2: Spinor) -> torch.Tensor:
    """
    Hermitian inner product ``s1† · s2`` over the component axis.

    Returns a tensor of batch shape (a 0-dim tensor for unbatched spinors).
    """
    _check_compatible(s1, s2)
    return torch.sum(torch.conj(s1.data) * s2.data, dim=-1)


def outer_product(s1: Spinor, s2: Spinor, conjugate: bool = True) -> torch.Tensor:
    """
    Outer product ``|s1><s2|`` with shape ``[..., N, N]``.

    With ``conjugate=True`` (default) this is ``s1 ⊗ s2†``, consistent with the
    Hermitian ``inner_product``; ``outer_product(s, s)`` of a unit spinor is a
    projector. Set ``conjugate=False`` for the plain tensor product ``s1 ⊗ s2ᵀ``.
    """
    _check_compatible(s1, s2)
    right = torch.conj(s2.data) if conjugate else s2.data
    return s1.data.unsqueeze(-1) * right.unsqueeze(-2)


def norm(s: Spinor) -> torch.Tensor:
    """L2 norm over the component axis. Always real-valued."""
    return torch.linalg.vector_norm(s.data, dim=-1)


def normalize(s: Spinor, eps: float = DEFAULT_EPS) -> Spinor:
    """Normalize to unit length. Norms below ``eps`` are clamped to ``eps``."""
    n = norm(s).clamp_min(eps)
    return s._wrap(s.data / n.unsqueeze(-1))


def conjugate(s: Spinor) -> Spinor:
    """Complex conjugate of a spinor."""
    return s._wrap(torch.conj(s.data))
