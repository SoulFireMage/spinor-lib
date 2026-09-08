"""SU(2)-equivariant building blocks for spinor-valued features.

A *feature map* here is a spinor of shape ``[..., C, 2]``: ``C`` channels of
2-component spinors, with the group acting on the last axis of every channel
at once (``rotate`` does exactly that, since a ``[2, 2]`` matrix broadcasts
over the channel axis). The functions in this module are the spin-½
Clebsch–Gordan products: two spin-½ objects combine into a spin-0 scalar or
a spin-1 vector, and a vector acting on a spinor gives a spinor.
"""

from typing import Optional

import torch

from .core import Spinor
from .observables import pauli
from .ops import inner_product
from .utils import complex_dtype_of


def _pair_dtype(a: Spinor, b: Spinor) -> torch.dtype:
    return torch.promote_types(complex_dtype_of(a.dtype), complex_dtype_of(b.dtype))


def _check_two(s: Spinor, name: str) -> None:
    if s.dim != 2:
        raise ValueError(f"{name} must be a 2-component spinor, got dim={s.dim}")


def singlet(a: Spinor, b: Spinor) -> torch.Tensor:
    """
    The antisymmetric invariant ``ε^{ij} a_i b_j = a₀ b₁ − a₁ b₀`` (no
    conjugation). Invariant under any matrix of determinant 1, so under SU(2)
    *and* SL(2, C); antisymmetric in its arguments. Complex-valued.
    """
    _check_two(a, "a")
    _check_two(b, "b")
    dtype = _pair_dtype(a, b)
    x, y = a.data.to(dtype), b.data.to(dtype)
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def overlap(a: Spinor, b: Spinor) -> torch.Tensor:
    """The Hermitian invariant ``⟨a|b⟩``; alias of :func:`spinor_lib.inner_product`."""
    return inner_product(a, b)


def vector(a: Spinor, b: Optional[Spinor] = None) -> torch.Tensor:
    """
    The spin-1 combination ``v_i = ⟨a|σ_i|b⟩``, shape ``[..., 3]``.

    Rotates as an ordinary 3-vector (``v ↦ M v`` with ``M = su2_to_so3(R)``)
    when both spinors are rotated by ``R``. With ``b=None`` it is ``⟨a|σ|a⟩``,
    which is real and equals the unnormalised Bloch vector; otherwise complex.
    """
    _check_two(a, "a")
    if b is None:
        b = a
    _check_two(b, "b")
    dtype = _pair_dtype(a, b)
    sig = pauli(dtype=dtype, device=a.device)
    v = torch.einsum("...i,kij,...j->...k", torch.conj(a.data.to(dtype)), sig, b.data.to(dtype))
    return torch.real(v) if b is a else v


def apply_vector(v: torch.Tensor, s: Spinor) -> Spinor:
    """
    The spin-1 ⊗ spin-½ → spin-½ product ``(v·σ) s`` for a real or complex
    vector ``v`` of shape ``[..., 3]``. Equivariant: rotating ``v`` by ``M``
    and ``s`` by ``R`` rotates the result by ``R``.
    """
    _check_two(s, "s")
    if v.ndim == 0 or v.shape[-1] != 3:
        raise ValueError(f"v must have shape [..., 3], got {tuple(v.shape)}")
    dtype = torch.promote_types(complex_dtype_of(v.dtype), complex_dtype_of(s.dtype))
    sig = pauli(dtype=dtype, device=s.device)
    v_sigma = torch.einsum("...k,kij->...ij", v.to(dtype), sig)
    out = torch.einsum("...ij,...j->...i", v_sigma, s.data.to(dtype))
    return Spinor(out, dim=2)


def gram(features: Spinor) -> torch.Tensor:
    """
    Gram matrix ``G_cc' = ⟨ψ_c|ψ_c'⟩`` of a feature map ``[..., C, N]``, shape
    ``[..., C, C]``, Hermitian. Every entry is invariant under a common
    unitary transformation of all channels.
    """
    x = features.data.to(complex_dtype_of(features.dtype))
    return torch.einsum("...ci,...di->...cd", torch.conj(x), x)


def invariant_features(features: Spinor) -> torch.Tensor:
    """
    Real invariant descriptor of a feature map ``[..., C, N]``: the ``C(C+1)/2``
    real parts of the upper triangle of the Gram matrix (norms and overlaps)
    followed by the ``C(C−1)/2`` imaginary parts of its strict upper triangle,
    giving ``C²`` real numbers. Invariant under SU(2) and global phases.
    """
    G = gram(features)
    C = G.shape[-1]
    rows, cols = torch.triu_indices(C, C, offset=0, device=G.device)
    upper = G[..., rows, cols]
    strict = rows != cols
    return torch.cat([torch.real(upper), torch.imag(upper[..., strict])], dim=-1)
