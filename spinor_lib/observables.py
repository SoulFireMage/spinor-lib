"""Spin-1/2 observables, Bloch-sphere geometry and density matrices.

Everything here works on batched spinors (``[..., N]``) and batched operators
(``[..., N, N]``), broadcasts over batch dimensions and preserves autograd.
Functions that are specific to two-component spinors say so in their docstring.
"""

from typing import Optional, Union

import torch

from .core import Spinor
from .ops import inner_product, normalize, outer_product
from .utils import DEFAULT_EPS, TensorLike, complex_dtype_of, real_dtype_of

DeviceLike = Optional[Union[str, torch.device]]
State = Union[Spinor, torch.Tensor]


# --------------------------------------------------------------- operators


def pauli(
    dtype: torch.dtype = torch.complex64, device: DeviceLike = None
) -> torch.Tensor:
    """The Pauli matrices stacked as a ``[3, 2, 2]`` tensor ``(σx, σy, σz)``."""
    return torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=dtype,
        device=device,
    )


def identity(
    n: int = 2, dtype: torch.dtype = torch.complex64, device: DeviceLike = None
) -> torch.Tensor:
    """The ``n × n`` identity operator."""
    return torch.eye(n, dtype=dtype, device=device)


def expectation(
    spinor: Spinor, operator: torch.Tensor, hermitian: bool = True
) -> torch.Tensor:
    """
    Expectation value ``⟨s|A|s⟩`` of an operator on a (not necessarily
    normalised) spinor.

    ``operator`` has shape ``[..., N, N]`` and broadcasts against the batch
    shape. With ``hermitian=True`` (the default, appropriate for observables)
    the real part is returned; otherwise the full complex value.
    """
    n = spinor.dim
    if operator.ndim < 2 or operator.shape[-2:] != (n, n):
        raise ValueError(
            f"operator must have shape [..., {n}, {n}], got {tuple(operator.shape)}"
        )
    out_dtype = torch.promote_types(complex_dtype_of(spinor.dtype), operator.dtype)
    s = spinor.data.to(out_dtype)
    a = operator.to(dtype=out_dtype, device=spinor.device)
    value = torch.einsum("...i,...ij,...j->...", torch.conj(s), a, s)
    return torch.real(value) if hermitian else value


# ------------------------------------------------------------ density matrix


def density_matrix(
    spinor: Spinor, normalize: bool = True, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """
    Density matrix ``ρ = |s⟩⟨s| / ⟨s|s⟩`` with shape ``[..., N, N]``.

    With ``normalize=False`` the projector is not divided by the norm squared.
    """
    rho = outer_product(spinor, spinor)
    if normalize:
        norm_sq = torch.real(inner_product(spinor, spinor)).clamp_min(eps)
        rho = rho / norm_sq[..., None, None]
    return rho


def _as_density(state: State) -> torch.Tensor:
    return state if isinstance(state, torch.Tensor) else density_matrix(state)


def purity(rho: torch.Tensor) -> torch.Tensor:
    """``Tr(ρ²)``: 1 for pure states, ``1/N`` for the maximally mixed state."""
    return torch.real(torch.einsum("...ij,...ji->...", rho, rho))


def density_from_bloch(
    bloch: TensorLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    Qubit density matrix ``ρ = ½ (I + n·σ)`` from a Bloch vector ``n`` of
    shape ``[..., 3]``. ``|n| < 1`` gives a mixed state, ``|n| = 1`` a pure one.
    """
    n = torch.as_tensor(bloch, device=device)
    if not n.is_floating_point():
        n = n.to(torch.get_default_dtype())
    if n.ndim == 0 or n.shape[-1] != 3:
        raise ValueError(f"bloch must have shape [..., 3], got {tuple(n.shape)}")
    cdtype = dtype or complex_dtype_of(n.dtype)
    sig = pauli(dtype=cdtype, device=n.device)
    n_sigma = torch.einsum("...i,ijk->...jk", n.to(cdtype), sig)
    return 0.5 * (identity(2, dtype=cdtype, device=n.device) + n_sigma)


# -------------------------------------------------------------- Bloch sphere


def bloch_vector(state: State) -> torch.Tensor:
    """
    Bloch vector ``n_i = Tr(ρ σ_i)`` of a two-component spinor or a ``2×2``
    density matrix, shape ``[..., 3]`` and real.

    For a spinor the state is normalised first, so the result lies on the unit
    sphere; a density matrix is used as given.
    """
    rho = _as_density(state)
    if rho.shape[-2:] != (2, 2):
        raise ValueError(
            f"Bloch vectors are defined for 2×2 states, got {tuple(rho.shape)}"
        )
    sig = pauli(dtype=complex_dtype_of(rho.dtype), device=rho.device)
    return torch.real(torch.einsum("...jk,ikj->...i", rho.to(sig.dtype), sig))


def from_bloch_vector(
    bloch: TensorLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
    eps: float = DEFAULT_EPS,
) -> Spinor:
    """
    Pure spinor ``(cos θ/2, e^{iφ} sin θ/2)`` pointing along a Bloch vector.

    ``bloch`` has shape ``[..., 3]`` and is normalised first. The global phase
    is fixed so that the first component is real and non-negative. The map is
    smooth except at the south pole, where the phase ``φ`` is undefined and
    ``(0, 1)`` is returned.
    """
    n = torch.as_tensor(bloch, device=device)
    if not n.is_floating_point():
        n = n.to(torch.get_default_dtype())
    if n.ndim == 0 or n.shape[-1] != 3:
        raise ValueError(f"bloch must have shape [..., 3], got {tuple(n.shape)}")
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(eps)
    nx, ny, nz = n.unbind(-1)

    cos_half = torch.sqrt(((1 + nz) / 2).clamp_min(0))
    sin_half = torch.sqrt(((1 - nz) / 2).clamp_min(0))
    r = torch.sqrt(nx * nx + ny * ny)
    on_axis = r < eps
    safe_r = torch.where(on_axis, torch.ones_like(r), r)
    phase = torch.complex(
        torch.where(on_axis, torch.ones_like(nx), nx / safe_r),
        torch.where(on_axis, torch.zeros_like(ny), ny / safe_r),
    )
    cdtype = dtype or complex_dtype_of(n.dtype)
    data = torch.stack(
        [cos_half.to(cdtype), phase.to(cdtype) * sin_half.to(cdtype)], dim=-1
    )
    return Spinor(data, dim=2)


# -------------------------------------------------------- distances / paths


def fidelity(a: State, b: State) -> torch.Tensor:
    """
    Fidelity between two states, in ``[0, 1]``.

    For two spinors this is ``|⟨a|b⟩|² / (⟨a|a⟩⟨b|b⟩)``. If either argument is
    a density matrix both are treated as ``2×2`` density matrices and the qubit
    closed form ``Tr(ρσ) + 2 √(det ρ · det σ)`` is used.
    """
    if isinstance(a, Spinor) and isinstance(b, Spinor):
        overlap = inner_product(a, b)
        denom = torch.real(inner_product(a, a)) * torch.real(inner_product(b, b))
        return (torch.abs(overlap) ** 2 / denom.clamp_min(DEFAULT_EPS)).clamp(0, 1)
    rho, sigma = _as_density(a), _as_density(b)
    if rho.shape[-2:] != (2, 2) or sigma.shape[-2:] != (2, 2):
        raise NotImplementedError(
            "Mixed-state fidelity is implemented for 2×2 states only"
        )
    dtype = torch.promote_types(rho.dtype, sigma.dtype)
    rho, sigma = rho.to(dtype), sigma.to(dtype)
    trace = torch.real(torch.einsum("...ij,...ji->...", rho, sigma))
    dets = torch.real(torch.linalg.det(rho) * torch.linalg.det(sigma)).clamp_min(0)
    return (trace + 2 * torch.sqrt(dets)).clamp(0, 1)


def fubini_study_distance(a: Spinor, b: Spinor) -> torch.Tensor:
    """
    Fubini–Study distance ``arccos |⟨a|b⟩|`` between two (unnormalised) spinors
    as points of projective space, in ``[0, π/2]``. Global phases are ignored.

    The gradient is unbounded when the states coincide or are orthogonal.
    """
    return torch.arccos(torch.sqrt(fidelity(a, b)).clamp(0, 1))


def slerp(
    a: Spinor, b: Spinor, t: Union[float, torch.Tensor], eps: float = 1e-6
) -> Spinor:
    """
    Geodesic interpolation between two states in projective space.

    Both inputs are normalised and ``b`` is phase-aligned with ``a`` so the
    path is the Fubini–Study geodesic. ``t`` is a scalar or a tensor of batch
    shape; ``t=0`` gives ``a`` and ``t=1`` gives ``b`` (up to a global phase).
    Orthogonal endpoints have no unique geodesic; one is chosen.
    """
    out_dtype = torch.promote_types(
        complex_dtype_of(a.dtype), complex_dtype_of(b.dtype)
    )
    na = normalize(a).to(dtype=out_dtype)
    nb = normalize(b).to(dtype=out_dtype)

    overlap = inner_product(na, nb)
    mag = torch.abs(overlap)
    phase = torch.where(
        mag > eps, overlap / mag.clamp_min(eps), torch.ones_like(overlap)
    )
    nb_data = nb.data * torch.conj(phase)[..., None]  # now ⟨a|b⟩ is real ≥ 0

    t = torch.as_tensor(t, dtype=real_dtype_of(out_dtype), device=na.device)
    omega = torch.arccos(mag.clamp(0, 1))
    small = omega < eps
    safe_omega = torch.where(small, torch.ones_like(omega), omega)
    sin_omega = torch.sin(safe_omega)
    w_a = torch.where(small, 1 - t, torch.sin((1 - t) * safe_omega) / sin_omega)
    w_b = torch.where(small, t, torch.sin(t * safe_omega) / sin_omega)
    data = (
        w_a.to(out_dtype)[..., None] * na.data
        + w_b.to(out_dtype)[..., None] * nb_data
    )
    return Spinor(data, dim=a.dim)
