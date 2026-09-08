"""Lorentz transformations of Weyl and Dirac spinors via SL(2, C).

Conventions
-----------
A 4-vector ``x = (x⁰, x¹, x², x³)`` is packed into a Hermitian matrix using
``σ_μ = (I, σx, σy, σz)`` for right-handed quantities and
``σ̄_μ = (I, −σx, −σy, −σz)`` for left-handed ones. An element ``A`` of
SL(2, C) acts as ``X ↦ A X A†``, which is a proper orthochronous Lorentz
transformation ``x ↦ Λx``; :func:`sl2c_to_lorentz` extracts ``Λ``.

Right-handed Weyl spinors transform as ``ψ_R ↦ A_R ψ_R`` with
``A_R = exp(−i θ·σ/2 + η·σ/2)`` and left-handed ones as ``ψ_L ↦ A_L ψ_L`` with
``A_L = exp(−i θ·σ/2 − η·σ/2) = (A_R†)⁻¹``, where ``θ`` is an axis-angle
rotation vector and ``η`` a rapidity vector. Rotations are shared with
:func:`spinor_lib.generate_su2_rotation`; boosts pick a handedness. A Dirac
spinor in the chiral (Weyl) basis is ``ψ = (ψ_L, ψ_R)`` and transforms with
``diag(A_L, A_R)``.

Apply any of these matrices to a spinor with :func:`spinor_lib.rotate`.
"""

from typing import Optional, Tuple, Union

import torch

from .core import Spinor
from .observables import identity, pauli
from .utils import TensorLike, complex_dtype_of, resolve_dtype_and_device

DeviceLike = Optional[Union[str, torch.device]]

HANDEDNESS_SIGN = {"left": -1.0, "right": 1.0}


def _sign(handedness: str) -> float:
    try:
        return HANDEDNESS_SIGN[handedness]
    except KeyError:
        raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}") from None


def _check_square(m: torch.Tensor, n: int, name: str) -> None:
    if m.ndim < 2 or m.shape[-2:] != (n, n):
        raise ValueError(f"{name} must have shape [..., {n}, {n}], got {tuple(m.shape)}")


def _check_vec3(v: torch.Tensor, name: str) -> None:
    if v.ndim == 0 or v.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3], got {tuple(v.shape)}")


# ------------------------------------------------------------ sigma matrices


def sigma_mu(
    handedness: str = "right",
    dtype: torch.dtype = torch.complex64,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    ``σ_μ = (I, σ)`` for ``handedness="right"`` or ``σ̄_μ = (I, −σ)`` for
    ``"left"``, stacked as ``[4, 2, 2]``.
    """
    sign = _sign(handedness)
    return torch.cat(
        [identity(2, dtype=dtype, device=device)[None], sign * pauli(dtype=dtype, device=device)]
    )


def vector_to_matrix(x: TensorLike, handedness: str = "right") -> torch.Tensor:
    """Pack a 4-vector ``[..., 4]`` into the Hermitian matrix ``x^μ σ_μ`` (or ``x^μ σ̄_μ``)."""
    x = torch.as_tensor(x)
    if not x.is_floating_point() and not x.is_complex():
        x = x.to(torch.get_default_dtype())
    if x.ndim == 0 or x.shape[-1] != 4:
        raise ValueError(f"x must have shape [..., 4], got {tuple(x.shape)}")
    cdtype = complex_dtype_of(x.dtype)
    S = sigma_mu(handedness, dtype=cdtype, device=x.device)
    return torch.einsum("...m,mab->...ab", x.to(cdtype), S)


def matrix_to_vector(X: torch.Tensor, handedness: str = "right") -> torch.Tensor:
    """Inverse of :func:`vector_to_matrix`: ``x^μ = ½ Tr(σ_μ X)``, returned real."""
    _check_square(X, 2, "X")
    X = X.to(complex_dtype_of(X.dtype))
    S = sigma_mu(handedness, dtype=X.dtype, device=X.device)
    return 0.5 * torch.real(torch.einsum("mab,...ba->...m", S, X))


def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x⁰y⁰ − x·y`` with the ``(+, −, −, −)`` metric, broadcasting over batch dimensions."""
    return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)


def minkowski_metric(dtype: torch.dtype = torch.float32, device: DeviceLike = None) -> torch.Tensor:
    """``diag(1, −1, −1, −1)``."""
    return torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=dtype, device=device))


# ------------------------------------------------------------- generators


def rapidity_from_velocity(beta: TensorLike) -> torch.Tensor:
    """Rapidity ``η = atanh(β)`` of a speed ``β = v/c`` (or a velocity vector's magnitude)."""
    return torch.atanh(torch.as_tensor(beta))


def velocity_from_rapidity(eta: TensorLike) -> torch.Tensor:
    """Speed ``β = tanh(η)``."""
    return torch.tanh(torch.as_tensor(eta))


def generate_boost(
    direction: TensorLike,
    rapidity: Union[float, torch.Tensor],
    *,
    handedness: str = "left",
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    Pure boost ``exp(± η/2 n·σ) = cosh(η/2) I ± sinh(η/2) n·σ`` along ``n``
    (``+`` for right-handed, ``−`` for left-handed spinors), shape ``[..., 2, 2]``.

    The corresponding Lorentz matrix (see :func:`sl2c_to_lorentz`) carries a
    particle at rest to velocity ``tanh(η) n``. Batches broadcast exactly as
    in :func:`spinor_lib.generate_su2_rotation`.
    """
    sign = _sign(handedness)
    real_dtype, device = resolve_dtype_and_device([direction, rapidity], dtype, device)
    n = torch.as_tensor(direction, dtype=real_dtype, device=device)
    eta = torch.as_tensor(rapidity, dtype=real_dtype, device=device)
    _check_vec3(n, "direction")
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)

    c = torch.cosh(eta / 2)
    s = sign * torch.sinh(eta / 2)
    nx, ny, nz = n.unbind(-1)
    c, sx, sy, sz = torch.broadcast_tensors(c, s * nx, s * ny, s * nz)
    zero = torch.zeros_like(c)
    # n·σ = [[nz, nx − i ny], [nx + i ny, −nz]]
    real = torch.stack([torch.stack([c + sz, sx], -1), torch.stack([sx, c - sz], -1)], -2)
    imag = torch.stack([torch.stack([zero, -sy], -1), torch.stack([sy, zero], -1)], -2)
    return torch.complex(real, imag)


def generate_lorentz(
    rotation: Optional[TensorLike] = None,
    boost: Optional[TensorLike] = None,
    *,
    handedness: str = "left",
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    General SL(2, C) transformation ``exp(−i θ·σ/2 ± η·σ/2)`` from an
    axis-angle rotation vector ``θ`` and a rapidity vector ``η``, both of
    shape ``[..., 3]`` (either may be omitted). Computed with the matrix
    exponential, so it is differentiable in both vectors.

    This is a single group element, not a rotation followed by a boost;
    compose those with ``@`` if that is what you want.
    """
    if rotation is None and boost is None:
        raise TypeError("generate_lorentz() needs at least one of rotation or boost")
    sign = _sign(handedness)
    inputs = [v for v in (rotation, boost) if v is not None]
    real_dtype, device = resolve_dtype_and_device(inputs, dtype, device)
    cdtype = complex_dtype_of(real_dtype)
    sig = pauli(dtype=cdtype, device=device)

    generator = None
    if rotation is not None:
        theta = torch.as_tensor(rotation, dtype=real_dtype, device=device)
        _check_vec3(theta, "rotation")
        generator = -0.5j * torch.einsum("...i,iab->...ab", theta.to(cdtype), sig)
    if boost is not None:
        eta = torch.as_tensor(boost, dtype=real_dtype, device=device)
        _check_vec3(eta, "boost")
        term = 0.5 * sign * torch.einsum("...i,iab->...ab", eta.to(cdtype), sig)
        generator = term if generator is None else generator + term
    return torch.linalg.matrix_exp(generator)


def other_handedness(A: torch.Tensor) -> torch.Tensor:
    """
    Convert an SL(2, C) matrix between handednesses: ``(A†)⁻¹``. Maps ``A_L``
    to ``A_R`` and back; rotations (unitary ``A``) are unchanged.
    """
    _check_square(A, 2, "A")
    return torch.linalg.inv(A.conj().mT)


# --------------------------------------------------------- Lorentz matrices


def sl2c_to_lorentz(A: torch.Tensor, handedness: str = "left") -> torch.Tensor:
    """
    The ``4×4`` Lorentz matrix ``Λ^μ_ν = ½ Tr(σ_μ A σ_ν A†)`` (with ``σ̄`` for
    ``handedness="left"``) of an SL(2, C) matrix, so that
    ``vector_to_matrix(Λx) = A · vector_to_matrix(x) · A†``.

    ``A_L`` with ``"left"`` and ``A_R = other_handedness(A_L)`` with ``"right"``
    give the same ``Λ``. Unitary ``A`` gives a pure rotation, with the ``3×3``
    block equal to :func:`spinor_lib.su2_to_so3`. ``A`` and ``−A`` map to the
    same ``Λ``.
    """
    _check_square(A, 2, "A")
    A = A.to(complex_dtype_of(A.dtype))
    S = sigma_mu(handedness, dtype=A.dtype, device=A.device)
    L = torch.einsum("mab,...bc,ncd,...ad->...mn", S, A, S, torch.conj(A))
    return 0.5 * torch.real(L)


def weyl_current(spinor: Spinor, handedness: str = "left") -> torch.Tensor:
    """
    The null 4-vector ``j^μ = ψ† σ_μ ψ`` (``σ̄`` for left-handed spinors),
    shape ``[..., 4]`` and real. Transforms as ``j ↦ Λ j`` when ``ψ ↦ A ψ``.
    """
    if spinor.dim != 2:
        raise ValueError(f"weyl_current expects a 2-component spinor, got dim={spinor.dim}")
    cdtype = complex_dtype_of(spinor.dtype)
    S = sigma_mu(handedness, dtype=cdtype, device=spinor.device)
    psi = spinor.data.to(cdtype)
    return torch.real(torch.einsum("...a,mab,...b->...m", torch.conj(psi), S, psi))


# ------------------------------------------------------------ Dirac spinors


def gamma_matrices(dtype: torch.dtype = torch.complex64, device: DeviceLike = None) -> torch.Tensor:
    """
    Dirac matrices ``γ^μ`` in the chiral (Weyl) basis, stacked as ``[4, 4, 4]``:
    ``γ^0 = [[0, I], [I, 0]]``, ``γ^i = [[0, σ_i], [−σ_i, 0]]``. They satisfy
    ``{γ^μ, γ^ν} = 2 η^{μν}`` with ``η = diag(1, −1, −1, −1)``.
    """
    zero = torch.zeros(2, 2, dtype=dtype, device=device)
    upper = sigma_mu("right", dtype=dtype, device=device)  # (I, σ)
    lower = sigma_mu("left", dtype=dtype, device=device)  # (I, −σ)
    top = torch.cat([zero.expand(4, 2, 2), upper], dim=-1)
    bottom = torch.cat([lower, zero.expand(4, 2, 2)], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def gamma5(dtype: torch.dtype = torch.complex64, device: DeviceLike = None) -> torch.Tensor:
    """``γ^5 = i γ^0 γ^1 γ^2 γ^3 = diag(−I, I)`` in the chiral basis."""
    return torch.diag(torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device))


def generate_dirac(
    rotation: Optional[TensorLike] = None,
    boost: Optional[TensorLike] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """
    Spinor representation ``S(Λ) = diag(A_L, A_R)`` of a Lorentz transformation
    acting on chiral-basis Dirac spinors, shape ``[..., 4, 4]``. Satisfies
    ``S⁻¹ γ^μ S = Λ^μ_ν γ^ν``.
    """
    A_L = generate_lorentz(rotation, boost, handedness="left", dtype=dtype, device=device)
    A_R = generate_lorentz(rotation, boost, handedness="right", dtype=dtype, device=device)
    batch = A_L.shape[:-2]
    S = torch.zeros(*batch, 4, 4, dtype=A_L.dtype, device=A_L.device)
    S[..., :2, :2] = A_L
    S[..., 2:, 2:] = A_R
    return S


def dirac_from_weyl(psi_L: Spinor, psi_R: Spinor) -> Spinor:
    """Stack left- and right-handed Weyl spinors into a chiral-basis Dirac spinor."""
    if psi_L.dim != 2 or psi_R.dim != 2:
        raise ValueError("dirac_from_weyl expects two 2-component spinors")
    dtype = torch.promote_types(psi_L.dtype, psi_R.dtype)
    left, right = torch.broadcast_tensors(psi_L.data.to(dtype), psi_R.data.to(dtype))
    return Spinor(torch.cat([left, right], dim=-1), dim=4)


def weyl_from_dirac(psi: Spinor) -> Tuple[Spinor, Spinor]:
    """Split a chiral-basis Dirac spinor into its ``(ψ_L, ψ_R)`` components."""
    if psi.dim != 4:
        raise ValueError(f"weyl_from_dirac expects a 4-component spinor, got dim={psi.dim}")
    return Spinor(psi.data[..., :2], dim=2), Spinor(psi.data[..., 2:], dim=2)


def dirac_adjoint(psi: Spinor) -> torch.Tensor:
    """The row vector ``ψ̄ = ψ† γ^0``, shape ``[..., 4]``."""
    if psi.dim != 4:
        raise ValueError(f"dirac_adjoint expects a 4-component spinor, got dim={psi.dim}")
    cdtype = complex_dtype_of(psi.dtype)
    g0 = gamma_matrices(dtype=cdtype, device=psi.device)[0]
    return torch.conj(psi.data.to(cdtype)) @ g0


def dirac_bilinear(psi: Spinor, chi: Spinor, gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    ``ψ̄ Γ χ`` for an optional ``4×4`` (or batched ``[..., 4, 4]``) matrix
    ``Γ``; ``Γ = None`` gives the scalar ``ψ̄ χ``. Complex-valued.
    """
    if chi.dim != 4:
        raise ValueError(f"dirac_bilinear expects 4-component spinors, got dim={chi.dim}")
    bar = dirac_adjoint(psi)
    dtype = torch.promote_types(bar.dtype, complex_dtype_of(chi.dtype))
    right = chi.data.to(dtype)
    if gamma is not None:
        _check_square(gamma, 4, "gamma")
        right = torch.einsum("...ab,...b->...a", gamma.to(dtype=dtype, device=right.device), right)
    return torch.einsum("...a,...a->...", bar.to(dtype), right)


def dirac_current(psi: Spinor) -> torch.Tensor:
    """The vector current ``j^μ = ψ̄ γ^μ ψ``, shape ``[..., 4]`` and real."""
    cdtype = complex_dtype_of(psi.dtype)
    gammas = gamma_matrices(dtype=cdtype, device=psi.device)
    bar = dirac_adjoint(psi)
    return torch.real(torch.einsum("...a,mab,...b->...m", bar, gammas, psi.data.to(cdtype)))
