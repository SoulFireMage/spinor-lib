"""The SU(2) → SO(3) double cover and unit-quaternion interop.

Conventions match :func:`spinor_lib.generate_su2_rotation`: the SU(2) matrix
for a unit quaternion ``q = (w, x, y, z) = (cos θ/2, n sin θ/2)`` is
``R = w·I − i (x σx + y σy + z σz)``, and the SO(3) matrix ``M`` satisfies
``R (v·σ) R† = (M v)·σ``, so rotating a spinor by ``R`` rotates its Bloch
vector by ``M``.
"""

from typing import Union
import torch

from .observables import pauli
from .utils import complex_dtype_of, real_dtype_of


def _check_square(m: torch.Tensor, n: int, name: str) -> None:
    if m.ndim < 2 or m.shape[-2:] != (n, n):
        raise ValueError(
            f"{name} must have shape [..., {n}, {n}], got {tuple(m.shape)}"
        )


def su2_to_so3(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) matrix ``M_ij = ½ Tr(σ_i R σ_j R†)`` of an SU(2) matrix, shape
    ``[..., 3, 3]`` and real. ``R`` and ``−R`` map to the same ``M``.
    """
    _check_square(R, 2, "R")
    R = R.to(complex_dtype_of(R.dtype))
    sig = pauli(dtype=R.dtype, device=R.device)
    M = torch.einsum("iab,...bc,jcd,...ad->...ij", sig, R, sig, torch.conj(R))
    return 0.5 * torch.real(M)


def su2_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Unit quaternion ``(w, x, y, z)`` of an SU(2) matrix, shape ``[..., 4]``.

    Only the entries ``R00`` and ``R10`` are read, so the input is assumed to
    already be in SU(2).
    """
    _check_square(R, 2, "R")
    r00, r10 = R[..., 0, 0], R[..., 1, 0]
    return torch.stack(
        [torch.real(r00), -torch.imag(r10), torch.real(r10), -torch.imag(r00)],
        dim=-1,
    )


def quaternion_to_su2(q: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    SU(2) matrix ``w·I − i (x σx + y σy + z σz)`` of a quaternion ``(w, x, y, z)``
    of shape ``[..., 4]``. With ``normalize=True`` the quaternion is scaled to
    unit length first, which makes this a smooth parameterisation of SU(2) from
    an unconstrained 4-vector.
    """
    q = torch.as_tensor(q)
    if not q.is_floating_point():
        q = q.to(torch.get_default_dtype())
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError(f"q must have shape [..., 4], got {tuple(q.shape)}")
    if normalize:
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    real = torch.stack([torch.stack([w, -y], -1), torch.stack([y, w], -1)], -2)
    imag = torch.stack([torch.stack([-z, -x], -1), torch.stack([-x, z], -1)], -2)
    return torch.complex(real, imag)


def so3_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Unit quaternion ``(w, x, y, z)`` of an SO(3) matrix, shape ``[..., 4]``.

    Uses Shepperd's method: all four candidate formulas are evaluated and the
    numerically best one (largest denominator) is chosen per matrix, so it is
    stable for every rotation including angle ``π``. The sign is not
    canonicalised; ``q`` and ``−q`` are the two SU(2) preimages.
    """
    _check_square(M, 3, "M")
    if not M.is_floating_point():
        M = M.to(torch.get_default_dtype())
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = M.reshape(*M.shape[:-2], 9).unbind(-1)
    trace = m00 + m11 + m22

    # Four times the squared magnitude of each of (w, x, y, z).
    four_sq = torch.stack(
        [
            1 + trace,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp_min(0)
    root = torch.sqrt(four_sq)

    candidates = torch.stack(
        [
            torch.stack([four_sq[..., 0], m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, four_sq[..., 1], m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, four_sq[..., 2], m12 + m21], -1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, four_sq[..., 3]], -1),
        ],
        dim=-2,
    )  # [..., candidate, component]
    candidates = candidates / (2 * root.clamp_min(0.1))[..., None]
    best = torch.argmax(four_sq, dim=-1)
    index = best[..., None, None].expand(*best.shape, 1, 4)
    q = torch.gather(candidates, -2, index).squeeze(-2)
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def so3_to_su2(M: torch.Tensor) -> torch.Tensor:
    """
    One of the two SU(2) preimages of an SO(3) matrix, shape ``[..., 2, 2]``.
    Round-tripping through :func:`su2_to_so3` returns ``M``; round-tripping an
    SU(2) matrix returns ``±R``.
    """
    return quaternion_to_su2(so3_to_quaternion(M), normalize=False)


def slerp_su2(
    R0: torch.Tensor,
    R1: torch.Tensor,
    t: Union[float, torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Geodesic interpolation between two SU(2) rotations.

    Works through unit quaternions and takes the shorter of the two arcs, so
    the result interpolates the *rotation* rather than the sign of ``R``.
    ``t`` is a scalar or a tensor of batch shape.
    """
    q0, q1 = su2_to_quaternion(R0), su2_to_quaternion(R1)
    dot = (q0 * q1).sum(-1)
    q1 = torch.where((dot < 0)[..., None], -q1, q1)
    dot = torch.abs(dot).clamp(0, 1)

    t = torch.as_tensor(t, dtype=real_dtype_of(q0.dtype), device=q0.device)
    omega = torch.arccos(dot)
    small = omega < eps
    safe_omega = torch.where(small, torch.ones_like(omega), omega)
    sin_omega = torch.sin(safe_omega)
    w0 = torch.where(small, 1 - t, torch.sin((1 - t) * safe_omega) / sin_omega)
    w1 = torch.where(small, t, torch.sin(t * safe_omega) / sin_omega)
    q = w0[..., None] * q0 + w1[..., None] * q1
    return quaternion_to_su2(q, normalize=True)
