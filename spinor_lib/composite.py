"""Multi-spinor (tensor-product) states.

A composite of subsystems with component counts ``dims = (d_0, d_1, ...)`` is
a spinor with ``prod(dims)`` components, ordered so that subsystem 0 is the
most significant index (``|i_0 i_1 ...⟩`` ↦ flat index
``i_0·d_1·d_2·… + i_1·d_2·… + …``), matching ``torch.kron``.
"""

import math
from typing import List, Sequence, Tuple, Union

import torch

from .core import Spinor
from .observables import density_matrix, identity
from .ops import inner_product
from .utils import DEFAULT_EPS

Subsystems = Union[int, Sequence[int]]


def _resolve_subsystems(
    dims: Sequence[int], keep: Subsystems
) -> Tuple[Tuple[int, ...], List[int]]:
    dims = tuple(int(d) for d in dims)
    if any(d < 1 for d in dims):
        raise ValueError(f"dims must be positive, got {dims}")
    keep_list = [keep] if isinstance(keep, int) else list(keep)
    if len(set(keep_list)) != len(keep_list):
        raise ValueError(f"keep contains duplicates: {keep_list}")
    for k in keep_list:
        if not 0 <= k < len(dims):
            raise IndexError(
                f"subsystem index {k} out of range for {len(dims)} subsystems"
            )
    return dims, keep_list


# ------------------------------------------------------------ constructions


def tensor_product(*spinors: Spinor) -> Spinor:
    """
    Tensor (Kronecker) product of spinors over the component axis.

    Batch dimensions broadcast. The result has ``dim = prod(s.dim for s)`` and
    keeps the autograd graph of every input.
    """
    if not spinors:
        raise TypeError("tensor_product() requires at least one spinor")
    out = spinors[0]
    for s in spinors[1:]:
        dtype = torch.promote_types(out.dtype, s.dtype)
        a, b = out.data.to(dtype), s.data.to(dtype)
        data = (a[..., :, None] * b[..., None, :]).flatten(-2)
        out = Spinor(data, dim=out.dim * s.dim)
    return out


def kron(*operators: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product of operators of shape ``[..., n_i, n_i]``."""
    if not operators:
        raise TypeError("kron() requires at least one operator")
    out = operators[0]
    for op in operators[1:]:
        dtype = torch.promote_types(out.dtype, op.dtype)
        a, b = out.to(dtype), op.to(dtype)
        m, n = a.shape[-1], b.shape[-1]
        prod = a[..., :, None, :, None] * b[..., None, :, None, :]
        out = prod.reshape(*prod.shape[:-4], m * n, m * n)
    return out


def embed_operator(
    operator: torch.Tensor, dims: Sequence[int], target: int
) -> torch.Tensor:
    """
    Lift an operator on subsystem ``target`` to the full composite space by
    tensoring with identities on every other subsystem. The result has shape
    ``[..., prod(dims), prod(dims)]`` and can be applied with ``rotate``.
    """
    dims, (target,) = _resolve_subsystems(dims, target)
    d = dims[target]
    if operator.ndim < 2 or operator.shape[-2:] != (d, d):
        raise ValueError(
            f"operator must have shape [..., {d}, {d}] for subsystem {target}, "
            f"got {tuple(operator.shape)}"
        )
    factors = [
        operator
        if i == target
        else identity(n, dtype=operator.dtype, device=operator.device)
        for i, n in enumerate(dims)
    ]
    return kron(*factors)


# ---------------------------------------------------------------- reductions


def partial_trace(
    rho: torch.Tensor, dims: Sequence[int], keep: Subsystems
) -> torch.Tensor:
    """
    Trace out every subsystem not listed in ``keep`` from a density matrix of
    shape ``[..., D, D]`` with ``D = prod(dims)``.

    The kept subsystems stay in their original order, so the result has shape
    ``[..., D_keep, D_keep]`` with ``D_keep = prod(dims[k] for k in keep)``.
    """
    dims, keep = _resolve_subsystems(dims, keep)
    total = math.prod(dims)
    if rho.ndim < 2 or rho.shape[-2:] != (total, total):
        raise ValueError(
            f"rho must have shape [..., {total}, {total}] for dims {dims}, "
            f"got {tuple(rho.shape)}"
        )
    if len(dims) > 26:
        raise ValueError("partial_trace supports at most 26 subsystems")

    batch = rho.shape[:-2]
    r = rho.reshape(*batch, *dims, *dims)
    keep_sorted = sorted(keep)
    rows = [chr(ord("a") + i) for i in range(len(dims))]
    cols = [
        chr(ord("A") + i) if i in keep_sorted else rows[i] for i in range(len(dims))
    ]
    out_rows = "".join(rows[i] for i in keep_sorted)
    out_cols = "".join(cols[i] for i in keep_sorted)
    equation = f"...{''.join(rows)}{''.join(cols)}->...{out_rows}{out_cols}"
    reduced = torch.einsum(equation, r)
    d_keep = math.prod(dims[k] for k in keep_sorted)
    return reduced.reshape(*batch, d_keep, d_keep)


def reduced_density_matrix(
    spinor: Spinor, dims: Sequence[int], keep: Subsystems
) -> torch.Tensor:
    """Density matrix of the subsystems in ``keep`` for a composite pure state."""
    return partial_trace(density_matrix(spinor), dims, keep)


# -------------------------------------------------------------- entanglement


def von_neumann_entropy(
    rho: torch.Tensor, base: float = math.e, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """
    Von Neumann entropy ``−Tr(ρ log ρ)`` of a density matrix, in the given
    ``base`` (``2`` for bits). Eigenvalues below ``eps`` contribute zero.

    The eigendecomposition's gradient is undefined where eigenvalues are
    degenerate (e.g. the maximally mixed state); this is a PyTorch limitation.
    """
    p = torch.linalg.eigvalsh(rho).clamp_min(0)
    safe_p = torch.where(p > eps, p, torch.ones_like(p))
    terms = torch.where(p > eps, p * torch.log(safe_p), torch.zeros_like(p))
    return -terms.sum(-1) / math.log(base)


def entanglement_entropy(
    spinor: Spinor, dims: Sequence[int], keep: Subsystems, base: float = math.e
) -> torch.Tensor:
    """
    Entanglement entropy of a pure composite state across the bipartition
    ``keep`` | rest: the von Neumann entropy of the reduced density matrix.
    Zero for product states, ``log(d)`` for maximally entangled ones.
    """
    return von_neumann_entropy(reduced_density_matrix(spinor, dims, keep), base=base)


def concurrence(spinor: Spinor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """
    Concurrence ``2 |a d − b c| / ⟨s|s⟩`` of a two-qubit pure state
    ``(a, b, c, d)``. Zero for product states, one for Bell states.
    """
    if spinor.dim != 4:
        raise ValueError(
            "concurrence is defined for two-qubit (4-component) states, "
            f"got dim={spinor.dim}"
        )
    a, b, c, d = spinor.data.unbind(-1)
    norm_sq = torch.real(inner_product(spinor, spinor)).clamp_min(eps)
    return 2 * torch.abs(a * d - b * c) / norm_sq
