"""Utility functions for the Spinor library."""

from typing import Optional, Sequence, Union
import torch
import numpy as np

TensorLike = Union[torch.Tensor, np.ndarray, list, tuple, float, complex]

# Threshold below which a norm is treated as zero (used to avoid divide-by-zero).
DEFAULT_EPS = 1e-12


def get_backend(tensor: object) -> str:
    """Detect the backend of an array-like object ("torch" or "numpy")."""
    if isinstance(tensor, torch.Tensor):
        return "torch"
    if isinstance(tensor, np.ndarray):
        return "numpy"
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def ensure_torch_tensor(
    data: TensorLike,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Convert input data to a PyTorch tensor.

    Existing tensors are returned as-is (so autograd leaves are preserved),
    NumPy arrays are wrapped without copying (memory is shared), and anything
    else goes through ``torch.as_tensor``.
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.as_tensor(data)

    if dtype is not None or device is not None:
        tensor = tensor.to(dtype=dtype, device=device)
    return tensor


def validate_spinor_dimension(tensor: torch.Tensor, dim: int = 2) -> None:
    """Validate that the last dimension matches the expected spinor dimension."""
    if tensor.ndim == 0:
        raise ValueError("Spinor tensor must have at least one dimension")
    if tensor.shape[-1] != dim:
        raise ValueError(
            f"Last dimension must be {dim} for spinors, got {tensor.shape[-1]}"
        )


def check_broadcastable(a: torch.Tensor, b: torch.Tensor) -> torch.Size:
    """Return the broadcast shape of two tensors, raising ValueError if impossible."""
    try:
        return torch.broadcast_shapes(a.shape, b.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Shapes {tuple(a.shape)} and {tuple(b.shape)} are not broadcastable"
        ) from exc


def real_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Return the real dtype underlying a (possibly complex) dtype."""
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex32:
        return torch.float16
    return dtype


def complex_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype whose real part has the given precision."""
    if dtype.is_complex:
        return dtype
    if dtype == torch.float64:
        return torch.complex128
    # float32 and everything lower-precision (or non-floating) map to complex64
    return torch.complex64


def resolve_dtype_and_device(
    inputs: Sequence[object],
    dtype: Optional[torch.dtype],
    device: Optional[Union[str, torch.device]],
) -> "tuple[torch.dtype, torch.device]":
    """
    Pick the real working dtype and device for a computation.

    If ``dtype`` is given it wins (a complex dtype is mapped to its real part).
    Otherwise, use the highest-precision floating dtype among any tensor inputs,
    falling back to the torch default. The device comes from ``device`` or the
    first tensor input, falling back to CPU.
    """
    if dtype is not None:
        real_dtype = real_dtype_of(dtype)
    else:
        real_dtype = None
        for x in inputs:
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                real_dtype = x.dtype if real_dtype is None else torch.promote_types(real_dtype, x.dtype)
        if real_dtype is None:
            real_dtype = torch.get_default_dtype()

    if device is None:
        for x in inputs:
            if isinstance(x, torch.Tensor):
                device = x.device
                break
    return real_dtype, torch.device(device) if device is not None else torch.device("cpu")
