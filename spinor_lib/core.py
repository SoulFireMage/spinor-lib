"""Core Spinor class definition."""

from typing import Any, Optional, Sequence, Union
import torch

from .utils import TensorLike, ensure_torch_tensor, validate_spinor_dimension


class Spinor:
    """
    A spinor that wraps a PyTorch tensor of shape ``[..., N]``.

    The leading dimensions are batch dimensions; the last dimension holds the
    ``N`` spinor components (``N == 2`` by default). All operations preserve the
    autograd graph of the wrapped tensor.

    Attributes:
        data: The underlying tensor, shape ``[..., N]``.
        dim: The number of spinor components ``N``.
    """

    def __init__(
        self,
        data: Optional[TensorLike] = None,
        *,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        dim: int = 2,
    ):
        """
        Create a spinor either from data or as zeros of a given shape.

        Args:
            data: Tensor, NumPy array, or nested list of components. A tensor is
                wrapped without copying, so gradients accumulate on the original.
            shape: Alternative to ``data`` — build a zero spinor of this shape.
                Defaults to ``torch.complex64`` unless ``dtype`` is given.
            dtype: Optional dtype to cast to.
            device: Optional device to move to.
            dim: Expected number of components in the last axis.
        """
        if (data is None) == (shape is None):
            raise TypeError("Spinor() requires exactly one of `data` or `shape`")
        if dim < 1:
            raise ValueError(f"dim must be a positive integer, got {dim}")

        if shape is not None:
            self.data = torch.zeros(
                tuple(shape), dtype=dtype or torch.complex64, device=device
            )
        else:
            self.data = ensure_torch_tensor(data, dtype=dtype, device=device)

        self.dim = dim
        validate_spinor_dimension(self.data, dim)

    # ------------------------------------------------------------------ props

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def batch_shape(self) -> torch.Size:
        """Shape of the batch dimensions (everything but the component axis)."""
        return self.data.shape[:-1]

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def requires_grad(self) -> bool:
        return self.data.requires_grad

    @property
    def grad(self) -> Optional[torch.Tensor]:
        return self.data.grad

    # ---------------------------------------------------------------- helpers

    def _wrap(self, tensor: torch.Tensor) -> "Spinor":
        """Wrap a tensor as a spinor with the same component count as ``self``."""
        return Spinor(tensor, dim=self.dim)

    # ---------------------------------------------------------------- dunders

    def __repr__(self) -> str:
        return f"Spinor(data={self.data!r}, dim={self.dim})"

    def __str__(self) -> str:
        return (
            f"Spinor(shape={tuple(self.data.shape)}, dtype={self.data.dtype}, "
            f"device={self.data.device})"
        )

    def __getitem__(self, key: Any) -> Union["Spinor", torch.Tensor]:
        """
        Index into the batch dimensions.

        Returns a ``Spinor`` when the result still ends in a component axis of
        size ``dim``; otherwise (e.g. ``s[..., 0]`` picking a single component)
        the raw tensor is returned.
        """
        result = self.data[key]
        if result.ndim >= 1 and result.shape[-1] == self.dim:
            return self._wrap(result)
        return result

    def __len__(self) -> int:
        if self.data.ndim < 2:
            raise TypeError("len() of an unbatched spinor; use .dim for components")
        return self.data.shape[0]

    def __add__(self, other: "Spinor") -> "Spinor":
        from .ops import add
        return add(self, other)

    def __sub__(self, other: "Spinor") -> "Spinor":
        from .ops import sub
        return sub(self, other)

    def __mul__(self, scalar: Union[float, complex, torch.Tensor]) -> "Spinor":
        from .ops import mul
        return mul(self, scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Union[float, complex, torch.Tensor]) -> "Spinor":
        from .ops import mul
        return mul(self, 1.0 / torch.as_tensor(scalar, device=self.device))

    def __neg__(self) -> "Spinor":
        return self._wrap(-self.data)

    # ---------------------------------------------------------------- methods

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Spinor":
        """Move/cast the spinor, mirroring ``torch.Tensor.to``."""
        return self._wrap(self.data.to(device=device, dtype=dtype))

    def reshape(self, *shape: int) -> "Spinor":
        """Reshape the batch dimensions; the component axis must be preserved."""
        return self._wrap(self.data.reshape(*shape))

    def clone(self) -> "Spinor":
        return self._wrap(self.data.clone())

    def detach(self) -> "Spinor":
        """Explicitly detach from the autograd graph."""
        return self._wrap(self.data.detach())

    def norm(self) -> torch.Tensor:
        """L2 norm over the component axis (real-valued)."""
        from .ops import norm
        return norm(self)

    def normalize(self) -> "Spinor":
        from .ops import normalize
        return normalize(self)

    def conj(self) -> "Spinor":
        from .ops import conjugate
        return conjugate(self)
