"""``torch.nn`` modules that act on spinors."""

import math
from typing import Optional

import torch
from torch import nn

from .core import Spinor
from .transformations import generate_su2_rotation_from_axis_angle, rotate


class SpinorLinear(nn.Module):
    """
    Learnable complex linear map on the component axis: ``s ↦ W s + b``.

    Batch dimensions pass through untouched, so an input of shape
    ``[..., in_components]`` gives ``[..., out_components]``.
    """

    def __init__(
        self,
        in_components: int = 2,
        out_components: int = 2,
        bias: bool = False,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        if not dtype.is_complex:
            raise ValueError("SpinorLinear expects a complex dtype")
        self.in_components = in_components
        self.out_components = out_components
        # torch.randn on a complex dtype draws real/imag parts with variance 1/2
        # each, so this matches a Kaiming-style 1/fan_in variance overall.
        self.weight = nn.Parameter(
            torch.randn(out_components, in_components, dtype=dtype) / math.sqrt(in_components)
        )
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_components, dtype=dtype)) if bias else None
        )

    def forward(self, spinor: Spinor) -> Spinor:
        if spinor.dim != self.in_components:
            raise ValueError(
                f"Expected {self.in_components}-component spinor, got {spinor.dim}"
            )
        x = spinor.data.to(torch.promote_types(spinor.dtype, self.weight.dtype))
        y = torch.matmul(x, self.weight.mT)
        if self.bias is not None:
            y = y + self.bias
        return Spinor(y, dim=self.out_components)

    def extra_repr(self) -> str:
        return (
            f"in_components={self.in_components}, out_components={self.out_components}, "
            f"bias={self.bias is not None}"
        )


class SU2Rotation(nn.Module):
    """
    Learnable SU(2) rotation parameterised by a single axis-angle vector.

    The parameter is initialised near (but not at) the identity: the gradient
    of the axis-angle map vanishes at exactly zero, which would leave the
    rotation stuck. Pass ``init`` for a specific starting rotation.
    """

    def __init__(self, init: Optional[torch.Tensor] = None, init_scale: float = 0.1):
        super().__init__()
        if init is None:
            init = torch.randn(3) * init_scale
        init = torch.as_tensor(init, dtype=torch.get_default_dtype())
        if init.shape != (3,):
            raise ValueError(f"init must have shape (3,), got {tuple(init.shape)}")
        self.axis_angle = nn.Parameter(init.clone())

    @property
    def matrix(self) -> torch.Tensor:
        """The current ``2×2`` SU(2) matrix."""
        return generate_su2_rotation_from_axis_angle(self.axis_angle)

    def forward(self, spinor: Spinor) -> Spinor:
        return rotate(spinor, self.matrix)
