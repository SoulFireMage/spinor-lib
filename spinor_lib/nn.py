"""``torch.nn`` modules that act on spinors."""

import math
from typing import Callable, Optional

import torch
from torch import nn

from .core import Spinor
from .equivariant import invariant_features
from .so3 import quaternion_to_su2
from .transformations import generate_su2_rotation_from_axis_angle, rotate
from .utils import complex_dtype_of


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
    rotation stuck. Pass ``init`` for a specific starting rotation. See
    :class:`QuaternionRotation` for a parameterisation without that dead spot.
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


class QuaternionRotation(nn.Module):
    """
    Learnable SU(2) rotation parameterised by an unconstrained 4-vector that
    is normalised to a unit quaternion ``(w, x, y, z)`` on every forward pass.

    Unlike :class:`SU2Rotation` this map has no dead spot: the gradient is
    well defined everywhere except the origin, which the parameter never
    reaches in practice. Initialises at the identity ``(1, 0, 0, 0)``.
    """

    def __init__(self, init: Optional[torch.Tensor] = None):
        super().__init__()
        if init is None:
            init = torch.tensor([1.0, 0.0, 0.0, 0.0])
        init = torch.as_tensor(init, dtype=torch.get_default_dtype())
        if init.shape != (4,):
            raise ValueError(f"init must have shape (4,), got {tuple(init.shape)}")
        self.quaternion = nn.Parameter(init.clone())

    @property
    def matrix(self) -> torch.Tensor:
        """The current ``2×2`` SU(2) matrix."""
        return quaternion_to_su2(self.quaternion)

    def forward(self, spinor: Spinor) -> Spinor:
        return rotate(spinor, self.matrix)


# ------------------------------------------------------ equivariant layers
#
# These act on feature maps: spinors of shape ``[..., C, 2]`` (``C`` channels).
# The group acts on the last axis of every channel, so anything that only
# mixes channels or rescales them by an invariant commutes with it.


class EquivariantLinear(nn.Module):
    """
    Complex linear mixing of channels: ``ψ'_o = Σ_c W_oc ψ_c``.

    Commutes with any transformation of the component axis (SU(2), phases,
    even SL(2, C)), so it is exactly equivariant. There is deliberately no
    bias: adding a fixed spinor would break equivariance.
    """

    def __init__(self, in_channels: int, out_channels: int, dtype: torch.dtype = torch.complex64):
        super().__init__()
        if not dtype.is_complex:
            raise ValueError("EquivariantLinear expects a complex dtype")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, dtype=dtype) / math.sqrt(in_channels)
        )

    def forward(self, features: Spinor) -> Spinor:
        x = features.data
        if x.ndim < 2 or x.shape[-2] != self.in_channels:
            raise ValueError(
                f"Expected features of shape [..., {self.in_channels}, {features.dim}], got {tuple(x.shape)}"
            )
        dtype = torch.promote_types(complex_dtype_of(x.dtype), self.weight.dtype)
        y = torch.einsum("oc,...ci->...oi", self.weight.to(dtype), x.to(dtype))
        return Spinor(y, dim=features.dim)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


class NormGate(nn.Module):
    """
    Gated nonlinearity ``ψ_c ↦ act(a_c ‖ψ_c‖² + b_c) · ψ_c`` with learnable
    per-channel scale ``a`` and shift ``b``.

    The gate depends only on the invariant norm, so the layer is equivariant
    under SU(2) and global phases. ``act`` defaults to the sigmoid.
    """

    def __init__(self, channels: int, act: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid):
        super().__init__()
        self.channels = channels
        self.act = act
        self.scale = nn.Parameter(torch.ones(channels))
        self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, features: Spinor) -> Spinor:
        x = features.data
        if x.ndim < 2 or x.shape[-2] != self.channels:
            raise ValueError(
                f"Expected features of shape [..., {self.channels}, {features.dim}], got {tuple(x.shape)}"
            )
        norm_sq = (x.abs() ** 2).sum(-1)
        gate = self.act(self.scale * norm_sq + self.shift)
        return Spinor(x * gate.to(x.dtype)[..., None], dim=features.dim)

    def extra_repr(self) -> str:
        return f"channels={self.channels}"


class InvariantReadout(nn.Module):
    """
    Maps a feature map ``[..., C, N]`` to ``out_features`` real invariants.

    Builds the Gram matrix of the channels (``C²`` real numbers: norms, and
    the real and imaginary parts of the pairwise overlaps) and applies a real
    linear layer, or a two-layer MLP with SiLU when ``hidden`` is given. The
    output is invariant under SU(2) and global phases. Nonlinear functions of
    overlaps (distances, angles) need the hidden layer.
    """

    def __init__(self, channels: int, out_features: int, hidden: Optional[int] = None, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.out_features = out_features
        n_invariants = channels * channels
        if hidden is None:
            self.head = nn.Linear(n_invariants, out_features, bias=bias)
        else:
            self.head = nn.Sequential(
                nn.Linear(n_invariants, hidden), nn.SiLU(), nn.Linear(hidden, out_features, bias=bias)
            )

    def forward(self, features: Spinor) -> torch.Tensor:
        x = features.data
        if x.ndim < 2 or x.shape[-2] != self.channels:
            raise ValueError(
                f"Expected features of shape [..., {self.channels}, {features.dim}], got {tuple(x.shape)}"
            )
        dtype = next(self.head.parameters()).dtype
        return self.head(invariant_features(features).to(dtype))

    def extra_repr(self) -> str:
        return f"channels={self.channels}, out_features={self.out_features}"
