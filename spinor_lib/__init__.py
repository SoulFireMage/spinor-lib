"""Spinor library public API."""

from .core import Spinor
from .ops import (
    add,
    sub,
    mul,
    inner_product,
    outer_product,
    norm,
    normalize,
    conjugate,
)
from .transformations import (
    generate_su2_rotation,
    generate_su2_rotation_from_axis_angle,
    rotate,
    global_phase,
    contract,
)
from .nn import SpinorLinear, SU2Rotation
from .utils import get_backend

__version__ = "0.1.0"

__all__ = [
    "Spinor",
    "add",
    "sub",
    "mul",
    "inner_product",
    "outer_product",
    "norm",
    "normalize",
    "conjugate",
    "generate_su2_rotation",
    "generate_su2_rotation_from_axis_angle",
    "rotate",
    "global_phase",
    "contract",
    "SpinorLinear",
    "SU2Rotation",
    "get_backend",
]
