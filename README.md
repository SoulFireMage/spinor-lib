# spinor-lib

Two-component (or N-component) spinors on top of PyTorch: Hermitian inner
products, SU(2) rotations, global phases, and `torch.nn` modules. Everything
broadcasts over batch dimensions and preserves autograd.

```
pip install -e ".[test]"
pytest
```

## Usage

```python
import torch
from spinor_lib import Spinor, inner_product, generate_su2_rotation, rotate

s = Spinor(torch.randn(32, 2, dtype=torch.complex64))   # batch of 32 spinors

overlap = inner_product(s, s)                              # shape [32], s†·s
R = generate_su2_rotation(axis=[0, 1, 0], angle=torch.pi / 4)
s_rot = rotate(s, R)                                       # dtypes are promoted

loss = (s_rot.norm() ** 2).sum()
loss.backward()
```

Rotation generators are batched — an axis of shape `[..., 3]` with an angle of
shape `[...]` gives matrices of shape `[..., 2, 2]`:

```python
angles = torch.linspace(0, torch.pi, 32)
R = generate_su2_rotation([0, 0, 1], angles)              # [32, 2, 2]
s_rot = rotate(s, R)                                       # one rotation per spinor
```

`SpinorLinear` and `SU2Rotation` in `spinor_lib.nn` are drop-in modules for
training; `SU2Rotation` learns an axis-angle vector.

Conventions: `inner_product(a, b) = a†·b`, `outer_product(a, b) = a ⊗ b†`,
`generate_su2_rotation(n, θ) = exp(−i θ/2 n·σ)`. See `SPEC.md` for the
original design brief.
