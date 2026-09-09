# spinor-lib

Two-component (or N-component) spinors on top of PyTorch: Hermitian inner
products, SU(2) rotations, global phases, and `torch.nn` modules. Everything
broadcasts over batch dimensions and preserves autograd.

```
pip install -e ".[test]"
pytest
```

## Interactive three.js demos

```
python demos/server.py        # then open http://localhost:8765
```

Five browser scenes driven live by the library through a tiny stdlib HTTP
server: the Bloch sphere and the 720° double cover, Dirac's belt trick,
relativistic aberration of the night sky, two-qubit entanglement, and a
`QuaternionRotation` learning a hidden rotation one gradient step at a time.
See `demos/README.md`.

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

## Observables and the Bloch sphere

`spinor_lib.observables` covers spin-1/2 physics on top of the core ops:

```python
from spinor_lib import (pauli, expectation, bloch_vector, from_bloch_vector,
                        density_matrix, purity, fidelity, fubini_study_distance, slerp)

sigma = pauli(dtype=torch.complex128)                      # [3, 2, 2]: σx, σy, σz
sz = expectation(s, sigma[2])                              # ⟨s|σz|s⟩, shape [32], real

n = bloch_vector(s)                                        # [32, 3] on the unit sphere
s2 = from_bloch_vector([1, 0, 0])                          # |+x⟩
rho = density_matrix(s)                                    # [32, 2, 2], Tr ρ = 1
purity(rho)                                                # 1 for pure states

fidelity(s, s2)                                            # |⟨s|s2⟩|² (also works on 2×2 density matrices)
fubini_study_distance(s, s2)                               # arccos |⟨s|s2⟩|, phase-invariant
path = slerp(s, s2, torch.linspace(0, 1, 10))              # geodesic in projective space
```

## SU(2), SO(3) and quaternions

`spinor_lib.so3` implements the double cover. Rotating a spinor by `R`
rotates its Bloch vector by `su2_to_so3(R)`:

```python
from spinor_lib import su2_to_so3, so3_to_su2, su2_to_quaternion, quaternion_to_su2, slerp_su2

M = su2_to_so3(R)                                          # [..., 3, 3], real orthogonal
R_again = so3_to_su2(M)                                    # ±R (one of the two preimages)
q = su2_to_quaternion(R)                                   # (w, x, y, z), unit length
R = quaternion_to_su2(torch.randn(4))                      # any 4-vector → SU(2), differentiable
R_mid = slerp_su2(R0, R1, 0.5)                             # shortest-arc interpolation
```

`quaternion_to_su2` on an unconstrained 4-vector is a smooth parameterisation
of SU(2) with no dead spot at the identity, so it makes a good learnable
rotation.

## Composite (multi-spinor) states

`spinor_lib.composite` handles tensor-product states. Subsystem 0 is the most
significant index, matching `torch.kron`:

```python
from spinor_lib import (tensor_product, embed_operator, kron, partial_trace,
                        reduced_density_matrix, entanglement_entropy, concurrence)

up = Spinor(torch.tensor([1, 0], dtype=torch.complex128))
psi = tensor_product(up, up)                               # 4-component |00⟩
psi = rotate(psi, embed_operator(H, dims=(2, 2), target=0))  # apply a 2×2 gate to qubit 0
psi = rotate(psi, CNOT)                                    # any [4, 4] operator

rho_0 = reduced_density_matrix(psi, dims=(2, 2), keep=0)   # trace out qubit 1
entanglement_entropy(psi, (2, 2), keep=0, base=2)          # 1 bit for a Bell state
concurrence(psi)                                           # 1 for a Bell state, 0 for products
```

`partial_trace` works for any number of subsystems of any sizes and keeps the
selected ones in their original order.

## Lorentz transformations: Weyl and Dirac spinors

`spinor_lib.lorentz` extends the group from SU(2) to SL(2,C). A boost is
just another 2×2 matrix applied with `rotate`, but it is not unitary and it
depends on the spinor's handedness (`A_L = (A_R†)⁻¹`; rotations are shared):

```python
from spinor_lib import (generate_boost, generate_lorentz, sl2c_to_lorentz, weyl_current,
                        rapidity_from_velocity, gamma_matrices, generate_dirac,
                        dirac_from_weyl, dirac_bilinear, dirac_current)

eta = rapidity_from_velocity(0.6)                          # β = 0.6 → η = atanh β
A_L = generate_boost([0, 0, 1], eta, handedness="left")   # exp(−η σz / 2), shape [2, 2]
A = generate_lorentz(rotation=[0.3, 0, 0], boost=[0, 0, eta])  # general element, via matrix_exp
Lam = sl2c_to_lorentz(A_L, "left")                         # the 4×4 Lorentz matrix
j = weyl_current(s, "left")                                # ψ†σ̄ᵘψ, a null 4-vector; j → Λ j

psi = dirac_from_weyl(s_left, s_right)                     # 4-component, chiral basis
S = generate_dirac(rotation=theta, boost=eta_vec)          # diag(A_L, A_R); S⁻¹γᵘS = Λᵘ_ν γ^ν
dirac_bilinear(psi, psi)                                   # ψ̄ψ, a Lorentz scalar
dirac_current(psi)                                         # ψ̄γᵘψ, a Lorentz vector
gamma_matrices()                                           # [4, 4, 4], {γᵘ, γ^ν} = 2 ηᵘ^ν
```

Conventions and the defining property `vector_to_matrix(Λx) = A X A†` are
documented at the top of `spinor_lib/lorentz.py`.

## Equivariant layers

`spinor_lib.equivariant` and `spinor_lib.nn` treat a spinor of shape
`[..., C, 2]` as `C` channels of features that all rotate together. Channel
mixing and norm-based gating commute with the group action, and the
Clebsch–Gordan products build invariants and vectors from pairs of channels:

```python
from torch import nn
from spinor_lib import (EquivariantLinear, NormGate, InvariantReadout, QuaternionRotation,
                        singlet, vector, apply_vector, invariant_features)

net = nn.Sequential(
    EquivariantLinear(4, 16),        # ψ'_o = Σ_c W_oc ψ_c (complex, no bias)
    NormGate(16),                    # ψ_c ↦ sigmoid(a‖ψ_c‖² + b) ψ_c
    EquivariantLinear(16, 16),
    NormGate(16),
)
head = InvariantReadout(16, 1, hidden=64)  # Gram-matrix invariants → MLP → scalar

# net(rotate(x, R)) == rotate(net(x), R), and head(net(rotate(x, R))) == head(net(x))

singlet(a, b)                  # a₀b₁ − a₁b₀: invariant under SU(2) and SL(2,C)
vector(a, b)                   # ⟨a|σ|b⟩: rotates as a 3-vector
apply_vector(v, s)             # (v·σ) s: vector ⊗ spinor → spinor

rot = QuaternionRotation()     # learnable SU(2) via a normalised 4-vector, no dead spot
```

`examples/learn_rotation.py` recovers a hidden rotation from before/after
pairs, and `examples/equivariant_regression.py` trains an equivariant
network and a plain MLP on the same invariant target and shows that only the
former is unchanged when the test set is rotated.

`SpinorLinear` and `SU2Rotation` in `spinor_lib.nn` are drop-in modules for
training; `SU2Rotation` learns an axis-angle vector (prefer `QuaternionRotation`).

Conventions: `inner_product(a, b) = a†·b`, `outer_product(a, b) = a ⊗ b†`,
`generate_su2_rotation(n, θ) = exp(−i θ/2 n·σ)`. See `SPEC.md` for the
original design brief.
