"""Recover an unknown SU(2) rotation from before/after pairs of spinors.

Two learnable parameterisations are compared on the same data: the axis-angle
``SU2Rotation`` and the unit-quaternion ``QuaternionRotation``. Both should
converge, but the quaternion form has no dead spot at the identity.

Run with:  python examples/learn_rotation.py
"""

import math

import torch

from spinor_lib import (
    QuaternionRotation,
    Spinor,
    SU2Rotation,
    fidelity,
    generate_su2_rotation,
    rotate,
    su2_to_so3,
    su2_to_quaternion,
)


def train(model, x, y, steps=300, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = (1 - fidelity(model(x), y)).mean()  # 1 − |⟨y|R x⟩|², phase-invariant
        loss.backward()
        opt.step()
    return loss.item()


def main():
    torch.manual_seed(0)

    # The hidden rotation and a batch of noisy observations.
    axis, angle = torch.randn(3), torch.tensor(2.1)
    target = generate_su2_rotation(axis, angle)
    x = Spinor(torch.randn(512, 2, dtype=torch.complex64))
    y = rotate(x, target)
    y = Spinor(y.data + 0.02 * torch.randn_like(y.data))

    print(f"hidden rotation: axis={axis / axis.norm()}, angle={angle.item():.3f} rad")
    print(f"hidden quaternion: {su2_to_quaternion(target)}\n")

    for name, model in [("SU2Rotation (axis-angle)", SU2Rotation()),
                        ("QuaternionRotation", QuaternionRotation())]:
        loss = train(model, x, y)
        R = model.matrix.detach()
        so3_err = (su2_to_so3(R) - su2_to_so3(target)).abs().max().item()
        print(f"{name}")
        print(f"  final loss        {loss:.2e}")
        print(f"  learned quaternion {su2_to_quaternion(R)}   (overall sign is the same rotation)")
        print(f"  max SO(3) error     {so3_err:.2e}\n")


if __name__ == "__main__":
    main()
