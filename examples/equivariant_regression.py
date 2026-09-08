"""Equivariance in practice: an SU(2)-equivariant network vs a plain MLP.

Each sample is a set of 4 spinors (4 channels). The target is a rotation
invariant of the set: the Fubini-Study distance between channels 0 and 1
plus the fidelity between channels 2 and 3. Both networks are trained on
unrotated data and then evaluated on the *same* data after a random SU(2)
rotation of every sample. The equivariant model's predictions do not change;
the MLP's do.

Run with:  python examples/equivariant_regression.py
"""

import torch
from torch import nn

from spinor_lib import (
    EquivariantLinear,
    InvariantReadout,
    NormGate,
    Spinor,
    fidelity,
    fubini_study_distance,
    generate_su2_rotation,
    rotate,
)

CHANNELS, HIDDEN = 4, 16


def make_data(n, generator):
    x = Spinor(torch.randn(n, CHANNELS, 2, dtype=torch.complex128, generator=generator))
    target = fubini_study_distance(x[:, 0], x[:, 1]) + fidelity(x[:, 2], x[:, 3])
    return x, target[:, None]


def random_rotations(n, generator):
    axis = torch.randn(n, 1, 3, dtype=torch.float64, generator=generator)
    angle = torch.rand(n, 1, dtype=torch.float64, generator=generator) * 6.283
    return generate_su2_rotation(axis, angle)  # one rotation per sample, shared by all channels


class EquivariantNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            EquivariantLinear(CHANNELS, HIDDEN, dtype=torch.complex128),
            NormGate(HIDDEN),
            EquivariantLinear(HIDDEN, HIDDEN, dtype=torch.complex128),
            NormGate(HIDDEN),
        )
        self.head = InvariantReadout(HIDDEN, 1, hidden=64)

    def forward(self, x):
        return self.head(self.body(x))


class PlainMLP(nn.Module):
    """Sees the raw real/imaginary components, so it has no idea about rotations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CHANNELS * 4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        flat = torch.view_as_real(x.data).flatten(1)
        return self.net(flat)


def train(model, x, y, steps, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def main():
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    x_train, y_train = make_data(2048, g)
    x_test, y_test = make_data(512, g)
    x_test_rot = rotate(x_test, random_rotations(512, g))

    for name, model in [("equivariant", EquivariantNet().double()), ("plain MLP", PlainMLP().double())]:
        train_loss = train(model, x_train, y_train, steps=600)
        with torch.no_grad():
            pred, pred_rot = model(x_test), model(x_test_rot)
            test_loss = nn.functional.mse_loss(pred, y_test).item()
            rot_loss = nn.functional.mse_loss(pred_rot, y_test).item()
            drift = (pred - pred_rot).abs().max().item()
        print(f"{name:12s} train MSE {train_loss:.4f} | test MSE {test_loss:.4f} | "
              f"rotated-test MSE {rot_loss:.4f} | max prediction change under rotation {drift:.2e}")


if __name__ == "__main__":
    main()
