import os
import math

import torch

import matplotlib.pyplot as plt

from flow_matching import ODEProcess, RungeKuttaIntegrator
from flow_matching.integrator_utils import RK4_TABLEAU

from examples.mixture.model import VectorField


def main():
    # consts
    torch.manual_seed(42)
    device = "cuda:0"

    # load model
    fname = os.path.join(os.path.dirname(__file__), "..", "trained.pt")
    vf = VectorField(in_dims=2, h_dims=128, t_dims=1).to(device)
    vf.load_state_dict(torch.load(fname))
    vf.eval()

    # ODE process setup
    process = ODEProcess(
        vf, integrator=RungeKuttaIntegrator(tableu=RK4_TABLEAU, device=device)
    )

    # get samples to integrate
    _x = torch.linspace(-10.0, 10.0, 50)
    _y = torch.linspace(-10.0, 10.0, 50)
    _x, _y = torch.meshgrid(_x, _y, indexing="ij")
    x = torch.stack([_x, _y], dim=-1)
    x = x.reshape(-1, 2).to(device)

    # set up time intervals for sampling
    t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
    t = t.expand(x.shape[0], 2)

    # make color for plot
    y = torch.zeros(x.shape[0])
    # class 1 mask
    mask_c1 = (x[:, 0] <= 8.0) & (x[:, 0] >= 4.0) & (x[:, 1] <= 7.0) & (x[:, 1] >= 5.0)
    mask_c2 = (
        (x[:, 0] >= -8.0) & (x[:, 0] <= -4.0) & (x[:, 1] >= -7.0) & (x[:, 1] <= -5.0)
    )
    y[mask_c1] = 1
    y[mask_c2] = 2

    y = y.detach().cpu().numpy()

    # integrate to get xt
    _, x_trajectory = process.sample(x, t, steps=10)
    xt = x_trajectory.detach().cpu().numpy()

    # plot sampling
    ax_cols = math.ceil(xt.shape[0] ** 0.5)
    ax_rows = math.ceil(xt.shape[0] / ax_cols)
    fig, axs = plt.subplots(ax_rows, ax_cols, figsize=(ax_cols * 4, ax_rows * 4))

    axs = axs.flatten()  # type: ignore
    for i, _xt in enumerate(xt):
        axs[i].scatter(_xt[:, 0], _xt[:, 1], c=y)

        axs[i].set_title(f"step = {i}")
        axs[i].set_xlim([-10, 10])
        axs[i].set_ylim([-10, 10])
        axs[i].set_aspect("equal")

    for i in range(xt.shape[0], len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
