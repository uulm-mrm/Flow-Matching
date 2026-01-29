import os

import torch

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np

from flow_matching import ODEProcess, RungeKuttaIntegrator
from flow_matching.integrator_utils import RK4_TABLEAU

from examples.deltas.model import VectorField


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

    _x = torch.linspace(-5.0, 5.0, 100)
    _y = torch.linspace(-5.0, 5.0, 100)
    _x, _y = torch.meshgrid(_x, _y, indexing="ij")
    x = torch.stack([_x, _y], dim=-1)
    x = x.reshape(-1, 2).to(device)

    # set up time intervals for sampling
    t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
    t = t.expand(x.shape[0], 2)

    # integrate to get xt
    t_trajectory, x_trajectory = process.sample(x, t, steps=10)

    t = t_trajectory.detach().cpu().numpy()  # [t, samples, 1]
    t = t[:, 0].flatten()  # [t,]

    xt = x_trajectory.detach().cpu().numpy()  # [t, samples, 2]

    plt.figure(figsize=(5, 5))

    for i in range(xt.shape[1]):
        x = xt[:, i, 0]
        y = xt[:, i, 1]

        points = np.stack([x, y], axis=1)
        segments = np.stack([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,  # type: ignore
            cmap="viridis",
            array=t,
            linewidth=0.5,
            alpha=0.2,
        )
        plt.gca().add_collection(lc)

    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.title("Flow trajectories over time")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
