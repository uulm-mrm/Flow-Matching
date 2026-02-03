import os

import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    _x = torch.linspace(-10.0, 10.0, 100)
    _y = torch.linspace(-10.0, 10.0, 100)
    _x, _y = torch.meshgrid(_x, _y, indexing="ij")
    x = torch.stack([_x, _y], dim=-1)
    x = x.reshape(-1, 2).to(device)

    # set up time intervals for sampling
    t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
    t = t.expand(x.shape[0], 2)

    # integrate to get xt
    t_trajectory, x_trajectory = process.sample(x, t, steps=100)

    t = t_trajectory.detach().cpu().numpy()  # [t, samples, 1]
    xt = x_trajectory.detach().cpu().numpy()  # [t, samples, 2]

    T, _, _ = xt.shape
    grid_size = 100  # since you created 100x100 grid

    fig, ax = plt.subplots()

    # Precompute global axis limits so they don't jump during animation
    xmin = xt[..., 0].min()
    xmax = xt[..., 0].max()
    ymin = xt[..., 1].min()
    ymax = xt[..., 1].max()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    def update(frame):
        ax.clear()

        _x = xt[frame, :, 0].reshape(grid_size, grid_size)
        _y = xt[frame, :, 1].reshape(grid_size, grid_size)

        # horizontal lines
        ax.plot(_x, _y, linewidth=0.5, c="black")
        # vertical lines
        ax.plot(_x.T, _y.T, linewidth=0.5, c="black")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"t index = {frame}")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=200, blit=False)  # type: ignore
    writer = animation.PillowWriter(fps=60)
    ani.save("flow_evolution.gif", writer=writer)
    plt.show()


if __name__ == "__main__":
    main()
