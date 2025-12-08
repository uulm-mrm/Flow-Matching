"""
Learns to generate moons distribution
"""

import math

from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_moons

from flow_matching import ODEProcess, MidpointIntegrator, AffinePath
from flow_matching.scheduler import OTScheduler

DEVICE = "cuda:0"

torch.manual_seed(42)


class VectorField(nn.Module):
    def __init__(self, in_dims: int, h_dims: int, t_dims: int = 1) -> None:
        super().__init__()

        self.in_dims = in_dims
        self.t_dims = t_dims

        self.mlp = nn.Sequential(
            nn.Linear(in_dims + t_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, in_dims),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        """Calculates the speed of each point x at t

        Args:
            xt (Tensor): x at point t in time, size (B, ...)
            t (Tensor): time, size (B | 1,) broadcasted to xt

        Returns:
            Tensor: speed for each component of the input x
        """
        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)


def x1_sampler(samples: int) -> Tensor:
    x, _ = make_moons(samples, noise=0.05)

    x = torch.from_numpy(x).float()

    return x


def main():
    batch_size = 1024

    in_dims = 2
    h_dims = 512
    epochs = 1_000

    vf = VectorField(in_dims=in_dims, h_dims=h_dims, t_dims=1).to(DEVICE)
    p = AffinePath(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        x0 = torch.randn(batch_size, in_dims).to(DEVICE)
        x1 = x1_sampler(batch_size).to(DEVICE)
        t = torch.rand((batch_size,), dtype=torch.float32, device=DEVICE)

        ps = p.sample(x0, x1, t)

        dxt_hat = vf.forward(ps.xt, ps.t)

        loss = (dxt_hat - ps.dxt).square().mean()

        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # integrate over time
    x0 = torch.randn((10_000, in_dims)).to(DEVICE)
    intervals = torch.tensor([[0.0, 1.0]], dtype=x0.dtype, device=x0.device)
    intervals = intervals.expand(x0.shape[0], 2)
    steps = 10

    vf = vf.eval()
    integrator = ODEProcess(vf, MidpointIntegrator())
    _, x_traj = integrator.sample(x0, intervals, steps=steps)

    # plot path
    sols = x_traj.detach().cpu().numpy()

    ax_cols = math.ceil(sols.shape[0] ** 0.5)
    ax_rows = math.ceil(sols.shape[0] / ax_cols)
    fig, axs = plt.subplots(ax_rows, ax_cols, figsize=(ax_cols * 4, ax_rows * 4))

    # yes you can flatten axes they are a np.array
    axs = axs.flatten()  # type: ignore
    for i, sol in enumerate(sols):
        H = axs[i].hist2d(sol[:, 0], sol[:, 1], 300, range=((-3, 3), (-3, 3)))

        cmin = 0.0
        cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()

        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)  # type: ignore

        _ = axs[i].hist2d(
            sol[:, 0], sol[:, 1], 300, range=((-3, 3), (-3, 3)), norm=norm
        )

        axs[i].set_title(f"step = {i}")
        axs[i].set_xlim([-3, 3])
        axs[i].set_ylim([-3, 3])
        axs[i].set_aspect("equal")

    for i in range(sols.shape[0], len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

    # visualize likelihood
    t = torch.tensor([[1.0, 0.0]], device=DEVICE)

    grid_size = 200
    grid_axis = torch.linspace(-3.0, 3.0, grid_size)

    x1 = torch.meshgrid(grid_axis, grid_axis, indexing="ij")
    x1 = torch.stack([x1[0].flatten(), x1[1].flatten()], dim=1).to(DEVICE)
    t = t.expand(x1.shape[0], 2)

    log_p0 = Independent(
        Normal(torch.zeros(2, device=DEVICE), torch.ones(2, device=DEVICE)), 1
    ).log_prob

    _, log_p1 = integrator.compute_likelihood(x1, t, log_p0, steps=10)

    log_p1 = torch.exp(log_p1).reshape(grid_size, grid_size)
    log_p1 = log_p1.detach().cpu().numpy()
    norm = cm.colors.Normalize(vmax=1.0, vmin=0.0)  # type: ignore
    plt.imshow(
        log_p1, extent=(-3.0, 3.0, -3.0, 3.0), cmap="viridis", norm=norm, origin="lower"
    )
    plt.show()


if __name__ == "__main__":
    main()
