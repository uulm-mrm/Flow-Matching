import math

from tqdm import tqdm

import torch
from torch import Tensor, nn

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

from flow_matching import AffinePath, ODEProcess, MidpointIntegrator
from flow_matching.scheduler import OTScheduler

DEVICE = "cuda:0"

torch.manual_seed(42)


class VectorField(nn.Module):
    def __init__(
        self, in_dims: int, h_dims: int, t_dims: int = 1, c_dims: int = 1
    ) -> None:
        super().__init__()

        self.in_dims = in_dims
        self.t_dims = t_dims

        self.mlp = nn.Sequential(
            nn.Linear(in_dims + t_dims + c_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, in_dims),
        )

    def forward(self, xt: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """Calculates the speed of each point x at t

        Args:
            xt (Tensor): x at point t in time, size (B, ...)
            t (Tensor): time, size (B | 1,) broadcasted to xt
            c (Tensor): context, size (B, C)

        Returns:
            Tensor: speed for each component of the input x
        """

        # the sampler iterates through time in tensors shaped ([])
        # so this broadcasts it to (B,1)
        # but if time is already in (B,) like for training then this just makes it a vector
        t = t.view(-1, self.t_dims).expand(xt.shape[0], self.t_dims)

        z = torch.cat([xt, t, c], dim=-1)

        return self.mlp(z)


def x1_sampler(samples: int) -> tuple[Tensor, Tensor]:
    x, y = make_moons(samples, noise=0.05)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)

    return x, y


def main():
    batch_size = 4096

    in_dims = 2
    h_dims = 512
    epochs = 1_000

    vf = VectorField(in_dims=in_dims, h_dims=h_dims, t_dims=1).to(DEVICE)
    p = AffinePath(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        x0 = torch.randn(batch_size, in_dims).to(DEVICE)
        x1, y = x1_sampler(batch_size)
        x1, y = x1.to(DEVICE), y.to(DEVICE)
        t = torch.rand((batch_size,), dtype=torch.float32, device=DEVICE)

        ps = p.sample(x0, x1, t)

        dxt_hat = vf.forward(ps.xt, ps.t, y)

        loss = (dxt_hat - ps.dxt).square().mean()

        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # integrate over time
    x0 = torch.randn((10_000, in_dims)).to(DEVICE)
    y = torch.randint(0, 2, (x0.shape[0], 1), device=DEVICE)
    intervals = torch.tensor([[0.0, 1.0]], dtype=x0.dtype, device=x0.device)
    intervals = intervals.expand(x0.shape[0], 2)
    steps = 10

    vf = vf.eval()
    integrator = ODEProcess(vf, MidpointIntegrator())
    _, x_traj = integrator.sample(x0, intervals, steps=steps, c=y)

    # plot path
    sols = x_traj.detach().cpu().numpy()
    y = y.detach().cpu().flatten().numpy()

    ax_cols = math.ceil(sols.shape[0] ** 0.5)
    ax_rows = math.ceil(sols.shape[0] / ax_cols)
    fig, axs = plt.subplots(ax_rows, ax_cols, figsize=(ax_cols * 4, ax_rows * 4))

    # yes you can flatten axes they are a np.array
    axs = axs.flatten()  # type: ignore
    for i, sol in enumerate(sols):
        axs[i].scatter(sol[:, 0], sol[:, 1], c=y, s=10)

        axs[i].set_title(f"step = {i}")
        axs[i].set_xlim([-3, 3])
        axs[i].set_ylim([-3, 3])
        axs[i].set_aspect("equal")

    for i in range(sols.shape[0], len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
