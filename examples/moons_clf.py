import math
from typing import Callable

from matplotlib import pyplot as plt
from matplotlib import cm

from tqdm import tqdm

import torch
from torch import Tensor, nn

from flow_matching.scheduler import OTScheduler
from flow_matching import Path, ODEProcess


class VectorField(nn.Module):
    def __init__(self, in_d: int, h_d: int, t_d: int) -> None:
        super().__init__()

        self.in_d = in_d
        self.t_d = t_d

        self.mlp = nn.Sequential(
            nn.Linear(in_d + t_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, in_d),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        t = t.view(-1, self.t_d).expand(xt.shape[0], self.t_d)

        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)


def xt_sampler(samples: int, bounds: tuple[float, float]) -> Tensor:
    """Samples a uniform distribution at given bounds"""
    x = torch.rand((samples, 2)) * (bounds[1] - bounds[0]) + bounds[0]

    return x


def push_forward(
    x0: Tensor,
    x1: Tensor,
    t_int: tuple[float, float],
    path: Path,
    vf_fwd: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    t0, t1 = t_int

    # s is local time for correct path interpolation
    s = torch.rand((x0.shape[0],)).float()

    # t is the map from local to global time for the model to predict
    t = s * (t1 - t0) + t0

    t = t.to(x0.device)
    s = s.to(x0.device)

    ps = path.sample(x0, x1, s)

    dxt_hat = vf_fwd(ps.xt, t)
    dxt = ps.dxt / (t1 - t0)

    return (dxt_hat - dxt).square().mean()


def main():
    torch.manual_seed(42)

    device = "cuda:0"

    batch_size = 4096
    x0_bounds = (0, 1)
    x1_bounds = (2, 3)

    in_dims = 2
    h_dims = 512
    epochs = 1_000

    vf = VectorField(in_d=in_dims, h_d=h_dims, t_d=1).to(device)
    p = Path(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        # sample shit here
        x0 = xt_sampler(batch_size, x0_bounds).to(device)
        x1 = xt_sampler(batch_size, x1_bounds).to(device)

        x_init = torch.randn_like(x1)

        loss1 = push_forward(x_init, x0, (0.0, 0.5), p, vf.forward)
        loss2 = push_forward(x0, x1, (0.5, 1.0), p, vf.forward)

        loss = loss1 + loss2

        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # integrate over time
    x_init = torch.randn((10_000, in_dims)).to(device)
    t = torch.linspace(0, 1, 11).to(device)

    vf = vf.eval()
    integrator = ODEProcess(vf)
    sols = integrator.sample(x_init, t, method="midpoint", step_size=0.05)

    # plot path
    sols = sols.detach().cpu().numpy()

    ax_cols = math.ceil(len(t) ** 0.5)
    ax_rows = math.ceil(len(t) / ax_cols)
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

        axs[i].set_title(f"t = {t[i]:.2f}")
        axs[i].set_xlim([-3, 3])
        axs[i].set_ylim([-3, 3])
        axs[i].set_aspect("equal")

    for i in range(len(t), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
