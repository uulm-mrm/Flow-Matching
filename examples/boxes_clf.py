import math

from matplotlib import pyplot as plt
from matplotlib import cm

from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from flow_matching import Path, ODEProcess, MidpointIntegrator, GoldenSectionSeeker
from flow_matching.utils import push_forward_all
from flow_matching.scheduler import OTScheduler


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
        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)


def xt_sampler(samples: int, bounds: tuple[float, float]) -> Tensor:
    """Samples a uniform distribution at given bounds"""
    x = torch.rand((samples, 2)) * (bounds[1] - bounds[0]) + bounds[0]

    return x


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

        loss = push_forward_all((x_init, x0, x1), (0.0, 1.0, 2.0), p, vf)

        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # integrate over time
    x_init = torch.randn((10_000, in_dims)).to(device)
    intervals = torch.tensor([[0.0, 2.0]], dtype=x_init.dtype, device=x_init.device)
    intervals = intervals.expand(x_init.shape[0], 2)
    steps = 10

    vf = vf.eval()
    integrator = ODEProcess(vf, MidpointIntegrator())
    _, x_traj = integrator.sample(x_init, intervals, steps=steps)

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

    # classification
    log_p0 = Independent(
        Normal(torch.zeros(2, device=device), torch.ones(2, device=device)), 1
    ).log_prob
    seeker = GoldenSectionSeeker(max_evals=20)
    interval = (0.0, 2.0)

    x0 = xt_sampler(samples=5, bounds=x0_bounds).to(device)
    min_t, min_p = integrator.classify(
        seeker, x0, log_p0, interval, steps=10, est_steps=1, eps=1e-8
    )
    print(f"Class t=0.5:\nt_pred: {min_t}\nlog_p: {min_p}")

    x1 = xt_sampler(samples=5, bounds=x1_bounds).to(device)
    min_t, min_p = integrator.classify(
        seeker, x1, log_p0, interval, steps=10, est_steps=1, eps=1e-8
    )
    print(f"Class t=1.0:\nt_pred: {min_t}\nlog_p: {min_p}")

    # OOD truly is at t=0
    ood = xt_sampler(samples=5, bounds=(-6.0, -5.0)).to(device)
    min_t, min_p = integrator.classify(
        seeker, ood, log_p0, interval, steps=10, est_steps=1, eps=1e-8
    )
    print(f"Class OOD:\nt_pred: {min_t}\nlog_p: {min_p}")


if __name__ == "__main__":
    main()
