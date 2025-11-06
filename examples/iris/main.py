from tqdm import tqdm

import torch
from torch import nn, Tensor

from flow_matching import (
    AffineMultiPath,
    AffinePath,
    ODEProcess,
    RungeKuttaIntegrator,
    tableaus,
)
from flow_matching.scheduler import CosineScheduler
from flow_matching.distributions import MultiIndependentNormal
from modules.utils import EMA

from examples.iris.data_utils import get_iris


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
            nn.Linear(h_d, in_d),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)


def main():
    torch.manual_seed(42)

    # consts
    device = "cuda:0"

    num_class = 3
    in_dims = 4
    h_dims = 512
    epochs = 5_000
    batch_size = 50

    sigma = 1.0
    r = 5.0

    # x0 sampler
    multi_normal = MultiIndependentNormal(
        c=num_class, shape=(in_dims,), r=r, sigma=sigma, device=device
    )

    # dataset
    x1, x2, x3 = get_iris(device=device)
    x = torch.stack([x1, x2, x3], dim=0)

    # fm stuff
    vf = VectorField(in_dims, h_dims, t_d=1).to(device)
    ema = EMA(vf, rate=0.999)

    path = AffineMultiPath(AffinePath(CosineScheduler()), num_paths=num_class)

    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        x0 = multi_normal.sample(batch_size)
        t = torch.rand((batch_size,), dtype=torch.float32, device=device)

        path_sample = path.sample(x0, x, t)
        dxt_hat = vf.forward(path_sample.xt, path_sample.t)

        loss = (dxt_hat - path_sample.dxt).square().mean()

        loss.backward()
        optim.step()

        ema.update_ema_t()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    ema.to_model()
    vf = vf.eval()

    proc = ODEProcess(vf, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device))
    x_init = torch.cat([x1, x2, x3, torch.rand_like(x1)], dim=0)
    intervals = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device).expand(
        x_init.shape[0], 2
    )

    _, x_traj = proc.sample(x_init, intervals, steps=100)
    sols = x_traj[-1]
    probs = multi_normal.log_likelihood(sols)

    for i, c in enumerate(probs.argmax(dim=1).chunk(4)[:-1]):
        print((c == i).sum())


if __name__ == "__main__":
    main()
