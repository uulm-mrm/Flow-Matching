from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from flow_matching import AffinePath, ODEProcess, RungeKuttaIntegrator, tableaus
from flow_matching.scheduler import CosineScheduler
from modules.utils import EMA

from examples.iris.data_utils import IrisDataset
from examples.iris.clf_utils import cosine_similarity, norm_decay, credal_measures


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
    torch.set_printoptions(precision=4, sci_mode=False)

    # consts
    device = "cuda:0"

    in_dims = 4
    h_dims = 512

    epochs = 5_000
    batch_size = 150

    # dataset
    ds = IrisDataset()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # fm stuff
    vf = VectorField(in_dims, h_dims, t_d=1).to(device)
    ema = EMA(vf, rate=0.999)

    path = AffinePath(CosineScheduler())

    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x, y in dl:
            optim.zero_grad()
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)

            t = torch.rand((batch_size,), dtype=torch.float32, device=device)

            path_sample = path.sample(y, x, t)
            dxt_hat = vf.forward(path_sample.xt, t.unsqueeze(1))

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss += loss.item()

        pbar.set_description(f"Loss: {epoch_loss:.3f}")

    ema.to_model()
    vf = vf.eval()

    proc = ODEProcess(vf, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device))

    ood = torch.randn((50, 4), dtype=torch.float32, device=device) + 10.0
    x_init = torch.cat([ds.x.to(device), ood], dim=0)

    intervals = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
    intervals = intervals.expand(x_init.shape[0], 2)

    _, x_traj = proc.sample(x_init, intervals, steps=100)
    sols = x_traj[-1]
    print("Solutions: ", sols.chunk(4))

    deltas = torch.zeros((3, in_dims), dtype=torch.float32, device=device)
    deltas[:, :3] = torch.eye(3, dtype=torch.float32, device=device)
    sims = cosine_similarity(sols, deltas)
    print("Similarities: ", sims.chunk(4))

    quality = norm_decay(sols)
    print("Quality: ", quality.chunk(4))

    # rescale similarities to [0, 1]
    measure = (sims + 1) * 0.5
    belief, vacuity = credal_measures(measure, quality, W=3.0)
    print("Belief: ", belief.chunk(4))
    print("Vacuity: ", vacuity.chunk(4))


if __name__ == "__main__":
    main()
