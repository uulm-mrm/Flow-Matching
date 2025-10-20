from pprint import pprint

from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.distributions import Independent, Normal

from flow_matching import Path, ODEProcess, MidpointIntegrator, GoldenSectionSeeker
from flow_matching.utils import push_forward_all
from flow_matching.scheduler import OTScheduler

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

    in_dims = 4
    h_dims = 256
    epochs = 1_000

    # dataset
    x1, x2, x3 = get_iris(device=device)

    # fm stuff
    vf = VectorField(in_d=in_dims, h_d=h_dims, t_d=1).to(device)
    p = Path(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        x_init = torch.randn_like(x1)

        loss = push_forward_all((x_init, x1, x2, x3), (0.0, 0.33, 0.66, 1.0), p, vf)

        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # classification
    vf.eval()
    integrator = ODEProcess(vf, MidpointIntegrator())
    seeker = GoldenSectionSeeker(max_evals=10)
    steps = 10

    log_p0 = Independent(
        Normal(torch.zeros(4, device=device), torch.ones(4, device=device)), 1
    ).log_prob

    min_t, min_p = integrator.classify(
        seeker, x1[:5], log_p0, steps=steps, est_steps=1, eps=1e-4
    )
    pprint(f"Class 0 @ 0.33:\nt_pred: {min_t}\nlog_p: {min_p}")

    min_t, min_p = integrator.classify(
        seeker, x2[:5], log_p0, steps=steps, est_steps=1, eps=1e-4
    )
    pprint(f"Class 1 @ 0.66:\nt_pred: {min_t}\nlog_p: {min_p}")

    min_t, min_p = integrator.classify(
        seeker, x3[:5], log_p0, steps=steps, est_steps=1, eps=1e-4
    )
    pprint(f"Class 2 @ 1.0:\nt_pred: {min_t}\nlog_p: {min_p}")

    min_t, min_p = integrator.classify(
        seeker,
        torch.rand((5, 4), device=device) - 3,
        log_p0,
        steps=steps,
        est_steps=1,
        eps=1e-4,
    )
    pprint(f"Class OOD:\nt_pred: {min_t}\nlog_p: {min_p}")


if __name__ == "__main__":
    main()
