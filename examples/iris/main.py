import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch import nn, Tensor

from flow_matching import Path, ODEProcess, MidpointIntegrator, NaiveMidpoints
from flow_matching.utils import push_forward_all
from flow_matching.scheduler import OTScheduler
from flow_matching.distributions import GaussianMixture

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
    batch_size = 50

    # dataset
    x1, x2, x3 = get_iris(device=device)
    anchors = (0.0, 0.3, 0.6, 1.0)

    x0_sampler = GaussianMixture(
        n=16, shape=(in_dims,), sigma=0.5, r=1.0, device=device
    )

    # fm stuff
    vf = VectorField(in_d=in_dims, h_d=h_dims, t_d=1).to(device)
    p = Path(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        # x_init = x0_sampler.sample(batch_size)

        shuffle_idx = torch.randperm(batch_size)
        loss = push_forward_all(
            (x1[shuffle_idx], x2[shuffle_idx], x3[shuffle_idx], x1[shuffle_idx]),
            anchors,
            p,
            vf,
        )

        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # postprocessing
    vf.eval()
    integrator = ODEProcess(vf, MidpointIntegrator())
    seeker = NaiveMidpoints(max_evals=50, iters=5)
    steps = 10
    log_p0 = x0_sampler.log_likelihood
    interval = (anchors[0], anchors[-1])

    # plot logp
    t_steps = 50
    batch_size = 50
    t = torch.linspace(0.0, anchors[-1], steps=t_steps, device=device).repeat(
        batch_size
    )
    intervals = torch.zeros(
        (t_steps * batch_size, 2), dtype=torch.float32, device=device
    )
    intervals[:, 0] = t

    x = x2.repeat_interleave(t_steps, dim=0)
    _, probs = integrator.compute_likelihood(
        x, intervals, log_p0, steps=steps, est_steps=10
    )

    intervals = intervals.chunk(50)
    probs = probs.chunk(50)

    plt.gca().invert_xaxis()
    for prob, interval in zip(probs, intervals):
        plt.plot(interval[:, 0].cpu().numpy(), torch.exp(prob).cpu().numpy())
    plt.show()

    # classify
    # x = torch.cat(
    #     [x1[:5], x2[:5], x3[:5], torch.rand((5, 4), device=device) - 3], dim=0
    # )
    # min_t, prob = integrator.classify(
    #     seeker, x, log_p0, interval, steps=steps, est_steps=5, eps=1e-8
    # )
    # print(f"t_pred:\n{min_t}")
    # print(f"prob:\n{prob}")


if __name__ == "__main__":
    main()
