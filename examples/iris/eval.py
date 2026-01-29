import os

import torch

from flow_matching import ODEProcess, RungeKuttaIntegrator, tableaus

from examples.iris.data import IrisDataset
from examples.iris.net import VectorField
from examples.iris.utils import cosine_similarity, norm_decay, credal_measures


def main():
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # consts
    device = "cpu"

    in_dims = 4
    h_dims = 512

    # load model
    vf = VectorField(in_dims, h_dims, t_d=1).to(device)
    vf.load_state_dict(
        torch.load(os.path.join(os.path.dirname(__file__), "trained.pt"))
    )

    # dataset
    ds = IrisDataset()

    # process setup
    proc = ODEProcess(vf, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device))

    ood = torch.randn((50, 4), dtype=torch.float32, device=device) + 10.0
    x_init = torch.cat([ds.x.to(device), ood], dim=0)

    intervals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)
    intervals = intervals.expand(x_init.shape[0], 2)

    # solve process
    _, x_traj = proc.sample(x_init, intervals, steps=100)
    sols = x_traj[-1]
    print("Solutions: ", sols.chunk(4))

    return

    # metrics
    deltas = torch.zeros((3, in_dims), dtype=torch.float32, device=device)
    deltas[:, :3] = torch.eye(3, dtype=torch.float32, device=device)
    sims = cosine_similarity(sols, deltas)
    print("Similarities: ", sims.chunk(4))

    quality = norm_decay(sols)
    print("Quality: ", quality.chunk(4))

    measure = (sims + 1) * 0.5  # rescale similarities to [0, 1]
    belief, vacuity = credal_measures(measure, quality, W=1.0)
    print("Belief: ", belief.chunk(4))
    print("Vacuity: ", vacuity.chunk(4))


if __name__ == "__main__":
    main()
