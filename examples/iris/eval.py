import os

import torch

from flow_matching import PotentialProcess, RungeKuttaIntegrator, tableaus

from examples.iris.data import IrisDataset, WineDataset
from examples.iris.net import PotentialField
from examples.iris.utils import cosine_similarity, norm_decay


def main():
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # consts
    device = "cuda:0"

    in_dims = 4
    h_dims = 256

    # load model
    pf = PotentialField(in_dims, h_dims, t_d=1).to(device)
    pf.load_state_dict(
        torch.load(os.path.join(os.path.dirname(__file__), "trained.pt"))
    )
    pf = pf.eval()

    # dataset
    id_ds = IrisDataset()
    ood_ds = WineDataset()

    # process setup
    proc = PotentialProcess(
        pf, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device)
    )

    x_id = id_ds.x.to(device)
    x_ood = ood_ds.x.to(device)
    x_init = torch.cat([x_id, x_ood], dim=0)

    # calculate energies of points at t=1
    t = torch.zeros((x_init.shape[0], 1), device=device, dtype=torch.float32)
    energies = pf.forward(x_init, t)
    print("ID Energies:\n", energies[: x_id.shape[0]].chunk(len(id_ds.categories)))
    print("OOD Energies:\n", energies[x_id.shape[0] :].chunk(len(ood_ds.categories)))

    # solve process
    intervals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)
    intervals = intervals.expand(x_init.shape[0], 2)

    _, x_traj = proc.sample(x_init, intervals, steps=100)
    sols = x_traj[-1]
    print("ID Solutions:\n", sols[: x_id.shape[0]].chunk(len(id_ds.categories)))
    print("OOD Solutions:\n", sols[x_id.shape[0] :].chunk(len(ood_ds.categories)))

    # metrics
    deltas = torch.zeros((3, in_dims), dtype=torch.float32, device=device)
    deltas[:, :3] = torch.eye(3, dtype=torch.float32, device=device)
    sims = cosine_similarity(sols, deltas)
    print("ID Similarities:\n", sims[: x_id.shape[0]].chunk(len(id_ds.categories)))
    print("OOD Similarities:\n", sims[x_id.shape[0] :].chunk(len(ood_ds.categories)))

    quality = norm_decay(sols)
    print("ID Quality:\n", quality[: x_id.shape[0]].chunk(len(id_ds.categories)))
    print("OOD Quality:\n", quality[x_id.shape[0] :].chunk(len(ood_ds.categories)))


if __name__ == "__main__":
    main()
