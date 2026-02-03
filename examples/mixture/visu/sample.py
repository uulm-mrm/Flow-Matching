import os

import torch

import matplotlib.pyplot as plt

from flow_matching import ODEProcess, RungeKuttaIntegrator
from flow_matching.integrator_utils import RK4_TABLEAU

from examples.mixture.model import VectorField
from examples.mixture import data


def main():
    # consts
    torch.manual_seed(42)
    device = "cuda:0"

    # load model
    fname = os.path.join(os.path.dirname(__file__), "..", "trained.pt")
    vf = VectorField(in_dims=2, h_dims=128, t_dims=1).to(device)
    vf.load_state_dict(torch.load(fname))
    vf.eval()

    # ODE process setup
    process = ODEProcess(
        vf, integrator=RungeKuttaIntegrator(tableu=RK4_TABLEAU, device=device)
    )

    # sample pdata
    pdata = data.get_pdata()
    ptarget = data.get_ptarget()

    _, x = data.sample_z(ptarget, pdata, samples=1000)
    x = x.to(device)

    # make time intervals for integration
    t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
    t = t.expand(x.shape[0], 2)

    # sample trajectoriy
    t_traj, x_traj = process.sample(x, t, steps=10)
    t_traj = t_traj.detach().cpu().numpy()
    x_traj = x_traj.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    for i in range(x_traj.shape[1]):
        plt.scatter(x_traj[:, i, 0], x_traj[:, i, 1], c=t_traj[:, i], alpha=0.3)

    plt.colorbar(label="Time")  # type: ignore
    plt.axis("equal")
    plt.title("Trajectories over time")
    plt.savefig("mixture_sample.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
