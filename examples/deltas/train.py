import os

from tqdm import tqdm

import torch

from flow_matching import AffinePath
from flow_matching.scheduler import CosineScheduler

from examples.deltas import data
from examples.deltas.model import VectorField


def main():
    # consts
    torch.manual_seed(42)
    device = "cuda:0"

    # data setup
    pdata = data.get_pdata()

    # training consts
    epochs = 5_000
    batch_size = 1024

    # vector field setup
    vf = VectorField(in_dims=2, h_dims=128, t_dims=1).to(device)

    # path setup
    p = AffinePath(CosineScheduler())

    # torch stuff
    optim = torch.optim.AdamW(vf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        optim.zero_grad()

        # sample x target and x data
        x_target, x_data = data.sample_z(pdata, samples=batch_size)
        x_target = x_target.to(device)
        x_data = x_data.to(device)

        # sample time
        t = torch.rand((batch_size,), dtype=torch.float32, device=device)

        # sample path
        path_sample = p.sample(x_data, x_target, t)

        # predict speed
        dxt_hat = vf.forward(path_sample.xt, path_sample.t)

        # get MSE of speeds
        loss = (dxt_hat - path_sample.dxt).square().mean()

        # update net
        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # save net after training
    fname = os.path.join(os.path.dirname(__file__), "trained.pt")
    torch.save(vf.state_dict(), fname)


if __name__ == "__main__":
    main()
