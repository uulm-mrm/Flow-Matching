import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import Tensor

from flow_matching import AffinePath
from flow_matching.scheduler import CosineScheduler
from modules.utils import EMA

from examples.iris.net import VectorField
from examples.iris.data import IrisDataset

"""
Maybe the vectorfield around the place of all input data is quite large and pointing towards the only dirac
somehow try to make the space super small around the known data, see if you can control it somehow
Anything that isn't in the known classes should flow outwards, not inward

If you were to plot it, there would be like a white hole around where the data is
followed by lines going towards the dirac deltas in a corridor
followed by a black hole around the dirac deltas near the end of time

So try to make this corridor as small as possible, and try to make the holes as small as possible
only for the things from ID to flow correctly, rest are in a repulsive field!!!
"""

"""
New goal:
1. sample t
2. sample x, y
3. sample path for xt
4. NEW STEP: network -> energy
5. NEW STEP: grad of energy = predicted velocity
6. (dxt - vt)**2 / n
"""


def main():
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # consts
    device = "cpu"

    in_dims = 4
    h_dims = 512

    epochs = 5_000
    batch_size = 50

    # dataset
    ds = IrisDataset(categories=(0,))
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

            path_sample = path.sample(x, y, t)
            dxt_hat = vf.forward(path_sample.xt, t.unsqueeze(1))

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss += loss.item()

        pbar.set_description(f"Loss: {epoch_loss:.4f}")

    ema.to_model()
    vf = vf.eval()

    torch.save(vf.state_dict(), os.path.join(os.path.dirname(__file__), "trained.pt"))


if __name__ == "__main__":
    main()
