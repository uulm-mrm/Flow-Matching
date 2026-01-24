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
    ds = IrisDataset(categories=(0, 1))
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

        pbar.set_description(f"Loss: {epoch_loss:.4f}")

    ema.to_model()
    vf = vf.eval()

    torch.save(vf.state_dict(), os.path.join(os.path.dirname(__file__), "trained.pt"))


if __name__ == "__main__":
    main()
