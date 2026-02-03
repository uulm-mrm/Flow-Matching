# pylint: disable=E1102

import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import functional as F

from flow_matching import AffinePath
from flow_matching.scheduler import CosineScheduler
from modules.utils import EMA

from examples.iris.net import PotentialField
from examples.iris.data import IrisDataset
from examples.iris.utils import gradient


# the ordinary loss for the learned vector field
def get_data_loss(x: Tensor, y: Tensor, pf: PotentialField, path: AffinePath) -> Tensor:
    # sample t
    t = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device)

    # sample path for xt
    path_sample = path.sample(x, y, t)
    xt = path_sample.xt.detach().requires_grad_(True)

    # get potential
    potential = pf.forward(xt, t.unsqueeze(1))

    # get speed as the negative gradient of the potential
    dxt_hat = -gradient(potential.sum(), xt, create_graph=True)

    # get difference between speeds
    return (dxt_hat - path_sample.dxt).square().mean()


# cos sim + grad norm
# grad norm to push them outward slowly and not mega quickly
# and also for it to be easily overridden by data loss
def get_noise_loss(
    x: Tensor, y: Tensor, pf: PotentialField, path: AffinePath
) -> Tensor:
    # sample x noise from U[-10, 10]
    x_noise = torch.empty_like(x).uniform_(-10.0, 10.0)

    # sample time for it
    t = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device)

    # sample noise path
    noise_path_sample = path.sample(x_noise, y, t)
    xt_noise = noise_path_sample.xt.detach().requires_grad_(True)

    # get velocity of noise
    potential = pf.forward(xt_noise, t.unsqueeze(1))
    dxt_noise = -gradient(potential.sum(), xt_noise, create_graph=True)

    grad_norm = dxt_noise.norm(2, dim=-1)

    # grad loss is E[(||Fi|| - 1)^2] try to minimize this
    grad_loss = (grad_norm - 1.0).square().mean()

    # try to also minimize cos sim for noise, so it flows "away" from target
    # since cossim is [-1, 1] minimizing it means "flow away"
    cos_sim = F.cosine_similarity(dxt_noise, noise_path_sample.dxt, dim=-1).mean()

    # return the total loss of noise
    grad_lambda = 0.1
    return cos_sim + grad_lambda * grad_loss


def main():
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # consts
    device = "cuda:0"

    in_dims = 4
    h_dims = 256

    epochs = 5_000
    batch_size = 50

    # dataset
    ds = IrisDataset(categories=(0, 1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # fm stuff
    pf = PotentialField(in_dims, h_dims, t_d=1).to(device)
    ema = EMA(pf, rate=0.999)

    path = AffinePath(CosineScheduler())

    optim = torch.optim.AdamW(pf.parameters(), lr=1e-3)

    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x, y in dl:
            optim.zero_grad()

            # sample x, y
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)

            # track these losses over time and see which does what
            loss_data = get_data_loss(x, y, pf, path)
            loss_noise = get_noise_loss(x, y, pf, path)

            loss = loss_data + loss_noise

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss += loss.item()

        pbar.set_description(f"Loss: {epoch_loss:.4f}")

    ema.to_model()
    pf = pf.eval()

    torch.save(pf.state_dict(), os.path.join(os.path.dirname(__file__), "trained.pt"))


if __name__ == "__main__":
    main()
