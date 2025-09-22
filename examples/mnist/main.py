from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from flow_matching import Path, Integrator
from flow_matching.scheduler import OTScheduler

from modules.utils import EMA

from examples.mnist.data_utils import get_mnist, sample_mnist
from examples.mnist.model import CNNVF

torch.manual_seed(42)


def main():
    # consts
    device = "cuda:0"

    classes = [3]
    batch_size = 512

    t_dims = 128
    lr = 1e-3
    epochs = 1_000

    # data prep
    ds = get_mnist("train")
    ds = sample_mnist(ds, classes)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # model prep
    vf = CNNVF(t_dims=t_dims).to(device)
    ema = EMA(vf, rate=0.999).to(device)

    path = Path(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=lr)

    # train the vf
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x1, _ in dl:
            optim.zero_grad()

            x1 = x1.to(device)

            # sample x0
            x0 = torch.randn_like(x1)

            # sample time
            t = torch.rand((x1.shape[0],), dtype=torch.float32, device=device)

            # path, speed and loss
            path_sample = path.sample(x0, x1, t)

            dxt_hat = vf.forward(path_sample.xt, t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            # update ema after optim
            ema.update_ema_t()

            epoch_loss += loss

        pbar.set_description(f"Loss: {(epoch_loss / len(dl)):.3f}")

    # after training send ema params to model
    ema.to_model()
    vf = vf.eval()

    # generate a few samples
    step_size = 0
    imgs = 1024
    top_k = 16

    x0 = torch.randn((imgs, 1, 28, 28), device=device)
    # we're interested in the end product not the path so no anchors
    t = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)

    integrator = Integrator(vf)
    sols = integrator.sample(x0, t, method="midpoint", step_size=step_size)
    fake_imgs = sols[-1]  # take the samples at t=1

    # find 16 images with the lowest losses compared to the first digit
    real_img: Tensor = ds[0][0].to(device).unsqueeze(0)

    # calculate mse vs real image, take the indices from torch.min and sample by them
    errors = (fake_imgs - real_img).square().sum(dim=(1, 2, 3)).div(28 * 28)
    sorted_errors = list(sorted(range(imgs), key=lambda idx: errors[idx]))
    sorted_errors = sorted_errors[:top_k]

    # take lowest top_k
    fake_imgs = fake_imgs[sorted_errors]

    fake_imgs = fake_imgs.cpu().detach().numpy()
    _, axs = plt.subplots(4, 4)

    axs = axs.flatten()
    for i in range(top_k):
        axs[i].imshow(fake_imgs[i, 0], "gray")
        axs[i].set_aspect("equal")

    plt.savefig("./examples/mnist/top_k_imgs.png")


if __name__ == "__main__":
    main()
