from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from flow_matching import Path, Integrator
from flow_matching.scheduler import OTScheduler

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
    epochs = 10_000

    # data prep
    ds = get_mnist("train")
    ds = sample_mnist(ds, classes)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # model prep
    vf = CNNVF(t_dims=t_dims).to(device)
    path = Path(OTScheduler())
    optim = torch.optim.AdamW(vf.parameters(), lr=lr)

    # train the vf
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x1, _ in dl:
            optim.zero_grad()

            # add noise to x1, because we sample new x0 noise each step
            # this will effectively mean the images are a bit blurry overall

            # the other way would be to have a fixed noise pool from the getgo
            # or have all the x0 points converge to x1 points
            # but the cardinality of x0 and x1 is then way off
            x1 = x1.to(device)
            x1_noise = torch.randn_like(x1) * torch.tensor(0.05**0.5, device=device)
            x1 += x1_noise

            # sample x0
            x0 = torch.randn_like(x1)

            # sample time
            t = torch.rand((batch_size,), dtype=torch.float32, device=device)

            # path, speed and loss
            path_sample = path.sample(x0, x1, t)

            dxt_hat = vf.forward(path_sample.xt, t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            epoch_loss += loss

        pbar.set_description(f"Loss: {(epoch_loss / len(dl)):.3f}")

    vf = vf.eval()

    # generate a few samples
    step_size = 0
    imgs = 16

    x0 = torch.randn((imgs, 1, 28, 28), device=device)
    # we're interested in the end product not the path so no anchors
    t = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)

    integrator = Integrator(vf)
    sols = integrator.sample(x0, t, method="midpoint", step_size=step_size)
    fake_imgs = sols[-1]  # take the samples at t=1

    fake_imgs = fake_imgs.cpu().detach().numpy()
    _, axs = plt.subplots(4, 4)

    axs = axs.flatten()
    for i in range(imgs):
        axs[i].imshow(fake_imgs[i, 0], "gray")
        axs[i].set_aspect("equal")

    plt.savefig("./examples/mnist/imgs.png")


if __name__ == "__main__":
    main()
