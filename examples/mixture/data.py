import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt


def get_pdata() -> tuple[MultivariateNormal, MultivariateNormal]:
    mean1 = torch.tensor([6.0, 6.0])
    mean2 = torch.tensor([-6.0, -6.0])

    # large variance in x  # small variance in y
    cov = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    return MultivariateNormal(mean1, cov), MultivariateNormal(mean2, cov)


def get_ptarget() -> tuple[MultivariateNormal, MultivariateNormal]:
    # simplex for 2 points is a line, scaled to radius=2
    mean1 = torch.tensor([2.0, 0.0])
    mean2 = torch.tensor([-2.0, 0.0])

    # isotropic but smaller variance
    cov = torch.tensor([[0.2, 0.0], [0.0, 0.2]])

    return MultivariateNormal(mean1, cov), MultivariateNormal(mean2, cov)


def sample_z(
    ptarget: tuple[MultivariateNormal, MultivariateNormal],
    pdata: tuple[MultivariateNormal, MultivariateNormal],
    samples: int,
) -> tuple[Tensor, Tensor]:

    assert samples % 2 == 0, "Samples should be divisible by 2"

    per_class_samples = samples // 2

    xtarget, xdata = [], []
    for _ptarget, _pdata in zip(ptarget, pdata):
        xtarget.append(_ptarget.sample((per_class_samples,)))
        xdata.append(_pdata.sample((per_class_samples,)))

    return torch.cat(xtarget, dim=0), torch.cat(xdata, dim=0)


def main():
    p0 = get_ptarget()
    p1 = get_pdata()

    x0, x1 = sample_z(p0, p1, 1000)

    plt.scatter(x0[:, 0], x0[:, 1])
    plt.axis("equal")
    plt.show()

    plt.scatter(x1[:, 0], x1[:, 1])
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
