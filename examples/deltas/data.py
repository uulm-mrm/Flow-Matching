import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt


def get_pdata() -> tuple[MultivariateNormal, MultivariateNormal]:
    mean1 = torch.tensor([3.0, 0.0])
    mean2 = torch.tensor([0.0, 3.0])

    # large variance in x  # small variance in y
    cov = torch.tensor([[0.3, 0.0], [0.0, 0.3]])

    return MultivariateNormal(mean1, cov), MultivariateNormal(mean2, cov)


def sample_z(
    pdata: tuple[MultivariateNormal, MultivariateNormal], samples: int
) -> tuple[Tensor, Tensor]:

    assert samples % 2 == 0, "Samples should be divisible by 2"

    per_class_samples = samples // 2

    xtarget, xdata = [], []
    for i, _pdata in enumerate(pdata):
        _xtarget = torch.zeros((per_class_samples, 2), dtype=torch.float32)
        _xtarget[:, i] = 1

        xtarget.append(_xtarget)
        xdata.append(_pdata.sample((per_class_samples,)))

    return torch.cat(xtarget, dim=0), torch.cat(xdata, dim=0)


def main():
    p1 = get_pdata()

    _, x1 = sample_z(p1, 1000)

    plt.scatter(x1[:, 0], x1[:, 1])
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
