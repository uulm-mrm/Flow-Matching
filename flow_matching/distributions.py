import math

import torch
from torch import Tensor


def simplex_in_sphere(
    n: int, shape: tuple[int, ...], device: str = "cpu", dtype=torch.float32
) -> Tensor:
    """
    Places points on a regular simplex in prod(shape) dims that is within a sphere of radius r
    Requires n <= prod(shape) + 1
    All norms = r
    All pair-wise distances = r * sqrt(2n / n-1)
    Deterministic for same seed
    """
    dims = math.prod(shape)

    assert n <= dims + 1, "For simplex points n needs to be less than dims+1"

    # build centered simplex in R^dims
    identity = torch.eye(n, dtype=dtype, device=device)
    ones = torch.ones((n, n), dtype=dtype, device=device)

    # the vertices are (n, n) but in reality they live in n-1 dimensional space
    # so decompose to get the actual n-1 dimensional basis
    vert = identity - ones / n  # center simplex at origin

    # orthonormal basis are the points then using the qr decomposition
    q, _ = torch.linalg.qr(vert)  # pylint: disable=E1102
    vert = q[:, : (n - 1)]  # (n, n-1)

    if dims < (n - 1):
        vert = vert[:, :dims]
    elif dims > (n - 1):
        pad = torch.zeros((n, dims - (n - 1)), dtype=dtype, device=device)
        vert = torch.cat([vert, pad], dim=1)

    # normalize to radius r=1.0
    vert = vert / vert.norm(p=2.0, dim=1, keepdim=True)

    return vert.reshape((n, *shape))


class MultiIndependentNormal:
    """
    Multiple independent Isotropic Gaussians in n-D space
    """

    def __init__(self, c: int, shape: tuple[int, ...], k: float, device: str) -> None:
        self.c = c

        self.shape = shape
        self.dims = math.prod(shape)

        self.sigma = 1 / (k * self.dims**0.5)
        self.device = device

        # (c, shape)
        self.means = simplex_in_sphere(c, shape, device=device)

        self.__inv_var = self.sigma ** (-2)
        self.__inv_dims = 1.0 / self.dims

        self.__log_sigma = self.dims * math.log(self.sigma)
        self.__log_2pi = 0.5 * self.dims * math.log(2 * math.pi)

    def sample(self, n: int) -> Tensor:
        """Samples n points from each Gaussian for a total of (c, n, D...) points"""
        base = torch.randn(
            size=(n, *self.shape), dtype=self.means.dtype, device=self.means.device
        )

        return base.unsqueeze(0) * self.sigma + self.means.unsqueeze(1)

    def sample_arbitrary(self, *ns: int) -> Tensor:
        """Samples an arbitrary amount of points from each Gaussian in order of list ns

        Args:
            ns int: integers for how many samples per Gaussian

        Returns:
            Tensor: n samples for each centroid given n in ns in order of c
        """
        assert len(ns) == self.c, "A sample amount must exist for all classes"

        total = sum(ns)
        base = torch.randn(
            size=(total, *self.shape), dtype=self.means.dtype, device=self.means.device
        )

        means = torch.repeat_interleave(
            self.means, torch.tensor(ns, device=self.means.device), dim=0
        )

        return base * self.sigma + means

    def log_likelihood(self, x: Tensor) -> Tensor:
        """Calculates the log likelihood of x w.r.t all gaussians returning (n, c)"""
        diffs = x.unsqueeze(1) - self.means.unsqueeze(0)  # (n, c, D...)
        diffs_sq = diffs.square().flatten(2).sum(dim=2)  # (n, c)

        log_exp = 0.5 * diffs_sq * self.__inv_var

        return (-self.__log_2pi - self.__log_sigma - log_exp) * self.__inv_dims


def main():
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    c = 3
    shape = (1, 32, 32)
    # shape = (2,)
    k = 3

    mn = MultiIndependentNormal(c=c, shape=shape, k=k, device="cpu")

    t = mn.sample_arbitrary(1, 0, 2)
    print(t.shape)

    # samples = mn.sample(1000)
    # for c in samples:
    #     plt.scatter(c[:, 0], c[:, 1])
    # plt.show()

    for m in mn.sample(10):
        plt.hist(m.flatten().numpy(), bins=50, edgecolor="black")

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tensor Values")
    plt.show()


if __name__ == "__main__":
    main()
