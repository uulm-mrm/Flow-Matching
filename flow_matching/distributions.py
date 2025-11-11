import math

from tqdm import tqdm

import torch
from torch import Tensor


def equidistant_on_sphere(
    n: int,
    shape: tuple[int, ...],
    r: float,
    steps: int = 5000,
    lr: float = 1e-3,
    p: int = 2,
    device: str = "cpu",
) -> Tensor:
    """
    Returns n points on a sphere equidistant one from another.
    For dims=prod(shape) >> 1 it will produce a trivial solution:
    points = +- 1 / sqrt(dims) for all vector elements
    """
    dims = math.prod(shape)

    # random init (n, dims)
    x = torch.randn(n, dims, dtype=torch.float32, requires_grad=True)  # type: ignore

    # reproject to r=1.0
    with torch.no_grad():
        x.data = x.data / x.data.norm(p=p, dim=-1, keepdim=True)

    optim = torch.optim.Adam([x], lr=lr)
    eps = 1e-6

    anull = torch.eye(n, dtype=torch.float32) * 1e9

    for _ in tqdm(range(steps), desc="Finding Equidistant Points"):
        optim.zero_grad()

        dists = torch.cdist(x, x, p=p)  # (n, n)
        dists = dists + anull

        inv_dists = dists ** (-p)

        energy = 0.5 * inv_dists.sum()

        energy.backward()
        optim.step()

        # reproject to 1.0 so that points don't escape
        with torch.no_grad():
            x.data = x.data / x.data.norm(p=p, dim=-1, keepdim=True).clamp_min(eps)

    # after optimizing on r=1 just rescale to wanted r
    return (r * x).detach().reshape((n, *shape)).float().to(device)


def simplex_in_sphere(
    n: int, shape: tuple[int, ...], r: float, device: str = "cpu", dtype=torch.float32
) -> Tensor:
    """
    Places points on a regular simplex in prod(shape) dims that is within a sphere of radius r
    Requires n <= prod(shape) + 1
    All norms = r
    All pair-wise distances = r * sqrt(2n / n-1)
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

    # randomly rotate to avoid sparse vectors (might be slow for large dims)
    # random rotation = orthonormal basis of gaussian
    a = torch.randn(dims, dims, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(a)  # pylint: disable=E1102
    vert = vert @ q.T

    # normalize to radius r
    vert = vert / vert.norm(p=2.0, dim=1, keepdim=True)
    vert = r * vert

    return vert.reshape((n, *shape))


class GaussianMixture:
    """
    Makes n Gaussian mixtures in an arbitrary dimension space

    Each Gaussian's mean is sampled from a sphere surface
    equidistant from other Gaussians.
    """

    def __init__(
        self,
        n: int,
        shape: tuple[int, ...],
        sigma: float,
        r: float,
        device: str,
    ) -> None:
        """
        Args:
            n (int): number of gaussians
            shape (tuple[int, ...]): actual shape of points that are sampled
            sigma (float): deviation of gaussians
            r (float): radius of the sphere centered around 0
            device (str): device on which to port
        """
        self.device = device

        self.n = torch.tensor(n, device=self.device)
        self.shape = shape
        self.dims = torch.prod(torch.tensor(shape, device=self.device))

        self.sigma = torch.tensor(sigma, device=self.device)
        self.r = torch.tensor(r, device=self.device)

        if n > 1:
            # make simplex if possible, otherwise optimize and hope for the best
            self.means = (
                simplex_in_sphere(n, shape, r, device)
                if n <= self.dims + 1
                else equidistant_on_sphere(
                    n, shape, r, steps=5_000, lr=1e-3, p=2, device=device
                )
            )
        else:
            self.means = torch.zeros((n, *shape), dtype=torch.float32, device=device)

    def sample(self, samples: int) -> Tensor:
        """
        Sample from the Gaussian mixture


        Starts with a N(0, I) Gaussian, then for each point sampled from it,
        chooses a mean that it will be centered around
        """
        noise = torch.randn((samples, *self.shape), device=self.device)  # type: ignore

        # evenly distribute across means. safer than randint for lower sample sizes
        uniform = samples // self.n
        leftover = samples % self.n

        uniform_idx = torch.arange(self.n, device=self.device).repeat(uniform)  # type: ignore
        leftover_idx = torch.randint(
            low=0, high=self.n, size=(leftover,), device=self.device  # type: ignore
        )

        comp_idx = torch.cat([uniform_idx, leftover_idx])

        sampled_points = self.means[comp_idx] + self.sigma * noise
        return sampled_points.reshape(samples, *self.shape)

    def log_likelihood(self, x: Tensor) -> Tensor:
        """
        Calculates the log likelihood of x given this Gaussian mixture
        """
        x = x.flatten(1)
        means = self.means.flatten(1)

        diffs = x[:, None] - means[None]

        sq_dists = (diffs**2).sum(dim=2)

        # log of gaussian w/o weight
        log_coeff = -0.5 * self.dims * torch.log(2 * torch.pi * self.sigma**2)
        log_probs = log_coeff - 0.5 * sq_dists / (self.sigma**2)

        # log( (1/n) * sum_i exp(log_probs_i) )
        return torch.logsumexp(log_probs, dim=1) - torch.log(self.n)


class MultiIndependentNormal:
    """
    Multiple independent Isotropic Gaussians in n-D space
    """

    def __init__(
        self, c: int, shape: tuple[int, ...], r: float, sigma: float, device: str
    ) -> None:
        self.c = c

        self.shape = shape
        self.dims = math.prod(shape)

        self.r = r
        self.sigma = sigma
        self.device = device

        # (c, shape)
        if c <= self.dims + 1:
            self.means = simplex_in_sphere(c, shape, r, device=device)

        else:
            self.means = equidistant_on_sphere(
                c, shape, r, steps=10_000, lr=1e-1, p=2, device=device
            )

        self.__inv_var = sigma ** (-2)

        self.__log_sigma = self.dims * math.log(self.sigma)
        self.__log_2pi = 0.5 * self.dims * math.log(2 * math.pi)

    def sample(self, n: int) -> Tensor:
        """Samples n points from each Gaussian for a total of (c, n, D...) points"""
        base = torch.randn(
            size=(n, *self.shape), dtype=self.means.dtype, device=self.means.device
        )

        return base.unsqueeze(0) * self.sigma + self.means.unsqueeze(1)

    def log_likelihood(self, x: Tensor) -> Tensor:
        """Calculates the log likelihood of x w.r.t all gaussians returning (n, c)"""
        diffs = x.unsqueeze(1) - self.means.unsqueeze(0)  # (n, c, D...)
        diffs_sq = diffs.square().flatten(2).sum(dim=2)  # (n, c)

        log_exp = 0.5 * diffs_sq * self.__inv_var

        return -self.__log_2pi - self.__log_sigma - log_exp


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    torch.manual_seed(42)

    c = 3
    dims = 2
    k = 3

    sigma = 1.0
    r = k * sigma * (dims) ** 0.5

    mn = MultiIndependentNormal(c=c, shape=(2,), r=r, sigma=sigma, device="cpu")
    print(mn.means)
    samples = mn.sample(1000)

    plt.gca().add_patch(Circle((0, 0), radius=r, fill=False, color="r"))

    for s in samples:
        plt.scatter(s[:, 0], s[:, 1])

    plt.xlim(-3 * r, 3 * r)
    plt.ylim(-3 * r, 3 * r)
    plt.show()


if __name__ == "__main__":
    main()
