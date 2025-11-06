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
    """Returns n points on a sphere equidistant one from another"""
    dims = math.prod(shape)

    # random init (n, dims)
    x = torch.randn(n, dims, dtype=torch.float32, requires_grad=True)  # type: ignore

    with torch.no_grad():
        x.data = r * x.data / x.data.norm(p=2.0, dim=-1, keepdim=True)

    optim = torch.optim.Adam([x], lr=lr)
    eps = 1e-6

    anull = torch.eye(n, dtype=torch.float32) * 1e9

    for _ in tqdm(range(steps), desc="Finding Equidistant Points"):
        optim.zero_grad()

        dists = torch.cdist(x, x, p=2)  # (n, n)
        dists = dists + anull

        inv_dists = dists ** (-p)

        energy = 0.5 * inv_dists.sum()

        energy.backward()
        optim.step()

        with torch.no_grad():
            x.data = r * x.data / x.data.norm(dim=-1, keepdim=True).clamp_min(eps)

    return x.detach().reshape((n, *shape)).float().to(device)


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
            self.means = equidistant_on_sphere(
                n, shape, r, steps=5_000, lr=1e-3, p=2, device=device
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
        self.means = (
            equidistant_on_sphere(
                c, shape, r, steps=10_000, lr=1e-1, p=2, device=device
            )
            if c > 1
            else torch.zeros((1, *self.shape), dtype=torch.float32, device=device)
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
