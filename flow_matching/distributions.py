from tqdm import tqdm

import torch
from torch import Tensor


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
            self.means = self.__find_means(steps=5_000, lr=0.1, p=2)
        else:
            self.means = torch.zeros((n, *shape), dtype=torch.float32, device=device)

    def __random_on_sphere(self) -> Tensor:
        x = torch.randn(self.n, self.dims, device=self.device)  # type: ignore

        return self.r * x / x.norm(dim=1, keepdim=True)

    def __find_means(self, steps: int, lr: float, p: int = 2) -> Tensor:
        means = self.__random_on_sphere()  # (n, dims)

        for it in tqdm(range(steps), desc="Finding Gaussian Mixture Centers"):
            # calculate m_i - m_j for all m in means
            diffs = means[:, None] - means[None]  # (n, n, dims)
            dists = diffs.norm(dim=2) + 1e-12  # (n, n)

            inv_dists = 1.0 / (dists ** (p + 1))  # (n, n)
            inv_dists.fill_diagonal_(0.0)  # anull self dist

            # calculate force for each m in means
            repulsion = (inv_dists[..., None] * diffs).sum(dim=1)  # (n, dims)

            # update means
            means = means + lr * repulsion

            # reproject means
            means = self.r * means / means.norm(dim=1, keepdim=True)

            # step learn rate decay every 10th of steps
            lr = lr * 0.97 if it % (steps // 10) == 0 else lr

        return means.reshape((self.n, *self.shape))

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
