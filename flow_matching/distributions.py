import math

import torch
from torch import Tensor


def simplex_in_sphere(
    n: int, shape: tuple[int, ...], r: float, device: str = "cpu", dtype=torch.float32
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

    # scale to radius r
    vert *= r

    return vert.reshape((n, *shape))


class MultiIndependentNormal:
    """
    Multiple independent Isotropic Gaussians in n-D space
    """

    def __init__(
        self, n: int, shape: tuple[int, ...], r: float, var_coef: float, device: str
    ) -> None:
        self.n = n

        self.shape = shape
        self.dims = math.prod(shape)

        self.device = device

        # (n, shape)
        self.means = simplex_in_sphere(n, shape, r=r, device=device)
        self.var_coef = torch.tensor(var_coef, device=self.device)

    def sample(self, *points: int) -> Tensor:
        """Samples p points from each Gaussian once, in order of arguments

        Args:
            points int: integers for how many samples per Gaussian

        Returns:
            Tensor: (sum(*points), D...) samples
        """

        assert len(points) == self.n, "A sample amount must exist for all classes"

        total = sum(points)
        base = torch.randn(
            size=(total, *self.shape), dtype=self.means.dtype, device=self.means.device
        )

        means = torch.repeat_interleave(
            self.means, torch.tensor(points, device=self.means.device), dim=0
        )

        return base * self.var_coef.sqrt() + means

    def log_likelihood(self, x: Tensor) -> Tensor:
        """Calculates -1/2 * 1/c * ||x-means||^2 w.r.t each Gaussian

        Args:
            x (Tensor): input tensor size (B, D...)

        Returns:
            Tensor: Energies for each of the Gaussians (B, n)
        """

        diffs = x.unsqueeze(1) - self.means.unsqueeze(0)  # (B, n, D...)

        diffs_sq = diffs.flatten(2).square().sum(dim=2)  # (B, n)

        # as dims -> inf, diffs -> var * dims
        # so divide by dims as well
        return -0.5 * diffs_sq / self.var_coef

    def get_scores(self, x: Tensor) -> Tensor:
        """Calculates the score of each input point w.r.t all Gaussians using:
        S_i = 1/var_coef * (<x, m_i> - 1/2 ||m_i|| ^2); where <x, m_i> is the dot product

        A direction compatibility test more-less

        Args:
            x (Tensor): input tensor size (B, D...)

        Returns:
            Tensor: Scores for each of the Gaussians (B, n)
        """

        # flatten for dot product
        x_flat = x.view(x.shape[0], -1)  # (B, D...)
        means_flat = self.means.view(self.means.shape[0], -1)  # (n, D...)

        # <x, m_i> for all m_i in means
        x_dot_m = torch.matmul(x_flat, means_flat.t())  # (B, n)

        # 1/2 ||m_i||^2
        mean_norms = 0.5 * torch.norm(means_flat, dim=1).square()

        return (x_dot_m - mean_norms) / self.var_coef

    def get_square_distances(self, x: Tensor, scores: Tensor | None = None) -> Tensor:
        """Based on points x, returns the distance from each of the Gaussian's means


        Args:
            x (Tensor): input tensor siize (B, D...)
            scores (Tensor | None, optional): scores if previously calculated size (B, n).
                To save time on calculating the distances since they're part of the score.
                Defaults to None.

        Returns:
            Tensor: Distances from each Gaussian's mean (the numerator of the Gaussian)
        """

        x_flat = x.view(x.shape[0], -1)  # (B, D...)
        x_norms_sq = torch.norm(x_flat, dim=1).square()  # (B,)

        if scores is not None:
            return x_norms_sq.unsqueeze(1) - (2 * self.var_coef * scores)

        means_flat = self.means.view(self.means.shape[0], -1)
        mean_norms_sq = torch.norm(means_flat, dim=1).square()

        return (
            x_norms_sq.unsqueeze(1)
            + mean_norms_sq.unsqueeze(1)
            - 2 * torch.dot(x_flat, means_flat.t())
        )

    def get_credability(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Based on points x calculates the belief and ucertainty for each.
        The class of the point is the argmax of the belief vector for that point.
        If uncertainty is above 0.5 then the point should be understood as OOD,
        but it still holds information for the closest Gaussian.

        Args:
            x (Tensor): input points size (B, D...)

        Returns:
            tuple[Tensor, Tensor]: (B, n) and (B,) tensors for belief in each of the
                Gaussians, and overall uncertainty of the placement of the point
        """

        # score and dist for belief and uncertainty
        scores = self.get_scores(x)  # (B, n)
        square_dists = self.get_square_distances(x, scores)  # (B, n)

        # distance -> quality
        mean_dist = self.dims * self.var_coef
        delta = (square_dists - mean_dist) / mean_dist

        alpha = 1.0  # tune
        quality = torch.exp(-alpha * torch.clamp(delta, min=0.0))

        # evidence of a point belonging to Gaussian N_i is then defined as
        # the score weighted by the "quality" of the point
        # normalize scores bcs exp is used on them
        scores = scores - scores.max(dim=1, keepdim=True).values
        evidence = torch.exp(scores) * quality  # (B, n)

        # belief and uncertainty are then calculated using Dirichelt-Based Credal Set theory
        # b = e_i / W + sum e_j; u = W / W + sum e_j
        # where W is the prior strenght, most usually the number of elements in the Credal Set
        denom = self.n + evidence.sum(dim=1, keepdim=True)  # (B, n)
        beliefs = evidence / denom  # (B, n)
        uncertainty = self.n / denom  # (B, 1)

        return beliefs, uncertainty

    def check_outlier(
        self, x: Tensor, scores: Tensor, sigma_threshold: int = 3
    ) -> Tensor:
        """Based on points x and scores for those points, checks whether they are outliers
        Does this by calculating the distance, and checking how many standard deviations
        it is from the mean with the best score. If it's above threshold deviations then it's OOD

        A distance compatibility test more-less

        Args:
            x (Tensor): input points to check whether they are outliers of size (B, D...)
            scores (Tensor): scores for those points of size (B, n)

        Returns:
            Tensor: a mask of size (B) with True if outlier
        """

        # get distances from best scores
        best_score_idx = torch.argmax(scores, dim=1)
        diffs_sq = self.get_square_distances(x, scores)[
            torch.arange(best_score_idx.shape[0]), best_score_idx
        ]

        # calculate threshold
        # for a Gaussian, ||x - m||^2 / var follows Chi-Squared
        # which for d -> inf approaches N(m=D*var, sigma^2=2D*var^2)
        # so OOD is distances which are > threshold * sigma of this N

        # mean of the above mentioned distance N distro
        expected_dist = self.dims * self.var_coef
        dist_sigma = (  # sigma of the above mentioned distance N distro
            torch.tensor(2.0 * self.dims, device=self.device).sqrt() * self.var_coef
        )

        # outlier mask
        return torch.abs(diffs_sq - expected_dist) > (sigma_threshold * dist_sigma)


def main():
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    n = 3
    # shape = (256, 7, 7)
    shape = (2,)
    r = 3.0
    var = 1.0

    mn = MultiIndependentNormal(n=n, shape=shape, r=r, var_coef=var, device="cpu")

    t = mn.sample(1, 0, 2)
    print(t.shape)

    for i, c in enumerate([100, 50, 20]):
        slist = [0] * n
        slist[i] = c

        samples = mn.sample(*slist)

        plt.scatter(samples[:, 0], samples[:, 1])

    plt.scatter(mn.means[:, 0], mn.means[:, 1], c="black")
    plt.show()


if __name__ == "__main__":
    main()
