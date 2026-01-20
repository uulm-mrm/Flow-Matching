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

        base = torch.zeros(
            size=(total, self.dims),
            dtype=self.means.dtype,
            device=self.means.device,
        )
        base[:, : self.n - 1] = torch.randn(
            size=(total, self.n - 1),
            dtype=self.means.dtype,
            device=self.means.device,
        )
        base = base.reshape(total, *self.shape)

        means = torch.repeat_interleave(
            self.means, torch.tensor(points, device=self.means.device), dim=0
        )

        return base * self.var_coef.sqrt() + means

    def get_square_distances(self, x: Tensor) -> Tensor:
        # if you train with sampling like above then this is what the distance needs to be
        # only up to the n-1 dimension of features
        x_flat = x.view(x.shape[0], self.dims)[:, : self.n - 1]
        means_flat = self.means.view(self.n, self.dims)[:, : self.n - 1]

        return torch.cdist(x_flat, means_flat).square()

    def get_credal_metrics(self, x: Tensor) -> dict[str, Tensor]:
        dist_sq = self.get_square_distances(x)

        dist_rbf = torch.exp(-dist_sq / (2.0 * self.var_coef))

        w = 1.0  # TODO: either self.n or 1.0 or 1/self.n?
        credal_denom = dist_rbf.sum(dim=-1) + w

        belief = dist_rbf / credal_denom.unsqueeze(1)
        vacuity = w / credal_denom

        return {"belief": belief, "vacuity": vacuity}


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(1)

    n = 3
    shape = (4,)
    r = 1.0
    var = 1 / 3.0

    mn = MultiIndependentNormal(n=n, shape=shape, r=r, var_coef=var, device="cpu")
    print(mn.means)

    samples = mn.sample(*[3] * n)
    ood_samples = torch.rand((3, *shape), device="cpu") + 10.0

    dsq = mn.get_square_distances(samples)
    rbf = torch.exp(-dsq / (2 * mn.var_coef))

    print("Samples: ", samples)
    print("RBF of d^2: ", rbf)
    print("Credal metrics: ", mn.get_credal_metrics(samples))


if __name__ == "__main__":
    main()
