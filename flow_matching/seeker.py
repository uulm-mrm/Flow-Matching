import torch
from torch import Tensor
from typing import Callable


class Seeker:
    def __init__(self, max_evals: int = 10) -> None:
        self.max_evals = max_evals

    def search(
        self,
        sfunc: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class GoldenSectionSeeker(Seeker):
    def __init__(self, max_evals: int = 10) -> None:
        super().__init__(max_evals)
        self.fi = (5**0.5 - 1.0) / 2.0

    def search(
        self,
        sfunc: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        """Performs a golden section minimum search over the interval [a, b]
        on the given function func

        Args:
            sfunc (Callable[[Tensor], Tensor]): 1D score function that takes a scalar
                and returns a score
            batch_size (int): number of elements in the batch
            a (Tensor): lower bound of the interval.
            b (Tensor): upper bound of the interval
            eps (float, optional): minimum interval size. Defaults to 1e-6.

        Returns:
            tuple[Tensor, Tensor]: the point which produces the minimum and its function value
        """

        # first eval points
        c = b - self.fi * (b - a)
        d = a + self.fi * (b - a)

        # first two evals
        fc = sfunc(c)
        fd = sfunc(d)

        evals = 2
        while ((b - a) > eps).any() and evals < self.max_evals:
            # true: min is in [a, d], false: min is in [c, b]
            mask = fc < fd

            # move intervals for mask
            b = torch.where(mask, d, b)
            a = torch.where(~mask, c, a)

            d = a + self.fi * (b - a)
            c = b - self.fi * (b - a)

            fc = sfunc(c)
            fd = sfunc(d)

            evals += 1

        # get the best from history
        mask = fc < fd
        min_x = torch.where(mask, c, d)
        min_fx = torch.where(mask, fc, fd)

        return min_x, min_fx


def main():
    optim = lambda x: (x - 2) ** 2

    gss = GoldenSectionSeeker(max_evals=20)
    a = torch.tensor([-2.2, 2.0, 1.5, 0.5])
    b = torch.tensor([2.2, 3.0, 2.2, 4.0])

    min_x, min_fx = gss.search(optim, a, b, eps=1e-8)
    print(min_x, min_fx)


if __name__ == "__main__":
    main()
