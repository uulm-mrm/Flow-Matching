from typing import Callable

import torch
from torch import Tensor


class Seeker:
    def __init__(self, max_evals: int = 10) -> None:
        self.max_evals = max_evals

    def search(
        self,
        score_func: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class GoldenSectionSeeker(Seeker):
    """
    Seeker class that searches for the optimum using the golden section method.
    Presumes the function is unimodal
    """

    def __init__(self, max_evals: int = 10) -> None:
        super().__init__(max_evals)
        self.fi = (5**0.5 - 1.0) / 2.0

    def search(
        self,
        score_func: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        """Performs a golden section minimum search over the interval [a, b]
        on the given function func

        Args:
            score_func (Callable[[Tensor], Tensor]): 1D score function that takes a scalar
                and returns a score
            a (Tensor): lower bound of the interval.
            b (Tensor): upper bound of the interval
            eps (float, optional): minimum interval size. Defaults to 1e-6.

        Returns:
            tuple[Tensor, Tensor]: the point which produces the minimum and its function value
        """

        # work on the (a, b] interval because of the classifier later
        original_b, fb = b.clone(), score_func(b)

        # first eval points
        c = b - self.fi * (b - a)
        d = a + self.fi * (b - a)

        # first two evals
        fc = score_func(c)
        fd = score_func(d)

        evals = 2
        while ((b - a) > eps).any() and evals < self.max_evals:
            # true: min is in [a, d], false: min is in [c, b]
            mask = fc < fd

            # move intervals for mask
            b = torch.where(mask, d, b)
            a = torch.where(~mask, c, a)

            d = a + self.fi * (b - a)
            c = b - self.fi * (b - a)

            fc = score_func(c)
            fd = score_func(d)

            evals += 1

        # get the best from history
        mask = fc < fd
        min_x = torch.where(mask, c, d)
        min_fx = torch.where(mask, fc, fd)

        mask = min_fx < fb
        min_x = torch.where(mask, min_x, original_b)
        min_fx = torch.where(mask, min_fx, fb)

        return min_x, min_fx


class NaiveMidpoints(Seeker):
    """
    Seeker that searches for the optimum by iteratively uniformly sampling the function
    Can work for non-unimodal functions
    """

    def __init__(self, max_evals: int = 10, iters: int = 2) -> None:
        assert (max_evals % iters) == 0

        super().__init__(max_evals)

        self.iters = iters
        self.iter_samples = (max_evals // iters) + 1

    def search(
        self,
        score_func: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        """
        1. Uniformly samples the interval.
        2. Evaluates the midpoints of all subintervals
        3. Finds the minimum midpoint
        4. New interval is one interval unit left and right of the min midpoint
        5. Repeats 1-4

        Args:
            score_func (Callable[[Tensor], Tensor]): 1D score function that takes a scalar
                and returns a score
            a (Tensor): lower bound of the interval, size (B,)
            b (Tensor): upper bound of the interval, size (B,)
            eps (float, optional): minimum interval size. Defaults to 1e-6.

        Returns:
            tuple[Tensor, Tensor]: the point which produces the minimum and its function value
        """

        lower = a.clone()
        upper = b.clone()

        iters = 0
        while iters < self.iters and ((upper - lower) > eps).any():
            # (B,)
            interval_units = (upper - lower) / self.iter_samples

            # (B, iter_samples)
            uniform_samples = lower.unsqueeze(1) + (upper - lower).unsqueeze(
                1
            ) * torch.linspace(0, 1, self.iter_samples, device=lower.device)

            # (B, iter_samples - 1)
            midpoints = (uniform_samples[:, 1:] + uniform_samples[:, :-1]) * 0.5

            midpoints = midpoints.flatten(0)

            scores = score_func(midpoints)

            scores = scores.reshape(a.shape[0], -1)  # (B, iter_samples - 1)
            midpoints = midpoints.reshape(a.shape[0], -1)  # (B, iter_samples - 1)

            min_scores = torch.argmin(scores, dim=1)  # (B,)

            # maybe n times interval units?
            rows = torch.arange(a.shape[0])
            upper = midpoints[rows, min_scores] + interval_units  # (B,)
            lower = midpoints[rows, min_scores] - interval_units  # (B,)

            # enforce interval size
            upper = torch.where(upper > b, b, upper)  # (B,)
            lower = torch.where(lower < a, a, lower)  # (B,)

            iters += 1

        return midpoints[rows, min_scores], scores[rows, min_scores]  # type: ignore


def main():
    nmp = NaiveMidpoints(max_evals=50, iters=5)
    a = torch.tensor([0.0, -1.0, -1.5, 0.5])
    b = torch.tensor([1.0, 1.0, 2.2, 4.0])

    t, x = nmp.search(lambda x: -torch.exp(x), a, b, eps=1e-3)
    print(t, x)


if __name__ == "__main__":
    main()
