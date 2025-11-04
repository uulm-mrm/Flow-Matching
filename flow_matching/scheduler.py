import torch
from torch import Tensor


class Scheduler:
    def alpha(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def sigma(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def d_alpha(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def d_sigma(self, t: Tensor) -> Tensor:
        raise NotImplementedError


class AnchorScheduler:
    def weight(self, t: Tensor, t_anc: Tensor) -> Tensor:
        """Weight tensor, equivalent to alpha/sigma

        Args:
            t (Tensor): time, size (B,)
            t_anc (Tensor): anchor times, size (N,)

        Returns:
            Tensor: w_i for each anchor t_i, size (N, B)
        """
        raise NotImplementedError

    def d_weight(self, t: Tensor, t_anc: Tensor) -> Tensor:
        """derivative of each of the weights for time, equivalent to d_alpha/d_sigma

        Args:
            t (Tensor): time, size (B,)
            t_anc (Tensor): anchor times, size (N,)

        Returns:
            Tensor: d_w_i for each anchor t_i, size (N, B)
        """
        raise NotImplementedError


class OTScheduler(Scheduler):
    def alpha(self, t: Tensor) -> Tensor:
        return t

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - t

    def d_alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def d_sigma(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)


class PolyScheduler(Scheduler):
    def __init__(self, n: float) -> None:
        super().__init__()

        self.n = n

    def alpha(self, t: Tensor) -> Tensor:
        return torch.pow(t, self.n)

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - torch.pow(t, self.n)

    def d_alpha(self, t: Tensor) -> Tensor:
        return self.n * torch.pow(t, self.n - 1)

    def d_sigma(self, t: Tensor) -> Tensor:
        return -self.n * torch.pow(t, self.n - 1)


class CosineMultiScheduler(AnchorScheduler):
    """Schedules the path with the following w_i

    w_i(t) = 1/2 * (1 + cos(pi/k * (t-ti)))
    d_wi(t) = -1/2 * pi/k * sin(pi/k * (t-ti))
    """

    def __init__(self, k: float) -> None:
        """
        Args:
            k (float): width of the bump of each cosine weight
        """
        super().__init__()

        self.k = k

    def weight(self, t: Tensor, t_anc: Tensor) -> Tensor:
        delta_t = t.unsqueeze(0) - t_anc.unsqueeze(1)  # (N, B)
        w = 0.5 * (1.0 + torch.cos(torch.pi * delta_t / self.k))  # (N, B)

        return torch.where(delta_t.abs() <= self.k, w, 0.0)

    def d_weight(self, t: Tensor, t_anc: Tensor) -> Tensor:
        delta_t = t.unsqueeze(0) - t_anc.unsqueeze(1)  # (N, B)
        dw = -0.5 * torch.pi / self.k * torch.sin(torch.pi * delta_t / self.k)  # (N, B)

        return torch.where(delta_t.abs() <= self.k, dw, 0.0)
