from dataclasses import dataclass, field

from torch import Tensor

from .scheduler import AffineScheduler, AnchorScheduler


def broadcast_to(x: Tensor, y: Tensor) -> Tensor:
    """Broadcasts x to y's dimensions

    Args:
        x (Tensor): the source to be broadcast, size (B,)
        y (Tensor): the target to broadcast to, size (B, N, M, ...)

    Returns:
        Tensor: broadcasted x from (B,) to (B, 1...) to match y
    """

    expand_to = len(y.shape) - len(x.shape)

    x = x.view(*x.shape, *([1] * expand_to))

    return x


@dataclass
class PathSample:
    """Provides x at t, speed of x at t, and t broadcast to their dimensions"""

    xt: Tensor = field(metadata={"help": "x at time t based on scheduler"})
    dxt: Tensor = field(
        metadata={"help": "what the speed of x should be at time t based on scheduler"}
    )
    t: Tensor = field(metadata={"help": "t broadcast to xt dimensions"})


class AffinePath:
    """Samples the probability path at different times for the vector field"""

    def __init__(self, sched: AffineScheduler) -> None:
        self.sched = sched

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor) -> PathSample:
        """Samples the probability path to return x at time t and speed of x at t

        Args:
            x0 (Tensor): initial x at time 0, size (B, ...)
            x1 (Tensor): target x at time 1, size (B, ...)
            t (Tensor): time to sample between, size (B,)

        Returns:
            PathSample: dataclass with xt, dxt, t
        """

        t = broadcast_to(t, x0)
        xt = self.sched.alpha(t) * x1 + self.sched.sigma(t) * x0
        dxt = self.sched.d_alpha(t) * x1 + self.sched.d_sigma(t) * x0

        return PathSample(xt, dxt, t)


class AnchoredPath:
    """
    Samples the probability path at different times for the vector field,
    For an arbitrary number of target densities
    """

    def __init__(self, sched: AnchorScheduler) -> None:
        self.sched = sched

    def sample(self, x_anchors: Tensor, t_anchors: Tensor, t: Tensor) -> PathSample:
        """Samples the probability path to return x at time t and speed of x at t

        Args:
            x_anchors (Tensor): anchors which to pass along the path, size (N, B, ...)
            t_anchors (Tensor): times that those anchors belong to, size (N,)
            t (Tensor): time to sample for, size (B,)

        Returns:
            PathSample: dataclass with xt, dxt, t
        """

        # make this quicker or sth
        weights = self.sched.weight(t, t_anchors)  # (N, B)
        weight_sum = weights.sum(dim=0)  # (B,)

        d_weights = self.sched.d_weight(t, t_anchors)  # (N, B)
        d_weight_sum = d_weights.sum(dim=0)  # (B,)

        norm_weights = weights / weight_sum  # (N, B)
        d_norm_weights = d_weights * weight_sum - d_weight_sum * weights / (
            weight_sum * weight_sum
        )  # (N, B)

        # (N, B, ...)
        norm_weights = broadcast_to(norm_weights, x_anchors)
        d_norm_weights = broadcast_to(d_norm_weights, x_anchors)

        xt = (norm_weights * x_anchors).sum(dim=0)  # (B, ...)
        dxt = (d_norm_weights * x_anchors).sum(dim=0)  # (B, ...)

        return PathSample(xt, dxt, broadcast_to(t, x_anchors[0]))


class AffineMultiPath:
    """
    Makes n Affine Paths of the same kind and calculates
    the probability paths independently for them
    """

    def __init__(self, base_path: AffinePath, num_paths: int) -> None:
        self.base_path = base_path
        self.num_paths = num_paths

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor) -> PathSample:
        """Samples from the affine multipath

        Args:
            x0 (Tensor): (num_paths, B, D...)
            x1 (Tensor): (num_paths, B, D...)
            t (Tensor): (B,)

        Returns:
            PathSample: (num_paths * B, D...) for xt, dxt and (num_paths * B,) for t
        """
        t = broadcast_to(t, x0[0])
        t = t.repeat(x0.shape[0], *[1] * len(t.shape))
        xt = self.base_path.sched.alpha(t) * x1 + self.base_path.sched.sigma(t) * x0
        dxt = (
            self.base_path.sched.d_alpha(t) * x1 + self.base_path.sched.d_sigma(t) * x0
        )

        return PathSample(xt.flatten(0, 1), dxt.flatten(0, 1), t.flatten(0, 1))
