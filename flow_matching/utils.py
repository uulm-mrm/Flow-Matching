import torch
from torch import nn, Tensor

from .path import Path


def push_forward(
    x0: Tensor,
    x1: Tensor,
    t_int: tuple[float, float],
    path: Path,
    vf: nn.Module,
    **vf_extras
) -> Tensor:
    """Pushes a path section from x0 to x1 within a given time interval"""
    t0, t1 = t_int

    # s is local time for correct path interpolation
    s = torch.rand((x0.shape[0], 1), device=x0.device)

    # t is the map from local to global time for the model to predict
    t = s * (t1 - t0) + t0

    ps = path.sample(x0, x1, s)

    dxt_hat = vf.forward(ps.xt, t, **vf_extras)
    dxt = ps.dxt / (t1 - t0)

    return (dxt_hat - dxt).square().mean()


def push_forward_all(
    anc_x: tuple[Tensor, ...],
    anc_t: tuple[float, ...],
    path: Path,
    vf: nn.Module,
    **vf_extras
) -> Tensor:
    """Pushes all segments from anc_x forward and returns the overall loss

    Args:
        anc_x (tuple[Tensor]): a tuple of anchor tensors
        anc_t (tuple[float, ...]): anchor times for each of the anchor tensors
        path (Path): path along which to flow
        vf (nn.Module): vector field model that learns the speeds of points in time

    Returns:
        Tensor: summed loss from all pushed segments
    """
    num_segments = len(anc_t) - 1

    loss = 0
    for x0, x1, t0, t1 in zip(anc_x[:-1], anc_x[1:], anc_t[:-1], anc_t[1:]):
        loss = loss + push_forward(x0, x1, (t0, t1), path, vf, **vf_extras)

    return loss / num_segments  # type: ignore
