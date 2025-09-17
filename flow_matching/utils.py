from torch import Tensor


def broadcast_to(x: Tensor, y: Tensor) -> Tensor:
    """Broadcasts x to y's dimensions

    Args:
        x (Tensor): the source to be broadcast, size (B,)
        y (Tensor): the target to broadcast to, size (B, N, M, ...)

    Returns:
        Tensor: broadcasted x from (B,) to (B, 1...) to match y
    """

    expand_to = len(y.shape) - 1

    x = x.view(-1, *([1] * expand_to))

    return x
