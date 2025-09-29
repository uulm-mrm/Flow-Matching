from torch import Tensor, nn
from torch.nn import functional as F

from modules.utils import FiLM, TimeDependentModule


class TimeConditionedConv(TimeDependentModule):
    """Ordinary convolution layer conditioned on time using FiLM

    Good for giving the convolution time context for each image
    """

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        ksize: int,
        t_dim: int,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        """
        Args:
            in_chan (int): input channels
            out_chan (int): output channels
            ksize (int): kernel size
            t_dim (int): time context dimensions
            stride (int, optional): conv stride. Defaults to 2.
            padding (int, optional): conv padding. Defaults to 1.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_chan, out_chan, ksize, stride, padding)
        self.film = FiLM(out_chan, t_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input size (B, C, H, W)
            t (Tensor): time size (B, D)

        Returns:
            Tensor: output size (B, C', H', W')
        """
        x = self.conv(x)
        x = self.film(x, t)

        return F.silu(x)
