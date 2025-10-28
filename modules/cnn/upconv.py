from torch import Tensor, nn
from torch.nn import functional as F

from ..utils import FiLM, TimeDependentModule


class TimeConditionedUpConv(TimeDependentModule):
    """Upconv conditioned on time

    Upconv uses pixelshuffle for upsampling, and FiLM for conditioning
    """

    def __init__(self, in_chan: int, out_chan: int, t_dim: int, scale: int = 2) -> None:
        """
        Args:
            in_chan (int): input channels
            out_chan (int): output channels
            t_dim (int): time context dimensions
            scale (int, optional): by how much to scale input. Defaults to 2.
        """
        super().__init__()

        # perserve H, W dimension, just make channels for pixel shuffle
        # for k=3, s=1 and p=1 perserves H and W
        self.conv = nn.Conv2d(
            in_chan, out_chan * (scale**2), kernel_size=3, stride=1, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.film = FiLM(out_chan, t_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input size (B, C, H, W)
            t (Tensor): time size (B, D)

        Returns:
            Tensor: output size (B, C', s*H, s*W)
        """

        x = self.conv(x)
        x = F.relu(x)

        x = self.pixel_shuffle(x)

        x = self.film(x, t)

        return x
