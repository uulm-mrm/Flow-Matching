import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention for the combined spatial and channel attention
    as proposed in the Convolutional Block Attention Module"""

    def __init__(self, in_c: int, reduction: int = 16) -> None:
        """
        Args:
            in_c (int): input channels
            reduction (int, optional): by how much to reduce the input channels in the MLP.
                Defaults to 16.
        """
        super().__init__()

        # an mlp for the max and avg pool accross channels
        # to get which channels are important and how
        self.mlp = nn.Sequential(
            nn.Linear(in_c, in_c // reduction),
            nn.SiLU(),
            nn.Linear(in_c // reduction, in_c),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size (B, C, H, W)

        Returns:
            Tensor: (B, C, H, W) tensor with attention applied to C
        """
        b, c, *_ = x.size()

        # gap and gmp on the image to get channel vectors
        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        gmp = F.adaptive_max_pool2d(x, 1).view(b, c)

        gap = self.mlp(gap)
        gmp = self.mlp(gmp)

        attn = F.sigmoid(gap + gmp).view(b, c, 1, 1)

        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention for the combined spatial and channel attention
    as proposed in the Convolutional Block Attention Module"""

    def __init__(self, k: int = 7) -> None:
        """
        Args:
            k (int, optional): spatial attention kernel size, keep odd. Defaults to 7.
        """
        super().__init__()

        # keep H, W constant based on k
        padding = (k - 1) // 2

        # shared conv for gap and gmp to figure out where to focus
        # gap is one channel, gmp is the other channel, so in_channels is 2
        # output channel is just 1 because we're interested in H, W attention
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=k,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size (B, C, H, W)

        Returns:
            Tensor: (B, C, H, W) tensor with attention applied to HxW
        """

        # gap and gmp across C dim
        gap = torch.mean(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)

        # concat gap and gmp in C dim
        attn = torch.cat([gap, gmp], dim=1)
        attn = self.conv(attn)
        attn = F.sigmoid(attn)

        return x * attn


class ConvBlockAttention(nn.Module):
    """Convolution Block Attention Module as proposed in the same named paper,
    Applies a channel attention followed by a spatial attention on the input
    """

    def __init__(self, in_c: int, reduction: int = 16, k: int = 7) -> None:
        """
        Args:
            in_c (int): input channels
            reduction (int, optional): by how much to reduce the input channels in the MLP.
                Defaults to 16.
            k (int, optional): spatial attention kernel size, keep odd. Defaults to 7.
        """
        super().__init__()

        self.cam = ChannelAttention(in_c, reduction)
        self.sam = SpatialAttention(k)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size (B, C, H, W)

        Returns:
            Tensor: tensor with attention applied to C then to HxW. out size (B, C, H, W)
        """

        x = self.cam(x)

        return self.sam(x)
