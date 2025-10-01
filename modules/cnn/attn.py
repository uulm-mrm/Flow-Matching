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


class ConvMHSA(nn.Module):
    """Implements convolution multi-head self-attention"""

    def __init__(self, in_c: int, heads: int = 4) -> None:
        """
        Args:
            in_c (int): number of input channels
            heads (int, optional): number of heads. Defaults to 4.
        """
        super().__init__()

        assert in_c % heads == 0, "Heads must divide input cahnnels"
        self.heads = heads
        self.head_dims = in_c // heads

        self.attn_scale = 1 / (self.head_dims**0.5)

        # linear projections for qkv, with squeezed space so conv1d
        # bit quicker to work in 1D with spatial flattened than to work in 2D
        self.qkv = nn.Conv1d(in_channels=in_c, out_channels=in_c * 3, kernel_size=1)

        # attn projection and scaling (same as starting with the output projection zeroed-out)
        self.out_proj = nn.Conv1d(in_c, in_c, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): tensor of size (B, C, H, W)

        Returns:
            Tensor: tensor of size (B, C, H, W) after self attention
        """
        b, c, h, w = x.size()
        x = x.reshape(b, c, -1)

        # project qkv
        q, k, v = self.qkv(x).chunk(3, dim=1)

        # (batch * heads, head_d, spatial) @ (batch * heads, head_d, spatial) =
        # (batch * heads, spatial, spatial)
        attn_w = torch.einsum(
            "bcn,bcm->bnm",  # a lil bit quicker to mul separately than div later
            q.view(b * self.heads, self.head_dims, h * w),
            k.view(b * self.heads, self.head_dims, h * w),
        )
        attn_w = attn_w * self.attn_scale
        attn_w = F.softmax(attn_w, dim=-1)

        # (batch * heads, spatial, spatial) @ (batch * heads, head_d, spatial) =
        # (batch * heads, head_d, spatial) -> (batch, heads * head_d, spatial)
        # heads * heads_d == in_c
        attn = torch.einsum(
            "bnm,bcm->bcn", attn_w, v.reshape(b * self.heads, self.head_dims, h * w)
        )
        attn = attn.reshape(b, -1, h * w)

        # project attn out
        attn = self.out_proj(attn)

        # since attn and x are (b, in_c, spatial) reshape only in the end
        return (self.gamma * attn + x).reshape(b, c, h, w)
