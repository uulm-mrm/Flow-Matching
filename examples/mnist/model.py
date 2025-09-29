import torch
from torch import Tensor, nn

from modules.cnn import TimeConditionedConv, TimeConditionedUpConv, ConvMHSA
from modules.utils import (
    SinusoidalTimeEmbedding,
    TimeDependentSequential,
    TimeDependentModule,
)


class TimeDependentLinearBlock(TimeDependentModule):
    def __init__(self, in_dims: int, out_dims: int, t_dims: int) -> None:
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_dims + t_dims, out_dims),
            nn.BatchNorm1d(out_dims),
            nn.SiLU(),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        z = torch.cat([x, t], dim=-1)

        return self.nn(z)


class CNNVF(nn.Module):
    def __init__(self, t_dims: int = 128) -> None:
        super().__init__()

        # time embedding
        self.time_embed = SinusoidalTimeEmbedding(t_dims)

        # downsampling
        self.down = TimeDependentSequential(
            TimeConditionedConv(1, 32, 3, t_dims, stride=1),  # B, 32, 28, 28
            nn.AdaptiveMaxPool2d(14),  # B, 32, 14, 14
            ConvMHSA(in_c=32, heads=4),
            TimeConditionedConv(32, 64, 3, t_dims, stride=1),  # B, 64, 14, 14
            nn.AdaptiveAvgPool2d(7),  # B, 64, 7, 7
            ConvMHSA(in_c=64, heads=8),
            TimeConditionedConv(64, 128, 3, t_dims, stride=1),  # B, 128, 7, 7
            nn.AdaptiveAvgPool2d(1),  # B, 128, 1, 1
        )

        # fc bottleneck
        self.fc = TimeDependentSequential(
            nn.Flatten(),
            TimeDependentLinearBlock(128, 256, t_dims=1),
            TimeDependentLinearBlock(256, 128, t_dims=1),
            nn.Unflatten(1, (128, 1, 1)),
        )

        # upsampling
        self.up = TimeDependentSequential(
            TimeConditionedUpConv(128, 128, t_dims, 7),  # B, 128, 7, 7
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # B, 64, 7, 7,
            ConvMHSA(in_c=64, heads=8),
            TimeConditionedUpConv(64, 64, t_dims, 2),  # B, 64, 14, 14
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # B, 32, 14, 14
            ConvMHSA(in_c=32, heads=4),
            TimeConditionedUpConv(32, 32, t_dims, 2),  # B, 32, 28, 28
            nn.Conv2d(32, 1, 3, stride=1, padding=1),  # B, 1, 28, 28
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input image size (B, 1, 28, 28)
            t (Tensor): time size (B,) or ()

        Returns:
            Tensor: output shape (B, 1, 28, 28)
        """
        t = t.view(-1, 1).expand(x.shape[0], 1)
        t_emb = self.time_embed.forward(t)

        # downsample
        x = self.down(x, t_emb)

        # fc bottleneck
        x = self.fc(x, t)

        # upsample
        x = self.up(x, t_emb)

        return x
