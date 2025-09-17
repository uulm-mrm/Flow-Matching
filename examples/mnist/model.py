from torch import Tensor, nn

from modules.cnn import TimeConditionedConv, TimeConditionedUpConv
from modules.utils import SinusoidalTimeEmbedding


class CNNVF(nn.Module):
    def __init__(self, t_dims: int = 128) -> None:
        super().__init__()

        # time embedding
        self.time_embed = SinusoidalTimeEmbedding(t_dims)

        # downsampling
        self.conv1 = TimeConditionedConv(1, 16, 3, t_dims)  # B, 16, 14, 14
        self.conv2 = TimeConditionedConv(16, 32, 3, t_dims)  # B, 32, 7, 7

        # fc bottleneck
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.SiLU(),
            nn.Linear(256, 32 * 7 * 7),
            nn.SiLU(),
            nn.Unflatten(1, (32, 7, 7)),
        )

        # upsampling
        self.upconv1 = TimeConditionedUpConv(32, 16, t_dims)  # B, 16, 14, 14
        self.upconv2 = TimeConditionedUpConv(16, 1, t_dims)  # B, 1, 28, 28

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input image size (B, 1, 28, 28)
            t (Tensor): time size (B,) or ()

        Returns:
            Tensor: output shape (B, 1, 28, 28)
        """

        t_emb = self.time_embed.forward(t)

        # downsample
        x = self.conv1.forward(x, t_emb)
        x = self.conv2.forward(x, t_emb)

        # fc bottleneck
        x = self.fc(x)

        # upsample
        x = self.upconv1.forward(x, t_emb)
        x = self.upconv2.forward(x, t_emb)

        return x
