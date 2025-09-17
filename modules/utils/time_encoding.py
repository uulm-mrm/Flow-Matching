import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        assert emb_dim % 2 == 0, "Embedding dimension needs to be divisible by 2"

        self.emb_dim = emb_dim
        self.log_10k = math.log(10_000)

    def forward(self, t: Tensor) -> Tensor:
        # t = (B,)
        half_dims = self.emb_dim // 2

        t_emb = self.log_10k / (half_dims - 1)
        t_emb = torch.exp(torch.arange(half_dims, device=t.device) * -t_emb)
        t_emb = t[:, None] * t_emb[None, :]  # (B, d/2)

        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)  # (B, d)

        return t_emb
