import torch
from torch import nn, Tensor


class PotentialField(nn.Module):
    def __init__(self, in_d: int, h_d: int, t_d: int) -> None:
        super().__init__()

        self.in_d = in_d
        self.t_d = t_d

        self.mlp = nn.Sequential(
            nn.Linear(in_d + t_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, 2 * h_d),
            nn.SiLU(),
            nn.Linear(2 * h_d, h_d // 2),
            nn.SiLU(),
            nn.Linear(h_d // 2, 1),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)
