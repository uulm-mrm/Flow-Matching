import torch
from torch import nn, Tensor


class VectorField(nn.Module):
    def __init__(self, in_dims: int, h_dims: int, t_dims: int = 1) -> None:
        super().__init__()

        self.in_dims = in_dims
        self.t_dims = t_dims

        self.mlp = nn.Sequential(
            nn.Linear(in_dims + t_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, h_dims),
            nn.SiLU(),
            nn.Linear(h_dims, in_dims),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        """Calculates the speed of each point x at t

        Args:
            xt (Tensor): x at point t in time, size (B, ...)
            t (Tensor): time, size (B | 1,) broadcasted to xt

        Returns:
            Tensor: speed for each component of the input x
        """
        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)
