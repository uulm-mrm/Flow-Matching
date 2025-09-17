from torch import Tensor, nn


class FiLM(nn.Module):
    """Apply FiLM (Feature-wise Linear Modulation) across input channels/blocks

    FiLM is a batch-norm equivalent which can be conditioned on certain data

    For more details see https://arxiv.org/abs/1709.07871
    """

    def __init__(self, chan: int, emb_dim: int) -> None:
        """
        Args:
            chan (int): number of channels to apply FiLM to
            emb_dim (int): embedded dimensions to make FiLM from
        """
        super().__init__()

        # this MLP makes gamma and beta parameters together (hence chan * 2)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, chan * 2),
            nn.SiLU(),
            nn.Linear(chan * 2, chan * 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Applies scaling (via gamma) and translation (via beta) to x.

        Gamma and beta are computed using an MLP that takes time embedding as context
        """
        gamma, beta = self.mlp.forward(t).chunk(2, dim=1)

        # span them accross H and W of input image
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        # FiLM equation
        return x * (1 + gamma) + beta
