from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BucherTableau:
    """A Butcher Tableau for RK integration methods"""

    a: Tensor  # the A matrix of size (S, S)
    b: Tensor  # the b vector of size (S,)
    c: Tensor  # the c vector of size (S,)


RK4_Tableau = BucherTableau(
    a=torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    b=torch.tensor([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
    c=torch.tensor([0.0, 0.5, 0.5, 1.0]),
)
