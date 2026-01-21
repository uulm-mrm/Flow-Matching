import torch
from torch import Tensor
from torch.utils.data import Dataset

from sklearn.datasets import load_iris


def get_iris(device: str) -> tuple[Tensor, Tensor, Tensor]:
    x, y = load_iris(return_X_y=True)

    x = torch.from_numpy(x)
    x = x.float().to(device)

    y = torch.from_numpy(y).to(device)

    return x[y == 0], x[y == 1], x[y == 2]


class IrisDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        x, y = load_iris(return_X_y=True)

        x = torch.from_numpy(x).float()
        self.x = (x - x.min(dim=0, keepdim=True).values) / (
            x.max(dim=0, keepdim=True).values - x.min(dim=0, keepdim=True).values
        )

        y = torch.from_numpy(y)
        y = torch.nn.functional.one_hot(y, num_classes=3)  # pylint: disable=E1102
        y = y.float()

        self.y = torch.zeros_like(x)
        self.y[:, : y.shape[1]] = y

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return self.y.shape[0]


def main():
    ds = IrisDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=24, shuffle=True, drop_last=True)

    for x, y in dl:
        print(x, y)


if __name__ == "__main__":
    main()
