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
    def __init__(self, categories: tuple[int, ...] = (0, 1, 2)) -> None:
        super().__init__()

        x, y = load_iris(return_X_y=True)

        # to tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)

        # subsample
        cats = torch.tensor(categories)
        mask = torch.isin(y, cats)
        x = x[mask]
        y = y[mask]

        # norm x
        xmin = x.min(dim=0, keepdim=True).values
        xmax = x.max(dim=0, keepdim=True).values
        self.x = (x - xmin) / (xmax - xmin)

        # y to deltas
        y = torch.nn.functional.one_hot(y, len(categories))  # pylint: disable=E1102
        y = y.float()

        self.y = torch.zeros_like(x)
        self.y[:, : y.shape[1]] = y

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return self.y.shape[0]


def main():
    ds = IrisDataset(categories=(0, 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=24, shuffle=True, drop_last=True)

    for x, y in dl:
        print(x, y)
        break


if __name__ == "__main__":
    main()
