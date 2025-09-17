from typing import Literal

from torch.utils.data import Dataset, Subset

from torchvision import transforms
from torchvision.datasets import MNIST


def get_mnist(subset: Literal["train", "test"]) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor(), lambda x: x * 2.0 - 1.0])

    mnist_dataset = MNIST(
        root="examples/mnist/dataset",
        train=(subset == "train"),
        download=True,
        transform=transform,
    )

    return mnist_dataset


def sample_mnist(dataset: Dataset, classes: list[int]) -> Dataset:
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]  # type: ignore
    sampled_dataset = Subset(dataset, indices)
    return sampled_dataset
