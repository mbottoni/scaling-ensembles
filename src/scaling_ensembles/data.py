from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from scaling_ensembles.config import DataConfig


@dataclass(frozen=True)
class DatasetInfo:
    input_shape: tuple[int, int, int]
    num_classes: int

    @property
    def input_dim(self) -> int:
        channels, height, width = self.input_shape
        return channels * height * width


def make_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader, DatasetInfo]:
    """Create train/eval loaders for small image classification datasets."""
    train_dataset, eval_dataset, info = make_datasets(config)
    if config.train_subset is not None:
        train_dataset = Subset(train_dataset, range(min(config.train_subset, len(train_dataset))))
    if config.eval_subset is not None:
        eval_dataset = Subset(eval_dataset, range(min(config.eval_subset, len(eval_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, info


def make_datasets(config: DataConfig):
    name = config.name.lower()

    if name == "mnist":
        transform = image_transform(mean=(0.1307,), std=(0.3081,))
        train = datasets.MNIST(config.root, train=True, transform=transform, download=True)
        eval_dataset = datasets.MNIST(config.root, train=False, transform=transform, download=True)
        return train, eval_dataset, DatasetInfo(input_shape=(1, 28, 28), num_classes=10)

    if name in {"fashionmnist", "fashion-mnist"}:
        transform = image_transform(mean=(0.2860,), std=(0.3530,))
        train = datasets.FashionMNIST(config.root, train=True, transform=transform, download=True)
        eval_dataset = datasets.FashionMNIST(config.root, train=False, transform=transform, download=True)
        return train, eval_dataset, DatasetInfo(input_shape=(1, 28, 28), num_classes=10)

    if name in {"cifar10", "cifar-10"}:
        transform = image_transform(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )
        train = datasets.CIFAR10(config.root, train=True, transform=transform, download=True)
        eval_dataset = datasets.CIFAR10(config.root, train=False, transform=transform, download=True)
        return train, eval_dataset, DatasetInfo(input_shape=(3, 32, 32), num_classes=10)

    raise ValueError(
        f"Unsupported dataset: {config.name}. Try MNIST, FashionMNIST, or CIFAR10."
    )


def image_transform(mean: tuple[float, ...], std: tuple[float, ...]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
