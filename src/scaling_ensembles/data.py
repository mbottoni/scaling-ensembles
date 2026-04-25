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
        train_transform = image_transform(mean=(0.1307,), std=(0.3081,))
        eval_transform = image_transform(
            mean=(0.1307,),
            std=(0.3081,),
            variant=config.eval_variant,
            noise_std=config.noise_std,
            blur_kernel_size=config.blur_kernel_size,
        )
        train = datasets.MNIST(config.root, train=True, transform=train_transform, download=True)
        eval_dataset = datasets.MNIST(config.root, train=False, transform=eval_transform, download=True)
        return train, eval_dataset, DatasetInfo(input_shape=(1, 28, 28), num_classes=10)

    if name in {"fashionmnist", "fashion-mnist"}:
        train_transform = image_transform(mean=(0.2860,), std=(0.3530,))
        eval_transform = image_transform(
            mean=(0.2860,),
            std=(0.3530,),
            variant=config.eval_variant,
            noise_std=config.noise_std,
            blur_kernel_size=config.blur_kernel_size,
        )
        train = datasets.FashionMNIST(config.root, train=True, transform=train_transform, download=True)
        eval_dataset = datasets.FashionMNIST(config.root, train=False, transform=eval_transform, download=True)
        return train, eval_dataset, DatasetInfo(input_shape=(1, 28, 28), num_classes=10)

    if name in {"cifar10", "cifar-10"}:
        train_transform = image_transform(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )
        eval_transform = image_transform(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
            variant=config.eval_variant,
            noise_std=config.noise_std,
            blur_kernel_size=config.blur_kernel_size,
        )
        train = datasets.CIFAR10(config.root, train=True, transform=train_transform, download=True)
        eval_dataset = datasets.CIFAR10(config.root, train=False, transform=eval_transform, download=True)
        return train, eval_dataset, DatasetInfo(input_shape=(3, 32, 32), num_classes=10)

    raise ValueError(
        f"Unsupported dataset: {config.name}. Try MNIST, FashionMNIST, or CIFAR10."
    )


def image_transform(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    variant: str = "clean",
    noise_std: float = 0.15,
    blur_kernel_size: int = 3,
) -> transforms.Compose:
    steps: list[object] = [transforms.ToTensor()]
    if variant == "clean":
        pass
    elif variant == "gaussian_noise":
        steps.append(AddGaussianNoise(noise_std))
    elif variant == "blur":
        steps.append(transforms.GaussianBlur(kernel_size=blur_kernel_size))
    else:
        raise ValueError(
            f"Unsupported eval_variant: {variant}. Try clean, gaussian_noise, or blur."
        )
    steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


class AddGaussianNoise:
    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + self.std * torch.randn_like(tensor)).clamp(0.0, 1.0)
