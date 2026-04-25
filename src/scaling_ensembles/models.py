from __future__ import annotations

import torch
from torch import nn

from scaling_ensembles.config import ModelConfig
from scaling_ensembles.data import DatasetInfo


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int,
        hidden_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = _activation(activation)
        layers: list[nn.Module] = [nn.Flatten()]
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current_dim, width), act()])
            current_dim = width
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int, width: int) -> None:
        super().__init__()
        channels, _, _ = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * width * 7 * 7, 4 * width),
            nn.ReLU(),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_model(config: ModelConfig, dataset: DatasetInfo, width: int) -> nn.Module:
    architecture = config.architecture.lower()
    if architecture == "mlp":
        return MLP(
            input_dim=dataset.input_dim,
            num_classes=dataset.num_classes,
            width=width,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
        )
    if architecture == "cnn":
        return SmallCNN(dataset.input_shape, dataset.num_classes, width)
    raise ValueError(f"Unsupported architecture: {config.architecture}. Try mlp or cnn.")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error
