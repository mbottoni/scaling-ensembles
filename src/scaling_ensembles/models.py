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
            MpsSafeAdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * width * 4 * 4, 4 * width),
            nn.ReLU(),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MpsSafeAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        output_height, output_width = self.output_size
        if x.device.type == "mps" and (
            height % output_height != 0 or width % output_width != 0
        ):
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)


class PatchTransformerClassifier(nn.Module):
    """Small DiT/ViT-style image classifier for function-similarity sweeps."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        width: int,
        depth: int,
        patch_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels, height, image_width = input_shape
        if height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError(
                f"Patch size {patch_size} must divide image shape {(height, image_width)}."
            )
        if width % num_heads != 0:
            raise ValueError(f"Width {width} must be divisible by num_heads {num_heads}.")

        num_patches = (height // patch_size) * (image_width // patch_size)
        self.patch_embed = nn.Conv2d(
            channels,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embed = nn.Parameter(torch.zeros(1, num_patches + 1, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)
        self._init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([class_tokens, tokens], dim=1) + self.position_embed
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)


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
    if architecture in {"patch_transformer", "dit", "dit_classifier"}:
        return PatchTransformerClassifier(
            input_shape=dataset.input_shape,
            num_classes=dataset.num_classes,
            width=width,
            depth=config.hidden_layers,
            patch_size=config.patch_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
    raise ValueError(
        f"Unsupported architecture: {config.architecture}. Try mlp, cnn, or patch_transformer."
    )


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
