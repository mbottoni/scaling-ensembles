from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scaling_ensembles.config import ExperimentConfig
from scaling_ensembles.data import DatasetInfo
from scaling_ensembles.models import make_model
from scaling_ensembles.train import evaluate


@dataclass(frozen=True)
class InterpolationPoint:
    width: int
    seed_a: int
    seed_b: int
    alpha: float
    loss: float
    accuracy: float
    loss_barrier: float


def linear_interpolation_barrier(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    config: ExperimentConfig,
    dataset_info: DatasetInfo,
    loader: DataLoader,
    steps: int,
    device: torch.device | str = "cpu",
) -> list[InterpolationPoint]:
    """Evaluate loss along theta(alpha) = (1-alpha) theta_a + alpha theta_b."""
    resolved_device = torch.device(device)
    ckpt_a = torch.load(checkpoint_a, map_location=resolved_device, weights_only=False)
    ckpt_b = torch.load(checkpoint_b, map_location=resolved_device, weights_only=False)
    if ckpt_a["width"] != ckpt_b["width"]:
        raise ValueError("Linear interpolation requires checkpoints from the same architecture width.")

    model = make_model(config.model, dataset_info, ckpt_a["width"]).to(resolved_device)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    accuracies: list[float] = []
    alphas = torch.linspace(0.0, 1.0, steps).tolist()

    for alpha in alphas:
        interpolated = interpolate_state_dicts(ckpt_a["state_dict"], ckpt_b["state_dict"], alpha)
        model.load_state_dict(interpolated)
        loss, accuracy = evaluate(model, loader, criterion, resolved_device)
        losses.append(loss)
        accuracies.append(accuracy)

    endpoint_loss = max(losses[0], losses[-1])
    return [
        InterpolationPoint(
            width=ckpt_a["width"],
            seed_a=ckpt_a["seed"],
            seed_b=ckpt_b["seed"],
            alpha=alpha,
            loss=loss,
            accuracy=accuracy,
            loss_barrier=loss - endpoint_loss,
        )
        for alpha, loss, accuracy in zip(alphas, losses, accuracies, strict=True)
    ]


def interpolate_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    return {
        key: (1.0 - alpha) * state_a[key] + alpha * state_b[key]
        for key in state_a
    }


def write_interpolation_results(results: list[InterpolationPoint], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(InterpolationPoint.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in writer.fieldnames})
