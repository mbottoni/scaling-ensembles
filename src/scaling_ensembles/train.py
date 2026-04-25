from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scaling_ensembles.config import ExperimentConfig, load_config
from scaling_ensembles.data import DatasetInfo, make_dataloaders
from scaling_ensembles.models import count_parameters, make_model


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    width: int
    seed: int
    parameter_count: int
    checkpoint_path: Path
    epochs_completed: int
    stop_reason: str
    train_loss: float
    train_accuracy: float
    eval_loss: float
    eval_accuracy: float


def resolve_device(device: str) -> torch.device:
    if device == "mps" and not torch.backends.mps.is_available():
        LOGGER.warning("Requested device 'mps', but MPS is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(
    config: ExperimentConfig,
    width: int,
    seed: int,
    output_dir: str | Path | None = None,
) -> TrainResult:
    LOGGER.info("Starting training run: width=%s seed=%s", width, seed)
    set_seed(seed)
    train_loader, eval_loader, dataset_info = make_dataloaders(config.data)
    device = resolve_device(config.training.device)
    model = make_model(config.model, dataset_info, width).to(device)
    parameter_count = count_parameters(model)
    LOGGER.info(
        "Prepared model: architecture=%s params=%s dataset=%s input_shape=%s device=%s",
        config.model.architecture,
        f"{parameter_count:,}",
        config.data.name,
        dataset_info.input_shape,
        device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    epochs_completed = 0
    stop_reason = "max_epochs"
    train_loss = float("nan")
    train_accuracy = float("nan")

    for epoch in range(config.training.epochs):
        LOGGER.info("Training epoch %s/%s", epoch + 1, config.training.epochs)
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        epochs_completed = epoch + 1
        should_check_target = (
            config.training.target_train_loss is not None
            and epochs_completed >= config.training.min_epochs
            and epochs_completed % config.training.eval_every_epochs == 0
        )
        if should_check_target:
            train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
            LOGGER.info(
                "Target-loss check epoch=%s train_loss=%.4f train_acc=%.4f target=%.4f",
                epochs_completed,
                train_loss,
                train_accuracy,
                config.training.target_train_loss,
            )
            if train_loss <= config.training.target_train_loss:
                stop_reason = "target_train_loss"
                LOGGER.info(
                    "Stopping early: train_loss %.4f <= target %.4f",
                    train_loss,
                    config.training.target_train_loss,
                )
                break

    LOGGER.info("Evaluating final train and eval metrics")
    train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
    eval_loss, eval_accuracy = evaluate(model, eval_loader, criterion, device)

    base_dir = Path(output_dir or config.output_dir)
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"width_{width}_seed_{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "width": width,
            "seed": seed,
            "parameter_count": parameter_count,
            "epochs_completed": epochs_completed,
            "stop_reason": stop_reason,
            "dataset": dataset_info,
            "model_config": config.model,
        },
        checkpoint_path,
    )
    LOGGER.info(
        "Finished width=%s seed=%s train_acc=%.4f eval_acc=%.4f eval_loss=%.4f checkpoint=%s",
        width,
        seed,
        train_accuracy,
        eval_accuracy,
        eval_loss,
        checkpoint_path,
    )

    return TrainResult(
        width=width,
        seed=seed,
        parameter_count=parameter_count,
        checkpoint_path=checkpoint_path,
        epochs_completed=epochs_completed,
        stop_reason=stop_reason,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        eval_loss=eval_loss,
        eval_accuracy=eval_accuracy,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> None:
    model.train()
    progress = tqdm(loader, desc=f"epoch {epoch + 1}", leave=False)
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    model.eval()
    resolved_device = torch.device(device)
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(resolved_device), targets.to(resolved_device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def load_checkpoint_model(
    checkpoint_path: str | Path,
    config: ExperimentConfig,
    dataset_info: DatasetInfo,
    device: torch.device | str = "cpu",
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(config.model, dataset_info, checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def write_train_results(results: list[TrainResult], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(TrainResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            row = {field: getattr(result, field) for field in writer.fieldnames}
            row["checkpoint_path"] = str(row["checkpoint_path"])
            writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train one model for a scaling-ensembles experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--width", required=True, type=int, help="Model width.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_one(config, width=args.width, seed=args.seed)
    write_train_results([result], Path(config.output_dir) / "train_single.csv")
    print(result)


if __name__ == "__main__":
    main()
