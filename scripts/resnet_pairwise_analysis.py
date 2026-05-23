#!/usr/bin/env python3
"""Compute pairwise diversity stats from ResNet checkpoints.

Loads saved checkpoints from outputs/series/cifar10-resnet-width-sweep/checkpoints/,
runs inference on CIFAR-10 test set, and computes pairwise disagreement and
ensemble gain. Writes results that can be used by compare_architectures.py.

Writes:
  outputs/series/cifar10-resnet-width-sweep/pairwise_similarity.csv
  outputs/series/cifar10-resnet-width-sweep/train_results.csv
"""
from __future__ import annotations

import csv
import logging
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scaling_ensembles.models import DatasetInfo, make_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

RESNET_DIR = Path("outputs/series/cifar10-resnet-width-sweep")
CKPT_DIR = RESNET_DIR / "checkpoints"
DATA_ROOT = Path("data")
DEVICE = "cpu"
BATCH_SIZE = 512
WIDTHS = [16, 32, 64, 128]
SEEDS = list(range(10))
DATASET_INFO = DatasetInfo(input_shape=(3, 32, 32), num_classes=10)


def get_test_loader() -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tfm)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


@torch.no_grad()
def get_predictions(model: torch.nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probs, all_targets = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_probs.append(logits.softmax(dim=1).cpu())
        all_targets.append(y)
    return torch.cat(all_probs), torch.cat(all_targets)


def load_checkpoint(width: int, seed: int) -> dict | None:
    path = CKPT_DIR / f"width_{width}_seed_{seed}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location=DEVICE, weights_only=False)


def main() -> None:
    loader = get_test_loader()
    LOGGER.info("Test loader ready")

    # Cache predictions per (width, seed)
    predictions: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    train_rows = []

    for width in WIDTHS:
        for seed in SEEDS:
            ck = load_checkpoint(width, seed)
            if ck is None:
                continue
            model = make_model(ck["model_config"], DATASET_INFO, width)
            model.load_state_dict(ck["state_dict"])
            model.to(DEVICE)

            probs, targets = get_predictions(model, loader)
            preds = probs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
            predictions[(width, seed)] = (probs, targets)

            train_rows.append({
                "width": width,
                "seed": seed,
                "parameter_count": ck["parameter_count"],
                "eval_accuracy": acc,
                "train_accuracy": ck.get("train_accuracy", float("nan")),
            })
            LOGGER.info("width=%d seed=%d acc=%.3f", width, seed, acc)

    # Write train_results.csv
    if train_rows:
        out = RESNET_DIR / "train_results.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
            writer.writeheader()
            writer.writerows(train_rows)
        LOGGER.info("Wrote %s", out)

    # Compute pairwise stats
    pairwise_rows = []
    for width in WIDTHS:
        seeds_available = [s for s in SEEDS if (width, s) in predictions]
        LOGGER.info("Width %d: %d seeds available for pairwise", width, len(seeds_available))

        if len(seeds_available) < 2:
            continue

        param_count = None
        for ck_row in train_rows:
            if int(ck_row["width"]) == width:
                param_count = int(ck_row["parameter_count"])
                break

        for sa, sb in combinations(seeds_available, 2):
            probs_a, targets = predictions[(width, sa)]
            probs_b, _ = predictions[(width, sb)]

            preds_a = probs_a.argmax(dim=1)
            preds_b = probs_b.argmax(dim=1)

            agreement = (preds_a == preds_b).float().mean().item()
            acc_a = (preds_a == targets).float().mean().item()
            acc_b = (preds_b == targets).float().mean().item()

            # Ensemble (average probs)
            ens_probs = 0.5 * (probs_a + probs_b)
            ens_acc = (ens_probs.argmax(dim=1) == targets).float().mean().item()

            # Jensen-Shannon divergence
            m = 0.5 * (probs_a + probs_b)
            kl_a = F.kl_div(m.clamp(1e-7).log(), probs_a.clamp(1e-7), reduction="batchmean")
            kl_b = F.kl_div(m.clamp(1e-7).log(), probs_b.clamp(1e-7), reduction="batchmean")
            js = (0.5 * (kl_a + kl_b)).item()

            pairwise_rows.append({
                "width": width,
                "seed_a": sa,
                "seed_b": sb,
                "parameter_count": param_count,
                "agreement": agreement,
                "model_a_accuracy": acc_a,
                "model_b_accuracy": acc_b,
                "ensemble_accuracy": ens_acc,
                "js_divergence": js,
            })

    if pairwise_rows:
        out = RESNET_DIR / "pairwise_similarity.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pairwise_rows)
        LOGGER.info("Wrote %s with %d rows", out, len(pairwise_rows))
    else:
        LOGGER.warning("No pairwise rows computed")


if __name__ == "__main__":
    main()
