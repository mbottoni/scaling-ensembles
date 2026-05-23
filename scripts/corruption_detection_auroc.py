#!/usr/bin/env python3
"""Corruption detection AUROC analysis.

Uses CIFAR-10 CNN models trained on clean data to distinguish clean
CIFAR-10 test examples from Gaussian-corrupted ones, using predictive
entropy as the uncertainty score.

Reads:
  outputs/series/cifar10-cnn-extended-width-sweep/cache/logits/
  outputs/series/cifar10-cnn-gaussian-noise-eval/cache/logits/

Writes:
  outputs/series/analysis/corruption_detection_auroc.csv
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs/series")
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DIR = OUTPUT_ROOT / "cifar10-cnn-extended-width-sweep" / "cache/logits"
NOISY_DIR = OUTPUT_ROOT / "cifar10-cnn-gaussian-noise-eval" / "cache/logits"

WIDTHS = [16, 128, 256]
SEEDS = list(range(5))


def load_logits(cache_dir: Path, width: int, seed: int) -> torch.Tensor | None:
    path = cache_dir / f"width_{width}_seed_{seed}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["logits"]


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs + 1e-8).log()).sum(dim=1)


def compute_auroc(
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
) -> tuple[float, float]:
    """Returns (single_model_auroc, ensemble_auroc) for one width."""
    n_models = clean_logits.shape[0]

    # Single model: use seed 0
    single_clean_ent = predictive_entropy(clean_logits[0].softmax(dim=1))
    single_noisy_ent = predictive_entropy(noisy_logits[0].softmax(dim=1))
    all_ent = torch.cat([single_clean_ent, single_noisy_ent]).numpy()
    labels = np.concatenate([np.zeros(len(single_clean_ent)), np.ones(len(single_noisy_ent))])
    single_auroc = roc_auc_score(labels, all_ent)

    # Ensemble: average softmax over all available models
    ens_clean_probs = clean_logits.softmax(dim=2).mean(dim=0)
    ens_noisy_probs = noisy_logits.softmax(dim=2).mean(dim=0)
    ens_clean_ent = predictive_entropy(ens_clean_probs)
    ens_noisy_ent = predictive_entropy(ens_noisy_probs)
    all_ent_ens = torch.cat([ens_clean_ent, ens_noisy_ent]).numpy()
    ens_auroc = roc_auc_score(labels, all_ent_ens)

    return single_auroc, ens_auroc


def main() -> None:
    rows = []
    for width in WIDTHS:
        clean_logits_list = []
        noisy_logits_list = []
        for seed in SEEDS:
            c = load_logits(CLEAN_DIR, width, seed)
            n = load_logits(NOISY_DIR, width, seed)
            if c is not None and n is not None:
                clean_logits_list.append(c)
                noisy_logits_list.append(n)

        if len(clean_logits_list) < 2:
            LOGGER.warning("Width %d: too few seeds (%d)", width, len(clean_logits_list))
            continue

        clean_stack = torch.stack(clean_logits_list)
        noisy_stack = torch.stack(noisy_logits_list)

        # Accuracy check
        targets = torch.load(CLEAN_DIR / f"width_{width}_seed_0.pt", map_location="cpu",
                             weights_only=False)["targets"]
        clean_acc = (clean_stack[0].softmax(1).argmax(1) == targets).float().mean().item()

        noisy_targets = torch.load(NOISY_DIR / f"width_{width}_seed_0.pt", map_location="cpu",
                                   weights_only=False)["targets"]
        noisy_acc = (noisy_stack[0].softmax(1).argmax(1) == noisy_targets).float().mean().item()

        single_auroc, ens_auroc = compute_auroc(clean_stack, noisy_stack)

        rows.append({
            "width": width,
            "n_seeds": len(clean_logits_list),
            "clean_accuracy": clean_acc,
            "noisy_accuracy": noisy_acc,
            "single_model_auroc": single_auroc,
            "ensemble_auroc": ens_auroc,
            "auroc_gain": ens_auroc - single_auroc,
        })

        LOGGER.info(
            "Width %d: clean acc=%.1f%%, noisy acc=%.1f%%, "
            "single AUROC=%.4f, ens AUROC=%.4f (+%.4f)",
            width, clean_acc * 100, noisy_acc * 100,
            single_auroc, ens_auroc, ens_auroc - single_auroc,
        )

    if rows:
        out_path = ANALYSIS_DIR / "corruption_detection_auroc.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
