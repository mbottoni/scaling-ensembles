#!/usr/bin/env python3
"""Corruption detection AUROC analysis.

Uses CIFAR-10 CNN models trained on clean data to distinguish clean
CIFAR-10 test examples from corrupted ones (Gaussian noise and Gaussian blur),
using predictive entropy as the uncertainty score.

Reads:
  outputs/series/cifar10-cnn-extended-width-sweep/cache/logits/
  outputs/series/cifar10-cnn-gaussian-noise-eval/cache/logits/
  outputs/series/cifar10-cnn-blur-eval/cache/logits/

Writes:
  outputs/series/analysis/corruption_detection_auroc.csv
  outputs/series/paper-plots/corruption_auroc.png
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs/series")
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
PLOT_DIR = OUTPUT_ROOT / "paper-plots"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DIR = OUTPUT_ROOT / "cifar10-cnn-extended-width-sweep" / "cache/logits"

CORRUPTION_DIRS = {
    "gaussian": OUTPUT_ROOT / "cifar10-cnn-gaussian-noise-eval" / "cache/logits",
    "blur": OUTPUT_ROOT / "cifar10-cnn-blur-eval" / "cache/logits",
}

WIDTHS = [16, 32, 64, 128, 256]
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
    # Store results for plotting: corruption_type -> width -> (single, ens)
    plot_data: dict[str, dict[int, tuple[float, float]]] = {c: {} for c in CORRUPTION_DIRS}

    for corruption_name, noisy_dir in CORRUPTION_DIRS.items():
        if not noisy_dir.exists():
            LOGGER.warning("Skipping %s (directory not found)", corruption_name)
            continue

        for width in WIDTHS:
            clean_logits_list = []
            noisy_logits_list = []
            for seed in SEEDS:
                c = load_logits(CLEAN_DIR, width, seed)
                n = load_logits(noisy_dir, width, seed)
                if c is not None and n is not None:
                    clean_logits_list.append(c)
                    noisy_logits_list.append(n)

            if len(clean_logits_list) < 2:
                LOGGER.warning("%s width %d: too few seeds (%d)", corruption_name, width, len(clean_logits_list))
                continue

            clean_stack = torch.stack(clean_logits_list)
            noisy_stack = torch.stack(noisy_logits_list)

            targets = torch.load(CLEAN_DIR / f"width_{width}_seed_0.pt", map_location="cpu",
                                 weights_only=False)["targets"]
            clean_acc = (clean_stack[0].softmax(1).argmax(1) == targets).float().mean().item()

            noisy_targets = torch.load(noisy_dir / f"width_{width}_seed_0.pt", map_location="cpu",
                                       weights_only=False)["targets"]
            noisy_acc = (noisy_stack[0].softmax(1).argmax(1) == noisy_targets).float().mean().item()

            single_auroc, ens_auroc = compute_auroc(clean_stack, noisy_stack)
            plot_data[corruption_name][width] = (single_auroc, ens_auroc)

            rows.append({
                "corruption": corruption_name,
                "width": width,
                "n_seeds": len(clean_logits_list),
                "clean_accuracy": clean_acc,
                "noisy_accuracy": noisy_acc,
                "single_model_auroc": single_auroc,
                "ensemble_auroc": ens_auroc,
                "auroc_gain": ens_auroc - single_auroc,
            })

            LOGGER.info(
                "%s width %d: clean acc=%.1f%%, noisy acc=%.1f%%, "
                "single AUROC=%.4f, ens AUROC=%.4f (+%.4f)",
                corruption_name, width, clean_acc * 100, noisy_acc * 100,
                single_auroc, ens_auroc, ens_auroc - single_auroc,
            )

    if rows:
        out_path = ANALYSIS_DIR / "corruption_detection_auroc.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %s", out_path)

    # Plot
    PARAM_COUNTS = {16: 38600, 32: 151900, 64: 602800, 128: 2402000, 256: 9587000}
    corruption_colors = {"gaussian": "#d62728", "blur": "#1f77b4"}
    corruption_labels = {"gaussian": "Gaussian noise", "blur": "Gaussian blur"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax_idx, (corruption_name, width_data) in enumerate(plot_data.items()):
        if not width_data:
            continue
        color = corruption_colors.get(corruption_name, "gray")
        label = corruption_labels.get(corruption_name, corruption_name)
        widths = sorted(width_data.keys())
        params = [PARAM_COUNTS.get(w, w) for w in widths]
        single_aurocs = [width_data[w][0] for w in widths]
        ens_aurocs = [width_data[w][1] for w in widths]

        for ax in axes:
            ax.semilogx(params, single_aurocs, "o--", color=color, markersize=5,
                        label=f"{label} (single)", alpha=0.7)
            ax.semilogx(params, ens_aurocs, "s-", color=color, markersize=6,
                        label=f"{label} (ensemble)")

    for ax in axes:
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
        ax.set_xlabel("# parameters")
        ax.set_ylabel("AUROC")
        ax.set_title("Corruption detection AUROC\n(clean vs. corrupted CIFAR-10)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 1.0)
        break  # only one panel

    # Right panel: AUROC gain (ensemble - single)
    ax2 = axes[1]
    for corruption_name, width_data in plot_data.items():
        if not width_data:
            continue
        color = corruption_colors.get(corruption_name, "gray")
        label = corruption_labels.get(corruption_name, corruption_name)
        widths = sorted(width_data.keys())
        params = [PARAM_COUNTS.get(w, w) for w in widths]
        gains = [width_data[w][1] - width_data[w][0] for w in widths]
        ax2.semilogx(params, gains, "o-", color=color, markersize=5, label=label)

    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("# parameters")
    ax2.set_ylabel("AUROC gain (ensemble − single)")
    ax2.set_title("Ensemble benefit for corruption detection\nvs. model capacity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = PLOT_DIR / "corruption_auroc.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
