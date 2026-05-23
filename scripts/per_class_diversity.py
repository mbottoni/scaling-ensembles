#!/usr/bin/env python3
"""Per-class diversity analysis for CIFAR-10.

Tests whether classes with higher error rates (harder classes) show more
pairwise disagreement — a fine-grained validation of the task-difficulty
hypothesis within a single dataset.

Reads cached logits from outputs/series/cifar10-cnn-extended-width-sweep/
Writes: outputs/series/analysis/cifar10_per_class_diversity.csv
        outputs/series/paper-plots/per_class_diversity.png
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


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

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
OUTPUT_ROOT = Path("outputs/series")
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
PLOT_DIR = OUTPUT_ROOT / "paper-plots"
CIFAR10_DIR = OUTPUT_ROOT / "cifar10-cnn-extended-width-sweep"

WIDTHS = [16, 32, 64, 128, 256]
SEEDS = list(range(10))
DIVERGED = {(256, 6)}
_MIN_ACC = 0.15


def load_cache(width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    path = CIFAR10_DIR / "cache/logits" / f"width_{width}_seed_{seed}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    logits, targets = data["logits"], data["targets"]
    probs = logits.softmax(dim=1)
    acc = (probs.argmax(dim=1) == targets).float().mean().item()
    if acc <= _MIN_ACC:
        return None
    return logits, targets


def compute_per_class_diversity(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    targets: torch.Tensor,
) -> dict[int, dict[str, float]]:
    probs_a = logits_a.softmax(dim=1)
    probs_b = logits_b.softmax(dim=1)
    preds_a = probs_a.argmax(dim=1)
    preds_b = probs_b.argmax(dim=1)

    results = {}
    for cls in range(10):
        mask = targets == cls
        if mask.sum() == 0:
            continue
        n = mask.sum().item()
        correct_a = (preds_a[mask] == targets[mask]).float()
        correct_b = (preds_b[mask] == targets[mask]).float()
        disagree = (preds_a[mask] != preds_b[mask]).float()
        results[cls] = {
            "error_rate_a": 1.0 - correct_a.mean().item(),
            "error_rate_b": 1.0 - correct_b.mean().item(),
            "disagreement": disagree.mean().item(),
            "n_examples": n,
        }
    return results


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect per-class diversity for each width
    width_class_stats: dict[int, dict[int, list[float]]] = {
        w: {cls: [] for cls in range(10)} for w in WIDTHS
    }
    width_class_error: dict[int, dict[int, list[float]]] = {
        w: {cls: [] for cls in range(10)} for w in WIDTHS
    }

    for width in WIDTHS:
        valid_seeds = [s for s in SEEDS if (width, s) not in DIVERGED]
        caches = {}
        for seed in valid_seeds:
            cached = load_cache(width, seed)
            if cached is not None:
                caches[seed] = cached

        seeds = list(caches.keys())
        n_pairs = 0
        for i, sa in enumerate(seeds):
            for sb in seeds[i + 1:]:
                logits_a, targets = caches[sa]
                logits_b, _ = caches[sb]
                per_cls = compute_per_class_diversity(logits_a, logits_b, targets)
                for cls, stats in per_cls.items():
                    width_class_stats[width][cls].append(stats["disagreement"])
                    width_class_error[width][cls].append(
                        0.5 * (stats["error_rate_a"] + stats["error_rate_b"])
                    )
                n_pairs += 1
        LOGGER.info("Width %d: %d pairs processed", width, n_pairs)

    # Write CSV
    rows = []
    for width in WIDTHS:
        for cls in range(10):
            disagree_vals = width_class_stats[width][cls]
            error_vals = width_class_error[width][cls]
            if not disagree_vals:
                continue
            rows.append({
                "width": width,
                "class": cls,
                "class_name": CIFAR10_CLASSES[cls],
                "disagreement_mean": np.mean(disagree_vals),
                "disagreement_std": np.std(disagree_vals),
                "error_rate_mean": np.mean(error_vals),
            })

    csv_path = ANALYSIS_DIR / "cifar10_per_class_diversity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s", csv_path)

    # Plot: per-class error rate vs disagreement (collapsed across widths)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: scatter of per-class error rate vs disagreement (all widths pooled)
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    by_class: dict[int, tuple[list, list]] = {cls: ([], []) for cls in range(10)}
    for r in rows:
        cls = int(r["class"])
        by_class[cls][0].append(float(r["error_rate_mean"]))
        by_class[cls][1].append(float(r["disagreement_mean"]))

    for cls in range(10):
        errs, disagrees = by_class[cls]
        ax.scatter(errs, disagrees, color=colors[cls], label=CIFAR10_CLASSES[cls],
                   s=60, alpha=0.85)

    # Fit and plot regression
    all_err = [float(r["error_rate_mean"]) for r in rows]
    all_dis = [float(r["disagreement_mean"]) for r in rows]
    corr = np.corrcoef(all_err, all_dis)[0, 1]
    z = np.polyfit(all_err, all_dis, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_err), max(all_err), 50)
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, label=f"r={corr:.2f}")
    ax.set_xlabel("Mean per-class error rate")
    ax.set_ylabel("Pairwise disagreement")
    ax.set_title("Per-class error rate vs. disagreement\n(CIFAR-10, all widths)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: per-class disagreement sorted by error rate, at two widths
    ax2 = axes[1]
    for width, color, marker in [(16, "#1f77b4", "o"), (128, "#d62728", "s")]:
        w_rows = [r for r in rows if int(r["width"]) == width]
        w_rows_sorted = sorted(w_rows, key=lambda r: float(r["error_rate_mean"]))
        names = [r["class_name"] for r in w_rows_sorted]
        disagrees = [float(r["disagreement_mean"]) * 100 for r in w_rows_sorted]
        x = np.arange(len(names))
        ax2.bar(x + (0.2 if width == 128 else -0.2), disagrees, width=0.35,
                label=f"w={width}", color=color, alpha=0.8)
    ax2.set_xticks(np.arange(10))
    ax2.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Pairwise disagreement (%)")
    ax2.set_title("Per-class disagreement (sorted by error rate)\nwidth 16 vs 128")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = PLOT_DIR / "per_class_diversity.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
