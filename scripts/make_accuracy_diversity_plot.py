#!/usr/bin/env python3
"""Create accuracy-vs-disagreement scatter plot across all architectures/datasets.

This is the key summary figure showing that accuracy (difficulty proxy)
governs disagreement, not architecture or parameter count.

Reads:
  outputs/series/*/pairwise_similarity.csv
  outputs/series/analysis/*_summary.csv

Writes:
  outputs/series/paper-plots/accuracy_vs_disagreement.png
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
PLOT_DIR = OUTPUT_ROOT / "paper-plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MIN_ACC = 0.15

DATASET_CONFIG = {
    "cifar10-cnn-extended-width-sweep": {
        "label": "CIFAR-10 (CNN)",
        "color": "#d62728",
        "marker": "o",
        "size": 80,
    },
    "cifar10-resnet-width-sweep": {
        "label": "CIFAR-10 (ResNet)",
        "color": "#ff7f0e",
        "marker": "s",
        "size": 80,
    },
}

SUMMARY_FILES = {
    "MNIST (MLP)": (OUTPUT_ROOT / "analysis/mnist_summary.csv", "#2ca02c", "^", 60),
    "FashionMNIST": (OUTPUT_ROOT / "analysis/fashionmnist_summary.csv", "#1f77b4", "D", 60),
    "SVHN": (OUTPUT_ROOT / "analysis/svhn_summary.csv", "#9467bd", "P", 60),
    "STL-10": (OUTPUT_ROOT / "analysis/stl10_summary.csv", "#8c564b", "X", 60),
}


def load_pairwise(exp_dir: Path) -> list[tuple[float, float, int]]:
    """Returns (accuracy, disagreement, params) tuples per width."""
    path = exp_dir / "pairwise_similarity.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))
    by_width: dict[int, list] = defaultdict(list)
    for r in rows:
        if float(r["model_a_accuracy"]) > MIN_ACC and float(r["model_b_accuracy"]) > MIN_ACC:
            by_width[int(r["width"])].append(r)

    result = []
    for width, pw in by_width.items():
        acc = np.mean([(float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) / 2 for r in pw])
        disagree = 1 - np.mean([float(r["agreement"]) for r in pw])
        params = int(pw[0]["parameter_count"])
        result.append((acc, disagree, params))
    return result


def load_summary(csv_path: Path) -> list[tuple[float, float, int]]:
    """Returns (accuracy, disagreement, params) tuples per width from summary CSV."""
    if not csv_path.exists():
        return []
    result = []
    for r in csv.DictReader(csv_path.open()):
        acc = float(r["eval_accuracy_mean"])
        disagree = float(r["disagreement_mean"])
        params = int(r["parameter_count"])
        result.append((acc, disagree, params))
    return result


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    all_acc, all_disagree = [], []

    # Plot pairwise data (CNN + ResNet on CIFAR-10)
    for exp_name, cfg in DATASET_CONFIG.items():
        points = load_pairwise(OUTPUT_ROOT / exp_name)
        if not points:
            LOGGER.warning("No pairwise data for %s", exp_name)
            continue
        accs = [p[0] for p in points]
        disagrees = [p[1] * 100 for p in points]
        params = [p[2] for p in points]
        sizes = [50 + 80 * (np.log10(p) - 4) for p in params]

        sc = ax.scatter(
            accs, disagrees, c=cfg["color"], marker=cfg["marker"],
            s=sizes, alpha=0.85, label=cfg["label"], edgecolors="white", linewidths=0.5,
        )
        # Annotate with parameter counts
        for a, d, p in zip(accs, disagrees, params):
            label = f"{p/1e6:.1f}M" if p >= 1e6 else f"{p/1e3:.0f}k"
            ax.annotate(label, (a, d), fontsize=7, xytext=(4, 2),
                        textcoords="offset points", color=cfg["color"], alpha=0.7)
        all_acc.extend(accs)
        all_disagree.extend([d / 100 for d in disagrees])

    # Plot summary data for other datasets
    for label, (csv_path, color, marker, size) in SUMMARY_FILES.items():
        points = load_summary(csv_path)
        if not points:
            continue
        accs = [p[0] for p in points]
        disagrees = [p[1] * 100 for p in points]
        params = [p[2] for p in points]
        # Use smaller markers for summary datasets (fewer seeds)
        ax.scatter(
            accs, disagrees, c=color, marker=marker,
            s=size, alpha=0.85, label=label, edgecolors="white", linewidths=0.5,
        )
        all_acc.extend(accs)
        all_disagree.extend([d / 100 for d in disagrees])

    # Fit and plot regression line
    if all_acc:
        errors = [1 - a for a in all_acc]
        disagrees_pct = [d * 100 for d in all_disagree]
        r, p_val = stats.pearsonr(errors, disagrees_pct)
        z = np.polyfit(errors, disagrees_pct, 1)
        p_poly = np.poly1d(z)
        x_line = np.linspace(min(errors), max(errors), 50)
        ax.plot([1 - x for x in x_line], p_poly(x_line),
                "k--", alpha=0.4, linewidth=1.5, label=f"OLS fit: r = {r:.3f} (n={len(all_acc)})")

    # Theoretical curve under independence: 2ε(1-ε)
    eps = np.linspace(0.01, 0.40, 100)
    ax.plot(1 - eps, 2 * eps * (1 - eps) * 100, "k:", alpha=0.3, linewidth=1,
            label=r"Theory: $2\varepsilon(1-\varepsilon)$")

    ax.set_xlabel("Individual model accuracy")
    ax.set_ylabel("Pairwise disagreement (%)")
    ax.set_title("Accuracy governs disagreement\nacross datasets, architectures, and widths")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Lower accuracy = harder task = more disagreement (right side)

    fig.tight_layout()
    path = PLOT_DIR / "accuracy_vs_disagreement.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
