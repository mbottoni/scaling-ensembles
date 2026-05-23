#!/usr/bin/env python3
"""Regenerate diversity_accuracy_plane.png with all datasets including ResNet.

Reads:
  outputs/series/cifar10-cnn-extended-width-sweep/pairwise_similarity.csv
  outputs/series/cifar10-resnet-width-sweep/pairwise_similarity.csv
  outputs/series/analysis/*_summary.csv

Writes:
  outputs/series/paper-plots/diversity_accuracy_plane.png
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

PAIRWISE_CONFIGS = {
    "cifar10-cnn-extended-width-sweep": {
        "label": "CIFAR-10 (CNN)",
        "color": "#d62728",
        "marker": "o",
    },
    "cifar10-resnet-width-sweep": {
        "label": "CIFAR-10 (ResNet)",
        "color": "#ff7f0e",
        "marker": "s",
    },
}

SUMMARY_CONFIGS = {
    "MNIST (MLP)": (OUTPUT_ROOT / "analysis/mnist_summary.csv", "#2ca02c", "^"),
    "FashionMNIST": (OUTPUT_ROOT / "analysis/fashionmnist_summary.csv", "#1f77b4", "D"),
    "SVHN": (OUTPUT_ROOT / "analysis/svhn_summary.csv", "#9467bd", "P"),
    "STL-10": (OUTPUT_ROOT / "analysis/stl10_summary.csv", "#8c564b", "X"),
}


def load_pairwise(exp_dir: Path) -> list[tuple[float, float, int]]:
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
        result.append((acc, disagree, params, width))
    return result


def load_summary(csv_path: Path) -> list[tuple[float, float, int, int]]:
    if not csv_path.exists():
        return []
    result = []
    for r in csv.DictReader(csv_path.open()):
        acc = float(r["eval_accuracy_mean"])
        disagree = float(r["disagreement_mean"])
        params = int(r["parameter_count"])
        width = int(r["width"])
        result.append((acc, disagree, params, width))
    return result


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    all_params = []
    all_points = []

    for exp_name, cfg in PAIRWISE_CONFIGS.items():
        points = load_pairwise(OUTPUT_ROOT / exp_name)
        if not points:
            LOGGER.warning("No pairwise data for %s", exp_name)
            continue
        all_points.extend([(p[2], exp_name) for p in points])
        all_params.extend([p[2] for p in points])

    for label, (csv_path, color, marker) in SUMMARY_CONFIGS.items():
        points = load_summary(csv_path)
        all_params.extend([p[2] for p in points])

    log_min = np.log10(min(all_params)) if all_params else 4
    log_max = np.log10(max(all_params)) if all_params else 7

    def param_to_size(p: int) -> float:
        return 40 + 200 * (np.log10(p) - log_min) / (log_max - log_min + 1e-9)

    # Plot pairwise data
    for exp_name, cfg in PAIRWISE_CONFIGS.items():
        points = load_pairwise(OUTPUT_ROOT / exp_name)
        if not points:
            continue
        accs = [p[0] * 100 for p in points]
        disagrees = [p[1] * 100 for p in points]
        params = [p[2] for p in points]
        widths = [p[3] for p in points]
        sizes = [param_to_size(p) for p in params]
        ax.scatter(
            disagrees, accs, c=cfg["color"], marker=cfg["marker"],
            s=sizes, alpha=0.85, label=cfg["label"], edgecolors="white", linewidths=0.5,
            zorder=3,
        )
        for d, a, p, w in zip(disagrees, accs, params, widths):
            ax.annotate(f"w={w}", (d, a), fontsize=7, xytext=(3, 2),
                        textcoords="offset points", color=cfg["color"], alpha=0.8)

    # Plot summary data
    for label, (csv_path, color, marker) in SUMMARY_CONFIGS.items():
        points = load_summary(csv_path)
        if not points:
            continue
        accs = [p[0] * 100 for p in points]
        disagrees = [p[1] * 100 for p in points]
        params = [p[2] for p in points]
        sizes = [param_to_size(p) for p in params]
        ax.scatter(
            disagrees, accs, c=color, marker=marker,
            s=sizes, alpha=0.85, label=label, edgecolors="white", linewidths=0.5,
            zorder=3,
        )

    # Size legend
    for log_val, label in [(4.5, "30k"), (5.5, "300k"), (6.5, "3M")]:
        p = 10 ** log_val
        ax.scatter([], [], c="gray", s=param_to_size(p), alpha=0.6,
                   label=f"{label} params", edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Pairwise disagreement (%)")
    ax.set_ylabel("Individual model accuracy (%)")
    ax.set_title("Diversity--accuracy plane\n(all datasets, architectures, and widths)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Annotate quadrants
    ax.axvline(15, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.axhline(85, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(2, 88, "Easy tasks\n(low diversity)", fontsize=8, color="gray", alpha=0.7)
    ax.text(22, 60, "Hard tasks\n(high diversity)", fontsize=8, color="gray", alpha=0.7)

    fig.tight_layout()
    path = PLOT_DIR / "diversity_accuracy_plane.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
