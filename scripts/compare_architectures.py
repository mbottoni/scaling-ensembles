#!/usr/bin/env python3
"""Compare functional diversity between CNN and ResNet on CIFAR-10.

Reads from:
  outputs/series/cifar10-cnn-extended-width-sweep/
  outputs/series/cifar10-resnet-width-sweep/

Produces:
  outputs/series/analysis/architecture_comparison.csv
  outputs/series/paper-plots/architecture_comparison.png
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
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
PLOT_DIR = OUTPUT_ROOT / "paper-plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

_MIN_ACC = 0.15


def load_pairwise(path: Path, diverged: set[tuple[int, int]] | None = None) -> dict[int, list[dict]]:
    if not path.exists():
        return {}
    diverged = diverged or set()
    rows = list(csv.DictReader(path.open()))
    by_width: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        w, sa, sb = int(r["width"]), int(r["seed_a"]), int(r["seed_b"])
        if (w, sa) not in diverged and (w, sb) not in diverged:
            by_width[w].append(r)
    return by_width


def load_train(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    diverged = set()
    for r in csv.DictReader(path.open()):
        if float(r["eval_accuracy"]) <= _MIN_ACC:
            diverged.add((int(r["width"]), int(r["seed"])))
    return diverged


def summarize(pairwise_by_width: dict[int, list[dict]], train_path: Path) -> list[dict]:
    train_by_width: dict[int, list[dict]] = defaultdict(list)
    if train_path.exists():
        for r in csv.DictReader(train_path.open()):
            if float(r["eval_accuracy"]) > _MIN_ACC:
                train_by_width[int(r["width"])].append(r)

    summary = []
    for width in sorted(pairwise_by_width):
        pw = pairwise_by_width[width]
        tr = train_by_width.get(width, [])
        if not pw:
            continue
        disagree = np.array([1 - float(r["agreement"]) for r in pw])
        single_acc = np.array([0.5 * (float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) for r in pw])
        ens_acc = np.array([float(r["ensemble_accuracy"]) for r in pw])
        gain = ens_acc - single_acc
        eval_acc = np.array([float(r["eval_accuracy"]) for r in tr]) if tr else np.array([np.nan])
        summary.append({
            "width": width,
            "parameter_count": int(pw[0]["parameter_count"]) if pw else 0,
            "n_pairs": len(pw),
            "eval_accuracy_mean": float(eval_acc.mean()),
            "disagreement_mean": float(disagree.mean()),
            "disagreement_std": float(disagree.std()),
            "ensemble_gain_mean": float(gain.mean()),
            "ensemble_gain_std": float(gain.std()),
        })
    return summary


def main() -> None:
    CNN_DIR = OUTPUT_ROOT / "cifar10-cnn-extended-width-sweep"
    RESNET_DIR = OUTPUT_ROOT / "cifar10-resnet-width-sweep"

    cnn_div = load_train(CNN_DIR / "train_results.csv")
    cnn_pw = load_pairwise(CNN_DIR / "pairwise_similarity.csv", cnn_div)
    cnn_summary = summarize(cnn_pw, CNN_DIR / "train_results.csv")

    resnet_div = load_train(RESNET_DIR / "train_results.csv")
    resnet_pw = load_pairwise(RESNET_DIR / "pairwise_similarity.csv", resnet_div)
    resnet_summary = summarize(resnet_pw, RESNET_DIR / "train_results.csv")

    if not cnn_summary:
        LOGGER.warning("No CNN summary data found")
        return
    if not resnet_summary:
        LOGGER.warning("No ResNet summary data found — training may still be in progress")

    # Save CSV
    all_rows = [{"arch": "CNN", **r} for r in cnn_summary] + \
               [{"arch": "ResNet", **r} for r in resnet_summary]
    if all_rows:
        out_path = ANALYSIS_DIR / "architecture_comparison.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        LOGGER.info("Wrote %s", out_path)

    if not resnet_summary:
        LOGGER.info("Skipping plots — ResNet data not yet available")
        return

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for summary, label, color, marker in [
        (cnn_summary, "SmallCNN", "#1f77b4", "o"),
        (resnet_summary, "ResNet", "#d62728", "s"),
    ]:
        params = [int(r["parameter_count"]) for r in summary]
        disagree_mean = [float(r["disagreement_mean"]) * 100 for r in summary]
        disagree_std = [float(r["disagreement_std"]) * 100 for r in summary]
        gain_mean = [float(r["ensemble_gain_mean"]) * 100 for r in summary]
        gain_std = [float(r["ensemble_gain_std"]) * 100 for r in summary]
        acc = [float(r["eval_accuracy_mean"]) * 100 for r in summary]

        axes[0].semilogx(params, disagree_mean, marker=marker, markersize=6, label=label, color=color)
        axes[0].fill_between(params,
                              [m - s for m, s in zip(disagree_mean, disagree_std)],
                              [m + s for m, s in zip(disagree_mean, disagree_std)],
                              alpha=0.15, color=color)

        axes[1].semilogx(params, gain_mean, marker=marker, markersize=6, label=label, color=color)
        axes[1].fill_between(params,
                              [m - s for m, s in zip(gain_mean, gain_std)],
                              [m + s for m, s in zip(gain_mean, gain_std)],
                              alpha=0.15, color=color)

        axes[2].semilogx(params, acc, marker=marker, markersize=6, label=label, color=color)

    for ax, title, ylabel in [
        (axes[0], "Prediction disagreement", "Disagreement (%)"),
        (axes[1], "Ensemble gain", "Gain over single model (pp)"),
        (axes[2], "Individual accuracy", "Eval accuracy (%)"),
    ]:
        ax.set_xlabel("# parameters")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n(CIFAR-10)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Architecture comparison: SmallCNN vs. ResNet on CIFAR-10", fontsize=11)
    fig.tight_layout()
    plot_path = PLOT_DIR / "architecture_comparison.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
