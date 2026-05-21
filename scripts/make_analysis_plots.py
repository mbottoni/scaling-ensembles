#!/usr/bin/env python3
"""Generate publication-quality plots from analysis outputs.

Reads CSVs from outputs/series/analysis/ and writes PNGs to
outputs/series/paper-plots/.
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
    "figure.dpi": 180,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

ANALYSIS_DIR = Path("outputs/series/analysis")
PLOT_DIR = Path("outputs/series/paper-plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_LABELS = {
    16: "38k",
    32: "152k",
    64: "603k",
    128: "2.4M",
    256: "9.6M",
}

WIDTH_COLORS = {
    16: "#1f77b4",
    32: "#ff7f0e",
    64: "#2ca02c",
    128: "#d62728",
    256: "#9467bd",
}


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# ── Ensemble size scaling ────────────────────────────────────────────────────
def plot_ensemble_size_scaling() -> None:
    rows = read_csv(ANALYSIS_DIR / "ensemble_size_scaling.csv")
    by_width: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_width[int(r["width"])].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: absolute accuracy
    ax = axes[0]
    for width, data in sorted(by_width.items()):
        ms = [int(d["m"]) for d in data]
        means = [float(d["accuracy_mean"]) * 100 for d in data]
        stds = [float(d["accuracy_std"]) * 100 for d in data]
        color = WIDTH_COLORS.get(width, None)
        ax.plot(ms, means, marker="o", markersize=4, label=f"w={width} ({PARAM_LABELS.get(width,'?')} params)", color=color)
        ax.fill_between(ms, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.15, color=color)
    ax.set_xlabel("Ensemble size M")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ensemble accuracy vs. size (CIFAR-10)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: marginal gain from M vs M=1
    ax = axes[1]
    for width, data in sorted(by_width.items()):
        ms = [int(d["m"]) for d in data]
        means = [float(d["accuracy_mean"]) * 100 for d in data]
        baseline = means[0]
        gains = [m - baseline for m in means]
        color = WIDTH_COLORS.get(width, None)
        ax.plot(ms, gains, marker="o", markersize=4, label=f"w={width} ({PARAM_LABELS.get(width,'?')} params)", color=color)
    ax.set_xlabel("Ensemble size M")
    ax.set_ylabel("Gain over M=1 (pp)")
    ax.set_title("Ensemble gain vs. size (CIFAR-10)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOT_DIR / "ensemble_size_scaling.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


# ── Calibration ECE ──────────────────────────────────────────────────────────
def plot_calibration_comparison() -> None:
    datasets = {
        "FashionMNIST\n(easy)": "fashionmnist_calibration.csv",
        "CIFAR-10\n(hard)": "cifar10_calibration.csv",
    }
    secondary = {}
    for name, fname in [("STL-10\n(harder)", "stl10_calibration.csv"), ("SVHN", "svhn_calibration.csv")]:
        if (ANALYSIS_DIR / fname).exists():
            secondary[name] = fname

    all_datasets = {**datasets, **secondary}
    fig, axes = plt.subplots(1, len(all_datasets), figsize=(4 * len(all_datasets), 4), sharey=False)
    if len(all_datasets) == 1:
        axes = [axes]

    for ax, (dataset_name, fname) in zip(axes, all_datasets.items()):
        path = ANALYSIS_DIR / fname
        if not path.exists():
            continue
        rows = read_csv(path)
        widths = [int(r["width"]) for r in rows]
        params = [width * width for width in widths]  # rough proxy
        single_eces = [float(r["single_ece_mean"]) * 100 for r in rows]
        single_stds = [float(r["single_ece_std"]) * 100 for r in rows]
        ens_eces = [float(r["ensemble_ece_mean"]) * 100 for r in rows]
        ens_stds = [float(r["ensemble_ece_std"]) * 100 for r in rows]

        x = range(len(widths))
        width_labels = [f"w={w}\n{PARAM_LABELS.get(w,'?')}" for w in widths]

        ax.bar([xi - 0.2 for xi in x], single_eces, width=0.35, label="Single model", color="#1f77b4", alpha=0.8,
               yerr=single_stds, capsize=3)
        ax.bar([xi + 0.2 for xi in x], ens_eces, width=0.35, label="2-model ensemble", color="#ff7f0e", alpha=0.8,
               yerr=ens_stds, capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(width_labels, fontsize=8)
        ax.set_ylabel("ECE (%)")
        ax.set_title(f"Expected Calibration Error\n{dataset_name}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = PLOT_DIR / "calibration_ece.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


# ── Summary: diversity vs. task difficulty with CI ───────────────────────────
def plot_diversity_with_ci() -> None:
    summaries = {
        "FashionMNIST": ANALYSIS_DIR / "fashionmnist_summary.csv",
        "CIFAR-10": ANALYSIS_DIR / "cifar10_extended_summary.csv",
    }
    for name in ["stl10_summary.csv", "svhn_summary.csv"]:
        if (ANALYSIS_DIR / name).exists():
            key = name.replace("_summary.csv", "").upper().replace("-", "")
            summaries[key] = ANALYSIS_DIR / name

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for dataset_name, csv_path in summaries.items():
        if not csv_path.exists():
            continue
        rows = read_csv(csv_path)
        params = [int(r["parameter_count"]) for r in rows]
        disagree_means = [float(r["disagreement_mean"]) * 100 for r in rows]
        disagree_stds = [float(r["disagreement_std"]) * 100 for r in rows]
        ens_gain_means = [float(r["ensemble_gain_mean"]) * 100 for r in rows]
        ens_gain_stds = [float(r["ensemble_gain_std"]) * 100 for r in rows]
        barrier_means = [float(r["barrier_mean"]) if r["barrier_mean"] != "nan" else float("nan") for r in rows]

        ax = axes[0]
        line = ax.semilogx(params, disagree_means, marker="o", markersize=5, label=dataset_name)[0]
        ax.fill_between(params,
                        [m - s for m, s in zip(disagree_means, disagree_stds)],
                        [m + s for m, s in zip(disagree_means, disagree_stds)],
                        alpha=0.15, color=line.get_color())

        ax2 = axes[1]
        line2 = ax2.semilogx(params, ens_gain_means, marker="o", markersize=5, label=dataset_name)[0]
        ax2.fill_between(params,
                         [m - s for m, s in zip(ens_gain_means, ens_gain_stds)],
                         [m + s for m, s in zip(ens_gain_means, ens_gain_stds)],
                         alpha=0.15, color=line2.get_color())

        ax3 = axes[2]
        valid = [(p, b) for p, b in zip(params, barrier_means) if b == b]
        if valid:
            ps, bs = zip(*valid)
            ax3.semilogx(ps, bs, marker="o", markersize=5, label=dataset_name)

    for ax, title, ylabel in [
        (axes[0], "Prediction disagreement vs. width", "Pairwise disagreement (%)"),
        (axes[1], "Ensemble gain vs. width", "Ensemble gain over single (pp)"),
        (axes[2], "Loss landscape barrier vs. width", "Max interpolation loss barrier"),
    ]:
        ax.set_xlabel("# parameters")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOT_DIR / "diversity_with_ci.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


# ── Barrier vs. diversity scatter ────────────────────────────────────────────
def plot_barrier_vs_diversity() -> None:
    """Key finding: low barriers ≠ low diversity (STL-10 example)."""
    summaries = {}
    for fname, label in [
        ("fashionmnist_summary.csv", "FashionMNIST"),
        ("cifar10_extended_summary.csv", "CIFAR-10"),
        ("stl10_summary.csv", "STL-10"),
        ("svhn_summary.csv", "SVHN"),
    ]:
        if (ANALYSIS_DIR / fname).exists():
            summaries[label] = read_csv(ANALYSIS_DIR / fname)

    fig, ax = plt.subplots(figsize=(6, 5))
    for dataset_name, rows in summaries.items():
        barriers = [float(r["barrier_mean"]) for r in rows if r["barrier_mean"] != "nan"]
        disagree = [float(r["disagreement_mean"]) * 100 for r in rows if r["barrier_mean"] != "nan"]
        params = [int(r["parameter_count"]) for r in rows if r["barrier_mean"] != "nan"]
        sizes = [40 + 120 * (np.log10(p) - 4) / 2 for p in params]
        sc = ax.scatter(barriers, disagree, s=sizes, alpha=0.8, label=dataset_name)
        for b, d, p in zip(barriers, disagree, params):
            ax.annotate(f"{p/1e6:.1f}M" if p > 1e6 else f"{p/1e3:.0f}k",
                        (b, d), fontsize=7, xytext=(4, 2), textcoords="offset points")

    ax.set_xlabel("Mean max interpolation loss barrier")
    ax.set_ylabel("Pairwise disagreement (%)")
    ax.set_title("Loss barriers vs. functional diversity\n(marker size ∝ log(params))")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOT_DIR / "barrier_vs_diversity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    plot_ensemble_size_scaling()
    plot_calibration_comparison()
    plot_diversity_with_ci()
    plot_barrier_vs_diversity()
    LOGGER.info("All plots written to %s", PLOT_DIR)
