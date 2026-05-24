#!/usr/bin/env python3
"""Plot p_corr vs p_same (correlated failure rate) across datasets.

p_same = P(both models predict same wrong class) = P(agree) - P(both right)
p_corr = P(ensemble right | models disagree) = gain/disagree + 0.5

Key finding: p_same is the mechanistic driver of p_corr variation across
datasets. MNIST has p_same=1% and p_corr=0.69; STL-10 has p_same=21% and
p_corr=0.57. The decomposition gain = dis × (p_corr - 0.5) with both
components explains all low-gain cases.

Reads:
  outputs/series/*/pairwise_similarity.csv  (with both_wrong column)

Writes:
  outputs/series/paper-plots/pcorr_psame.png
"""
from __future__ import annotations

import csv
import logging
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

EXPERIMENT_LABELS = {
    "mnist-mlp-extended-width-sweep": ("MNIST (MLP)", "#2ca02c", "^", 80),
    "dataset-difficulty-fashionmnist": ("FashionMNIST", "#1f77b4", "D", 70),
    "svhn-cnn-width-sweep": ("SVHN", "#9467bd", "P", 70),
    "cifar10-cnn-extended-width-sweep": ("CIFAR-10 (CNN)", "#d62728", "o", 70),
    "cifar10-resnet-width-sweep": ("CIFAR-10 (ResNet)", "#ff7f0e", "s", 70),
    "stl10-cnn-width-sweep": ("STL-10 (data-scarce)", "#8c564b", "X", 100),
}

CORRUPT_LABELS = {
    "cifar10-cnn-gaussian-noise-eval": ("CIFAR-10 + Gaussian noise", "#aec7e8", "v", 60),
    "cifar10-cnn-blur-eval": ("CIFAR-10 + Blur", "#ffbb78", "^", 60),
}


def load_pairs(path: Path) -> tuple[float, float, float, float] | None:
    rows = list(csv.DictReader(path.open()))
    if not rows or "both_wrong" not in rows[0]:
        return None
    dis = np.mean([float(r["disagreement"]) for r in rows])
    bw = np.mean([float(r["both_wrong"]) for r in rows])
    avg_acc = np.mean([(float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) / 2 for r in rows])
    gain = np.mean([float(r["ensemble_accuracy"]) - (float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) / 2 for r in rows])

    if dis < 0.01:
        return None
    p_corr = gain / dis + 0.5
    # p_same = P(agree) - P(both right) = P(both wrong AND same class)
    # P(both right) ≈ 2(1-ε) - (1-P(both wrong)) = P(A right) + P(B right) - P(A or B right)
    # P(A or B right) = 1 - P(both wrong)
    err = 1 - avg_acc
    p_both_right = (1 - err) + (1 - err) - (1 - bw)
    agree = np.mean([float(r["agreement"]) for r in rows])
    p_same = max(0, agree - p_both_right)
    return dis, p_corr, p_same, gain


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: p_corr vs p_same (main scatter)
    ax = axes[0]
    all_psame = []
    all_pcorr = []

    for exp_dir, (label, color, marker, size) in EXPERIMENT_LABELS.items():
        sim_path = OUTPUT_ROOT / exp_dir / "pairwise_similarity.csv"
        if not sim_path.exists():
            continue
        result = load_pairs(sim_path)
        if result is None:
            continue
        dis, p_corr, p_same, gain = result
        ax.scatter(p_same * 100, p_corr, c=color, marker=marker, s=size,
                   alpha=0.9, label=label, edgecolors="white", linewidths=0.5, zorder=3)
        all_psame.append(p_same)
        all_pcorr.append(p_corr)

    # Add corruption experiments (lighter / outlined)
    for exp_dir, (label, color, marker, size) in CORRUPT_LABELS.items():
        sim_path = OUTPUT_ROOT / exp_dir / "pairwise_similarity.csv"
        if not sim_path.exists():
            continue
        result = load_pairs(sim_path)
        if result is None:
            continue
        dis, p_corr, p_same, gain = result
        ax.scatter(p_same * 100, p_corr, c=color, marker=marker, s=size,
                   alpha=0.7, label=label, edgecolors="gray", linewidths=0.8, zorder=2)
        all_psame.append(p_same)
        all_pcorr.append(p_corr)

    # Fit line
    if len(all_psame) >= 3:
        slope, intercept, r, p_val, _ = stats.linregress(all_psame, all_pcorr)
        x_line = np.linspace(0, max(all_psame) * 1.05, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line * 100, y_line, "k--", alpha=0.5, linewidth=1.4,
                label=f"OLS: $r={r:.3f}$, $p={p_val:.3f}$")
        LOGGER.info("p_corr vs p_same regression: r=%.3f, p=%.4f", r, p_val)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance ($p_{\\rm corr}=0.5$)")
    ax.set_xlabel(r"$p_{\rm same}$ (\% of examples: both wrong, same class)")
    ax.set_ylabel(r"$p_{\rm corr} = P(\text{ens right}\mid\text{disagree})$")
    ax.set_title("Correlated failure rate drives $p_{\\rm corr}$\n(and therefore ensemble gain)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Panel 2: gain decomposition (dis × (p_corr - 0.5))
    ax = axes[1]
    for exp_dir, (label, color, marker, size) in EXPERIMENT_LABELS.items():
        sim_path = OUTPUT_ROOT / exp_dir / "pairwise_similarity.csv"
        if not sim_path.exists():
            continue
        result = load_pairs(sim_path)
        if result is None:
            continue
        dis, p_corr, p_same, gain = result
        predicted_gain = dis * (p_corr - 0.5)
        ax.scatter(gain * 100, predicted_gain * 100, c=color, marker=marker, s=size,
                   alpha=0.9, label=label, edgecolors="white", linewidths=0.5, zorder=3)

    ax.plot([0, 6], [0, 6], "k-", alpha=0.3, linewidth=1, label="$y=x$")
    ax.set_xlabel("Observed gain (pp)")
    ax.set_ylabel(r"$\mathrm{dis} \times (p_{\rm corr} - 0.5)$ (pp)")
    ax.set_title("Gain formula validation:\ngain $=$ dis $\\times$ $(p_{\\rm corr}-\\frac{1}{2})$")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = PLOT_DIR / "pcorr_psame.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()
