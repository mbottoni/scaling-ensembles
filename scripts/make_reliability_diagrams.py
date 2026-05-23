#!/usr/bin/env python3
"""Generate reliability diagrams and error-decomposition plots."""
from __future__ import annotations

import csv
import itertools
import logging
from collections import defaultdict
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

PLOT_DIR = Path("outputs/series/paper-plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 15
MIN_ACC = 0.15


def load_logits(output_dir: Path, width: int, seed: int):
    path = output_dir / "cache/logits" / f"width_{width}_seed_{seed}.pt"
    if not path.exists():
        return None, None
    cached = torch.load(path, map_location="cpu", weights_only=False)
    logits, targets = cached["logits"], cached["targets"]
    acc = (logits.softmax(1).argmax(1) == targets).float().mean().item()
    if acc <= MIN_ACC:
        return None, None
    return logits, targets


def reliability_diagram_data(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = N_BINS):
    confidences = probs.max(1).values
    preds = probs.argmax(1)
    correct = (preds == targets).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n = mask.sum().item()
        if n == 0:
            continue
        bin_centers.append(((lo + hi) / 2).item())
        bin_acc.append(correct[mask].mean().item())
        bin_conf.append(confidences[mask].mean().item())
        bin_counts.append(n)
    return bin_centers, bin_acc, bin_conf, bin_counts


def plot_reliability_diagrams() -> None:
    """Compare reliability curves for individual models and ensembles at two widths."""
    CIFAR10_EXT = Path("outputs/series/cifar10-cnn-extended-width-sweep")
    FMNIST = Path("outputs/series/dataset-difficulty-fashionmnist")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for row_idx, (exp_dir, dataset_name, widths, seeds_per_width) in enumerate([
        (FMNIST, "FashionMNIST", [16, 128], 5),
        (CIFAR10_EXT, "CIFAR-10", [16, 128], 10),
    ]):
        for col_idx, width in enumerate(widths):
            ax_single = axes[row_idx, col_idx * 2]
            ax_ens = axes[row_idx, col_idx * 2 + 1]

            # Collect valid models
            valid = []
            for seed in range(seeds_per_width):
                logits, targets = load_logits(exp_dir, width, seed)
                if logits is not None:
                    valid.append((logits, targets))
            if not valid:
                continue

            targets = valid[0][1]

            # Individual models
            all_single_acc, all_single_conf = [], []
            for logits, tgts in valid:
                probs = logits.softmax(1)
                centers, acc, conf, _ = reliability_diagram_data(probs, tgts)
                all_single_acc.append(acc)
                all_single_conf.append(conf)
            mean_acc = [np.mean([a[i] if i < len(a) else np.nan for a in all_single_acc]) for i in range(N_BINS)]
            mean_conf = [np.mean([c[i] if i < len(c) else np.nan for c in all_single_conf]) for i in range(N_BINS)]

            ax_single.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
            valid_pts = [(c, a) for c, a in zip(centers, mean_acc) if not np.isnan(a)]
            if valid_pts:
                cs, ms = zip(*valid_pts)
                ax_single.plot(cs, ms, "o-", color="#1f77b4", markersize=4, label="Single model")
            ax_single.set_xlim(0, 1)
            ax_single.set_ylim(0, 1)
            ax_single.set_xlabel("Confidence")
            ax_single.set_ylabel("Accuracy")
            ax_single.set_title(f"{dataset_name} w={width}\nSingle model")
            ax_single.legend(fontsize=7)
            ax_single.grid(True, alpha=0.3)

            # Ensemble (all valid models)
            all_logits = torch.stack([l for l, _ in valid])
            ens_probs = all_logits.softmax(2).mean(0)
            centers_e, acc_e, conf_e, _ = reliability_diagram_data(ens_probs, targets)

            ax_ens.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
            ax_ens.plot(centers_e, acc_e, "o-", color="#ff7f0e", markersize=4, label=f"Ensemble (M={len(valid)})")
            ax_ens.set_xlim(0, 1)
            ax_ens.set_ylim(0, 1)
            ax_ens.set_xlabel("Confidence")
            ax_ens.set_ylabel("Accuracy")
            ax_ens.set_title(f"{dataset_name} w={width}\nEnsemble M={len(valid)}")
            ax_ens.legend(fontsize=7)
            ax_ens.grid(True, alpha=0.3)

    fig.suptitle("Reliability Diagrams: Individual vs. Ensemble Calibration", fontsize=12)
    fig.tight_layout()
    path = PLOT_DIR / "reliability_diagrams.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


def plot_error_decomposition() -> None:
    """Decompose ensemble gain into error-rate and error-correlation components."""
    CIFAR10_EXT = Path("outputs/series/cifar10-cnn-extended-width-sweep")
    widths = [16, 32, 64, 128, 256]
    seeds_per_width = 10

    decomp: dict[int, dict] = {}
    for width in widths:
        valid = []
        for seed in range(seeds_per_width):
            logits, targets = load_logits(CIFAR10_EXT, width, seed)
            if logits is not None:
                valid.append((logits, targets))
        if len(valid) < 2:
            continue
        targets = valid[0][1]

        error_rates = [1 - (l.softmax(1).argmax(1) == t).float().mean().item() for l, t in valid]
        mean_error = np.mean(error_rates)

        # Pairwise error covariance: E[err_a * err_b] - E[err_a] * E[err_b]
        pair_covs = []
        for (la, ta), (lb, tb) in itertools.combinations(valid, 2):
            wrong_a = (la.softmax(1).argmax(1) != ta).float()
            wrong_b = (lb.softmax(1).argmax(1) != tb).float()
            cov = (wrong_a * wrong_b).mean().item() - wrong_a.mean().item() * wrong_b.mean().item()
            pair_covs.append(cov)

        decomp[width] = {
            "mean_error": mean_error,
            "error_covariance": np.mean(pair_covs),
            "n_models": len(valid),
        }

    params = [38570, 151882, 602762, 2401546, 9587210]
    widths_valid = [w for w in widths if w in decomp]
    ps = [params[widths.index(w)] for w in widths_valid]
    mean_errs = [decomp[w]["mean_error"] * 100 for w in widths_valid]
    covs = [decomp[w]["error_covariance"] * 100 for w in widths_valid]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.semilogx(ps, mean_errs, "o-", color="#1f77b4", markersize=6, label="Mean individual error rate")
    ax.set_xlabel("# parameters")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Individual error rate vs. width")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogx(ps, covs, "o-", color="#d62728", markersize=6, label="Mean pairwise error covariance")
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.set_xlabel("# parameters")
    ax2.set_ylabel("Error covariance (pp²)")
    ax2.set_title("Pairwise error covariance vs. width\n(CIFAR-10)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Error decomposition: individual error rate and pairwise covariance", fontsize=11)
    fig.tight_layout()
    path = PLOT_DIR / "error_decomposition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)

    # Print table
    print("\nError Decomposition (CIFAR-10):")
    print(f"{'Width':>6} {'Params':>10} {'MeanErr':>10} {'PairwiseCov':>14}")
    for w in widths_valid:
        p = params[widths.index(w)]
        print(f"{w:>6} {p:>10,} {decomp[w]['mean_error']*100:>9.3f}% {decomp[w]['error_covariance']*100:>13.4f}pp²")


if __name__ == "__main__":
    plot_reliability_diagrams()
    plot_error_decomposition()
    LOGGER.info("Done")
