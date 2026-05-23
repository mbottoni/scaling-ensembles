#!/usr/bin/env python3
"""Ensemble gain vs model error rate scatter plot.

Shows that ensemble accuracy gain is a strong linear function of error rate
when error is driven by intrinsic task ambiguity (r=0.962, n=23), but
STL-10 (data-scarce, 5k train images) lies far below the regression line:
high error from underfitting produces correlated failures and low gain.

Reads:
  outputs/series/analysis/{mnist,fashionmnist,svhn,stl10}_summary.csv
  outputs/series/analysis/cifar10_extended_summary.csv
  outputs/series/cifar10-resnet-width-sweep/pairwise_similarity.csv
  outputs/series/analysis/cifar100_summary.csv  (if present)

Writes:
  outputs/series/paper-plots/gain_vs_error.png
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

DATASET_STYLES = {
    "MNIST":          {"color": "#2ca02c", "marker": "^", "size": 60, "label": "MNIST (MLP)"},
    "FashionMNIST":   {"color": "#1f77b4", "marker": "D", "size": 60, "label": "FashionMNIST"},
    "SVHN":           {"color": "#9467bd", "marker": "P", "size": 60, "label": "SVHN"},
    "STL10":          {"color": "#8c564b", "marker": "X", "size": 80, "label": "STL-10 (data-scarce)"},
    "CIFAR10-CNN":    {"color": "#d62728", "marker": "o", "size": 60, "label": "CIFAR-10 (CNN)"},
    "CIFAR10-ResNet": {"color": "#ff7f0e", "marker": "s", "size": 60, "label": "CIFAR-10 (ResNet)"},
    "CIFAR100":       {"color": "#e377c2", "marker": "h", "size": 60, "label": "CIFAR-100"},
}


def load_summary(path: Path, name: str) -> list[tuple[str, float, float, float]]:
    """Return (name, params, error, gain_pp) tuples."""
    result = []
    for row in csv.DictReader(path.open()):
        err = 1.0 - float(row["eval_accuracy_mean"])
        gain = float(row["ensemble_gain_mean"]) * 100.0
        params = float(row["parameter_count"])
        result.append((name, params, err, gain))
    return result


def load_resnet_pairwise(path: Path) -> list[tuple[str, float, float, float]]:
    by_width: dict[int, dict] = {}
    for row in csv.DictReader(path.open()):
        w = int(row["width"])
        gain = float(row["ensemble_accuracy"]) - (
            float(row["model_a_accuracy"]) + float(row["model_b_accuracy"])
        ) / 2
        acc = (float(row["model_a_accuracy"]) + float(row["model_b_accuracy"])) / 2
        if w not in by_width:
            by_width[w] = {"gains": [], "accs": [], "params": int(row["parameter_count"])}
        by_width[w]["gains"].append(gain * 100)
        by_width[w]["accs"].append(acc)
    result = []
    for w, d in sorted(by_width.items()):
        err = 1.0 - float(np.mean(d["accs"]))
        gain = float(np.mean(d["gains"]))
        result.append(("CIFAR10-ResNet", d["params"], err, gain))
    return result


def main() -> None:
    analysis = OUTPUT_ROOT / "analysis"
    all_points: list[tuple[str, float, float, float]] = []

    for name, fname in [
        ("MNIST", "mnist_summary.csv"),
        ("FashionMNIST", "fashionmnist_summary.csv"),
        ("SVHN", "svhn_summary.csv"),
        ("STL10", "stl10_summary.csv"),
    ]:
        p = analysis / fname
        if p.exists():
            all_points.extend(load_summary(p, name))

    cifar10_path = analysis / "cifar10_extended_summary.csv"
    if cifar10_path.exists():
        all_points.extend(load_summary(cifar10_path, "CIFAR10-CNN"))

    resnet_path = OUTPUT_ROOT / "cifar10-resnet-width-sweep" / "pairwise_similarity.csv"
    if resnet_path.exists():
        all_points.extend(load_resnet_pairwise(resnet_path))

    cifar100_path = analysis / "cifar100_summary.csv"
    if cifar100_path.exists():
        all_points.extend(load_summary(cifar100_path, "CIFAR100"))

    # Regression: exclude STL-10 (data-scarce outlier)
    regression_points = [p for p in all_points if p[0] != "STL10"]
    errors_reg = np.array([p[2] for p in regression_points])
    gains_reg = np.array([p[3] for p in regression_points])
    r, p_val = stats.pearsonr(errors_reg, gains_reg)
    slope, intercept, _, _, _ = stats.linregress(errors_reg, gains_reg)
    n_reg = len(regression_points)
    LOGGER.info("Regression (n=%d, excl. STL-10): r=%.3f p=%.2e", n_reg, r, p_val)
    LOGGER.info("gain ≈ %.2f × error + %.3f", slope, intercept)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Plot by dataset group
    plotted = set()
    for name, params, err, gain in all_points:
        style = DATASET_STYLES.get(name, {"color": "gray", "marker": ".", "size": 40, "label": name})
        label = style["label"] if name not in plotted else None
        ax.scatter(
            err * 100, gain,
            c=style["color"], marker=style["marker"],
            s=style["size"], alpha=0.85, label=label,
            edgecolors="white", linewidths=0.5,
            zorder=3 if name == "STL10" else 2,
        )
        plotted.add(name)

    # Draw regression line
    x_line = np.linspace(0, max(errors_reg) * 1.05, 80)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line * 100, y_line, "k--", alpha=0.55, linewidth=1.5,
        label=f"OLS (excl. STL-10, n={n_reg}): $r={r:.3f}$",
    )

    # Annotate STL-10 outlier
    stl10_pts = [(p[2], p[3]) for p in all_points if p[0] == "STL10"]
    if stl10_pts:
        avg_err = np.mean([p[0] for p in stl10_pts]) * 100
        avg_gain = np.mean([p[1] for p in stl10_pts])
        predicted = slope * avg_err / 100 + intercept
        ax.annotate(
            "STL-10: high error\nfrom data scarcity\n(5k train images)",
            xy=(avg_err, avg_gain),
            xytext=(avg_err - 12, avg_gain + 1.2),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#8c564b", lw=1.0),
            color="#8c564b",
        )

    ax.set_xlabel("Individual model error rate (%)")
    ax.set_ylabel("Ensemble accuracy gain (pp)")
    ax.set_title(
        "Ensemble gain scales with intrinsic task difficulty\n"
        r"(gain $\approx$ 14\% $\times$ error rate; $r=0.962$, excl. data-scarce STL-10)"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = PLOT_DIR / "gain_vs_error.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()
