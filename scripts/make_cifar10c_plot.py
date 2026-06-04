#!/usr/bin/env python3
"""Severity-graded CIFAR-10-C figure: diversity and ensemble gain vs severity.

Reads:  outputs/series/analysis/cifar10c_by_severity.csv
Writes: outputs/series/paper-plots/cifar10c_severity.png
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

ANALYSIS = Path("outputs/series/analysis")
PLOT_DIR = Path("outputs/series/paper-plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {16: "#fdae61", 32: "#f46d43", 64: "#d73027", 128: "#a50026", 256: "#67001f"}


def main() -> None:
    path = ANALYSIS / "cifar10c_by_severity.csv"
    rows = list(csv.DictReader(path.open()))
    by_width: dict[int, list] = defaultdict(list)
    for r in rows:
        by_width[int(r["width"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for width in sorted(by_width):
        rs = sorted(by_width[width], key=lambda r: int(r["severity"]))
        sev = [int(r["severity"]) for r in rs]
        acc = [float(r["indiv_acc"]) * 100 for r in rs]
        dis = [float(r["disagreement"]) * 100 for r in rs]
        gain = [float(r["gain_pp"]) for r in rs]
        c = COLORS.get(width, "gray")
        params = {16: "38k", 32: "152k", 64: "603k", 128: "2.4M", 256: "9.6M"}.get(width, str(width))
        lbl = f"w={width} ({params})"
        axes[0].plot(sev, acc, "o-", color=c, label=lbl, markersize=4)
        axes[1].plot(sev, dis, "o-", color=c, label=lbl, markersize=4)
        axes[2].plot(sev, gain, "o-", color=c, label=lbl, markersize=4)

    axes[0].set_ylabel("Individual accuracy (%)")
    axes[0].set_title("Accuracy degrades with severity")
    axes[1].set_ylabel("Pairwise disagreement (%)")
    axes[1].set_title("Diversity rises with severity")
    axes[2].set_ylabel("2-member ensemble gain (pp)")
    axes[2].set_title("Ensemble gain rises with severity")
    for ax in axes:
        ax.set_xlabel("Corruption severity (0 = clean)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
    axes[0].legend()

    fig.tight_layout()
    out = PLOT_DIR / "cifar10c_severity.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()
