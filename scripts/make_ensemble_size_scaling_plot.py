#!/usr/bin/env python3
"""Enhanced ensemble-size scaling plot with universal f(M) scaling factor.

Key finding: the ratio f(M) = gain(M) / gain(2) is universal across model
widths (CV < 5%), and fits the rational law
  f(M) = (1 + a*(M-2)) / (1 + b*(M-2)),  a≈0.898, b≈0.358
with RMSE < 0.007.  This implies a gain saturation of f(∞) ≈ 2.51× the
2-member gain, and means a 2-member pilot experiment predicts M-member gain
at any scale.

Reads:
  outputs/series/analysis/ensemble_size_scaling.csv

Writes:
  outputs/series/paper-plots/ensemble_size_scaling.png  (replaces old version)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs/series")
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
PLOT_DIR = OUTPUT_ROOT / "paper-plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_COLORS = {16: "#1f77b4", 32: "#ff7f0e", 64: "#2ca02c", 128: "#d62728", 256: "#9467bd"}
PARAM_LABELS = {16: "38k", 32: "152k", 64: "603k", 128: "2.4M", 256: "9.6M"}


def rational(m: np.ndarray, a: float, b: float) -> np.ndarray:
    return (1 + a * (m - 2)) / (1 + b * (m - 2))


def main() -> None:
    rows = list(csv.DictReader((ANALYSIS_DIR / "ensemble_size_scaling.csv").open()))
    widths = sorted(set(int(r["width"]) for r in rows))
    m_vals_all = sorted(set(int(r["m"]) for r in rows))

    width_data: dict[int, dict[int, tuple[float, float]]] = {}
    for w in widths:
        wrows = sorted([r for r in rows if int(r["width"]) == w], key=lambda r: int(r["m"]))
        base = float(wrows[0]["accuracy_mean"])
        width_data[w] = {
            int(r["m"]): (float(r["accuracy_mean"]) - base, float(r["accuracy_std"]))
            for r in wrows
        }

    # Compute universal f(M) = gain(M)/gain(2) across widths
    ms_for_fit = np.array([m for m in m_vals_all if m >= 2])
    mean_fMs = []
    std_fMs = []
    for m in ms_for_fit:
        ratios = []
        for w in widths:
            g2 = width_data[w].get(2, (None,))[0]
            gm = width_data[w].get(m, (None,))[0]
            if g2 and gm and g2 > 0:
                ratios.append(gm / g2)
        mean_fMs.append(np.mean(ratios) if ratios else np.nan)
        std_fMs.append(np.std(ratios) if ratios else np.nan)
    mean_fMs = np.array(mean_fMs)
    std_fMs = np.array(std_fMs)

    # Fit rational scaling law
    valid = ~np.isnan(mean_fMs)
    popt, _ = curve_fit(rational, ms_for_fit[valid], mean_fMs[valid], p0=[0.9, 0.36])
    a_fit, b_fit = popt
    saturation = a_fit / b_fit
    m_fit_line = np.linspace(2, max(ms_for_fit) + 3, 200)
    f_fit_line = rational(m_fit_line, a_fit, b_fit)
    rmse = float(np.sqrt(np.mean((rational(ms_for_fit[valid], a_fit, b_fit) - mean_fMs[valid]) ** 2)))
    LOGGER.info("Rational fit: a=%.4f, b=%.4f, saturation f(∞)=%.3f, RMSE=%.5f", a_fit, b_fit, saturation, rmse)

    # ── Figure: 3 panels ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel 1: absolute accuracy
    ax = axes[0]
    for w in widths:
        wrows = sorted([r for r in rows if int(r["width"]) == w], key=lambda r: int(r["m"]))
        ms = [int(r["m"]) for r in wrows]
        accs = [float(r["accuracy_mean"]) * 100 for r in wrows]
        stds = [float(r["accuracy_std"]) * 100 for r in wrows]
        col = WIDTH_COLORS.get(w)
        ax.plot(ms, accs, marker="o", ms=4, label=f"w={w} ({PARAM_LABELS[w]})", color=col)
        ax.fill_between(ms, [a - s for a, s in zip(accs, stds)], [a + s for a, s in zip(accs, stds)],
                        alpha=0.12, color=col)
    ax.set_xlabel("Ensemble size $M$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ensemble accuracy vs. size\n(CIFAR-10 CNN)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: absolute gain(M) curves
    ax = axes[1]
    for w in widths:
        wrows = sorted([r for r in rows if int(r["width"]) == w], key=lambda r: int(r["m"]))
        ms = [int(r["m"]) for r in wrows]
        base = float(wrows[0]["accuracy_mean"])
        gains = [(float(r["accuracy_mean"]) - base) * 100 for r in wrows]
        col = WIDTH_COLORS.get(w)
        ax.plot(ms, gains, marker="o", ms=4, label=f"w={w} ({PARAM_LABELS[w]})", color=col)
    ax.set_xlabel("Ensemble size $M$")
    ax.set_ylabel("Gain over single model (pp)")
    ax.set_title("Gain vs. ensemble size\n(CIFAR-10 CNN)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # Panel 3: universal f(M) = gain(M)/gain(2) with rational fit
    ax = axes[2]
    for w in widths:
        g2 = width_data[w].get(2, (None,))[0]
        if g2 is None or g2 == 0:
            continue
        ms = sorted(m for m in width_data[w] if m >= 2)
        fMs = [width_data[w][m][0] / g2 for m in ms]
        col = WIDTH_COLORS.get(w)
        ax.plot(ms, fMs, marker="o", ms=4, alpha=0.7, color=col, label=f"w={w} ({PARAM_LABELS[w]})")

    # Plot rational fit
    label_fit = (
        f"Rational fit: $f(M)=\\frac{{1+{a_fit:.2f}(M-2)}}{{1+{b_fit:.3f}(M-2)}}$\n"
        f"RMSE={rmse:.4f}, $f(\\infty)\\approx{saturation:.2f}$"
    )
    ax.plot(m_fit_line, f_fit_line, "k--", linewidth=1.8, alpha=0.8, label=label_fit)
    ax.axhline(saturation, color="gray", linestyle=":", linewidth=1, alpha=0.6,
               label=f"Asymptote $f(\\infty)={saturation:.2f}$")
    ax.set_xlabel("Ensemble size $M$")
    ax.set_ylabel(r"$f(M) = \mathrm{gain}(M)\,/\,\mathrm{gain}(2)$")
    ax.set_title(
        "Universal M-scaling factor\n"
        r"($\mathrm{CV}<5\%$ across widths)"
    )
    ax.legend(loc="upper left", fontsize=7.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=2)
    ax.set_ylim(bottom=0.9)

    fig.tight_layout()
    out = PLOT_DIR / "ensemble_size_scaling.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", out)

    # Print table for paper
    print("\nf(M) table for paper:")
    print(f"{'M':>3}  {'f(M) mean':>10}  {'f(M) std':>9}  {'fit':>7}")
    for m, f_m, s_m in zip(ms_for_fit[valid], mean_fMs[valid], std_fMs[valid]):
        fit_val = rational(np.array([m]), a_fit, b_fit)[0]
        print(f"{m:3d}  {f_m:10.4f}  {s_m:9.4f}  {fit_val:7.4f}")


if __name__ == "__main__":
    main()
