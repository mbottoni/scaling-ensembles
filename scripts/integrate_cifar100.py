#!/usr/bin/env python3
"""Integrate CIFAR-100 results into paper once training is complete.

Run this after all CIFAR-100 checkpoints exist.  It:
1. Generates cifar100_summary.csv via run_analysis logic
2. Regenerates all paper plots
3. Prints the key statistics to update in the paper

Usage: .venv/bin/python scripts/integrate_cifar100.py
"""
from __future__ import annotations

import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs/series")
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
CIFAR100_DIR = OUTPUT_ROOT / "cifar100-cnn-width-sweep"
CKPT_DIR = CIFAR100_DIR / "checkpoints"


def check_checkpoints() -> bool:
    expected = [f"width_{w}_seed_{s}.pt" for w in [16, 32, 64, 128] for s in range(5)]
    missing = [f for f in expected if not (CKPT_DIR / f).exists()]
    if missing:
        LOGGER.warning("Missing %d checkpoints: %s", len(missing), missing[:5])
        return False
    LOGGER.info("All 20 CIFAR-100 checkpoints present.")
    return True


def run_analysis() -> None:
    LOGGER.info("Running run_analysis.py to generate cifar100_summary.csv...")
    result = subprocess.run(
        [sys.executable, "scripts/run_analysis.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        LOGGER.error("run_analysis.py failed (returncode=%d)", result.returncode)
        sys.exit(1)
    LOGGER.info("run_analysis.py complete.")


def regenerate_plots() -> None:
    for script in [
        "scripts/make_accuracy_diversity_plot.py",
        "scripts/make_overview_plot.py",
        "scripts/make_gain_vs_error_plot.py",
    ]:
        LOGGER.info("Running %s...", script)
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            LOGGER.warning("%s failed (returncode=%d)", script, result.returncode)


def compute_cross_dataset_stats() -> None:
    summary_files = {
        "MNIST": ANALYSIS_DIR / "mnist_summary.csv",
        "FashionMNIST": ANALYSIS_DIR / "fashionmnist_summary.csv",
        "SVHN": ANALYSIS_DIR / "svhn_summary.csv",
        "STL10": ANALYSIS_DIR / "stl10_summary.csv",
        "CIFAR10": ANALYSIS_DIR / "cifar10_extended_summary.csv",
        "CIFAR100": ANALYSIS_DIR / "cifar100_summary.csv",
    }

    dataset_means: dict[str, tuple[float, float]] = {}
    for name, path in summary_files.items():
        if not path.exists():
            LOGGER.warning("Missing %s", path)
            continue
        rows = list(csv.DictReader(path.open()))
        accs = [float(r["eval_accuracy_mean"]) for r in rows]
        disags = [float(r["disagreement_mean"]) for r in rows]
        dataset_means[name] = (np.mean(accs), np.mean(disags))

    print("\n=== Cross-dataset correlation ===")
    for name, (acc, disag) in sorted(dataset_means.items()):
        print(f"  {name:12s}: acc={acc:.3f}, disagree={disag:.3f}")

    errors = [1 - v[0] for v in dataset_means.values()]
    disags = [v[1] for v in dataset_means.values()]
    r, p = stats.pearsonr(errors, disags)
    print(f"\nCross-dataset r={r:.3f}, p={p:.4e} (n={len(errors)})")

    # CIFAR-100 stats
    if "CIFAR100" in dataset_means:
        acc, disag = dataset_means["CIFAR100"]
        print(f"\nCIFAR-100: acc={acc:.3f}, disagree={disag:.3f}")
        print(f"  Update paper: 'seven benchmarks'")
        print(f"  Cross-dataset r={r:.3f}, p={p:.4e}")


def print_paper_updates() -> None:
    summary = ANALYSIS_DIR / "cifar100_summary.csv"
    if not summary.exists():
        return

    print("\n=== Paper updates needed ===")
    rows = list(csv.DictReader(summary.open()))
    for row in rows:
        w = int(row["width"])
        params = int(row["parameter_count"])
        acc = float(row["eval_accuracy_mean"])
        disag = float(row["disagreement_mean"])
        gain = float(row["ensemble_gain_mean"]) * 100
        print(f"  CIFAR-100 w={w:3d} ({params/1e3:.0f}k): acc={acc:.3f}, disagree={disag:.3f}, gain={gain:.2f}pp")

    print("\n  1. Abstract: 'six benchmarks' → 'seven benchmarks'")
    print("  2. Intro contributions: update cross-dataset r and p-value")
    print("  3. Section 3.1: update cross-dataset correlation text")
    print("  4. Tab 1: add CIFAR-100 row")
    print("  5. Section conclusion: 'six' → 'seven'")
    print("  6. n count: 414 → 418 (4 more aggregates)")


def main() -> None:
    if not check_checkpoints():
        LOGGER.error("Not all CIFAR-100 checkpoints are ready yet. Exiting.")
        sys.exit(1)

    run_analysis()
    regenerate_plots()
    compute_cross_dataset_stats()
    print_paper_updates()


if __name__ == "__main__":
    main()
