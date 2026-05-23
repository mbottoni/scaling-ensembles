#!/usr/bin/env python3
"""Run post-hoc analysis on existing experiment outputs.

Produces outputs/series/analysis/*.csv including:
  ensemble_size_scaling.csv, cifar10_calibration_ts.csv,
  fashionmnist_calibration_ts.csv, significance_tests.csv,
  and per-dataset summary CSVs with mean±std CIs.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from scaling_ensembles.analysis import (
    bootstrap_difference_test,
    calibration_with_temperature_scaling,
    ensemble_size_scaling,
    experiment_summary_with_ci,
    load_logits_cache,
    write_ensemble_scaling_csv,
    write_summary_csv,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("outputs/series")
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── CIFAR-10 extended sweep (10 seeds × 5 widths) ───────────────────────────
CIFAR10_EXT = OUTPUT_ROOT / "cifar10-cnn-extended-width-sweep"
cifar10_widths = [16, 32, 64, 128, 256]
cifar10_seeds = list(range(10))

LOGGER.info("Loading CIFAR-10 extended logit cache...")
cifar10_cache = load_logits_cache(CIFAR10_EXT, cifar10_widths, cifar10_seeds)
valid_widths_cifar10 = sorted({w for (w, _) in cifar10_cache})
LOGGER.info("Loaded %d (width, seed) pairs", len(cifar10_cache))

LOGGER.info("Computing ensemble-size scaling for CIFAR-10...")
scaling = ensemble_size_scaling(cifar10_cache, valid_widths_cifar10, n_bootstrap=500)
write_ensemble_scaling_csv(scaling, ANALYSIS_DIR / "ensemble_size_scaling.csv")

LOGGER.info("Computing calibration with temperature scaling for CIFAR-10...")
cifar10_calib = calibration_with_temperature_scaling(cifar10_cache, valid_widths_cifar10)
from scaling_ensembles.analysis import write_calibration_csv
write_calibration_csv(cifar10_calib, ANALYSIS_DIR / "cifar10_calibration_ts.csv")
LOGGER.info("Wrote cifar10_calibration_ts.csv")

cifar10_summary = experiment_summary_with_ci(CIFAR10_EXT)
write_summary_csv(cifar10_summary, ANALYSIS_DIR / "cifar10_extended_summary.csv")

# ── FashionMNIST ─────────────────────────────────────────────────────────────
FMNIST = OUTPUT_ROOT / "dataset-difficulty-fashionmnist"
fmnist_widths = [16, 32, 64, 128]
fmnist_seeds = list(range(5))
LOGGER.info("Loading FashionMNIST logit cache...")
fmnist_cache = load_logits_cache(FMNIST, fmnist_widths, fmnist_seeds)
fmnist_valid_widths = sorted({w for (w, _) in fmnist_cache})

LOGGER.info("Computing calibration with temperature scaling for FashionMNIST...")
fmnist_calib = calibration_with_temperature_scaling(fmnist_cache, fmnist_valid_widths)
write_calibration_csv(fmnist_calib, ANALYSIS_DIR / "fashionmnist_calibration_ts.csv")
fmnist_summary = experiment_summary_with_ci(FMNIST)
write_summary_csv(fmnist_summary, ANALYSIS_DIR / "fashionmnist_summary.csv")
LOGGER.info("Wrote FashionMNIST analysis")

# ── STL-10 ───────────────────────────────────────────────────────────────────
STL10 = OUTPUT_ROOT / "stl10-cnn-width-sweep"
if (STL10 / "cache/logits").exists():
    stl10_widths = [16, 32, 64, 128]
    stl10_seeds = list(range(3))
    stl10_cache = load_logits_cache(STL10, stl10_widths, stl10_seeds)
    stl10_valid_widths = sorted({w for (w, _) in stl10_cache})
    stl10_calib = calibration_with_temperature_scaling(stl10_cache, stl10_valid_widths)
    write_calibration_csv(stl10_calib, ANALYSIS_DIR / "stl10_calibration_ts.csv")
    stl10_summary = experiment_summary_with_ci(STL10)
    write_summary_csv(stl10_summary, ANALYSIS_DIR / "stl10_summary.csv")
    LOGGER.info("Wrote STL-10 analysis")

# ── SVHN ─────────────────────────────────────────────────────────────────────
SVHN = OUTPUT_ROOT / "svhn-cnn-width-sweep"
if (SVHN / "cache/logits").exists():
    svhn_widths = [16, 32, 64, 128]
    svhn_seeds = list(range(3))
    svhn_cache = load_logits_cache(SVHN, svhn_widths, svhn_seeds)
    svhn_valid_widths = sorted({w for (w, _) in svhn_cache})
    svhn_calib = calibration_with_temperature_scaling(svhn_cache, svhn_valid_widths)
    write_calibration_csv(svhn_calib, ANALYSIS_DIR / "svhn_calibration_ts.csv")
    svhn_summary = experiment_summary_with_ci(SVHN)
    write_summary_csv(svhn_summary, ANALYSIS_DIR / "svhn_summary.csv")
    LOGGER.info("Wrote SVHN analysis")

# ── MNIST MLP extended sweep (10 seeds × 6 widths) ──────────────────────────
MNIST = OUTPUT_ROOT / "mnist-mlp-extended-width-sweep"
if (MNIST / "cache/logits").exists():
    mnist_widths = [32, 64, 128, 256, 512, 1024]
    mnist_seeds = list(range(10))
    LOGGER.info("Loading MNIST MLP logit cache...")
    mnist_cache = load_logits_cache(MNIST, mnist_widths, mnist_seeds)
    mnist_valid_widths = sorted({w for (w, _) in mnist_cache})
    mnist_calib = calibration_with_temperature_scaling(mnist_cache, mnist_valid_widths)
    write_calibration_csv(mnist_calib, ANALYSIS_DIR / "mnist_calibration_ts.csv")
    mnist_summary = experiment_summary_with_ci(MNIST)
    write_summary_csv(mnist_summary, ANALYSIS_DIR / "mnist_summary.csv")
    LOGGER.info("Wrote MNIST analysis")

# ── Bootstrap significance tests ─────────────────────────────────────────────
LOGGER.info("Running bootstrap significance tests...")

# Test: disagreement is higher on CIFAR-10 than FashionMNIST at each width
sig_rows = []
for w_c, w_f in [(16, 16), (64, 64), (128, 128)]:
    cifar_pairwise_path = CIFAR10_EXT / "pairwise_similarity.csv"
    fmnist_pairwise_path = FMNIST / "pairwise_similarity.csv"

    def read_disagree(path, width, diverged=None):
        diverged = diverged or set()
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return [
            1.0 - float(r["agreement"])
            for r in rows
            if int(r["width"]) == width
            and (width, int(r["seed_a"])) not in diverged
            and (width, int(r["seed_b"])) not in diverged
        ]

    cifar_diverged = {(256, 6)}
    c_vals = read_disagree(cifar_pairwise_path, w_c, cifar_diverged)
    f_vals = read_disagree(fmnist_pairwise_path, w_f)
    if not c_vals or not f_vals:
        continue
    result = bootstrap_difference_test(c_vals, f_vals, n_bootstrap=2000)
    sig_rows.append({
        "test": f"disagreement CIFAR10 w={w_c} > FashionMNIST w={w_f}",
        **result,
    })
    LOGGER.info(
        "  CIFAR10 w=%d vs FashionMNIST w=%d: diff=%.4f p=%.4f",
        w_c, w_f, result["observed_difference"], result["p_value"],
    )

# Test: ensemble gain is higher on CIFAR-10 than FashionMNIST
def read_ens_gain(path, width, diverged=None):
    diverged = diverged or set()
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        float(r["ensemble_accuracy"]) - 0.5 * (float(r["model_a_accuracy"]) + float(r["model_b_accuracy"]))
        for r in rows
        if int(r["width"]) == width
        and (width, int(r["seed_a"])) not in diverged
        and (width, int(r["seed_b"])) not in diverged
    ]

for w in [16, 64, 128]:
    c_gain = read_ens_gain(CIFAR10_EXT / "pairwise_similarity.csv", w, {(256, 6)})
    f_gain = read_ens_gain(FMNIST / "pairwise_similarity.csv", w)
    if not c_gain or not f_gain:
        continue
    result = bootstrap_difference_test(c_gain, f_gain, n_bootstrap=2000)
    sig_rows.append({
        "test": f"ensemble_gain CIFAR10 w={w} > FashionMNIST w={w}",
        **result,
    })
    LOGGER.info(
        "  Ensemble gain CIFAR10 w=%d vs FashionMNIST w=%d: diff=%.4f p=%.4f",
        w, w, result["observed_difference"], result["p_value"],
    )

if sig_rows:
    sig_path = ANALYSIS_DIR / "significance_tests.csv"
    with sig_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sig_rows)
    LOGGER.info("Wrote significance_tests.csv")

LOGGER.info("Analysis complete. Output: %s", ANALYSIS_DIR)
