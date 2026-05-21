#!/usr/bin/env python3
"""Run post-hoc analysis on existing experiment outputs.

Produces:
  outputs/series/analysis/ensemble_size_scaling.csv
  outputs/series/analysis/calibration_summary.csv
  outputs/series/analysis/cifar10_extended_summary.csv
  outputs/series/analysis/fashionmnist_summary.csv
  outputs/series/analysis/stl10_summary.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

from scaling_ensembles.analysis import (
    calibration_summary,
    ensemble_size_scaling,
    experiment_summary_with_ci,
    load_logits_cache,
    write_calibration_csv,
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
LOGGER.info("Loaded %d (width, seed) pairs across widths %s", len(cifar10_cache), valid_widths_cifar10)

# Ensemble-size scaling
LOGGER.info("Computing ensemble-size scaling for CIFAR-10...")
scaling = ensemble_size_scaling(cifar10_cache, valid_widths_cifar10, n_bootstrap=500)
write_ensemble_scaling_csv(scaling, ANALYSIS_DIR / "ensemble_size_scaling.csv")
LOGGER.info("Wrote ensemble_size_scaling.csv")

# Calibration
LOGGER.info("Computing calibration (ECE) for CIFAR-10...")
calib = calibration_summary(cifar10_cache, valid_widths_cifar10)
write_calibration_csv(calib, ANALYSIS_DIR / "cifar10_calibration.csv")
LOGGER.info("Wrote cifar10_calibration.csv")

# Summary with CI
cifar10_summary = experiment_summary_with_ci(CIFAR10_EXT)
write_summary_csv(cifar10_summary, ANALYSIS_DIR / "cifar10_extended_summary.csv")
LOGGER.info("Wrote cifar10_extended_summary.csv")

# ── FashionMNIST ─────────────────────────────────────────────────────────────
FMNIST = OUTPUT_ROOT / "dataset-difficulty-fashionmnist"
if (FMNIST / "cache/logits").exists():
    fmnist_widths = [16, 32, 64, 128]
    fmnist_seeds = list(range(5))
    LOGGER.info("Loading FashionMNIST logit cache...")
    fmnist_cache = load_logits_cache(FMNIST, fmnist_widths, fmnist_seeds)
    fmnist_valid_widths = sorted({w for (w, _) in fmnist_cache})
    fmnist_calib = calibration_summary(fmnist_cache, fmnist_valid_widths)
    write_calibration_csv(fmnist_calib, ANALYSIS_DIR / "fashionmnist_calibration.csv")
    fmnist_summary = experiment_summary_with_ci(FMNIST)
    write_summary_csv(fmnist_summary, ANALYSIS_DIR / "fashionmnist_summary.csv")
    LOGGER.info("Wrote FashionMNIST analysis")
else:
    LOGGER.info("FashionMNIST logit cache not found, skipping calibration analysis")

# ── STL-10 ───────────────────────────────────────────────────────────────────
STL10 = OUTPUT_ROOT / "stl10-cnn-width-sweep"
if (STL10 / "cache/logits").exists():
    stl10_widths = [16, 32, 64, 128]
    stl10_seeds = list(range(3))
    LOGGER.info("Loading STL-10 logit cache...")
    stl10_cache = load_logits_cache(STL10, stl10_widths, stl10_seeds)
    stl10_valid_widths = sorted({w for (w, _) in stl10_cache})
    stl10_calib = calibration_summary(stl10_cache, stl10_valid_widths)
    write_calibration_csv(stl10_calib, ANALYSIS_DIR / "stl10_calibration.csv")
    stl10_summary = experiment_summary_with_ci(STL10)
    write_summary_csv(stl10_summary, ANALYSIS_DIR / "stl10_summary.csv")
    LOGGER.info("Wrote STL-10 analysis")
else:
    LOGGER.info("STL-10 logit cache not found, skipping calibration analysis")

# ── SVHN ─────────────────────────────────────────────────────────────────────
SVHN = OUTPUT_ROOT / "svhn-cnn-width-sweep"
if (SVHN / "cache/logits").exists():
    svhn_widths = [16, 32, 64, 128]
    svhn_seeds = list(range(3))
    svhn_cache = load_logits_cache(SVHN, svhn_widths, svhn_seeds)
    svhn_valid_widths = sorted({w for (w, _) in svhn_cache})
    svhn_calib = calibration_summary(svhn_cache, svhn_valid_widths)
    write_calibration_csv(svhn_calib, ANALYSIS_DIR / "svhn_calibration.csv")
    svhn_summary = experiment_summary_with_ci(SVHN)
    write_summary_csv(svhn_summary, ANALYSIS_DIR / "svhn_summary.csv")
    LOGGER.info("Wrote SVHN analysis")

LOGGER.info("Analysis complete. Output: %s", ANALYSIS_DIR)
