"""Post-hoc analysis utilities for scaling-ensembles experiments.

Computes ensemble-size scaling, calibration (ECE), and statistical summaries
from existing cached logits — no re-training required.
"""
from __future__ import annotations

import csv
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scaling_ensembles.metrics import expected_calibration_error


LOGGER = logging.getLogger(__name__)

# Runs whose accuracy is at or below random-chance are treated as diverged.
_MIN_ACCURACY = 0.15


def load_logits_cache(
    output_dir: Path,
    widths: list[int],
    seeds: list[int],
    cache_subdir: str = "cache/logits",
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Load all cached (logits, targets) tensors for (width, seed) pairs."""
    cache_dir = output_dir / cache_subdir
    result: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for width in widths:
        for seed in seeds:
            path = cache_dir / f"width_{width}_seed_{seed}.pt"
            if not path.exists():
                LOGGER.warning("Missing logit cache: %s", path)
                continue
            cached = torch.load(path, map_location="cpu", weights_only=False)
            logits, targets = cached["logits"], cached["targets"]
            probs = logits.softmax(dim=1)
            accuracy = (probs.argmax(dim=1) == targets).float().mean().item()
            if accuracy <= _MIN_ACCURACY:
                LOGGER.warning(
                    "Skipping diverged run width=%s seed=%s (accuracy=%.4f)", width, seed, accuracy
                )
                continue
            result[(width, seed)] = (logits, targets)
    return result


def ensemble_size_scaling(
    logits_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    widths: list[int],
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Compute ensemble accuracy for M=1..N members via bootstrap sampling.

    Returns {width: {M: (mean_accuracy, std_accuracy)}}.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    result: dict[int, dict[int, tuple[float, float]]] = {}
    for width in widths:
        seeds = [s for (w, s) in logits_cache if w == width]
        if not seeds:
            continue
        n_models = len(seeds)
        all_logits = torch.stack([logits_cache[(width, s)][0] for s in seeds])
        targets = logits_cache[(width, seeds[0])][1]

        scaling: dict[int, tuple[float, float]] = {}
        for m in range(1, n_models + 1):
            accuracies = []
            for _ in range(n_bootstrap):
                chosen = rng.choice(n_models, size=m, replace=False)
                ensemble_probs = all_logits[chosen].softmax(dim=2).mean(dim=0)
                acc = (ensemble_probs.argmax(dim=1) == targets).float().mean().item()
                accuracies.append(acc)
            scaling[m] = (float(np.mean(accuracies)), float(np.std(accuracies)))
        result[width] = scaling
    return result


def calibration_summary(
    logits_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    widths: list[int],
) -> dict[int, dict[str, float]]:
    """Compute ECE for individual models and 2-member ensembles per width.

    Returns {width: {single_ece_mean, single_ece_std, ensemble_ece_mean, ensemble_ece_std, ece_reduction}}.
    """
    result: dict[int, dict[str, float]] = {}
    for width in widths:
        seeds = [s for (w, s) in logits_cache if w == width]
        if not seeds:
            continue

        single_eces: list[float] = []
        for seed in seeds:
            logits, targets = logits_cache[(width, seed)]
            single_eces.append(expected_calibration_error(logits.softmax(dim=1), targets))

        pair_eces: list[float] = []
        for sa, sb in itertools.combinations(seeds, 2):
            logits_a, targets = logits_cache[(width, sa)]
            logits_b, _ = logits_cache[(width, sb)]
            ens_probs = 0.5 * (logits_a.softmax(dim=1) + logits_b.softmax(dim=1))
            pair_eces.append(expected_calibration_error(ens_probs, targets))

        single_mean = float(np.mean(single_eces))
        ens_mean = float(np.mean(pair_eces)) if pair_eces else float("nan")
        result[width] = {
            "single_ece_mean": single_mean,
            "single_ece_std": float(np.std(single_eces)),
            "ensemble_ece_mean": ens_mean,
            "ensemble_ece_std": float(np.std(pair_eces)) if pair_eces else float("nan"),
            "ece_reduction": single_mean - ens_mean,
        }
    return result


def experiment_summary_with_ci(
    output_dir: Path,
) -> list[dict[str, float | int | str]]:
    """Summarize an experiment with mean ± std confidence intervals across seeds.

    Filters diverged runs (accuracy ≤ _MIN_ACCURACY) before computing statistics.
    """
    train_path = output_dir / "train_results.csv"
    pairwise_path = output_dir / "pairwise_similarity.csv"
    barrier_path = output_dir / "interpolation_barriers.csv"

    with train_path.open() as f:
        train_rows = list(csv.DictReader(f))
    with pairwise_path.open() as f:
        pairwise_rows = list(csv.DictReader(f))
    with barrier_path.open() as f:
        barrier_rows = list(csv.DictReader(f))

    # Identify diverged (width, seed) pairs
    diverged: set[tuple[int, int]] = set()
    for row in train_rows:
        if float(row["eval_accuracy"]) <= _MIN_ACCURACY:
            diverged.add((int(row["width"]), int(row["seed"])))

    train_by_width: dict[int, list[dict]] = defaultdict(list)
    pairwise_by_width: dict[int, list[dict]] = defaultdict(list)
    barrier_by_pair: dict[tuple[int, int, int], list[float]] = defaultdict(list)

    for row in train_rows:
        w, s = int(row["width"]), int(row["seed"])
        if (w, s) not in diverged:
            train_by_width[w].append(row)

    for row in pairwise_rows:
        w, sa, sb = int(row["width"]), int(row["seed_a"]), int(row["seed_b"])
        if (w, sa) not in diverged and (w, sb) not in diverged:
            pairwise_by_width[w].append(row)

    for row in barrier_rows:
        w, sa, sb = int(row["width"]), int(row["seed_a"]), int(row["seed_b"])
        if (w, sa) not in diverged and (w, sb) not in diverged:
            barrier_by_pair[(w, sa, sb)].append(float(row["loss_barrier"]))

    barrier_by_width: dict[int, list[float]] = defaultdict(list)
    for (w, *_), barriers in barrier_by_pair.items():
        barrier_by_width[w].append(max(barriers))

    summary = []
    for width in sorted(train_by_width):
        tr = train_by_width[width]
        pw = pairwise_by_width[width]
        if not tr or not pw:
            continue

        eval_acc = np.array([float(r["eval_accuracy"]) for r in tr])
        disagreement = np.array([1.0 - float(r["agreement"]) for r in pw])
        single_acc = np.array([0.5 * (float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) for r in pw])
        ens_acc = np.array([float(r["ensemble_accuracy"]) for r in pw])
        ens_gain = ens_acc - single_acc
        js_vals = [float(r["js_divergence"]) for r in pw if float(r["js_divergence"]) == float(r["js_divergence"])]
        barriers = barrier_by_width.get(width, [])

        row_out: dict[str, float | int | str] = {
            "width": width,
            "parameter_count": int(tr[0]["parameter_count"]),
            "n_runs": len(tr),
            "eval_accuracy_mean": float(eval_acc.mean()),
            "eval_accuracy_std": float(eval_acc.std()),
            "disagreement_mean": float(disagreement.mean()),
            "disagreement_std": float(disagreement.std()),
            "ensemble_gain_mean": float(ens_gain.mean()),
            "ensemble_gain_std": float(ens_gain.std()),
            "js_divergence_mean": float(np.mean(js_vals)) if js_vals else float("nan"),
            "barrier_mean": float(np.mean(barriers)) if barriers else float("nan"),
            "barrier_std": float(np.std(barriers)) if barriers else float("nan"),
        }
        summary.append(row_out)
    return summary


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ensemble_scaling_csv(
    scaling: dict[int, dict[int, tuple[float, float]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "m", "accuracy_mean", "accuracy_std"])
        for width in sorted(scaling):
            for m in sorted(scaling[width]):
                mean, std = scaling[width][m]
                writer.writerow([width, m, mean, std])


def write_calibration_csv(
    calibration: dict[int, dict[str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "single_ece_mean", "single_ece_std", "ensemble_ece_mean", "ensemble_ece_std", "ece_reduction"])
        for width in sorted(calibration):
            d = calibration[width]
            writer.writerow([
                width,
                d["single_ece_mean"], d["single_ece_std"],
                d["ensemble_ece_mean"], d["ensemble_ece_std"],
                d["ece_reduction"],
            ])
