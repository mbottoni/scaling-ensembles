#!/usr/bin/env python3
"""Compare functional diversity with vs without data augmentation.

Tests whether the task-difficulty invariance of ensemble diversity survives
once large-width overfitting is regularized away by standard augmentation
(random crop + horizontal flip).

For each (dataset) it pairs the augmented sweep against its non-augmented
baseline and reports, per shared width: individual eval accuracy, the
train-eval gap (overfitting), pairwise disagreement, and 2-member ensemble
gain.  If diversity were merely an artifact of the memorization regime, the
augmented columns would collapse; if it is governed by task difficulty, the
disagreement and gain should persist.

Usage: .venv/bin/python scripts/compare_augmentation.py
Reads:  outputs/series/<exp>/pairwise_similarity.csv  (+ train_results.csv)
Writes: outputs/series/analysis/augmentation_comparison.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("outputs/series")
ANALYSIS = ROOT / "analysis"

# dataset -> (augmented_exp, baseline_exp)
PAIRS = {
    "CIFAR-100": ("cifar100-cnn-augmented", "cifar100-cnn-width-sweep"),
    "CIFAR-10": ("cifar10-cnn-augmented", "cifar10-cnn-extended-width-sweep"),
}
MIN_ACC = 0.15


def load_pairwise(exp: str) -> dict[int, dict[str, float]]:
    """Per width: mean disagreement, ensemble gain, individual eval acc."""
    path = ROOT / exp / "pairwise_similarity.csv"
    if not path.exists():
        return {}
    by_width: dict[int, list] = defaultdict(list)
    for r in csv.DictReader(path.open()):
        a, b = float(r["model_a_accuracy"]), float(r["model_b_accuracy"])
        if a <= MIN_ACC or b <= MIN_ACC:
            continue
        by_width[int(r["width"])].append(r)
    out = {}
    for w, rows in by_width.items():
        disagree = np.mean([float(r["disagreement"]) for r in rows])
        indiv = np.mean([(float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) / 2 for r in rows])
        ens = np.mean([float(r["ensemble_accuracy"]) for r in rows])
        gain = np.mean([
            float(r["ensemble_accuracy"]) - (float(r["model_a_accuracy"]) + float(r["model_b_accuracy"])) / 2
            for r in rows
        ])
        out[w] = {
            "disagreement": disagree,
            "indiv_acc": indiv,
            "ensemble_acc": ens,
            "gain_pp": gain * 100,
            "n_pairs": len(rows),
        }
    return out


def load_train_gap(exp: str) -> dict[int, float]:
    """Per width: mean (train_acc - eval_acc) overfitting gap."""
    path = ROOT / exp / "train_results.csv"
    if not path.exists():
        return {}
    by_width: dict[int, list] = defaultdict(list)
    for r in csv.DictReader(path.open()):
        by_width[int(r["width"])].append(
            float(r["train_accuracy"]) - float(r["eval_accuracy"])
        )
    return {w: float(np.mean(v)) for w, v in by_width.items()}


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for dataset, (aug_exp, base_exp) in PAIRS.items():
        aug, base = load_pairwise(aug_exp), load_pairwise(base_exp)
        aug_gap, base_gap = load_train_gap(aug_exp), load_train_gap(base_exp)
        shared = sorted(set(aug) & set(base))
        if not shared:
            print(f"[skip] {dataset}: no shared widths yet "
                  f"(aug={sorted(aug)}, base={sorted(base)})")
            continue
        print(f"\n=== {dataset}: augmentation vs baseline ===")
        print(f"{'width':>6} | {'indiv acc (base→aug)':>24} | {'gap (base→aug)':>18} | "
              f"{'disagree (base→aug)':>22} | {'gain pp (base→aug)':>20}")
        print("-" * 100)
        for w in shared:
            b, a = base[w], aug[w]
            bg = base_gap.get(w, float("nan"))
            ag = aug_gap.get(w, float("nan"))
            print(f"{w:>6} | {b['indiv_acc']:>10.3f} → {a['indiv_acc']:<10.3f} | "
                  f"{bg:>7.3f} → {ag:<7.3f} | "
                  f"{b['disagreement']*100:>9.1f}% → {a['disagreement']*100:<9.1f}% | "
                  f"{b['gain_pp']:>8.2f} → {a['gain_pp']:<8.2f}")
            out_rows.append({
                "dataset": dataset, "width": w,
                "indiv_acc_base": round(b["indiv_acc"], 4),
                "indiv_acc_aug": round(a["indiv_acc"], 4),
                "train_eval_gap_base": round(bg, 4),
                "train_eval_gap_aug": round(ag, 4),
                "disagreement_base": round(b["disagreement"], 4),
                "disagreement_aug": round(a["disagreement"], 4),
                "gain_pp_base": round(b["gain_pp"], 3),
                "gain_pp_aug": round(a["gain_pp"], 3),
            })

    if out_rows:
        out_path = ANALYSIS / "augmentation_comparison.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
