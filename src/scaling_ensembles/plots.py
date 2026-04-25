from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_similarity_vs_params(pairwise_csv: str | Path):
    rows = read_csv(pairwise_csv)
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        params = int(row["parameter_count"])
        grouped[params]["agreement"].append(float(row["agreement"]))
        grouped[params]["js_divergence"].append(float(row["js_divergence"]))
        grouped[params]["ensemble_gain"].append(
            float(row["ensemble_accuracy"])
            - 0.5 * (float(row["model_a_accuracy"]) + float(row["model_b_accuracy"]))
        )

    params = np.array(sorted(grouped))
    agreement = np.array([np.mean(grouped[p]["agreement"]) for p in params])
    js_divergence = np.array([np.mean(grouped[p]["js_divergence"]) for p in params])
    ensemble_gain = np.array([np.mean(grouped[p]["ensemble_gain"]) for p in params])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(params, agreement, marker="o")
    axes[0].set_title("Prediction agreement")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("# parameters")

    axes[1].plot(params, js_divergence, marker="o", color="tab:orange")
    axes[1].set_title("JS divergence")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("# parameters")

    axes[2].plot(params, ensemble_gain, marker="o", color="tab:green")
    axes[2].set_title("Ensemble accuracy gain")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("# parameters")

    fig.tight_layout()
    return fig


def plot_loss_barriers(interpolation_csv: str | Path):
    rows = read_csv(interpolation_csv)
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (int(row["width"]), int(row["seed_a"]), int(row["seed_b"]))
        grouped[key].append(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, group in grouped.items():
        group = sorted(group, key=lambda row: float(row["alpha"]))
        alphas = [float(row["alpha"]) for row in group]
        losses = [float(row["loss"]) for row in group]
        ax.plot(alphas, losses, alpha=0.45, label=f"w={key[0]}, {key[1]}-{key[2]}")

    ax.set_xlabel("Interpolation alpha")
    ax.set_ylabel("Eval loss")
    ax.set_title("Linear interpolation loss")
    if len(grouped) <= 8:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
