from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS = (
    "mnist-width-sweep",
    "cifar10-cnn-width-sweep",
    "cifar10-patch-transformer-width-sweep",
)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_paper_plots(
    output_root: str | Path = "outputs",
    experiments: tuple[str, ...] = DEFAULT_EXPERIMENTS,
    report_dir: str | Path | None = None,
) -> dict[str, Path]:
    output_root = Path(output_root)
    report_dir = Path(report_dir) if report_dir is not None else output_root / "paper-plots"
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        experiment: summarize_experiment(output_root / experiment)
        for experiment in experiments
    }
    written = {
        "diversity_accuracy": report_dir / "diversity_accuracy_plane.png",
        "similarity_scaling": report_dir / "similarity_vs_parameter_count.png",
        "loss_barriers": report_dir / "linear_interpolation_barriers.png",
    }
    plot_diversity_accuracy_plane(summaries, written["diversity_accuracy"])
    plot_similarity_scaling(summaries, written["similarity_scaling"])
    plot_loss_barrier_scaling(summaries, written["loss_barriers"])

    for experiment in experiments:
        matrix_path = report_dir / f"{experiment}_final_minima_similarity.png"
        plot_final_minima_similarity(output_root / experiment, matrix_path)
        written[f"{experiment}_matrices"] = matrix_path

    LOGGER.info("Wrote paper-style plots to %s", report_dir)
    return written


def summarize_experiment(output_dir: Path) -> list[dict[str, float | int]]:
    train_rows = read_csv(output_dir / "train_results.csv")
    pairwise_rows = read_csv(output_dir / "pairwise_similarity.csv")
    interpolation_rows = read_csv(output_dir / "interpolation_barriers.csv")

    train_by_width: dict[int, list[dict[str, str]]] = defaultdict(list)
    pairwise_by_width: dict[int, list[dict[str, str]]] = defaultdict(list)
    barrier_by_width: dict[int, list[float]] = defaultdict(list)

    for row in train_rows:
        train_by_width[int(row["width"])].append(row)
    for row in pairwise_rows:
        pairwise_by_width[int(row["width"])].append(row)

    barrier_by_pair: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in interpolation_rows:
        key = (int(row["width"]), int(row["seed_a"]), int(row["seed_b"]))
        barrier_by_pair[key].append(float(row["loss_barrier"]))
    for (width, *_), barriers in barrier_by_pair.items():
        barrier_by_width[width].append(max(barriers))

    summary = []
    for width in sorted(train_by_width):
        train_group = train_by_width[width]
        pair_group = pairwise_by_width[width]
        eval_acc = np.array([float(row["eval_accuracy"]) for row in train_group])
        disagreement = np.array([1.0 - float(row["agreement"]) for row in pair_group])
        single_acc = np.array(
            [
                0.5 * (float(row["model_a_accuracy"]) + float(row["model_b_accuracy"]))
                for row in pair_group
            ]
        )
        ensemble_acc = np.array([float(row["ensemble_accuracy"]) for row in pair_group])
        summary.append(
            {
                "width": width,
                "parameter_count": int(train_group[0]["parameter_count"]),
                "eval_accuracy": float(eval_acc.mean()),
                "eval_accuracy_std": float(eval_acc.std()),
                "prediction_disagreement": float(disagreement.mean()),
                "agreement": float(np.mean([float(row["agreement"]) for row in pair_group])),
                "js_divergence": float(np.mean([float(row["js_divergence"]) for row in pair_group])),
                "ensemble_gain": float((ensemble_acc - single_acc).mean()),
                "max_loss_barrier": float(np.mean(barrier_by_width[width])),
            }
        )
    return summary


def plot_diversity_accuracy_plane(
    summaries: dict[str, list[dict[str, float | int]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for experiment, rows in summaries.items():
        params = np.array([float(row["parameter_count"]) for row in rows])
        sizes = 50 + 220 * (np.log10(params) - np.log10(params).min()) / (
            np.log10(params).max() - np.log10(params).min() + 1e-9
        )
        ax.scatter(
            [100 * float(row["prediction_disagreement"]) for row in rows],
            [100 * float(row["eval_accuracy"]) for row in rows],
            s=sizes,
            alpha=0.75,
            label=experiment,
        )
        for row in rows:
            ax.annotate(
                f"w={row['width']}\n{format_params(int(row['parameter_count']))}",
                (
                    100 * float(row["prediction_disagreement"]),
                    100 * float(row["eval_accuracy"]),
                ),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_title("Diversity-accuracy plane")
    ax.set_xlabel("Prediction disagreement (%)")
    ax.set_ylabel("Mean eval accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similarity_scaling(
    summaries: dict[str, list[dict[str, float | int]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for experiment, rows in summaries.items():
        params = [int(row["parameter_count"]) for row in rows]
        axes[0].plot(params, [100 * float(row["agreement"]) for row in rows], marker="o", label=experiment)
        axes[1].plot(params, [float(row["js_divergence"]) for row in rows], marker="o", label=experiment)
        axes[2].plot(params, [100 * float(row["ensemble_gain"]) for row in rows], marker="o", label=experiment)

    for ax, title, ylabel in [
        (axes[0], "Function similarity", "Agreement (%)"),
        (axes[1], "Predictive distribution distance", "JS divergence"),
        (axes[2], "Ensemble benefit", "Accuracy gain (pp)"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("# parameters")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_barrier_scaling(
    summaries: dict[str, list[dict[str, float | int]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for experiment, rows in summaries.items():
        ax.plot(
            [int(row["parameter_count"]) for row in rows],
            [float(row["max_loss_barrier"]) for row in rows],
            marker="o",
            label=experiment,
        )
    ax.set_title("Linear interpolation barrier vs parameter count")
    ax.set_xlabel("# parameters")
    ax.set_ylabel("Mean max loss barrier")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_minima_similarity(output_dir: Path, output_path: Path) -> None:
    train_rows = read_csv(output_dir / "train_results.csv")
    pairwise_rows = read_csv(output_dir / "pairwise_similarity.csv")
    largest_width = max(int(row["width"]) for row in train_rows)
    width_rows = [row for row in train_rows if int(row["width"]) == largest_width]
    seeds = [int(row["seed"]) for row in width_rows]
    weight_matrix = np.eye(len(seeds))
    disagreement_matrix = np.zeros((len(seeds), len(seeds)))

    vectors = {
        int(row["seed"]): flatten_checkpoint(row["checkpoint_path"])
        for row in width_rows
    }
    for i, seed_i in enumerate(seeds):
        for j, seed_j in enumerate(seeds):
            if i == j:
                continue
            weight_matrix[i, j] = torch.nn.functional.cosine_similarity(
                vectors[seed_i],
                vectors[seed_j],
                dim=0,
            ).item()

    for row in pairwise_rows:
        if int(row["width"]) != largest_width:
            continue
        i = seeds.index(int(row["seed_a"]))
        j = seeds.index(int(row["seed_b"]))
        disagreement = 1.0 - float(row["agreement"])
        disagreement_matrix[i, j] = disagreement
        disagreement_matrix[j, i] = disagreement

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, matrix, title, cmap in [
        (axes[0], weight_matrix, "Weight cosine similarity", "viridis"),
        (axes[1], disagreement_matrix, "Prediction disagreement", "magma"),
    ]:
        image = ax.imshow(matrix, cmap=cmap)
        ax.set_title(f"{title}\nwidth={largest_width}")
        ax.set_xticks(range(len(seeds)), labels=seeds)
        ax.set_yticks(range(len(seeds)), labels=seeds)
        ax.set_xlabel("seed")
        ax.set_ylabel("seed")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def flatten_checkpoint(checkpoint_path: str | Path) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensors = [
        value.detach().flatten().float()
        for value in checkpoint["state_dict"].values()
        if torch.is_floating_point(value)
    ]
    return torch.cat(tensors)


def format_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate paper-style plots from experiment outputs.")
    parser.add_argument("--output-root", default="outputs", help="Root directory containing experiment outputs.")
    parser.add_argument("--report-dir", default=None, help="Directory where plots should be written.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(DEFAULT_EXPERIMENTS),
        help="Experiment output directory names to include.",
    )
    args = parser.parse_args()

    outputs = make_paper_plots(
        output_root=args.output_root,
        experiments=tuple(args.experiments),
        report_dir=args.report_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
