from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from scaling_ensembles.config import load_config
from scaling_ensembles.data import make_dataloaders
from scaling_ensembles.interpolate import (
    InterpolationPoint,
    linear_interpolation_barrier,
    write_interpolation_results,
)
from scaling_ensembles.metrics import (
    PairwiseSimilarity,
    collect_logits,
    compare_predictions,
    write_pairwise_results,
)
from scaling_ensembles.train import (
    TrainResult,
    load_checkpoint_model,
    resolve_device,
    train_one,
    write_train_results,
)


def run_sweep(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_results: list[TrainResult] = []
    for width in config.model.widths:
        for seed in config.training.seeds:
            result = train_one(config, width=width, seed=seed, output_dir=output_dir)
            train_results.append(result)

    train_csv = output_dir / "train_results.csv"
    write_train_results(train_results, train_csv)

    _, eval_loader, dataset_info = make_dataloaders(config.data)
    device = resolve_device(config.training.device)
    by_width: dict[int, list[TrainResult]] = defaultdict(list)
    for result in train_results:
        by_width[result.width].append(result)

    pairwise_results: list[PairwiseSimilarity] = []
    interpolation_results: list[InterpolationPoint] = []
    for width, results in by_width.items():
        pairs = list(itertools.combinations(results, 2))
        if config.similarity.max_pairs_per_width is not None:
            pairs = pairs[: config.similarity.max_pairs_per_width]

        logits_cache = {}
        targets_cache = None
        for result in results:
            model = load_checkpoint_model(result.checkpoint_path, config, dataset_info, device)
            logits, targets = collect_logits(model, eval_loader, device)
            logits_cache[result.seed] = logits
            targets_cache = targets

        if targets_cache is None:
            continue

        for result_a, result_b in tqdm(pairs, desc=f"compare width={width}", leave=False):
            pairwise_results.append(
                compare_predictions(
                    logits_cache[result_a.seed],
                    logits_cache[result_b.seed],
                    targets_cache,
                    width=width,
                    seed_a=result_a.seed,
                    seed_b=result_b.seed,
                    parameter_count=result_a.parameter_count,
                )
            )
            if config.similarity.interpolation_steps > 1:
                interpolation_results.extend(
                    linear_interpolation_barrier(
                        result_a.checkpoint_path,
                        result_b.checkpoint_path,
                        config,
                        dataset_info,
                        eval_loader,
                        config.similarity.interpolation_steps,
                        device,
                    )
                )

    pairwise_csv = output_dir / "pairwise_similarity.csv"
    write_pairwise_results(pairwise_results, pairwise_csv)
    interpolation_csv = output_dir / "interpolation_barriers.csv"
    write_interpolation_results(interpolation_results, interpolation_csv)

    return {
        "train": train_csv,
        "pairwise": pairwise_csv,
        "interpolation": interpolation_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a width sweep for functional similarity.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()

    outputs = run_sweep(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
