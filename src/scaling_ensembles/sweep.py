from __future__ import annotations

import argparse
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scaling_ensembles.config import ExperimentConfig, load_config
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
from scaling_ensembles.tracking import MlflowTracker


LOGGER = logging.getLogger(__name__)


def run_sweep(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.training.device)
    LOGGER.info("Starting sweep config=%s", config_path)
    LOGGER.info(
        "Experiment=%s dataset=%s architecture=%s widths=%s seeds=%s epochs=%s device=%s output_dir=%s",
        config.name,
        config.data.name,
        config.model.architecture,
        list(config.model.widths),
        list(config.training.seeds),
        config.training.epochs,
        device,
        output_dir,
    )

    with MlflowTracker(config, config_path) as tracker:
        train_results: list[TrainResult] = []
        total_runs = len(config.model.widths) * len(config.training.seeds)
        run_index = 0
        for width in tqdm(config.model.widths, desc="width sweep"):
            for seed in tqdm(config.training.seeds, desc=f"seeds width={width}", leave=False):
                run_index += 1
                LOGGER.info("Run %s/%s: width=%s seed=%s", run_index, total_runs, width, seed)
                result = train_one(config, width=width, seed=seed, output_dir=output_dir)
                tracker.log_train_result(result)
                train_results.append(result)

        train_csv = output_dir / "train_results.csv"
        write_train_results(train_results, train_csv)
        tracker.log_artifact(train_csv)
        LOGGER.info("Wrote training results: %s", train_csv)

        _, eval_loader, dataset_info = make_dataloaders(config.data)
        by_width: dict[int, list[TrainResult]] = defaultdict(list)
        for result in train_results:
            by_width[result.width].append(result)

        pairwise_results: list[PairwiseSimilarity] = []
        interpolation_results: list[InterpolationPoint] = []
        for width, results in by_width.items():
            pairs = list(itertools.combinations(results, 2))
            if config.similarity.max_pairs_per_width is not None:
                pairs = pairs[: config.similarity.max_pairs_per_width]
            LOGGER.info(
                "Computing similarities for width=%s with %s checkpoints and %s pairs",
                width,
                len(results),
                len(pairs),
            )

            logits_cache = {}
            targets_cache = None
            for result in tqdm(results, desc=f"logits width={width}", leave=False):
                LOGGER.info("Collecting logits width=%s seed=%s", width, result.seed)
                logits, targets = collect_or_load_logits(
                    config,
                    output_dir,
                    result,
                    eval_loader,
                    dataset_info,
                    device,
                )
                logits_cache[result.seed] = logits
                targets_cache = targets

            if targets_cache is None:
                continue

            for result_a, result_b in tqdm(pairs, desc=f"compare width={width}", leave=False):
                LOGGER.info(
                    "Comparing width=%s seeds=(%s,%s)",
                    width,
                    result_a.seed,
                    result_b.seed,
                )
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
                    LOGGER.info(
                        "Interpolating width=%s seeds=(%s,%s) steps=%s",
                        width,
                        result_a.seed,
                        result_b.seed,
                        config.similarity.interpolation_steps,
                    )
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
        tracker.log_artifact(pairwise_csv)
        LOGGER.info("Wrote pairwise similarity results: %s", pairwise_csv)
        interpolation_csv = output_dir / "interpolation_barriers.csv"
        write_interpolation_results(interpolation_results, interpolation_csv)
        tracker.log_artifact(interpolation_csv)
        LOGGER.info("Wrote interpolation barrier results: %s", interpolation_csv)

        return {
            "train": train_csv,
            "pairwise": pairwise_csv,
            "interpolation": interpolation_csv,
        }


def collect_or_load_logits(
    config: ExperimentConfig,
    output_dir: Path,
    result: TrainResult,
    eval_loader,
    dataset_info,
    device,
):
    cache_path = output_dir / config.cache.cache_dir / "logits" / f"width_{result.width}_seed_{result.seed}.pt"
    if config.cache.enabled and config.cache.reuse_logits and cache_path.exists():
        LOGGER.info("Reusing cached logits: %s", cache_path)
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["logits"], cached["targets"]

    model = load_checkpoint_model(result.checkpoint_path, config, dataset_info, device)
    logits, targets = collect_logits(model, eval_loader, device)
    if config.cache.enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits.cpu(), "targets": targets.cpu()}, cache_path)
        LOGGER.info("Cached logits: %s", cache_path)
    return logits, targets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run a width sweep for functional similarity.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()

    outputs = run_sweep(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
