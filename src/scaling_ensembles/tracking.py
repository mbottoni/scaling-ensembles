from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import platform
import shutil
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from scaling_ensembles.config import ExperimentConfig
from scaling_ensembles.interpolate import InterpolationPoint
from scaling_ensembles.metrics import PairwiseSimilarity
from scaling_ensembles.train import TrainResult


LOGGER = logging.getLogger(__name__)


class MlflowTracker:
    def __init__(self, config: ExperimentConfig, config_path: str | Path) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.mlflow: Any | None = None
        self.active_run_id: str | None = None

    def __enter__(self) -> "MlflowTracker":
        if not self.config.tracking.mlflow_enabled:
            return self

        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "MLflow tracking is enabled, but mlflow is not installed. "
                "Install with `python -m pip install -e '.[tracking]'`."
            ) from error

        self.mlflow = mlflow
        if self.config.tracking.tracking_uri is not None:
            mlflow.set_tracking_uri(self.config.tracking.tracking_uri)
        mlflow.set_experiment(self.config.tracking.experiment_name or self.config.name)
        active_run = mlflow.start_run(run_name=self.config.tracking.run_name or self.config.name)
        self.active_run_id = active_run.info.run_id
        self.log_params()
        self.log_environment()
        self.log_artifact(self.config_path, artifact_path="configs")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.mlflow is None:
            return
        if exc_value is not None:
            self.mlflow.set_tag("status", "failed")
            self.mlflow.set_tag("error", str(exc_value))
        else:
            self.mlflow.set_tag("status", "completed")
        self.mlflow.end_run()

    def log_params(self) -> None:
        if self.mlflow is None:
            return
        params = flatten_dict("config", dataclasses.asdict(self.config))
        params["config_path"] = str(self.config_path)
        params["cache_signature"] = cache_signature(self.config)
        params.update(
            {
                "dataset": self.config.data.name,
                "eval_variant": self.config.data.eval_variant,
                "architecture": self.config.model.architecture,
                "widths": ",".join(map(str, self.config.model.widths)),
                "seeds": ",".join(map(str, self.config.training.seeds)),
                "epochs": self.config.training.epochs,
                "target_train_loss": self.config.training.target_train_loss,
                "device": self.config.training.device,
                "lr": self.config.training.optimizer.lr,
                "weight_decay": self.config.training.optimizer.weight_decay,
                "cache_enabled": self.config.cache.enabled,
                "cache_reuse_checkpoints": self.config.cache.reuse_checkpoints,
                "cache_reuse_logits": self.config.cache.reuse_logits,
                "cache_force_retrain": self.config.cache.force_retrain,
                "mlflow_cache_enabled": self.config.tracking.use_mlflow_cache,
            }
        )
        self.mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_environment(self) -> None:
        if self.mlflow is None or not self.config.tracking.log_environment:
            return
        self.mlflow.log_params(
            {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "torch_version": str(torch.__version__),
                "mps_available": str(torch.backends.mps.is_available()),
                "cuda_available": str(torch.cuda.is_available()),
            }
        )
        git_commit = run_git_command("rev-parse", "HEAD")
        git_dirty = run_git_command("status", "--short")
        if git_commit is not None:
            self.mlflow.set_tag("git_commit", git_commit)
        if git_dirty is not None:
            self.mlflow.set_tag("git_dirty", bool(git_dirty.strip()))

    def log_train_result(self, result: TrainResult) -> None:
        if self.mlflow is None:
            return
        prefix = f"width_{result.width}_seed_{result.seed}"
        self.mlflow.log_metrics(
            {
                f"{prefix}_train_loss": result.train_loss,
                f"{prefix}_train_accuracy": result.train_accuracy,
                f"{prefix}_eval_loss": result.eval_loss,
                f"{prefix}_eval_accuracy": result.eval_accuracy,
                f"{prefix}_epochs_completed": result.epochs_completed,
            }
        )
        self.mlflow.set_tag(f"{prefix}_stop_reason", result.stop_reason)
        if self.config.tracking.log_checkpoints:
            self.log_artifact(result.checkpoint_path, artifact_path="checkpoints")

    def log_pairwise_results(self, results: list[PairwiseSimilarity]) -> None:
        if self.mlflow is None:
            return
        for result in results:
            row = dataclasses.asdict(result)
            prefix = f"pair_width_{result.width}_seeds_{result.seed_a}_{result.seed_b}"
            self.mlflow.log_metrics(
                {
                    f"{prefix}_{key}": float(value)
                    for key, value in row.items()
                    if key not in {"width", "seed_a", "seed_b", "parameter_count"}
                }
            )
        self.log_summary_metrics("pairwise", [dataclasses.asdict(result) for result in results])

    def log_interpolation_results(self, results: list[InterpolationPoint]) -> None:
        if self.mlflow is None:
            return
        for result in results:
            prefix = f"interp_width_{result.width}_seeds_{result.seed_a}_{result.seed_b}"
            step = int(round(result.alpha * 1000))
            self.mlflow.log_metric(f"{prefix}_loss", result.loss, step=step)
            self.mlflow.log_metric(f"{prefix}_accuracy", result.accuracy, step=step)
            self.mlflow.log_metric(f"{prefix}_loss_barrier", result.loss_barrier, step=step)
        self.log_summary_metrics("interpolation", [dataclasses.asdict(result) for result in results])

    def log_summary_metrics(self, group: str, rows: list[dict[str, Any]]) -> None:
        if self.mlflow is None or not rows:
            return
        numeric_keys = [
            key
            for key, value in rows[0].items()
            if isinstance(value, int | float) and key not in {"width", "seed", "seed_a", "seed_b"}
        ]
        metrics: dict[str, float] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
            if not values:
                continue
            metrics[f"{group}_{key}_mean"] = sum(values) / len(values)
            metrics[f"{group}_{key}_min"] = min(values)
            metrics[f"{group}_{key}_max"] = max(values)
        self.mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.mlflow is None or not self.config.tracking.log_artifacts:
            return
        local_path = Path(path)
        if local_path.exists():
            self.mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        else:
            LOGGER.warning("Skipping missing MLflow artifact: %s", local_path)

    def log_logits(self, logits_path: str | Path) -> None:
        if self.config.tracking.log_logits:
            self.log_artifact(logits_path, artifact_path="cache/logits")

    def restore_artifact(self, output_dir: str | Path, artifact_path: str) -> Path | None:
        if (
            self.mlflow is None
            or not self.config.tracking.use_mlflow_cache
            or self.config.cache.force_retrain
        ):
            return None

        output_root = Path(output_dir)
        local_path = output_root / artifact_path
        if local_path.exists():
            return local_path

        experiment = self.mlflow.get_experiment_by_name(
            self.config.tracking.experiment_name or self.config.name
        )
        if experiment is None:
            return None

        client = self.mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=50,
        )
        for run in runs:
            if run.info.run_id == self.active_run_id:
                continue
            if not self._run_matches_config(run):
                continue
            try:
                downloaded_path = client.download_artifacts(
                    run.info.run_id,
                    artifact_path,
                    dst_path=str(output_root),
                )
            except Exception:
                continue
            downloaded = Path(downloaded_path)
            if downloaded != local_path and downloaded.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, local_path)
            LOGGER.info("Restored MLflow artifact cache: %s", local_path)
            return local_path
        return None

    def _run_matches_config(self, run: Any) -> bool:
        params = run.data.params
        return (
            params.get("cache_signature") == cache_signature(self.config)
            or (
                params.get("dataset") == self.config.data.name
                and params.get("eval_variant") == self.config.data.eval_variant
                and params.get("architecture") == self.config.model.architecture
                and params.get("widths") == ",".join(map(str, self.config.model.widths))
                and params.get("seeds") == ",".join(map(str, self.config.training.seeds))
                and params.get("epochs") == str(self.config.training.epochs)
                and params.get("target_train_loss") == str(self.config.training.target_train_loss)
                and params.get("lr") == str(self.config.training.optimizer.lr)
                and params.get("weight_decay") == str(self.config.training.optimizer.weight_decay)
            )
        )


def cache_signature(config: ExperimentConfig) -> str:
    payload = {
        "data": dataclasses.asdict(config.data),
        "model": dataclasses.asdict(config.model),
        "training": dataclasses.asdict(config.training),
        "similarity": dataclasses.asdict(config.similarity),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def flatten_dict(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key, nested_value in value.items():
            flattened.update(flatten_dict(f"{prefix}.{key}", nested_value))
        return flattened
    if isinstance(value, list | tuple):
        return {prefix: ",".join(map(str, value))}
    return {prefix: str(value)}


def run_git_command(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return completed.stdout.strip()
