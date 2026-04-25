from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any

from scaling_ensembles.config import ExperimentConfig
from scaling_ensembles.train import TrainResult


LOGGER = logging.getLogger(__name__)


class MlflowTracker:
    def __init__(self, config: ExperimentConfig, config_path: str | Path) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.mlflow: Any | None = None

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
        mlflow.start_run(run_name=self.config.tracking.run_name or self.config.name)
        self.log_params()
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
        self.mlflow.log_params(
            {
                "config_path": str(self.config_path),
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
            }
        )

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

    def log_artifact(self, path: str | Path) -> None:
        if self.mlflow is None or not self.config.tracking.log_artifacts:
            return
        artifact_path = Path(path)
        if artifact_path.exists():
            self.mlflow.log_artifact(str(artifact_path))
        else:
            LOGGER.warning("Skipping missing MLflow artifact: %s", artifact_path)
