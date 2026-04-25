from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str = "MNIST"
    root: str = "data"
    batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 0
    train_subset: int | None = None
    eval_subset: int | None = None
    eval_variant: str = "clean"
    noise_std: float = 0.15
    blur_kernel_size: int = 3


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "mlp"
    widths: tuple[int, ...] = (32, 64, 128)
    hidden_layers: int = 2
    activation: str = "relu"
    patch_size: int = 4
    num_heads: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "auto"
    target_train_loss: float | None = None
    min_epochs: int = 1
    eval_every_epochs: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class SimilarityConfig:
    max_pairs_per_width: int | None = None
    interpolation_steps: int = 11


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    reuse_checkpoints: bool = True
    reuse_logits: bool = True
    force_retrain: bool = False
    cache_dir: str = "cache"


@dataclass(frozen=True)
class TrackingConfig:
    mlflow_enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    log_artifacts: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "mnist-width-sweep"
    output_dir: str = "outputs/mnist-width-sweep"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config into typed dataclasses."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    default_config = ExperimentConfig()
    data = DataConfig(**raw.get("data", {}))
    model_raw = raw.get("model", {})
    model = ModelConfig(
        **{
            **model_raw,
            "widths": tuple(model_raw.get("widths", ModelConfig().widths)),
        }
    )
    optimizer = OptimizerConfig(**raw.get("training", {}).get("optimizer", {}))
    training_raw = {k: v for k, v in raw.get("training", {}).items() if k != "optimizer"}
    training = TrainingConfig(
        **{
            **training_raw,
            "seeds": tuple(training_raw.get("seeds", TrainingConfig().seeds)),
            "optimizer": optimizer,
        }
    )
    similarity = SimilarityConfig(**raw.get("similarity", {}))
    cache = CacheConfig(**raw.get("cache", {}))
    tracking = TrackingConfig(**raw.get("tracking", {}))
    return ExperimentConfig(
        name=raw.get("name", default_config.name),
        output_dir=raw.get("output_dir", default_config.output_dir),
        data=data,
        model=model,
        training=training,
        similarity=similarity,
        cache=cache,
        tracking=tracking,
    )
