from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ExperimentConfig:
    project: str
    run_name: Optional[str]
    seed: int
    output_dir: str


@dataclass
class DataConfig:
    validation_fraction: float
    batch_size: int
    shuffle_buffer: int
    cache: bool
    prefetch: bool


@dataclass
class ModelConfig:
    conv_filters: List[int]
    dense_units: int
    dropout: float


@dataclass
class TrainingConfig:
    epochs: int
    optimizer: str
    learning_rate: float
    early_stopping_patience: int
    reduce_lr_patience: int
    require_gpu: bool


@dataclass
class LoggingConfig:
    use_wandb: bool
    log_model_artifact: bool


class AppConfig:
    """Strongly-typed application configuration with validation and CLI override."""

    def __init__(self, raw: Dict[str, Any]):
        self.experiment = ExperimentConfig(**raw["experiment"])
        self.data = DataConfig(**raw["data"])
        self.model = ModelConfig(**raw["model"])
        self.training = TrainingConfig(**raw["training"])
        self.logging = LoggingConfig(**raw["logging"])
        self._validate()

    def _validate(self):
        if not (0.0 < self.data.validation_fraction < 0.5):
            raise ValueError("validation_fraction must be in (0.0, 0.5)")
        if self.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0.0 <= self.model.dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": vars(self.experiment),
            "data": vars(self.data),
            "model": vars(self.model),
            "training": vars(self.training),
            "logging": vars(self.logging),
        }

    def save_yaml(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)