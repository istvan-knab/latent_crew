"""2D detection config: YAML file -> typed dataclasses, validated on load.

Mirrors the style of ``latent_crew.config`` (segmentation) but holds the keys the YOLO detector needs.
Loading this module does not import torch or ultralytics, so it stays cheap to parse a config.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """Reads a config; a top-level ``base: other.yaml`` (relative to this file) is loaded first and overridden."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    return merge(read_yaml(path.parent / base), raw) if base else raw


def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge: values of ``override`` win, nested sections are merged key by key."""
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@dataclass
class ExperimentConfig:
    """Run name, random seed and where run artifacts go (``experiment:`` in the YAML)."""

    name: str
    seed: int
    output_dir: str


@dataclass
class DatasetConfig:
    """Where the processed data and the SUN 2D box metadata live, and how the YOLO dataset is built."""

    name: str
    root: str  # processed/<name> with index.jsonl
    raw_root: str  # holds <name>/metadata/SUNRGBDMeta2DBB_v2.mat
    yolo_root: str  # generated YOLO dataset (images + labels + data.yaml)
    num_classes: int  # keep the N most frequent class names
    min_box_size: int = 4  # drop boxes smaller than this many pixels on a side


@dataclass
class ModelConfig:
    """Pretrained YOLO weights and the training/val image size (``model:``)."""

    weights: str
    imgsz: int = 640


@dataclass
class TrainingConfig:
    """Optimisation hyperparameters passed through to ultralytics (``training:``)."""

    epochs: int
    batch_size: int
    optimizer: str = "auto"
    lr0: float = 0.01
    patience: int = 15
    workers: int = 8
    cache: bool = False


@dataclass
class HardwareConfig:
    """Device (cuda | cpu); fails early if cuda is requested but unavailable (``hardware:``)."""

    device: str


@dataclass
class LoggingConfig:
    """Weights & Biases project and mode (``logging:``)."""

    project: str
    mode: str = "online"


@dataclass
class DetectionConfig:
    """The full config of a 2D detection run: one dataclass per YAML section, validated on construction."""

    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    hardware: HardwareConfig
    logging: LoggingConfig

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DetectionConfig":
        """Builds the config from the parsed YAML dict; unknown or missing keys raise a TypeError."""
        return cls(
            experiment=ExperimentConfig(**raw["experiment"]),
            dataset=DatasetConfig(**raw["dataset"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            hardware=HardwareConfig(**raw["hardware"]),
            logging=LoggingConfig(**raw["logging"]),
        )

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "DetectionConfig":
        """Reads and validates a YAML config file (resolving ``base:`` inheritance, see ``read_yaml``)."""
        return cls.from_dict(read_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        """Plain nested dict, same layout as the YAML file (logged to W&B and saved next to checkpoints)."""
        return asdict(self)

    def save_yaml(self, dest: Path) -> None:
        """Writes the resolved config next to the run output, so every run can be reproduced."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @property
    def run_dir(self) -> Path:
        """<output_dir>/<name>: where the YOLO run (weights, metrics, plots) is written (absolute)."""
        return (Path(self.experiment.output_dir) / self.experiment.name).resolve()

    def _validate(self) -> None:
        """Fails early with a readable message instead of deep inside training."""
        d, t = self.dataset, self.training
        if d.name != "sunrgbd":
            raise ValueError("Only SUN RGB-D is supported for 2D detection (dataset.name: sunrgbd)")
        if d.num_classes <= 0:
            raise ValueError("dataset.num_classes must be positive (keep the N most frequent class names)")
        if d.min_box_size < 0:
            raise ValueError("dataset.min_box_size must be >= 0")
        if self.model.imgsz % 32:
            raise ValueError("model.imgsz must be a multiple of 32 (YOLO stride)")
        if min(t.epochs, t.batch_size, t.workers) <= 0:
            raise ValueError("training.epochs, batch_size and workers must be positive")
        if t.patience < 0:
            raise ValueError("training.patience must be >= 0 (0 = off)")
        if t.optimizer not in ("auto", "SGD", "Adam", "AdamW", "sgd", "adam", "adamw"):
            raise ValueError("training.optimizer must be auto, SGD, Adam or AdamW")
        if self.hardware.device not in ("cuda", "cpu"):
            raise ValueError("hardware.device must be cuda or cpu")
        if self.logging.mode not in ("online", "offline", "disabled"):
            raise ValueError("logging.mode must be online, offline or disabled")
