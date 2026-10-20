"""Typed configuration for segmentation training and evaluation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Task label of every run; logged to W&B as tag and job type. Only 2d_segmentation is implemented so far.
TASKS = ("2d_segmentation", "3d_segmentation", "2d_detection", "3d_detection")
SUPPORTED_WANDB_MODES = ("online", "offline", "disabled")
SUPPORTED_DATASETS = ("sunrgbd", "nyu_depth_v2")
SUPPORTED_DEVICES = ("cuda", "mps", "cpu")
# Output classes without "unlabeled": SUN RGB-D 37, NYU Depth V2 mapped to NYU40
NUM_CLASSES = {"sunrgbd": 37, "nyu_depth_v2": 40}
SUPPORTED_MODALITIES = ("rgb", "depth")
SUPPORTED_OPTIMIZERS = ("sgd", "adamw")
SUPPORTED_SCHEDULERS = ("poly", "cosine")
SUPPORTED_CLASS_WEIGHTING = ("none", "median_frequency")


@dataclass
class ExperimentConfig:
    name: str
    task: str
    seed: int
    output_dir: str


@dataclass
class LoggingConfig:
    project: str
    entity: str | None
    mode: str
    num_sample_images: int
    tags: List[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    name: str
    root: str
    num_classes: int
    ignore_index: int
    modalities: List[str]


@dataclass
class ModelConfig:
    decoder: str
    encoder: str
    encoder_weights: str | None
    decoder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    scheduler: str
    warmup_iters: int
    encoder_lr_mult: float
    freeze_encoder_epochs: int
    class_weighting: str
    amp: bool
    val_fraction: float
    eval_every: int
    log_every: int


@dataclass
class AugmentationConfig:
    crop_size: List[int]
    scale_range: List[float]
    hflip: bool
    color_jitter: float


@dataclass
class HardwareConfig:
    device: str
    num_workers: int


class SegmentationConfig:
    """Segmentation config loaded from YAML, validated on load."""

    def __init__(self, raw: Dict[str, Any]):
        self.experiment = ExperimentConfig(**raw["experiment"])
        self.dataset = DatasetConfig(**raw["dataset"])
        self.model = ModelConfig(**raw["model"])
        self.training = TrainingConfig(**raw["training"])
        self.augmentation = AugmentationConfig(**raw["augmentation"])
        self.hardware = HardwareConfig(**raw["hardware"])
        self.logging = LoggingConfig(**raw["logging"])
        self._validate()

    @property
    def wandb_tags(self) -> List[str]:
        """The task label is always the first tag, so runs can be filtered by task in W&B."""
        return [self.experiment.task] + [t for t in self.logging.tags if t != self.experiment.task]

    @property
    def in_channels(self) -> int:
        """RGB gives 3 input channels, depth adds 1 (early fusion)."""
        return 3 * ("rgb" in self.dataset.modalities) + ("depth" in self.dataset.modalities)

    def model_spec(self) -> Dict[str, Any]:
        """Everything needed to rebuild the network; stored in every checkpoint."""
        return {
            "decoder": self.model.decoder,
            "encoder": self.model.encoder,
            "encoder_weights": self.model.encoder_weights,
            "decoder_kwargs": dict(self.model.decoder_kwargs),
            "in_channels": self.in_channels,
            "num_classes": self.dataset.num_classes,
        }

    def _validate(self) -> None:
        # Imported here so that loading a config does not pull in torch at import time
        from latent_crew.models.segmentation import check_model_names

        if self.experiment.task not in TASKS:
            raise ValueError(f"experiment.task must be one of {TASKS}")
        if self.experiment.task != "2d_segmentation":
            raise NotImplementedError(f"{self.experiment.task} is a planned task, only 2d_segmentation is implemented")
        if self.logging.mode not in SUPPORTED_WANDB_MODES:
            raise ValueError(f"logging.mode must be one of {SUPPORTED_WANDB_MODES}")
        if self.dataset.name not in SUPPORTED_DATASETS:
            raise ValueError(f"dataset.name must be one of {SUPPORTED_DATASETS}")
        if not self.dataset.modalities or not set(self.dataset.modalities) <= set(SUPPORTED_MODALITIES):
            raise ValueError(f"dataset.modalities must be a non-empty subset of {SUPPORTED_MODALITIES}")
        if self.dataset.num_classes != NUM_CLASSES[self.dataset.name]:
            raise ValueError(f"dataset.num_classes must be {NUM_CLASSES[self.dataset.name]} for {self.dataset.name}")
        check_model_names(self.model.decoder, self.model.encoder)
        if self.training.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"training.optimizer must be one of {SUPPORTED_OPTIMIZERS}")
        if self.training.scheduler not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"training.scheduler must be one of {SUPPORTED_SCHEDULERS}")
        if self.training.class_weighting not in SUPPORTED_CLASS_WEIGHTING:
            raise ValueError(f"training.class_weighting must be one of {SUPPORTED_CLASS_WEIGHTING}")
        if self.training.lr <= 0 or self.training.batch_size <= 0 or self.training.epochs <= 0:
            raise ValueError("training.lr, batch_size and epochs must be positive")
        if not 0.0 <= self.training.encoder_lr_mult <= 1.0:
            raise ValueError("training.encoder_lr_mult must be in [0.0, 1.0]")
        if self.hardware.device not in SUPPORTED_DEVICES:
            raise ValueError(f"hardware.device must be one of {SUPPORTED_DEVICES}")
        if not 0.0 < self.training.val_fraction < 0.5:
            raise ValueError("training.val_fraction must be in (0.0, 0.5)")
        if self.training.eval_every <= 0 or self.training.log_every <= 0:
            raise ValueError("training.eval_every and log_every must be positive")
        if not 0 <= self.training.freeze_encoder_epochs < self.training.epochs:
            raise ValueError("training.freeze_encoder_epochs must be in [0, epochs)")
        if len(self.augmentation.crop_size) != 2 or any(s % 32 for s in self.augmentation.crop_size):
            raise ValueError("augmentation.crop_size must be [height, width], both multiples of 32")

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "SegmentationConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": asdict(self.experiment),
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "augmentation": asdict(self.augmentation),
            "hardware": asdict(self.hardware),
            "logging": asdict(self.logging),
        }

    def save_yaml(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
