"""Segmentation config: YAML file -> typed dataclasses, validated on load."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """Reads a config; a top-level `base: other.yaml` (relative to this file) is loaded first and overridden.

    Experiment configs therefore only list what they change, e.g. configs/experiments/dice_ce.yaml.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    return merge(read_yaml(path.parent / base), raw) if base else raw


def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge: values of `override` win, nested sections are merged key by key."""
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@dataclass
class ExperimentConfig:
    """Run name, random seed and where checkpoints go (`experiment:` in the YAML)."""

    name: str
    seed: int
    output_dir: str


@dataclass
class DatasetConfig:
    """Processed dataset location, class count, ignore label and input modalities (`dataset:`)."""

    name: str
    root: str
    num_classes: int
    ignore_index: int
    modalities: List[str]


@dataclass
class ModelConfig:
    """smp decoder + encoder names and pretrained weights (`model:`); see docs/segmentation_models.md."""

    decoder: str
    encoder: str
    encoder_weights: str | None
    decoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.0  # Dropout2d before the segmentation head; 0 = off


@dataclass
class TrainingConfig:
    """Optimisation hyperparameters and the validation / logging cadence (`training:`)."""

    epochs: int
    batch_size: int
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    scheduler: str
    warmup_iters: int
    encoder_lr_mult: float
    class_weighting: str
    amp: bool
    eval_every: int
    log_every: int
    early_stopping_patience: int = 0  # validations without val mIoU gain before stopping; 0 = off
    loss: str = "ce"  # ce | dice_ce
    label_smoothing: float = 0.0  # cross-entropy label smoothing; 0 = off
    l1_lambda: float = 0.0  # weight of the L1 penalty on conv / linear weights; 0 = off


@dataclass
class AugmentationConfig:
    """Training-time augmentation; geometric ops hit rgb, depth and mask alike (`augmentation:`)."""

    crop_size: List[int]
    scale_range: List[float]
    hflip: bool
    color_jitter: float


@dataclass
class HardwareConfig:
    """Device (cuda | mps | cpu) and DataLoader worker count (`hardware:`)."""

    device: str
    num_workers: int


@dataclass
class LoggingConfig:
    """Weights & Biases project, mode and sample-image count (`logging:`)."""

    project: str
    entity: str | None
    mode: str
    num_sample_images: int
    tags: List[str] = field(default_factory=list)


@dataclass
class SegmentationConfig:
    """The full config of a segmentation run: one dataclass per YAML section, validated on construction."""

    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    hardware: HardwareConfig
    logging: LoggingConfig

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SegmentationConfig":
        """Builds the config from the parsed YAML dict; unknown or missing keys raise a TypeError."""
        return cls(
            experiment=ExperimentConfig(**raw["experiment"]),
            dataset=DatasetConfig(**raw["dataset"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            augmentation=AugmentationConfig(**raw["augmentation"]),
            hardware=HardwareConfig(**raw["hardware"]),
            logging=LoggingConfig(**raw["logging"]),
        )

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "SegmentationConfig":
        """Reads and validates a YAML config file (resolving `base:` inheritance, see `read_yaml`)."""
        return cls.from_dict(read_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        """Plain nested dict, same layout as the YAML file (logged to W&B and stored in checkpoints)."""
        return asdict(self)

    def save_yaml(self, dest: Path) -> None:
        """Writes the config next to the checkpoints, so every run can be reproduced."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

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
            "dropout": self.model.dropout,
            "in_channels": self.in_channels,
            "num_classes": self.dataset.num_classes,
        }

    def _validate(self) -> None:
        """Fails early with a readable message instead of deep inside training."""
        # Imported here so that loading a config does not pull in torch at import time
        from latent_crew.models.segmentation import check_model_names

        d, t = self.dataset, self.training
        if d.name != "sunrgbd" or d.num_classes != 37:
            raise ValueError("Only SUN RGB-D with 37 classes is supported (dataset.name: sunrgbd, num_classes: 37)")
        if not d.modalities or not set(d.modalities) <= {"rgb", "depth"}:
            raise ValueError("dataset.modalities must be a non-empty subset of [rgb, depth]")
        check_model_names(self.model.decoder, self.model.encoder)
        if t.optimizer not in ("sgd", "adamw"):
            raise ValueError("training.optimizer must be sgd or adamw")
        if t.scheduler not in ("poly", "cosine"):
            raise ValueError("training.scheduler must be poly or cosine")
        if t.loss not in ("ce", "dice_ce"):
            raise ValueError("training.loss must be ce or dice_ce")
        if not 0.0 <= t.label_smoothing < 1.0:
            raise ValueError("training.label_smoothing must be in [0, 1)")
        if t.class_weighting not in ("none", "median_frequency"):
            raise ValueError("training.class_weighting must be none or median_frequency")
        if min(t.lr, t.batch_size, t.epochs, t.eval_every, t.log_every) <= 0:
            raise ValueError("training.lr, batch_size, epochs, eval_every and log_every must be positive")
        if t.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be >= 0 (0 = off)")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if t.l1_lambda < 0:
            raise ValueError("training.l1_lambda must be >= 0 (0 = off)")
        if not 0.0 <= t.encoder_lr_mult <= 1.0:
            raise ValueError("training.encoder_lr_mult must be in [0, 1]")
        if len(self.augmentation.crop_size) != 2 or any(s % 32 for s in self.augmentation.crop_size):
            raise ValueError("augmentation.crop_size must be [height, width], both multiples of 32")
        if self.hardware.device not in ("cuda", "mps", "cpu"):
            raise ValueError("hardware.device must be cuda, mps or cpu")
        if self.logging.mode not in ("online", "offline", "disabled"):
            raise ValueError("logging.mode must be online, offline or disabled")
