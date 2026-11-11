"""PyTorch Dataset for 2D (RGB or RGB-D) semantic segmentation on the processed index.jsonl."""

import json
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from latent_crew.config import SegmentationConfig
from latent_crew.data.transforms import EvalTransform, TrainTransform

# ImageNet statistics, because the encoders are pretrained on ImageNet (0-1 scale)
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Filled depth over valid pixels, mm (data/reports/data_evaluation.md)
DEPTH_STATS_MM = {"sunrgbd": (2398.3, 1248.9), "nyu_depth_v2": (2796.3, 1386.1)}


def build_transform(config: SegmentationConfig, augment: bool) -> TrainTransform | EvalTransform:
    d, a = config.dataset, config.augmentation
    depth_mean, _ = DEPTH_STATS_MM[d.name]
    if not augment:
        return EvalTransform(ignore_index=d.ignore_index, depth_fill=depth_mean)
    return TrainTransform(
        crop_size=a.crop_size,
        scale_range=a.scale_range,
        hflip=a.hflip,
        color_jitter=a.color_jitter,
        ignore_index=d.ignore_index,
        depth_fill=depth_mean,
    )


class SegmentationDataset(Dataset):
    """Returns {"image": float [C, H, W], "mask": long [H, W], "id": str} per sample.

    Channels follow `modalities`: rgb (3, ImageNet-normalised), then depth (1, dataset-normalised).
    Labels are remapped: 0 (unlabeled) -> ignore_index, k -> k - 1, so the classes are 0 .. num_classes - 1.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        dataset: str,
        modalities: List[str],
        num_classes: int,
        ignore_index: int,
        transform: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]] | None = None,
    ):
        if dataset == "nyu_depth_v2":
            # processed/ stores the 894 original NYU labels; training needs NYU40
            raise NotImplementedError("NYU needs the 894 -> 40 class mapping, which is not in processed/ yet")
        self.root = Path(root)
        self.modalities = modalities
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.transform = transform
        self.depth_mean, self.depth_std = DEPTH_STATS_MM[dataset]

        index = self.root / "index.jsonl"
        if not index.exists():
            raise FileNotFoundError(f"{index} not found: run data/scripts/collect_data.py first")
        with open(index, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.samples = [r for r in rows if r["split"] == split]
        if not self.samples:
            raise ValueError(f"No {split!r} samples in {index}")
        self.class_names = json.loads((self.root / "classes.json").read_text(encoding="utf-8"))[1:]

    @classmethod
    def from_config(cls, config: SegmentationConfig, split: str, augment: bool | None = None) -> "SegmentationDataset":
        """Train split gets random augmentation, every other split only padding (override with augment)."""
        d = config.dataset
        augment = split == "train" if augment is None else augment
        return cls(d.root, split, d.name, d.modalities, d.num_classes, d.ignore_index, build_transform(config, augment))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | str]:
        row = self.samples[i]
        sample = {
            # OpenCV reads BGR; the pretrained encoders expect RGB
            "rgb": cv2.cvtColor(self._read(row["rgb"]), cv2.COLOR_BGR2RGB),
            "depth": self._read(row["depth"]).astype(np.float32),  # uint16 mm
            "mask": self._remap_labels(self._read(row["semantic"])),
        }
        if self.transform is not None:
            sample = self.transform(sample)

        channels = []
        if "rgb" in self.modalities:
            channels.append((sample["rgb"].astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD)
        if "depth" in self.modalities:
            channels.append(((sample["depth"] - self.depth_mean) / self.depth_std)[..., None])
        image = np.concatenate(channels, axis=-1).transpose(2, 0, 1)  # HWC -> CHW

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            # Flips and crops leave numpy views that torch cannot wrap, hence the copies
            "mask": torch.from_numpy(np.ascontiguousarray(sample["mask"])),
            "id": row["id"],
        }

    def label_mask(self, i: int) -> np.ndarray:
        """Remapped label mask of sample i at full resolution, without reading the images (for class statistics)."""
        return self._remap_labels(self._read(self.samples[i]["semantic"]))

    def denormalize(self, image: torch.Tensor) -> Dict[str, np.ndarray]:
        """Inverse of the normalisation in __getitem__: {"rgb": uint8 [H, W, 3], "depth": mm [H, W]}."""
        channels = image.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        out = {}
        if "rgb" in self.modalities:
            rgb = (channels[..., :3] * RGB_STD + RGB_MEAN) * 255.0
            out["rgb"] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        if "depth" in self.modalities:
            out["depth"] = channels[..., -1] * self.depth_std + self.depth_mean
        return out

    def _read(self, relative_path: str) -> np.ndarray:
        path = self.root / relative_path
        # IMREAD_UNCHANGED keeps the uint16 depth and label PNGs as they are
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise OSError(f"Cannot read {path}")
        return array

    def _remap_labels(self, semantic: np.ndarray) -> np.ndarray:
        labels = semantic.astype(np.int64)
        if labels.max() > self.num_classes:
            raise ValueError(f"Label {labels.max()} exceeds num_classes={self.num_classes}")
        mask = labels - 1
        mask[labels == 0] = self.ignore_index
        return mask
