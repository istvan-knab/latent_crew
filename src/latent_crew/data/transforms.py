"""Paired transforms: every geometric op is applied identically to rgb, depth and mask.

Samples are dicts of numpy arrays: rgb uint8 [H, W, 3], depth float32 mm [H, W], mask int64 [H, W].
Randomness comes from Python's `random`, which DataLoader reseeds in every worker (as it does torch and NumPy),
so workers draw different augmentations and seed_everything() makes runs reproducible.
"""

import random
from typing import Dict, List

import cv2
import numpy as np

Sample = Dict[str, np.ndarray]

# Padding the RGB with the ImageNet mean pixel makes padded areas 0 after normalisation
RGB_FILL = (124, 116, 104)


def resize(sample: Sample, height: int, width: int) -> Sample:
    """Bilinear for rgb; nearest for depth (no fake depths across edges) and mask (no fake classes)."""
    return {
        "rgb": cv2.resize(sample["rgb"], (width, height), interpolation=cv2.INTER_LINEAR),
        "depth": cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST),
        # cv2 cannot resize int64, the labels fit in int32
        "mask": cv2.resize(sample["mask"].astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64),
    }


def pad_to(sample: Sample, height: int, width: int, ignore_index: int, depth_fill: float) -> Sample:
    """Pads bottom/right up to (height, width); padded mask pixels are ignored by the loss."""
    h, w = sample["mask"].shape
    pad = ((0, max(height - h, 0)), (0, max(width - w, 0)))
    if pad == ((0, 0), (0, 0)):
        return sample
    rgb = np.stack([np.pad(sample["rgb"][..., c], pad, constant_values=RGB_FILL[c]) for c in range(3)], axis=-1)
    return {
        "rgb": rgb,
        "depth": np.pad(sample["depth"], pad, constant_values=depth_fill),
        "mask": np.pad(sample["mask"], pad, constant_values=ignore_index),
    }


def color_jitter(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Random brightness, contrast and saturation in [1 - strength, 1 + strength]; RGB only."""
    image = rgb.astype(np.float32)
    image *= random.uniform(1 - strength, 1 + strength)  # brightness
    mean = image.mean()
    image = (image - mean) * random.uniform(1 - strength, 1 + strength) + mean  # contrast
    gray = image.mean(axis=-1, keepdims=True)
    image = (image - gray) * random.uniform(1 - strength, 1 + strength) + gray  # saturation
    return np.clip(image, 0, 255).astype(np.uint8)


class TrainTransform:
    """Random scale -> pad -> random crop -> horizontal flip -> colour jitter."""

    def __init__(
        self,
        crop_size: List[int],
        scale_range: List[float],
        hflip: bool,
        color_jitter: float,
        ignore_index: int,
        depth_fill: float,
    ):
        self.crop_h, self.crop_w = crop_size
        self.scale_range = scale_range
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.ignore_index = ignore_index
        self.depth_fill = depth_fill

    def __call__(self, sample: Sample) -> Sample:
        h, w = sample["mask"].shape
        scale = random.uniform(*self.scale_range)
        sample = resize(sample, round(h * scale), round(w * scale))
        sample = pad_to(sample, self.crop_h, self.crop_w, self.ignore_index, self.depth_fill)

        h, w = sample["mask"].shape
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        sample = {k: v[top : top + self.crop_h, left : left + self.crop_w] for k, v in sample.items()}

        if self.hflip and random.random() < 0.5:
            sample = {k: v[:, ::-1] for k, v in sample.items()}
        if self.color_jitter > 0:
            sample["rgb"] = color_jitter(sample["rgb"], self.color_jitter)
        return sample


class EvalTransform:
    """Keeps the full resolution, only pads up to a multiple of 32 (the encoder's total stride).

    The padded mask pixels are ignored, so the metric only counts the original image.
    """

    def __init__(self, ignore_index: int, depth_fill: float, multiple: int = 32):
        self.ignore_index = ignore_index
        self.depth_fill = depth_fill
        self.multiple = multiple

    def __call__(self, sample: Sample) -> Sample:
        h, w = sample["mask"].shape
        m = self.multiple
        return pad_to(sample, -(-h // m) * m, -(-w // m) * m, self.ignore_index, self.depth_fill)
