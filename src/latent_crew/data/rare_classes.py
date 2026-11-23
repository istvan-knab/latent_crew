"""Rare-class augmentation: class statistics, copy-paste of rare objects and repeat-factor sampling weights.

The weak classes of SUN RGB-D (bag, paper, box, ...) are small objects that cover few pixels. Flipping or cropping
the few images that show them only repeats the same views, so the tools here show those objects in new contexts:

- `class_pixel_counts` / `rare_classes`: which classes are rare, measured on the train split (cached on disk).
- `CopyPaste`: pastes rare-class objects (RGB, depth and label) cut from other training images into the sample.
- `repeat_factors`: per-image sampling weights, so images with rare classes are drawn more often (LVIS, Gupta 2019).
The rare-class-centred crop lives in `transforms.TrainTransform` (`rare_crop_prob`).

Randomness uses Python's `random`, like transforms.py, so DataLoader workers differ and seeded runs reproduce.
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

Sample = Dict[str, np.ndarray]


# --------------------------------------------------------------------------------------------------------------------
# Class statistics
# --------------------------------------------------------------------------------------------------------------------


def class_pixel_counts(load_mask: Callable[[int], np.ndarray], ids: Sequence[str], num_classes: int,
                       ignore_index: int, cache: Path) -> np.ndarray:
    """Pixel count of every class in every image, int64 [num_images, num_classes].

    Reading all label PNGs takes a while, so the result is cached in `cache` (.npz) and only recomputed when the
    image ids change (e.g. after a re-split).
    """
    if cache.exists():
        data = np.load(cache)
        if data["ids"].tolist() == list(ids) and data["counts"].shape[1] == num_classes:
            return data["counts"]
    counts = np.zeros((len(ids), num_classes), dtype=np.int64)
    for i in tqdm(range(len(ids)), desc="Class pixel counts", unit="img"):
        mask = load_mask(i)
        counts[i] = np.bincount(mask[mask != ignore_index], minlength=num_classes)
    try:
        # Write to a temporary file first, so parallel runs (sweep agents) never read a half-written cache
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, ids=np.array(ids), counts=counts)
        os.replace(tmp, cache)
    except OSError:
        pass  # read-only dataset folder: the counts are simply recomputed next time
    return counts


def rare_classes(counts: np.ndarray, n: int) -> List[int]:
    """The `n` classes with the smallest share of labelled pixels (classes that never occur are skipped)."""
    pixels = counts.sum(axis=0)
    present = np.flatnonzero(pixels > 0)
    return sorted(present[np.argsort(pixels[present], kind="stable")][:n].tolist())


def repeat_factors(counts: np.ndarray, threshold: float) -> np.ndarray:
    """Repeat-factor sampling weight per image (LVIS): r_i = max over the classes c in image i of r_c,

    with r_c = max(1, sqrt(threshold / f_c)) and f_c the fraction of images that contain class c.
    Classes in more than `threshold` of the images get r_c = 1, rarer classes proportionally more.
    """
    present = counts > 0
    freq = present.mean(axis=0)
    class_factor = np.ones(counts.shape[1])
    seen = freq > 0
    class_factor[seen] = np.maximum(1.0, np.sqrt(threshold / freq[seen]))
    # Images without any labelled pixel keep weight 1
    return np.where(present, class_factor, 1.0).max(axis=1, initial=1.0)


# --------------------------------------------------------------------------------------------------------------------
# Copy-paste
# --------------------------------------------------------------------------------------------------------------------


class CopyPaste:
    """Pastes rare-class objects from other training images into a sample (ClassMix / copy-paste style).

    One object is one connected component of a rare class in a donor image. It is pasted at a random position and
    occludes whatever was there, so its pixels take the object's label. Depth stays geometrically consistent:
    the object is moved to the scene depth at the paste position, its depth values are scaled by the same factor,
    and its size is scaled by the inverse (pinhole camera: apparent size ~ 1 / distance).
    Works on raw samples (rgb uint8, depth float32 mm, mask int64), before the random scale / crop.
    """

    def __init__(self, load: Callable[[int], Sample], counts: np.ndarray, classes: Sequence[int], prob: float,
                 max_objects: int, min_area: int = 64, scale_limits: Tuple[float, float] = (0.5, 2.0)):
        """`load(i)` returns raw training sample i; `counts` comes from `class_pixel_counts` of the same images."""
        self.load = load
        self.prob = prob
        self.max_objects = max_objects
        self.min_area = min_area
        self.scale_limits = scale_limits
        # Donor images per rare class: those with at least `min_area` pixels of it
        self.donors = {c: np.flatnonzero(counts[:, c] >= min_area) for c in classes}
        self.donors = {c: d for c, d in self.donors.items() if d.size}
        self.classes = sorted(self.donors)

    def __call__(self, sample: Sample, index: int | None = None) -> Sample:
        """Returns the sample with 1..max_objects rare objects pasted in, with probability `prob`."""
        if not self.classes or random.random() >= self.prob:
            return sample
        sample = {k: v.copy() for k, v in sample.items()}  # never modify the caller's arrays
        for _ in range(random.randint(1, self.max_objects)):
            c = random.choice(self.classes)  # class-balanced: every rare class is equally likely
            donor = int(random.choice(self.donors[c]))
            if donor == index:
                continue  # an object pasted into its own image adds no new context
            obj = self._cut_object(self.load(donor), c)
            if obj is not None:
                self._paste(sample, *obj, c)
        return sample

    def _cut_object(self, donor: Sample, c: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """A random connected component of class c: (rgb, depth, binary region) cropped to its bounding box."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats((donor["mask"] == c).astype(np.uint8), connectivity=8)
        components = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.min_area]
        if not components:
            return None
        k = random.choice(components)
        x, y, w, h = stats[k, :4]
        box = (slice(y, y + h), slice(x, x + w))
        return donor["rgb"][box], donor["depth"][box], labels[box] == k

    def _paste(self, sample: Sample, rgb: np.ndarray, depth: np.ndarray, region: np.ndarray, c: int) -> None:
        """Places one object into `sample` in place, at the scene depth of a random position."""
        H, W = sample["mask"].shape
        h, w = region.shape
        cy, cx = random.randrange(H), random.randrange(W)

        # Scene depth around the paste position vs. the object's own depth decides its new size
        window = sample["depth"][max(cy - h // 2, 0) : cy + h // 2 + 1, max(cx - w // 2, 0) : cx + w // 2 + 1]
        scale = 1.0
        object_depth, scene_depth = _median_positive(depth[region]), _median_positive(window)
        if object_depth > 0 and scene_depth > 0:
            scale = float(np.clip(object_depth / scene_depth, *self.scale_limits))
        scale = min(scale, H / h, W / w)  # the object must fit into the image
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))

        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # Nearest for depth and region (no fake depths or labels across the object's edge);
        # depth / scale moves the object to the distance at which it has this apparent size
        depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST) / scale
        region = cv2.resize(region.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)

        top = min(max(cy - nh // 2, 0), H - nh)
        left = min(max(cx - nw // 2, 0), W - nw)
        box = (slice(top, top + nh), slice(left, left + nw))
        sample["rgb"][box][region] = rgb[region]
        sample["depth"][box][region] = depth[region]
        sample["mask"][box][region] = c


def _median_positive(depth: np.ndarray) -> float:
    """Median over valid (> 0) depth values, 0 if there are none."""
    valid = depth[depth > 0]
    return float(np.median(valid)) if valid.size else 0.0
