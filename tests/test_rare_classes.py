"""Rare-class augmentation tests: class statistics, copy-paste, rare-class crop and repeat-factor sampling."""

import random
from pathlib import Path

import numpy as np
import pytest

from latent_crew.data.rare_classes import CopyPaste, class_pixel_counts, rare_classes, repeat_factors
from latent_crew.data.transforms import TrainTransform

IGNORE = 255


def test_rare_classes_are_the_smallest_pixel_shares() -> None:
    counts = np.array([[100, 5, 0, 30], [100, 0, 0, 30]])  # class 2 never occurs
    assert rare_classes(counts, 2) == [1, 3]


def test_repeat_factors_follow_lvis() -> None:
    # Class 0 in every image, class 1 only in image 0 (f = 0.25)
    counts = np.array([[10, 1], [10, 0], [10, 0], [10, 0]])
    weights = repeat_factors(counts, threshold=1.0)
    assert weights == pytest.approx([2.0, 1.0, 1.0, 1.0])  # sqrt(1 / 0.25) = 2
    assert repeat_factors(counts, threshold=0.1) == pytest.approx([1.0] * 4)  # nothing below 10 %


def test_class_pixel_counts_are_cached(tmp_path: Path) -> None:
    masks = [np.array([[0, 1], [IGNORE, 1]]), np.array([[2, 2], [2, 2]])]
    calls = []

    def load(i: int) -> np.ndarray:
        calls.append(i)
        return masks[i]

    cache = tmp_path / "counts.npz"
    first = class_pixel_counts(load, ["a", "b"], 3, IGNORE, cache)
    assert first.tolist() == [[1, 2, 0], [0, 0, 4]]
    assert np.array_equal(class_pixel_counts(load, ["a", "b"], 3, IGNORE, cache), first) and calls == [0, 1]
    class_pixel_counts(load, ["a", "c"], 3, IGNORE, cache)  # other ids: recomputed
    assert calls == [0, 1, 0, 1]


def make_sample(h: int, w: int, label: int, depth: float, color: int) -> dict:
    return {"rgb": np.full((h, w, 3), color, np.uint8), "depth": np.full((h, w), depth, np.float32),
            "mask": np.full((h, w), label, np.int64)}


def test_copy_paste_moves_object_to_scene_depth() -> None:
    """A rare object at 1 m pasted into a scene at 2 m: half the size, depth doubled, label and colour kept."""
    donor = make_sample(60, 80, label=0, depth=1000.0, color=10)
    donor["mask"][10:30, 20:40] = 5  # a 20 x 20 object of rare class 5
    donor["rgb"][10:30, 20:40] = 200
    target = make_sample(60, 80, label=1, depth=2000.0, color=50)
    counts = np.zeros((2, 6), np.int64)
    counts[0, 5] = 400
    paste = CopyPaste(lambda i: donor, counts, classes=[5], prob=1.0, max_objects=1)

    random.seed(0)
    out = paste(target, index=1)
    pasted = out["mask"] == 5
    assert pasted.sum() == 100  # 10 x 10
    assert (out["rgb"][pasted] == 200).all() and (out["depth"][pasted] == 2000.0).all()
    assert (out["mask"][~pasted] == 1).all() and (out["rgb"][~pasted] == 50).all()
    assert (target["mask"] == 1).all()  # the input sample is not modified


def test_copy_paste_skips_the_sample_itself() -> None:
    """The only donor of class 0 is the sample itself, so nothing is pasted."""
    sample = make_sample(20, 20, label=5, depth=1000.0, color=0)
    sample["mask"][:] = 0
    paste = CopyPaste(lambda i: make_sample(20, 20, label=0, depth=1000.0, color=99), np.array([[400]]),
                      classes=[0], prob=1.0, max_objects=3)
    assert (paste(sample, index=0)["rgb"] == 0).all()


def test_rare_crop_keeps_the_rare_pixel() -> None:
    mask = np.zeros((200, 300), np.int64)
    mask[190, 5] = 7  # one rare pixel near a corner
    sample = {"rgb": np.zeros((200, 300, 3), np.uint8), "depth": np.zeros((200, 300), np.float32), "mask": mask}
    transform = TrainTransform([64, 96], [1.0, 1.0], hflip=False, color_jitter=0.0, ignore_index=IGNORE,
                               depth_fill=0.0, rare_classes=[7], rare_crop_prob=1.0)
    random.seed(0)
    assert all((transform(sample)["mask"] == 7).any() for _ in range(20))

