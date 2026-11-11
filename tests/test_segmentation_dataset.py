"""Dataset and transform tests on a tiny synthetic dataset in the processed/ format."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from latent_crew.config import SegmentationConfig
from latent_crew.data.segmentation_dataset import SegmentationDataset
from latent_crew.data.transforms import EvalTransform, TrainTransform
from latent_crew.models.segmentation import build_segmentation_model

NUM_CLASSES = 37
IGNORE = 255
HEIGHT, WIDTH = 90, 130  # deliberately not multiples of 32, like the real sensors


def write_fake_dataset(root: Path, n_train: int = 3, n_test: int = 2) -> None:
    """Writes rgb JPEG, uint16 mm depth PNG and uint8 semantic PNG plus index.jsonl, like collect_data.py."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_train + n_test):
        split = "train" if i < n_train else "test"
        stem = f"{i + 1:06d}"
        row = {"id": f"sun_{stem}", "dataset": "sunrgbd", "split": split}
        for field, ext in (("rgb", "jpg"), ("depth", "png"), ("semantic", "png")):
            row[field] = f"{split}/{field}/{stem}.{ext}"
            (root / split / field).mkdir(parents=True, exist_ok=True)
        semantic = rng.integers(0, NUM_CLASSES + 1, (HEIGHT, WIDTH), dtype=np.uint8)
        semantic[:10] = 0  # an unlabeled band
        depth = rng.integers(500, 8000, (HEIGHT, WIDTH)).astype(np.uint16)
        rgb = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.imwrite(str(root / row["rgb"]), rgb)
        cv2.imwrite(str(root / row["depth"]), depth)
        cv2.imwrite(str(root / row["semantic"]), semantic)
        rows.append(row)
    (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    classes = ["unlabeled"] + [f"class_{k}" for k in range(1, NUM_CLASSES + 1)]
    (root / "classes.json").write_text(json.dumps(classes), encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_fake_dataset(tmp_path)
    return tmp_path


def make_dataset(root: Path, split: str, modalities: list, transform=None) -> SegmentationDataset:
    return SegmentationDataset(root, split, "sunrgbd", modalities, NUM_CLASSES, IGNORE, transform)


def test_split_filtering_and_class_names(root: Path) -> None:
    assert len(make_dataset(root, "train", ["rgb"])) == 3
    assert len(make_dataset(root, "test", ["rgb"])) == 2
    assert len(make_dataset(root, "train", ["rgb"]).class_names) == NUM_CLASSES


def test_label_remap(root: Path) -> None:
    sample = make_dataset(root, "train", ["rgb"])[0]
    semantic = cv2.imread(str(root / "train/semantic/000001.png"), cv2.IMREAD_UNCHANGED)
    mask = sample["mask"].numpy()
    assert (mask[semantic == 0] == IGNORE).all()
    assert (mask[semantic > 0] == semantic[semantic > 0].astype(np.int64) - 1).all()
    assert mask[mask != IGNORE].max() <= NUM_CLASSES - 1


@pytest.mark.parametrize(("modalities", "channels"), [(["rgb"], 3), (["rgb", "depth"], 4)])
def test_eval_sample_is_padded_to_multiple_of_32(root: Path, modalities: list, channels: int) -> None:
    transform = EvalTransform(ignore_index=IGNORE, depth_fill=2398.3)
    sample = make_dataset(root, "test", modalities, transform)[0]
    assert sample["image"].shape == (channels, 96, 160)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.int64
    # Padding is ignored by the loss and neutral (0) after normalisation
    assert (sample["mask"][HEIGHT:] == IGNORE).all() and (sample["mask"][:, WIDTH:] == IGNORE).all()
    assert sample["image"][:, HEIGHT:].abs().max() < 0.05


def test_train_transform_keeps_depth_and_mask_aligned() -> None:
    """Depth encodes the label, so any misalignment between the two shows up directly."""
    mask = np.repeat(np.arange(13), 10)[None, :].repeat(90, axis=0).astype(np.int64)  # vertical stripes
    sample = {"rgb": np.zeros((90, 130, 3), np.uint8), "depth": (mask * 100 + 1000).astype(np.float32), "mask": mask}
    transform = TrainTransform([64, 96], [0.5, 2.0], hflip=True, color_jitter=0.0, ignore_index=IGNORE, depth_fill=0.0)
    random.seed(0)
    for _ in range(20):
        out = transform(sample)
        assert out["mask"].shape == out["depth"].shape == (64, 96) and out["rgb"].shape == (64, 96, 3)
        valid = out["mask"] != IGNORE
        assert np.array_equal(out["depth"][valid], out["mask"][valid] * 100 + 1000)


def test_nyu_without_class_mapping_fails(root: Path) -> None:
    with pytest.raises(NotImplementedError):
        SegmentationDataset(root, "train", "nyu_depth_v2", ["rgb"], 40, IGNORE)


def test_batch_runs_through_model(root: Path) -> None:
    transform = TrainTransform([64, 96], [0.5, 2.0], True, 0.2, IGNORE, 2398.3)
    loader = DataLoader(make_dataset(root, "train", ["rgb", "depth"], transform), batch_size=2, shuffle=True)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 4, 64, 96) and batch["mask"].shape == (2, 64, 96)

    spec = {"decoder": "deeplabv3plus", "encoder": "mobilenet_v2", "encoder_weights": None, "in_channels": 4, "num_classes": NUM_CLASSES}
    logits = build_segmentation_model(spec, pretrained=False)(batch["image"])
    loss = torch.nn.functional.cross_entropy(logits, batch["mask"], ignore_index=IGNORE)
    assert torch.isfinite(loss)


def test_from_config(root: Path) -> None:
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs" / "segmentation.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["root"] = str(root)
    raw["augmentation"]["crop_size"] = [64, 96]
    config = SegmentationConfig(raw)
    assert SegmentationDataset.from_config(config, "train")[0]["image"].shape == (4, 64, 96)
    assert SegmentationDataset.from_config(config, "test")[0]["image"].shape == (4, 96, 160)


def test_denormalize_inverts_normalisation(root: Path) -> None:
    dataset = make_dataset(root, "test", ["rgb", "depth"])
    views = dataset.denormalize(dataset[0]["image"])
    rgb = cv2.cvtColor(cv2.imread(str(root / "test/rgb/000004.jpg")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(root / "test/depth/000004.png"), cv2.IMREAD_UNCHANGED)
    assert np.abs(views["rgb"].astype(int) - rgb).max() <= 1
    assert np.allclose(views["depth"], depth, atol=0.5)


def test_render_sample_writes_png(root: Path, tmp_path: Path) -> None:
    from latent_crew.utils.visualization import render_sample

    dataset = make_dataset(root, "train", ["rgb", "depth"])
    sample = dataset[0]
    fig = render_sample(dataset.denormalize(sample["image"]), sample["mask"].numpy(), dataset.class_names, IGNORE, "test")
    assert len(fig.axes) == 4 + 1  # four panels + depth colorbar
    fig.savefig(tmp_path / "sample.png")
    assert (tmp_path / "sample.png").stat().st_size > 0
