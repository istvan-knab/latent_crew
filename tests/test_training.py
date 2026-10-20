"""End-to-end smoke test: dataset check -> training -> checkpoint -> test evaluation, on synthetic data, CPU, W&B off."""

import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from latent_crew.config import SegmentationConfig
from latent_crew.data.prepare import count_missing_files, ensure_dataset
from latent_crew.evaluation.metrics import ConfusionMatrix
from latent_crew.evaluation.test_segmentation import test as run_test
from latent_crew.training.train_segmentation import lr_lambda, train
from test_segmentation_dataset import write_fake_dataset

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "segmentation.yaml"


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SegmentationConfig:
    root = tmp_path / "data" / "processed" / "sunrgbd"
    root.mkdir(parents=True)
    write_fake_dataset(root, n_train=8, n_test=2)
    monkeypatch.chdir(tmp_path)  # W&B files go to ./outputs
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["experiment"].update(name="smoke", output_dir=str(tmp_path / "checkpoints"))
    raw["dataset"]["root"] = str(root)
    raw["model"].update(encoder="mobilenet_v2", encoder_weights=None)
    raw["training"].update(epochs=2, batch_size=2, warmup_iters=2, log_every=1, freeze_encoder_epochs=1,
                           class_weighting="median_frequency", val_fraction=0.25)
    raw["augmentation"]["crop_size"] = [64, 96]
    raw["hardware"].update(device="cpu", num_workers=0)
    raw["logging"].update(mode="disabled", num_sample_images=1)
    return SegmentationConfig(raw)


def test_train_then_test(config: SegmentationConfig) -> None:
    best = train(config)
    assert best.exists() and (best.parent / "last.pt").exists() and (best.parent / "config.yaml").exists()
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    assert checkpoint["model_spec"] == config.model_spec()
    assert checkpoint["task"] == "2d_segmentation"

    metrics = run_test(config, best, "test")
    assert 0.0 <= metrics["mIoU"] <= 1.0
    assert (best.parent / "best_test_metrics.json").exists()


def test_existing_dataset_is_not_downloaded(config: SegmentationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("collector must not run when the data is complete")

    monkeypatch.setattr(subprocess, "run", fail)
    ensure_dataset("sunrgbd", config.dataset.root)


def test_missing_dataset_runs_collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data" / "processed" / "sunrgbd"
    calls = []

    def fake_collector(command, check):
        calls.append(command)
        root.mkdir(parents=True)
        write_fake_dataset(root)  # what collect_data.py would produce

    monkeypatch.setattr(subprocess, "run", fake_collector)
    ensure_dataset("sunrgbd", root)
    assert len(calls) == 1
    assert calls[0][1].endswith("data/scripts/collect_data.py")
    assert calls[0][-4:] == ["--datasets", "sunrgbd", "--data-root", str(tmp_path / "data")]
    assert count_missing_files(root) == 0


def test_incomplete_dataset_is_detected(config: SegmentationConfig) -> None:
    root = Path(config.dataset.root)
    (root / "train/semantic/000001.png").unlink()
    assert count_missing_files(root) == 1


def test_root_outside_processed_layout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ensure_dataset("sunrgbd", tmp_path / "somewhere")


def test_confusion_matrix_iou() -> None:
    confusion = ConfusionMatrix(num_classes=3, ignore_index=255)
    target = torch.tensor([[[0, 0, 1, 255]]])
    logits = torch.nn.functional.one_hot(torch.tensor([[[0, 1, 1, 2]]]), 3).permute(0, 3, 1, 2).float()
    confusion.update(logits, target)
    metrics = confusion.compute(["a", "b", "c"])
    # class a: TP 1, FN 1 -> 0.5; class b: TP 1, FP 1 -> 0.5; class c never in ground truth -> excluded
    assert metrics["per_class_iou"] == pytest.approx({"a": 0.5, "b": 0.5})
    assert metrics["mIoU"] == pytest.approx(0.5)
    assert metrics["pixel_accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("scheduler", ("poly", "cosine"))
def test_lr_schedule_warms_up_then_decays(scheduler: str) -> None:
    factor = lr_lambda(scheduler, warmup_iters=10, total_iters=110)
    assert factor(0) == pytest.approx(0.1) and factor(9) == pytest.approx(1.0)
    assert factor(10) == pytest.approx(1.0) and factor(110) == pytest.approx(0.0, abs=1e-9)
    assert factor(50) > factor(90)
