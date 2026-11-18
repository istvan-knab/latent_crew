"""End-to-end smoke test: dataset check -> training -> checkpoint -> test evaluation, on synthetic data, CPU, W&B off."""

import io
import subprocess
from pathlib import Path

import pytest
import torch
import wandb
import yaml

from latent_crew.config import SegmentationConfig
from latent_crew.data.prepare import count_missing_files, ensure_dataset
from latent_crew.evaluation.metrics import ConfusionMatrix
from latent_crew.evaluation.test_segmentation import test as run_test
from latent_crew.training.train_segmentation import SegmentationTrainer, lr_lambda, train
from test_segmentation_dataset import write_fake_dataset

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "segmentation.yaml"


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SegmentationConfig:
    root = tmp_path / "data" / "processed" / "sunrgbd"
    root.mkdir(parents=True)
    write_fake_dataset(root, n_train=6, n_test=2, n_val=2)
    monkeypatch.chdir(tmp_path)  # W&B files go to ./outputs
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["experiment"].update(name="smoke", output_dir=str(tmp_path / "checkpoints"))
    raw["dataset"]["root"] = str(root)
    raw["model"].update(encoder="mobilenet_v2", encoder_weights=None)
    raw["training"].update(epochs=2, batch_size=2, warmup_iters=2, log_every=1, class_weighting="median_frequency")
    raw["augmentation"]["crop_size"] = [64, 96]
    raw["hardware"].update(device="cpu", num_workers=0)
    raw["logging"].update(mode="disabled", num_sample_images=1)
    return SegmentationConfig.from_dict(raw)


def test_train_then_test(config: SegmentationConfig) -> None:
    best = train(config)
    assert best.exists() and (best.parent / "last.pt").exists() and (best.parent / "config.yaml").exists()
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    assert checkpoint["model_spec"] == config.model_spec()

    metrics = run_test(config, best, "test")
    assert 0.0 <= metrics["mIoU"] <= 1.0
    assert (best.parent / "best_test_metrics.json").exists()


def test_training_logs_recoverable_model_artifact(config: SegmentationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    config.logging.mode = "offline"
    config.logging.num_sample_images = 0
    monkeypatch.setenv("WANDB_CACHE_DIR", str(Path(config.experiment.output_dir).parent / "wandb-cache"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(Path(config.experiment.output_dir).parent / "wandb-data"))
    captured = []
    original = wandb.Run.log_artifact

    def capture(run, artifact, *args, **kwargs):
        files = {name: Path(entry.local_path).read_bytes() for name, entry in artifact.manifest.entries.items()}
        captured.append((artifact, files, run.id))
        return original(run, artifact, *args, **kwargs)

    monkeypatch.setattr(wandb.Run, "log_artifact", capture)
    best = train(config)

    # W&B also stores logged tables (the per-class IoU chart) as artifacts; only the model artifact matters here
    models = [entry for entry in captured if entry[0].type == "model"]
    assert len(models) == 1
    artifact, files, run_id = models[0]
    assert artifact.type == "model"
    assert set(files) == {"best.pt", "last.pt", "config.yaml"}
    assert run_id in artifact.name
    for name, content in files.items():
        assert content == (best.parent / name).read_bytes()
    checkpoint = torch.load(io.BytesIO(files["best.pt"]), map_location="cpu", weights_only=True)
    assert checkpoint["model_spec"] == artifact.metadata["model_spec"] == config.model_spec()
    assert checkpoint["epoch"] == artifact.metadata["best_epoch"]
    assert checkpoint["val_metrics"]["mIoU"] == artifact.metadata["best_val_mIoU"]
    assert torch.load(io.BytesIO(files["last.pt"]), weights_only=True)["epoch"] == config.training.epochs
    assert yaml.safe_load(files["config.yaml"]) == config.to_dict()


def test_disabled_logging_skips_model_artifact(config: SegmentationConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("disabled W&B must not construct or upload model artifacts")

    monkeypatch.setattr(wandb, "Artifact", fail)
    trainer = SegmentationTrainer.__new__(SegmentationTrainer)  # no data or model needed for this check
    trainer.config, trainer.out_dir = config, tmp_path / "missing-checkpoints"
    trainer.log_checkpoint_artifact()


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


def test_training_logs_the_three_sections(config: SegmentationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every validated epoch logs loss/train + loss/val, the metrics/ keys and the samples/ images."""
    import latent_crew.training.train_segmentation as trainer_module
    from latent_crew.utils.wandb_sections import WandbSections

    logged = {}

    class RecordingSections(WandbSections):
        """Records every logged key, also when W&B itself is disabled."""

        def __init__(self, run):
            super().__init__(run)
            log = run.log
            self.run = type("RecordingRun", (), {"log": lambda _, data, **kw: (logged.update(data), log(data, **kw))})()

    monkeypatch.setattr(trainer_module, "WandbSections", RecordingSections)
    train(config)
    for key in ("loss/train", "loss/val", "metrics/mIoU", "metrics/pixel_accuracy", "metrics/mean_class_accuracy",
                "metrics/per_class_iou", "metrics/lr_decoder", "loss/train_vs_val", "samples/validation",
                "optim/train_loss_iter"):
        assert key in logged, key
    assert logged["loss/val"] > 0 and logged["loss/train"] > 0


def test_trainer_uses_the_val_split_of_the_index(config: SegmentationConfig) -> None:
    """Validation reads the `val` rows of index.jsonl; nothing is carved off the train split any more."""
    trainer = SegmentationTrainer(config)
    assert {s["split"] for s in trainer.train_set.samples} == {"train"}
    assert {s["split"] for s in trainer.val_set.samples} == {"val"}
    assert (len(trainer.train_set), len(trainer.val_set)) == (6, 2)


def test_early_stopping_counts_validations_without_gain() -> None:
    from latent_crew.training.train_segmentation import EarlyStopping

    stopper = EarlyStopping(patience=2)
    assert [stopper.should_stop(m) for m in (0.1, 0.2, 0.2, 0.15)] == [False, False, False, True]
    assert not any(EarlyStopping(patience=0).should_stop(0.0) for _ in range(10))  # 0 = off


def test_training_stops_early(config: SegmentationConfig) -> None:
    config.training.epochs, config.training.early_stopping_patience = 5, 1
    best = train(config)
    last = torch.load(best.parent / "last.pt", map_location="cpu", weights_only=True)
    assert last["epoch"] < 5  # random weights on random labels: mIoU stalls quickly


def test_training_with_dice_ce_loss(config: SegmentationConfig) -> None:
    config.training.epochs, config.training.loss = 1, "dice_ce"
    assert train(config).exists()
