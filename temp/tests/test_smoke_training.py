from pathlib import Path
import numpy as np
from src.config import DataConfig, ModelConfig, TrainingConfig
from src.data import MNISTDataModule
from src.evaluator import ClassifierEvaluator
from src.experiment_logger import ExperimentLogger
from src.model import MNISTClassifierBuilder
from src.trainer import ClassifierTrainer


def test_smoke_training_run(tmp_path: Path):
    data_cfg = DataConfig(validation_fraction=0.1, batch_size=32, shuffle_buffer=100, cache=False, prefetch=False)
    dm = MNISTDataModule(data_cfg)
    dm.prepare_data()

    train_ds = dm.get_train_dataset(limit_batches=2)
    val_ds = dm.get_val_dataset(limit_batches=2)

    model_cfg = ModelConfig(conv_filters=[8, 16], dense_units=32, dropout=0.1)
    model = MNISTClassifierBuilder.build(model_cfg)

    train_cfg = TrainingConfig(
        epochs=1,
        optimizer="adam",
        learning_rate=0.001,
        early_stopping_patience=2,
        reduce_lr_patience=1,
        require_gpu=False,
    )

    exp_logger = ExperimentLogger(enabled=False)
    trainer = ClassifierTrainer(model, train_cfg, tmp_path / "checkpoints", exp_logger)
    history = trainer.train(train_ds, val_ds, epochs=1)

    assert "loss" in history.history
    assert np.isfinite(history.history["loss"][0])
    assert (tmp_path / "checkpoints" / "best.keras").exists()

    evaluator = ClassifierEvaluator(trainer.model, tmp_path)
    metrics = evaluator.evaluate(dm.x_test[:64], dm.y_test[:64])
    assert "test_accuracy" in metrics