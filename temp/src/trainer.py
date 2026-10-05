import logging
from pathlib import Path
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger
from src.config import TrainingConfig
from src.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """Coordinates model training, callbacks, best weights persistence."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: TrainingConfig,
        checkpoint_dir: Path,
        exp_logger: ExperimentLogger,
    ):
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.exp_logger = exp_logger
        self.best_checkpoint_path = self.checkpoint_dir / "best.keras"

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
    ) -> tf.keras.callbacks.History:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.best_checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        if self.exp_logger.enabled:
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))

        logger.info(f"Starting training for {epochs} epochs...")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        # Ensure the model in memory has the exact best saved weights
        if self.best_checkpoint_path.exists():
            logger.info("Reloading best saved checkpoint from disk...")
            self.model = tf.keras.models.load_model(str(self.best_checkpoint_path))

        return history