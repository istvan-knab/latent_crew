import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.config import DataConfig


class MNISTDataModule:
    """Handles MNIST downloading, splitting, normalization, and tf.data pipelines."""

    def __init__(self, config: DataConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def prepare_data(self) -> None:
        (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Convert to float32 and normalize [0, 1]
        x_train_full = x_train_full.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
        x_train_full = np.expand_dims(x_train_full, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        # Stratified train/val split for class distribution balance
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=self.config.validation_fraction,
            random_state=self.seed,
            stratify=y_train_full,
        )

        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test

        self._validate_tensors()

    def _validate_tensors(self) -> None:
        assert self.x_train.shape[1:] == (28, 28, 1), f"Invalid shape: {self.x_train.shape}"
        assert 0.0 <= np.min(self.x_train) and np.max(self.x_train) <= 1.0, "Data not in [0, 1]"
        assert set(np.unique(self.y_train)).issubset(set(range(10))), "Labels out of range 0-9"

    def get_train_dataset(self, limit_batches: int | None = None) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        if self.config.cache:
            ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.config.shuffle_buffer, seed=self.seed)
        ds = ds.batch(self.config.batch_size)
        if limit_batches:
            ds = ds.take(limit_batches)
        if self.config.prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_val_dataset(self, limit_batches: int | None = None) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        if self.config.cache:
            ds = ds.cache()
        ds = ds.batch(self.config.batch_size)
        if limit_batches:
            ds = ds.take(limit_batches)
        if self.config.prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_test_dataset(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        ds = ds.batch(self.config.batch_size)
        if self.config.prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds