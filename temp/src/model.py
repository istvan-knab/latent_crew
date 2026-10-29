import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import ModelConfig


class MNISTClassifierBuilder:
    """Builds and compiles the baseline Convolutional Neural Network."""

    @staticmethod
    def build(config: ModelConfig, learning_rate: float = 0.001) -> models.Sequential:
        filters = config.conv_filters
        model = models.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(filters[0], (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(filters[1], (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(config.dense_units, activation="relu"),
                layers.Dropout(config.dropout),
                layers.Dense(10, activation="softmax"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model