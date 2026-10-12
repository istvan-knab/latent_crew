import numpy as np
from src.config import DataConfig
from src.data import MNISTDataModule


def test_data_shapes_and_ranges():
    config = DataConfig(
        validation_fraction=0.1,
        batch_size=64,
        shuffle_buffer=1000,
        cache=False,
        prefetch=False,
    )
    dm = MNISTDataModule(config, seed=42)
    dm.prepare_data()

    assert dm.x_train.shape[1:] == (28, 28, 1)
    assert dm.x_val.shape[1:] == (28, 28, 1)
    assert dm.x_test.shape[1:] == (28, 28, 1)

    assert 0.0 <= np.min(dm.x_train) and np.max(dm.x_train) <= 1.0
    assert set(np.unique(dm.y_train)).issubset(set(range(10)))

    # Dataset pipelines batch check
    train_batch_x, train_batch_y = next(iter(dm.get_train_dataset()))
    assert train_batch_x.shape == (64, 28, 28, 1)
    assert train_batch_y.shape == (64,)