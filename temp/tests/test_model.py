import numpy as np
from src.config import ModelConfig
from src.model import MNISTClassifierBuilder


def test_model_architecture_and_forward_pass():
    cfg = ModelConfig(conv_filters=[16, 32], dense_units=64, dropout=0.2)
    model = MNISTClassifierBuilder.build(cfg)

    dummy_input = np.random.rand(8, 28, 28, 1).astype(np.float32)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (8, 10)
    # Check Softmax sums to 1
    assert np.allclose(np.sum(output, axis=1), 1.0, atol=1e-5)