import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Enable op-level determinism if supported
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass