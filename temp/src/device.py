import logging
import sys
import tensorflow as tf

logger = logging.getLogger(__name__)


class DeviceManager:
    """Verifies and manages computation devices (GPU / CPU)."""

    @staticmethod
    def setup_device(require_gpu: bool = True) -> dict:
        gpus = tf.config.list_physical_devices("GPU")
        device_info = {
            "gpu_available": len(gpus) > 0,
            "gpu_count": len(gpus),
            "gpu_names": [gpu.name for gpu in gpus],
        }

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU(s) successfully detected: {device_info['gpu_names']}")
            except RuntimeError as e:
                logger.warning(f"Memory growth configuration error: {e}")
        else:
            if require_gpu:
                err_msg = (
                    "CRITICAL: GPU is required (--require-gpu or require_gpu: true), "
                    "but no physical GPU was found by TensorFlow!"
                )
                logger.error(err_msg)
                raise SystemError(err_msg)
            else:
                logger.warning("No GPU found. Proceeding on CPU.")

        return device_info