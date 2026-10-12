#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DataConfig
from src.data import MNISTDataModule
from src.evaluator import ClassifierEvaluator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvaluateClassifier")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST classifier checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .keras checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    logger.info(f"Loading weights from {ckpt_path}...")
    model = tf.keras.models.load_model(str(ckpt_path))

    data_config = DataConfig(
        validation_fraction=0.1,
        batch_size=128,
        shuffle_buffer=1000,
        cache=False,
        prefetch=False,
    )
    dm = MNISTDataModule(data_config)
    dm.prepare_data()

    evaluator = ClassifierEvaluator(model=model, output_dir=Path(args.output_dir))
    metrics = evaluator.evaluate(dm.x_test, dm.y_test)

    logger.info("Evaluation Complete.")
    logger.info(f"Test Accuracy: {metrics['test_accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()