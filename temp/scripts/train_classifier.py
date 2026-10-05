#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

# Ensure temp directory is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.data import MNISTDataModule
from src.device import DeviceManager
from src.evaluator import ClassifierEvaluator
from src.experiment_logger import ExperimentLogger
from src.model import MNISTClassifierBuilder
from src.trainer import ClassifierTrainer
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainClassifier")


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline MNIST classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--run-name", type=str, default=None, help="Override W&B run name")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Limit batches for fast smoke testing")
    parser.add_argument("--require-gpu", action="store_true", default=None, help="Enforce GPU usage")
    parser.add_argument("--allow-cpu", action="store_true", default=False, help="Allow running on CPU")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable Weights & Biases")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    app_config = AppConfig.load_from_yaml(config_path)

    # CLI Overrides
    if args.epochs is not None:
        app_config.training.epochs = args.epochs
    if args.batch_size is not None:
        app_config.data.batch_size = args.batch_size
    if args.learning_rate is not None:
        app_config.training.learning_rate = args.learning_rate
    if args.run_name is not None:
        app_config.experiment.run_name = args.run_name
    if args.allow_cpu:
        app_config.training.require_gpu = False
    elif args.require_gpu is not None:
        app_config.training.require_gpu = args.require_gpu
    if args.no_wandb:
        app_config.logging.use_wandb = False

    output_dir = Path(app_config.experiment.output_dir)
    set_seed(app_config.experiment.seed)

    # Check hardware
    device_info = DeviceManager.setup_device(require_gpu=app_config.training.require_gpu)

    # Init experiment tracker
    exp_logger = ExperimentLogger(enabled=app_config.logging.use_wandb)
    exp_logger.start_run(
        project=app_config.experiment.project,
        config=app_config.to_dict(),
        run_name=app_config.experiment.run_name,
        system_info=device_info,
    )

    try:
        # Data loading
        data_mod = MNISTDataModule(config=app_config.data, seed=app_config.experiment.seed)
        data_mod.prepare_data()
        train_ds = data_mod.get_train_dataset(limit_batches=args.limit_train_batches)
        val_ds = data_mod.get_val_dataset(limit_batches=args.limit_train_batches)

        # Model creation
        model = MNISTClassifierBuilder.build(
            config=app_config.model,
            learning_rate=app_config.training.learning_rate,
        )
        model.summary(print_fn=logger.info)

        # Training
        trainer = ClassifierTrainer(
            model=model,
            config=app_config.training,
            checkpoint_dir=output_dir / "checkpoints",
            exp_logger=exp_logger,
        )
        history = trainer.train(train_ds, val_ds, epochs=app_config.training.epochs)

        # Plot curves
        curves_path = output_dir / "figures" / "training_curves.png"
        ClassifierEvaluator.plot_training_curves(history, curves_path)

        # Final Evaluation
        evaluator = ClassifierEvaluator(model=trainer.model, output_dir=output_dir)
        metrics = evaluator.evaluate(data_mod.x_test, data_mod.y_test)
        logger.info(f"Final Test Evaluation: Accuracy = {metrics['test_accuracy'] * 100:.2f}%")

        # Save config backup
        app_config.save_yaml(output_dir / "reports" / "resolved_config.yaml")

        # Log artifacts to W&B
        if app_config.logging.log_model_artifact:
            exp_logger.log_artifact(trainer.best_checkpoint_path, "model", "mnist-best-model")
            exp_logger.log_artifact(curves_path, "figure", "training-curves")
            exp_logger.log_artifact(output_dir / "figures" / "confusion_matrix.png", "figure", "confusion-matrix")

    finally:
        exp_logger.finish()


if __name__ == "__main__":
    main()