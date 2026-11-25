"""Trains a 2D object detector (Ultralytics YOLO26) on SUN RGB-D and logs the run to Weights & Biases.

The detector starts from pretrained COCO weights (``model.weights``) and is fine-tuned on the SUN RGB-D 2D
boxes. Model selection uses the ``val`` split of index.jsonl (a fixed, seeded part of the official train set
written by data/scripts/collect_data.py); the official ``test`` split is only touched by
``latent_crew.evaluation.test_detection``.

The YOLO dataset (images + YOLO-format labels + data.yaml) is built once from the processed data and the SUN
2D box metadata by ``latent_crew.data.detection_dataset`` and reused on later runs.

Usage:
    python -m latent_crew.training.train_detection --config configs/detection.yaml
"""

import argparse
import shutil
from pathlib import Path

from latent_crew.data.detection_dataset import build_yolo_dataset
from latent_crew.data.prepare import ensure_dataset
from latent_crew.detection_config import DetectionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains the 2D object detector (YOLO26) on SUN RGB-D.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--device", default=None, help="Override hardware.device (cuda | cpu)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases for this run")
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild the YOLO dataset even if it exists")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    """Returns the ultralytics device string, failing loudly instead of silently training on the CPU."""
    import torch

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "hardware.device is cuda, but CUDA is not available (CPU-only torch build or no GPU access)"
            )
        return "0"  # first CUDA device, in ultralytics notation
    return "cpu"


def main() -> None:
    args = parse_args()
    config = DetectionConfig.load_from_yaml(args.config)

    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.hardware.device = args.device
    if args.no_wandb:
        config.logging.mode = "disabled"

    device = resolve_device(config.hardware.device)

    # 1) Make sure the processed SUN RGB-D data exists (downloads only what is missing), then build the YOLO dataset.
    ensure_dataset(config.dataset.name, config.dataset.root)
    data_yaml = build_yolo_dataset(
        processed_root=config.dataset.root,
        raw_root=config.dataset.raw_root,
        yolo_root=config.dataset.yolo_root,
        num_classes=config.dataset.num_classes,
        min_box_size=config.dataset.min_box_size,
        force=args.rebuild_data,
    )

    # 2) Weights & Biases. Ultralytics has a built-in W&B callback; enable/disable it through its settings.
    from ultralytics import settings as yolo_settings

    yolo_settings.update({"wandb": config.logging.mode != "disabled"})
    if config.logging.mode != "disabled":
        import os

        os.environ.setdefault("WANDB_PROJECT", config.logging.project)
        os.environ["WANDB_MODE"] = config.logging.mode

    # 3) Train. Ultralytics seeds Python/NumPy/Torch from `seed`, saves the best/last checkpoints and metric plots
    #    to <project>/<name>/, and does early stopping with `patience`.
    from ultralytics import YOLO

    model = YOLO(config.model.weights)
    # Absolute path so ultralytics writes to <output_dir>/<name> directly, not under its default runs/detect/.
    run_parent = Path(config.experiment.output_dir).resolve()
    results = model.train(
        data=str(data_yaml),
        epochs=config.training.epochs,
        batch=config.training.batch_size,
        imgsz=config.model.imgsz,
        optimizer=config.training.optimizer,
        lr0=config.training.lr0,
        patience=config.training.patience,
        workers=config.training.workers,
        cache=config.training.cache,
        seed=config.experiment.seed,
        deterministic=True,
        device=device,
        project=str(run_parent),
        name=config.experiment.name,
        exist_ok=True,
    )

    # 4) Save the resolved config next to the run so the training is reproducible from the checkpoint folder.
    run_dir = Path(results.save_dir) if getattr(results, "save_dir", None) else config.run_dir
    config.save_yaml(run_dir / "resolved_config.yaml")
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, run_dir / "best.pt")
    print(f"Training done. Run directory: {run_dir}")
    print(f"Best checkpoint: {best if best.exists() else '(not found)'}")


if __name__ == "__main__":
    main()
