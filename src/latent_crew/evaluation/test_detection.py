"""Evaluates a trained 2D detector (YOLO26) and reports the standard detection metrics.

Runs the Ultralytics validator on a split of the SUN RGB-D YOLO dataset and prints / saves mAP@0.5,
mAP@0.5:0.95, precision and recall (the metrics used throughout 2D object detection). By default it uses
the ``val`` split (model selection); pass ``--split test`` for the held-out official test split.

Usage:
    python -m latent_crew.evaluation.test_detection --config configs/detection.yaml \
        --checkpoint checkpoints/sunrgbd_yolo26n_2d/best.pt
"""

import argparse
import json
from pathlib import Path

from latent_crew.data.detection_dataset import build_yolo_dataset
from latent_crew.detection_config import DetectionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates a trained YOLO26 detector on SUN RGB-D.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint (best.pt)")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate")
    parser.add_argument("--device", default=None, help="Override hardware.device (cuda | cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionConfig.load_from_yaml(args.config)
    if args.device is not None:
        config.hardware.device = args.device

    import torch

    device = "0" if (config.hardware.device == "cuda" and torch.cuda.is_available()) else "cpu"

    data_yaml = build_yolo_dataset(
        processed_root=config.dataset.root,
        raw_root=config.dataset.raw_root,
        yolo_root=config.dataset.yolo_root,
        num_classes=config.dataset.num_classes,
        min_box_size=config.dataset.min_box_size,
    )

    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=config.model.imgsz,
        device=device,
        project=str(config.run_dir),
        name=f"eval_{args.split}",
        exist_ok=True,
    )

    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "mAP50-95": round(float(metrics.box.map), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    print("\n=== Detection metrics ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    out = config.run_dir / f"metrics_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
