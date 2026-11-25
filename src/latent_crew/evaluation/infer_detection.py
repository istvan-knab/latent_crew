"""Runs the trained 2D detector on one image (or a folder) and saves the image with the boxes drawn on it.

Each detected object becomes a bounding box labelled with its class name and confidence. If no image is given,
the first image of the YOLO ``val`` split is used, so the command works right after training with no extra input.

Usage:
    python -m latent_crew.evaluation.infer_detection --config configs/detection.yaml \
        --checkpoint checkpoints/sunrgbd_yolo26n_2d/best.pt --image path/to/image.jpg
"""

import argparse
from pathlib import Path

from latent_crew.data.detection_dataset import build_yolo_dataset
from latent_crew.detection_config import DetectionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs YOLO26 detection on an image and draws the boxes.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint (best.pt)")
    parser.add_argument("--image", default=None, help="Image or folder; default: first val image of the dataset")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing a box")
    parser.add_argument("--device", default=None, help="Override hardware.device (cuda | cpu)")
    return parser.parse_args()


def _default_image(config: DetectionConfig) -> str:
    """First image of the YOLO val split, so inference works with no --image after training."""
    data_yaml = build_yolo_dataset(
        processed_root=config.dataset.root,
        raw_root=config.dataset.raw_root,
        yolo_root=config.dataset.yolo_root,
        num_classes=config.dataset.num_classes,
        min_box_size=config.dataset.min_box_size,
    )
    val_dir = data_yaml.parent / "images" / "val"
    images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise FileNotFoundError(f"No val images found in {val_dir}")
    return str(images[0])


def main() -> None:
    args = parse_args()
    config = DetectionConfig.load_from_yaml(args.config)
    if args.device is not None:
        config.hardware.device = args.device

    import torch

    device = "0" if (config.hardware.device == "cuda" and torch.cuda.is_available()) else "cpu"
    image = args.image or _default_image(config)

    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    out_dir = config.run_dir / "predictions"
    results = model.predict(
        source=image,
        conf=args.conf,
        imgsz=config.model.imgsz,
        device=device,
        save=True,  # writes the image with boxes drawn to <project>/<name>/
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
    )

    for result in results:
        boxes = result.boxes
        print(f"\n{Path(result.path).name}: {len(boxes)} object(s)")
        for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            name = result.names[int(cls_id)]
            x1, y1, x2, y2 = (round(v, 1) for v in xyxy)
            print(f"  {name:<16} conf={conf:.2f}  box=({x1}, {y1}, {x2}, {y2})")
        print(f"Saved annotated image to {Path(result.save_dir) / Path(result.path).name}")


if __name__ == "__main__":
    main()
