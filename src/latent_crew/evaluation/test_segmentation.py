"""Evaluates a trained segmentation checkpoint on the test split (mIoU, per-class IoU) and logs it to W&B."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import wandb
from torch.utils.data import DataLoader

from latent_crew.config import SegmentationConfig
from latent_crew.data.prepare import ensure_dataset
from latent_crew.data.segmentation_dataset import SegmentationDataset
from latent_crew.evaluation.evaluate import evaluate, sample_images
from latent_crew.models.segmentation import build_segmentation_model
from latent_crew.utils.runtime import init_wandb, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates a trained segmentation checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file (dataset, hardware, logging)")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    return parser.parse_args()


def test(config: SegmentationConfig, checkpoint_path: str | Path, split: str = "test") -> Dict:
    """The network comes from the checkpoint's model spec; the config only supplies data, hardware and logging."""
    d = config.dataset
    seed_everything(config.experiment.seed)
    device = resolve_device(config.hardware.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = checkpoint["model_spec"]
    if spec["in_channels"] != config.in_channels or spec["num_classes"] != d.num_classes:
        raise ValueError(f"Checkpoint expects {spec['in_channels']} channels / {spec['num_classes']} classes, "
                         f"config gives {config.in_channels} / {d.num_classes}")

    model = build_segmentation_model(spec, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    ensure_dataset(d.name, d.root)
    dataset = SegmentationDataset.from_config(config, split, augment=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=config.hardware.num_workers, pin_memory=device.type == "cuda")
    metrics = evaluate(model, loader, d.num_classes, d.ignore_index, device, config.training.amp).compute(dataset.class_names)

    print(f"{split}: mIoU {metrics['mIoU']:.4f}, pixel accuracy {metrics['pixel_accuracy']:.4f}, "
          f"mean class accuracy {metrics['mean_class_accuracy']:.4f} ({len(dataset)} images)")
    for name, iou in sorted(metrics["per_class_iou"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {iou:.4f}")

    out = Path(checkpoint_path).with_name(f"{Path(checkpoint_path).stem}_{split}_metrics.json")
    out.write_text(json.dumps({"checkpoint": str(checkpoint_path), "split": split, **metrics}, indent=2), encoding="utf-8")
    print(f"Saved {out}")

    run = init_wandb(config, f"{config.experiment.name}_{split}", extra_tags=["evaluation"])
    # The evaluated network is the checkpoint's, which may differ from the model section of the passed config
    run.config.update({"checkpoint_model_spec": spec, "checkpoint_epoch": checkpoint["epoch"]})
    table = wandb.Table(columns=["class", "IoU"], data=[[k, v] for k, v in metrics["per_class_iou"].items()])
    log = {f"{split}/mIoU": metrics["mIoU"], f"{split}/pixel_accuracy": metrics["pixel_accuracy"],
           f"{split}/mean_class_accuracy": metrics["mean_class_accuracy"], f"{split}/per_class_iou": table}
    if config.logging.num_sample_images > 0:
        log[f"{split}/samples"] = sample_images(model, dataset, dataset, config.logging.num_sample_images, d.ignore_index, device)
    run.log(log)
    run.summary.update({f"{split}_mIoU": metrics["mIoU"], "checkpoint": str(checkpoint_path)})
    run.finish()
    return metrics


def main() -> None:
    args = parse_args()
    test(SegmentationConfig.load_from_yaml(args.config), args.checkpoint, args.split)


if __name__ == "__main__":
    main()
