"""Trains the 2D segmentation model end to end (pretrained encoder + random decoder) and logs to W&B."""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from latent_crew.config import SegmentationConfig
from latent_crew.data.prepare import ensure_dataset
from latent_crew.data.segmentation_dataset import SegmentationDataset
from latent_crew.evaluation.evaluate import evaluate, sample_images
from latent_crew.models.segmentation import build_segmentation_model, param_groups, set_encoder_trainable
from latent_crew.utils.runtime import autocast, init_wandb, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains the segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    return parser.parse_args()


def split_train_val(n: int, val_fraction: float, seed: int) -> tuple[List[int], List[int]]:
    """Fixed random split of the official train split; the test split stays untouched until the final evaluation."""
    order = np.random.default_rng(seed).permutation(n).tolist()
    n_val = max(1, round(n * val_fraction))
    return order[n_val:], order[:n_val]


def median_frequency_weights(dataset: SegmentationDataset, indices: List[int], num_classes: int, ignore_index: int) -> torch.Tensor:
    """weight_c = median(freq) / freq_c, freq_c = pixels of c / pixels of the images containing c (Eigen & Fergus).

    Rare classes get weights > 1, frequent ones (wall, floor) < 1. Classes that never occur get 0.
    """
    class_pixels = np.zeros(num_classes)
    image_pixels = np.zeros(num_classes)
    for i in tqdm(indices, desc="Class frequencies", unit="img"):
        mask = dataset.label_mask(i)
        counts = np.bincount(mask[mask != ignore_index], minlength=num_classes)
        class_pixels += counts
        image_pixels[counts > 0] += mask.size
    present = class_pixels > 0
    freq = np.zeros(num_classes)
    freq[present] = class_pixels[present] / image_pixels[present]
    weights = np.zeros(num_classes)
    weights[present] = np.median(freq[present]) / freq[present]
    return torch.tensor(weights, dtype=torch.float32)


def lr_lambda(scheduler: str, warmup_iters: int, total_iters: int):
    """Linear warmup, then poly (power 0.9) or cosine decay; a factor on each group's base LR, applied per iteration."""

    def factor(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = min(1.0, (it - warmup_iters) / max(1, total_iters - warmup_iters))
        if scheduler == "poly":
            return (1.0 - progress) ** 0.9
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def build_optimizer(model: nn.Module, config: SegmentationConfig) -> torch.optim.Optimizer:
    t = config.training
    groups = param_groups(model, t.lr, t.encoder_lr_mult)
    if t.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
    return torch.optim.AdamW(groups, lr=t.lr, weight_decay=t.weight_decay)


def save_checkpoint(path: Path, model: nn.Module, config: SegmentationConfig, epoch: int, metrics: Dict, class_names: List[str]) -> None:
    """The model spec travels with the weights, so evaluation rebuilds the same network whatever the config says later."""
    torch.save(
        {
            "model_spec": config.model_spec(),
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_metrics": metrics,
            "class_names": class_names,
            "task": config.experiment.task,
            "config": config.to_dict(),
        },
        path,
    )


def train(config: SegmentationConfig) -> Path:
    """Runs the full training; returns the path of the best checkpoint (by validation mIoU)."""
    t, d = config.training, config.dataset
    if d.name == "nyu_depth_v2":
        raise NotImplementedError("NYU needs the 894 -> 40 class mapping first (checked before downloading anything)")
    seed_everything(config.experiment.seed)
    device = resolve_device(config.hardware.device)
    ensure_dataset(d.name, d.root)

    # Same samples twice: augmented for training, plain (full resolution) for validation
    train_base = SegmentationDataset.from_config(config, "train", augment=True)
    val_base = SegmentationDataset.from_config(config, "train", augment=False)
    train_idx, val_idx = split_train_val(len(train_base), t.val_fraction, config.experiment.seed)
    train_set, val_set = Subset(train_base, train_idx), Subset(val_base, val_idx)
    if len(train_set) < t.batch_size:
        raise ValueError(f"Only {len(train_set)} training samples for batch_size={t.batch_size}")

    loader_args = {"num_workers": config.hardware.num_workers, "pin_memory": device.type == "cuda",
                   "persistent_workers": config.hardware.num_workers > 0}
    generator = torch.Generator().manual_seed(config.experiment.seed)
    # drop_last: a final batch of 1 would break BatchNorm in train mode
    train_loader = DataLoader(train_set, batch_size=t.batch_size, shuffle=True, drop_last=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, batch_size=1, **loader_args)

    model = build_segmentation_model(config.model_spec()).to(device)
    optimizer = build_optimizer(model, config)
    total_iters = t.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(t.scheduler, t.warmup_iters, total_iters))

    weight = None
    if t.class_weighting == "median_frequency":
        weight = median_frequency_weights(train_base, train_idx, d.num_classes, d.ignore_index).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=d.ignore_index)

    out_dir = Path(config.experiment.output_dir) / config.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(out_dir / "config.yaml")
    run = init_wandb(config, config.experiment.name)
    run.summary["train_samples"], run.summary["val_samples"] = len(train_set), len(val_set)
    print(f"{config.experiment.task}: {len(train_set)} train / {len(val_set)} val samples, "
          f"{config.model.encoder} + {config.model.decoder}, {config.in_channels} input channels, device {device}")

    best_miou, best_path, step = -1.0, out_dir / "best.pt", 0
    for epoch in range(1, t.epochs + 1):
        # Optional warmup: only the random decoder learns while the encoder is frozen
        set_encoder_trainable(model, epoch > t.freeze_encoder_epochs)
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{t.epochs}", unit="batch")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if not (mask != d.ignore_index).any():
                continue  # a batch with only ignored pixels has no loss
            with autocast(device, t.amp):
                logits = model(image)
            loss = criterion(logits.float(), mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % t.log_every == 0:
                lrs = {f"train/lr_{g['name']}": g["lr"] for g in optimizer.param_groups}
                run.log({"train/loss": loss.item(), "epoch": epoch, **lrs}, step=step)
                progress.set_postfix(loss=f"{loss.item():.3f}")

        if epoch % t.eval_every == 0 or epoch == t.epochs:
            metrics = evaluate(model, val_loader, d.num_classes, d.ignore_index, device, t.amp).compute(val_base.class_names)
            log = {"val/mIoU": metrics["mIoU"], "val/pixel_accuracy": metrics["pixel_accuracy"],
                   "val/mean_class_accuracy": metrics["mean_class_accuracy"], "epoch": epoch}
            log.update({f"val_iou/{name}": iou for name, iou in metrics["per_class_iou"].items()})
            if config.logging.num_sample_images > 0:
                log["val/samples"] = sample_images(model, val_set, val_base, config.logging.num_sample_images, d.ignore_index, device)
            run.log(log, step=step)
            print(f"Epoch {epoch}: val mIoU {metrics['mIoU']:.4f}, pixel accuracy {metrics['pixel_accuracy']:.4f}")

            save_checkpoint(out_dir / "last.pt", model, config, epoch, metrics, val_base.class_names)
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                save_checkpoint(best_path, model, config, epoch, metrics, val_base.class_names)
                run.summary["best_val_mIoU"], run.summary["best_epoch"] = best_miou, epoch

    run.finish()
    print(f"Best val mIoU {best_miou:.4f}, checkpoint: {best_path}")
    return best_path


def main() -> None:
    args = parse_args()
    train(SegmentationConfig.load_from_yaml(args.config))


if __name__ == "__main__":
    main()
