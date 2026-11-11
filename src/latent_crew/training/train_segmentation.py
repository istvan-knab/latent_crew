"""Trains the 2D segmentation model end to end (pretrained encoder + random decoder) and logs to W&B."""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import wandb
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


def log_checkpoint_artifact(run: wandb.Run, config: SegmentationConfig, out_dir: Path) -> None:
    """Attach recoverable best/last checkpoints to this run; offline runs stage them for later sync."""
    if config.logging.mode == "disabled":
        return

    artifact_name = f"segmentation-{run.id}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"Best and last segmentation checkpoints for {config.experiment.name}.",
        metadata={
            "task": config.experiment.task,
            "dataset": config.dataset.name,
            "modalities": config.dataset.modalities,
            "model_spec": config.model_spec(),
            "best_epoch": run.summary["best_epoch"],
            "best_val_mIoU": run.summary["best_val_mIoU"],
            "last_epoch": config.training.epochs,
        },
    )
    for filename in ("best.pt", "last.pt", "config.yaml"):
        artifact.add_file(str(out_dir / filename), name=filename)
    run.log_artifact(artifact)
    run.summary["model_artifact"] = artifact_name


def load_train_val(config: SegmentationConfig) -> tuple[SegmentationDataset, SegmentationDataset, List[int], List[int]]:
    """Checks (and if needed prepares) the dataset, then splits the official train split into train and validation."""
    d = config.dataset
    if d.name == "nyu_depth_v2":
        raise NotImplementedError("NYU needs the 894 -> 40 class mapping first (checked before downloading anything)")
    ensure_dataset(d.name, d.root)
    # Same samples twice: augmented for training, plain (full resolution) for validation
    train_base = SegmentationDataset.from_config(config, "train", augment=True)
    val_base = SegmentationDataset.from_config(config, "train", augment=False)
    train_idx, val_idx = split_train_val(len(train_base), config.training.val_fraction, config.experiment.seed)
    if len(train_idx) < config.training.batch_size:
        raise ValueError(f"Only {len(train_idx)} training samples for batch_size={config.training.batch_size}")
    return train_base, val_base, train_idx, val_idx


def loader_args(config: SegmentationConfig, device: torch.device) -> Dict:
    return {"num_workers": config.hardware.num_workers, "pin_memory": device.type == "cuda",
            "persistent_workers": config.hardware.num_workers > 0}


def train_loader_for(train_set: Subset, config: SegmentationConfig, device: torch.device, generator: torch.Generator) -> DataLoader:
    # drop_last: a final batch of 1 would break BatchNorm in train mode
    return DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True, drop_last=True,
                      generator=generator, **loader_args(config, device))


def build_scheduler(optimizer: torch.optim.Optimizer, config: SegmentationConfig, iters_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    t = config.training
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(t.scheduler, t.warmup_iters, t.epochs * iters_per_epoch))


def build_criterion(config: SegmentationConfig, class_weight: torch.Tensor | None) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weight, ignore_index=config.dataset.ignore_index)


def class_weight_for(config: SegmentationConfig, train_base: SegmentationDataset, train_idx: List[int]) -> torch.Tensor | None:
    d = config.dataset
    if config.training.class_weighting == "median_frequency":
        return median_frequency_weights(train_base, train_idx, d.num_classes, d.ignore_index)
    return None


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler, criterion: nn.Module,
                    config: SegmentationConfig, device: torch.device, epoch: int, step: int, run: wandb.Run) -> int:
    """One pass over the training loader; returns the global iteration counter after the epoch."""
    t, d = config.training, config.dataset
    # Optional warmup: only the random decoder learns while the encoder is frozen
    set_encoder_trainable(model, epoch > t.freeze_encoder_epochs)
    model.train()
    progress = tqdm(loader, desc=f"Epoch {epoch}/{t.epochs}", unit="batch")
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
    return step


def validate(model: nn.Module, val_loader: DataLoader, val_set: Subset, val_base: SegmentationDataset, config: SegmentationConfig,
             device: torch.device, epoch: int, step: int, run: wandb.Run, num_sample_images: int) -> Dict:
    """Full-resolution validation, logged to W&B; returns the metrics dict."""
    d = config.dataset
    metrics = evaluate(model, val_loader, d.num_classes, d.ignore_index, device, config.training.amp).compute(val_base.class_names)
    log = {"val/mIoU": metrics["mIoU"], "val/pixel_accuracy": metrics["pixel_accuracy"],
           "val/mean_class_accuracy": metrics["mean_class_accuracy"], "epoch": epoch}
    log.update({f"val_iou/{name}": iou for name, iou in metrics["per_class_iou"].items()})
    if num_sample_images > 0:
        log["val/samples"] = sample_images(model, val_set, val_base, num_sample_images, d.ignore_index, device)
    run.log(log, step=step)
    print(f"Epoch {epoch}: val mIoU {metrics['mIoU']:.4f}, pixel accuracy {metrics['pixel_accuracy']:.4f}")
    return metrics


def train(config: SegmentationConfig) -> Path:
    """Runs the full training; returns the path of the best checkpoint (by validation mIoU)."""
    t = config.training
    seed_everything(config.experiment.seed)
    device = resolve_device(config.hardware.device)
    train_base, val_base, train_idx, val_idx = load_train_val(config)
    train_set, val_set = Subset(train_base, train_idx), Subset(val_base, val_idx)

    generator = torch.Generator().manual_seed(config.experiment.seed)
    train_loader = train_loader_for(train_set, config, device, generator)
    val_loader = DataLoader(val_set, batch_size=1, **loader_args(config, device))

    model = build_segmentation_model(config.model_spec()).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    class_weight = class_weight_for(config, train_base, train_idx)
    criterion = build_criterion(config, None if class_weight is None else class_weight.to(device))

    out_dir = Path(config.experiment.output_dir) / config.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(out_dir / "config.yaml")
    run = init_wandb(config, config.experiment.name)
    run.summary["train_samples"], run.summary["val_samples"] = len(train_set), len(val_set)
    print(f"{config.experiment.task}: {len(train_set)} train / {len(val_set)} val samples, "
          f"{config.model.encoder} + {config.model.decoder}, {config.in_channels} input channels, device {device}")

    best_miou, best_path, step = -1.0, out_dir / "best.pt", 0
    for epoch in range(1, t.epochs + 1):
        step = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, config, device, epoch, step, run)

        if epoch % t.eval_every == 0 or epoch == t.epochs:
            metrics = validate(model, val_loader, val_set, val_base, config, device, epoch, step, run,
                               config.logging.num_sample_images)

            save_checkpoint(out_dir / "last.pt", model, config, epoch, metrics, val_base.class_names)
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                save_checkpoint(best_path, model, config, epoch, metrics, val_base.class_names)
                run.summary["best_val_mIoU"], run.summary["best_epoch"] = best_miou, epoch

    try:
        log_checkpoint_artifact(run, config, out_dir)
    finally:
        run.finish()
    print(f"Best val mIoU {best_miou:.4f}, checkpoint: {best_path}")
    return best_path


def main() -> None:
    args = parse_args()
    train(SegmentationConfig.load_from_yaml(args.config))


if __name__ == "__main__":
    main()
