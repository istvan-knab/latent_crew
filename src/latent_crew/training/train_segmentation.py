"""Trains the 2D semantic segmentation model end to end and logs the run to Weights & Biases.

The encoder starts from pretrained ImageNet weights, the decoder and the classifier head start random, and both are
trained together. Validation runs on a fixed hold-out of the official train split; the official test split is only
used by `latent_crew.evaluation.test_segmentation`.

Usage:
    python -m latent_crew.training.train_segmentation --config configs/segmentation.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List

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
from latent_crew.models.segmentation import build_segmentation_model, param_groups
from latent_crew.utils.runtime import autocast, init_wandb, resolve_device, seed_everything


# --------------------------------------------------------------------------------------------------------------------
# Pure helpers (no state, easy to test on their own)
# --------------------------------------------------------------------------------------------------------------------


def split_train_val(n: int, val_fraction: float, seed: int) -> tuple[List[int], List[int]]:
    """Splits the indices 0..n-1 into (train, val) with a fixed random permutation."""
    order = np.random.default_rng(seed).permutation(n).tolist()
    n_val = max(1, round(n * val_fraction))
    return order[n_val:], order[:n_val]


def median_frequency_weights(dataset: SegmentationDataset, indices: List[int], num_classes: int,
                             ignore_index: int) -> torch.Tensor:
    """Class weights against imbalance: weight_c = median(freq) / freq_c (Eigen & Fergus, 2015).

    freq_c is the number of pixels of class c divided by the number of pixels of the images that contain c.
    Rare classes get weights > 1, frequent ones (wall, floor) < 1, and classes that never occur get 0.
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


def lr_lambda(scheduler: str, warmup_iters: int, total_iters: int) -> Callable[[int], float]:
    """Learning-rate factor per iteration: linear warmup, then poly (power 0.9) or cosine decay to 0."""

    def factor(it: int) -> float:
        """Multiplier on the base LR at iteration `it`."""
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = min(1.0, (it - warmup_iters) / max(1, total_iters - warmup_iters))
        if scheduler == "poly":
            return (1.0 - progress) ** 0.9
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


# --------------------------------------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------------------------------------


class SegmentationTrainer:
    """Owns everything one training run needs: data, model, optimizer, schedule, loss and the W&B run.

    Typical use:
        best_checkpoint = SegmentationTrainer(config).fit()
    """

    def __init__(self, config: SegmentationConfig):
        self.config = config
        seed_everything(config.experiment.seed)
        self.device = resolve_device(config.hardware.device)
        self.out_dir = Path(config.experiment.output_dir) / config.experiment.name

        self._build_data()
        self._build_model()
        self.step = 0  # global iteration counter, used as the W&B step

    # ---- setup -----------------------------------------------------------------------------------------------------

    def _build_data(self) -> None:
        """Datasets and loaders. The same train-split samples are read twice: augmented and plain."""
        c, d, t = self.config, self.config.dataset, self.config.training
        ensure_dataset(d.name, d.root)  # downloads and prepares the data only if something is missing

        self.train_base = SegmentationDataset.from_config(c, "train", augment=True)
        self.val_base = SegmentationDataset.from_config(c, "train", augment=False)
        self.train_idx, val_idx = split_train_val(len(self.train_base), t.val_fraction, c.experiment.seed)
        if len(self.train_idx) < t.batch_size:
            raise ValueError(f"Only {len(self.train_idx)} training samples for batch_size={t.batch_size}")
        self.train_set = Subset(self.train_base, self.train_idx)
        self.val_set = Subset(self.val_base, val_idx)

        workers = c.hardware.num_workers
        loader_args = {"num_workers": workers, "pin_memory": self.device.type == "cuda",
                       "persistent_workers": workers > 0}
        # drop_last: a final batch of 1 would break BatchNorm in train mode
        self.train_loader = DataLoader(self.train_set, batch_size=t.batch_size, shuffle=True, drop_last=True,
                                       generator=torch.Generator().manual_seed(c.experiment.seed), **loader_args)
        # Batch size 1: validation runs at full resolution and image sizes differ between sensors
        self.val_loader = DataLoader(self.val_set, batch_size=1, **loader_args)

    def _build_model(self) -> None:
        """Network, optimizer (two LR groups), per-iteration LR schedule and loss."""
        t, d = self.config.training, self.config.dataset
        self.model = build_segmentation_model(self.config.model_spec()).to(self.device)

        # The pretrained encoder moves slower (lr * encoder_lr_mult) than the random decoder (lr)
        groups = param_groups(self.model, t.lr, t.encoder_lr_mult)
        if t.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(groups, lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(groups, lr=t.lr, weight_decay=t.weight_decay)

        total_iters = t.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda(t.scheduler, t.warmup_iters, total_iters))

        class_weight = None
        if t.class_weighting == "median_frequency":
            class_weight = median_frequency_weights(self.train_base, self.train_idx, d.num_classes, d.ignore_index)
            class_weight = class_weight.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=d.ignore_index)

    # ---- main loop -------------------------------------------------------------------------------------------------

    def fit(self) -> Path:
        """Trains for `training.epochs` epochs; returns the path of the best checkpoint (by validation mIoU)."""
        c, t = self.config, self.config.training
        self.out_dir.mkdir(parents=True, exist_ok=True)
        c.save_yaml(self.out_dir / "config.yaml")
        self.run = init_wandb(c, c.experiment.name)
        self.run.summary["train_samples"] = len(self.train_set)
        self.run.summary["val_samples"] = len(self.val_set)
        print(f"{len(self.train_set)} train / {len(self.val_set)} val samples, {c.model.encoder} + {c.model.decoder}, "
              f"{c.in_channels} input channels, device {self.device}")

        best_miou, best_path = -1.0, self.out_dir / "best.pt"
        try:
            for epoch in range(1, t.epochs + 1):
                self.train_one_epoch(epoch)
                if epoch % t.eval_every != 0 and epoch != t.epochs:
                    continue
                metrics = self.validate(epoch)
                self.save_checkpoint(self.out_dir / "last.pt", epoch, metrics)
                if metrics["mIoU"] > best_miou:
                    best_miou = metrics["mIoU"]
                    self.save_checkpoint(best_path, epoch, metrics)
                    self.run.summary["best_val_mIoU"], self.run.summary["best_epoch"] = best_miou, epoch
            self.log_checkpoint_artifact()
        finally:
            self.run.finish()

        print(f"Best val mIoU {best_miou:.4f}, checkpoint: {best_path}")
        return best_path

    def train_one_epoch(self, epoch: int) -> None:
        """One pass over the training loader with one optimizer and scheduler step per batch."""
        t, ignore = self.config.training, self.config.dataset.ignore_index
        self.model.train()
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{t.epochs}", unit="batch")
        for batch in progress:
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            if not (mask != ignore).any():
                continue  # a batch with only ignored pixels has no loss

            with autocast(self.device, t.amp):
                logits = self.model(image)
            loss = self.criterion(logits.float(), mask)  # loss in float32 for numerical stability

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.step % t.log_every == 0:
                lrs = {f"train/lr_{g['name']}": g["lr"] for g in self.optimizer.param_groups}
                self.run.log({"train/loss": loss.item(), "epoch": epoch, **lrs}, step=self.step)
                progress.set_postfix(loss=f"{loss.item():.3f}")

    def validate(self, epoch: int) -> Dict:
        """Full-resolution validation (mIoU, pixel and class accuracy, per-class IoU), logged to W&B."""
        c, d = self.config, self.config.dataset
        confusion = evaluate(self.model, self.val_loader, d.num_classes, d.ignore_index, self.device, c.training.amp)
        metrics = confusion.compute(self.val_base.class_names)

        log = {"val/mIoU": metrics["mIoU"], "val/pixel_accuracy": metrics["pixel_accuracy"],
               "val/mean_class_accuracy": metrics["mean_class_accuracy"], "epoch": epoch}
        log.update({f"val_iou/{name}": iou for name, iou in metrics["per_class_iou"].items()})
        if c.logging.num_sample_images > 0:
            log["val/samples"] = sample_images(self.model, self.val_set, self.val_base, c.logging.num_sample_images,
                                               d.ignore_index, self.device)
        self.run.log(log, step=self.step)
        print(f"Epoch {epoch}: val mIoU {metrics['mIoU']:.4f}, pixel accuracy {metrics['pixel_accuracy']:.4f}")
        return metrics

    # ---- outputs ---------------------------------------------------------------------------------------------------

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict) -> None:
        """Saves the weights together with the model spec, so evaluation can rebuild the exact same network."""
        torch.save(
            {
                "model_spec": self.config.model_spec(),
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_metrics": metrics,
                "class_names": self.val_base.class_names,
                "config": self.config.to_dict(),
            },
            path,
        )

    def log_checkpoint_artifact(self) -> None:
        """Attaches best.pt, last.pt and config.yaml to the run as a W&B `model` artifact.

        Online runs upload it, offline runs stage it for a later `wandb sync`, disabled logging skips it.
        """
        c = self.config
        if c.logging.mode == "disabled":
            return
        name = f"segmentation-{self.run.id}"
        artifact = wandb.Artifact(
            name=name,
            type="model",
            description=f"Best and last segmentation checkpoints for {c.experiment.name}.",
            metadata={
                "dataset": c.dataset.name,
                "modalities": c.dataset.modalities,
                "model_spec": c.model_spec(),
                "best_epoch": self.run.summary["best_epoch"],
                "best_val_mIoU": self.run.summary["best_val_mIoU"],
                "last_epoch": c.training.epochs,
            },
        )
        for filename in ("best.pt", "last.pt", "config.yaml"):
            artifact.add_file(str(self.out_dir / filename), name=filename)
        self.run.log_artifact(artifact)
        self.run.summary["model_artifact"] = name


def train(config: SegmentationConfig) -> Path:
    """Convenience wrapper: trains with `config` and returns the best checkpoint path."""
    return SegmentationTrainer(config).fit()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Trains the segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    train(SegmentationConfig.load_from_yaml(parser.parse_args().config))


if __name__ == "__main__":
    main()
