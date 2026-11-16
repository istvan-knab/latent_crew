"""Groups the training logs into fixed W&B sections, so every run's dashboard looks the same.

W&B puts every key with the same prefix (the part before the first "/") into one panel section:

    loss/      train and validation loss per epoch, in one place to spot over- and underfitting
    metrics/   semantic segmentation metrics on the validation set: mIoU, pixel accuracy,
               mean class accuracy and a per-class IoU bar chart
    samples/   the same validation images every epoch: RGB, depth, ground truth, prediction, overlay
    optim/     per-iteration training loss and learning rates (for debugging the schedule)

The epoch-level sections use `epoch` as their x-axis, the `optim/` section the global iteration.
"""

from typing import Dict, List

import wandb


class WandbSections:
    """Thin wrapper around a W&B run with one logging method per dashboard section."""

    def __init__(self, run: "wandb.Run"):
        self.run = run
        # Epoch-level charts are plotted against the epoch, not against the iteration counter
        run.define_metric("epoch")
        for section in ("loss", "metrics", "samples"):
            run.define_metric(f"{section}/*", step_metric="epoch")
        # mIoU is the model-selection metric: show its best value in the run summary
        run.define_metric("metrics/mIoU", step_metric="epoch", summary="max")

    def log_iteration(self, step: int, loss: float, learning_rates: Dict[str, float]) -> None:
        """optim/ section: raw training loss of one batch and the current learning rate per parameter group."""
        lrs = {f"optim/lr_{name}": lr for name, lr in learning_rates.items()}
        self.run.log({"optim/train_loss_iter": loss, **lrs}, step=step)

    def log_losses(self, step: int, epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        """loss/ section: mean training loss of the epoch and, on validated epochs, the validation loss."""
        log = {"epoch": epoch, "loss/train": train_loss}
        if val_loss is not None:
            log["loss/val"] = val_loss
        self.run.log(log, step=step)

    def log_metrics(self, step: int, epoch: int, metrics: Dict) -> None:
        """metrics/ section: the standard semantic segmentation metrics on the validation set."""
        self.run.log({
            "epoch": epoch,
            "metrics/mIoU": metrics["mIoU"],
            "metrics/pixel_accuracy": metrics["pixel_accuracy"],
            "metrics/mean_class_accuracy": metrics["mean_class_accuracy"],
            "metrics/per_class_iou": self._per_class_iou_chart(metrics["per_class_iou"]),
        }, step=step)

    def log_samples(self, step: int, epoch: int, images: List["wandb.Image"]) -> None:
        """samples/ section: the model's predictions on the same validation images every epoch."""
        if images:
            self.run.log({"epoch": epoch, "samples/validation": images}, step=step)

    @staticmethod
    def _per_class_iou_chart(per_class_iou: Dict[str, float]):
        """Bar chart of the IoU of every class, sorted from best to worst, for the current epoch."""
        rows = sorted(per_class_iou.items(), key=lambda kv: -kv[1])
        table = wandb.Table(columns=["class", "IoU"], data=[[name, iou] for name, iou in rows])
        return wandb.plot.bar(table, "class", "IoU", title="Per-class IoU (validation)")
