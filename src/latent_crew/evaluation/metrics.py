"""Confusion-matrix based segmentation metrics: mIoU, per-class IoU, pixel accuracy."""

from typing import Dict, List

import torch


class ConfusionMatrix:
    """Accumulates a [num_classes, num_classes] matrix (rows = ground truth, cols = prediction) over batches."""

    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """logits [B, C, H, W], target [B, H, W]; ignored pixels are skipped."""
        pred = logits.argmax(dim=1)
        valid = target != self.ignore_index
        index = target[valid] * self.num_classes + pred[valid]
        counts = torch.bincount(index.cpu(), minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self, class_names: List[str]) -> Dict[str, float | Dict[str, float]]:
        """IoU = TP / (TP + FP + FN) per class; mIoU averages the classes that occur in the ground truth."""
        m = self.matrix.double()
        tp = m.diag()
        gt = m.sum(dim=1)
        iou = tp / (gt + m.sum(dim=0) - tp).clamp(min=1)
        present = gt > 0
        return {
            "mIoU": iou[present].mean().item() if present.any() else 0.0,
            "pixel_accuracy": (tp.sum() / m.sum().clamp(min=1)).item(),
            "mean_class_accuracy": (tp[present] / gt[present]).mean().item() if present.any() else 0.0,
            "per_class_iou": {class_names[k]: iou[k].item() for k in range(self.num_classes) if present[k]},
        }
