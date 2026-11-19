"""Segmentation losses selectable with `training.loss` in the config."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice loss (equal weights by default).

    Cross-entropy is dominated by the large classes (wall, floor); the Dice term scores every class by its overlap
    regardless of its pixel count, which pushes the rare classes up. Ignored pixels are left out of both terms, and
    the Dice term is averaged over the classes that occur in the batch's ground truth.
    """

    def __init__(self, ignore_index: int, weight: torch.Tensor | None = None, dice_weight: float = 1.0,
                 smooth: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.smooth = smooth  # keeps the ratio finite for classes with (almost) no pixels

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits [B, C, H, W] (float32), target [B, H, W] with class ids or ignore_index."""
        valid = (target != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
        probs = logits.softmax(dim=1) * valid
        # One-hot of the target; ignored pixels become class 0 here but are zeroed by `valid`
        one_hot = F.one_hot(target.masked_fill(~valid[:, 0], 0), logits.shape[1]).permute(0, 3, 1, 2) * valid

        dims = (0, 2, 3)  # sum over batch and pixels, one Dice score per class
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        present = one_hot.sum(dims) > 0
        dice_loss = 1 - dice[present].mean() if present.any() else logits.sum() * 0
        return self.ce(logits, target) + self.dice_weight * dice_loss


def build_criterion(name: str, ignore_index: int, class_weight: torch.Tensor | None,
                    label_smoothing: float = 0.0) -> nn.Module:
    """`ce`: plain (optionally class-weighted) cross-entropy; `dice_ce`: cross-entropy + soft Dice.

    `label_smoothing` (0 = off) softens the cross-entropy targets, so the network is not pushed towards
    over-confident logits on the training pixels (a regulariser against overfitting).
    """
    if name == "dice_ce":
        return DiceCELoss(ignore_index, weight=class_weight, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index, label_smoothing=label_smoothing)
