"""Evaluation loop and W&B sample images, shared by training (validation) and test_segmentation."""

from typing import List

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset

from latent_crew.data.segmentation_dataset import SegmentationDataset
from latent_crew.evaluation.metrics import ConfusionMatrix
from latent_crew.utils.runtime import autocast
from latent_crew.utils.visualization import render_sample


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
    amp: bool,
) -> ConfusionMatrix:
    """Full-resolution evaluation; the loader must use batch_size=1 because image sizes differ."""
    model.eval()
    confusion = ConfusionMatrix(num_classes, ignore_index)
    for batch in loader:
        with autocast(device, amp):
            logits = model(batch["image"].to(device, non_blocking=True))
        confusion.update(logits.float(), batch["mask"].to(device, non_blocking=True))
    return confusion


@torch.no_grad()
def sample_images(
    model: nn.Module,
    dataset: Dataset,
    base: SegmentationDataset,
    count: int,
    ignore_index: int,
    device: torch.device,
) -> List["wandb.Image"]:
    """RGB, depth, label and prediction side by side for the first `count` samples of `dataset`.

    `base` is the underlying SegmentationDataset (for denormalisation and class names), `dataset` may be a Subset of it.
    """
    model.eval()
    images = []
    for i in range(min(count, len(dataset))):
        sample = dataset[i]
        prediction = model(sample["image"][None].to(device)).argmax(dim=1)[0].cpu().numpy()
        fig = render_sample(
            base.denormalize(sample["image"]),
            sample["mask"].numpy(),
            base.class_names,
            ignore_index,
            sample["id"],
            prediction=prediction,
        )
        images.append(wandb.Image(fig, caption=sample["id"]))
        plt.close(fig)
    return images
