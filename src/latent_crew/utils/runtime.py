"""Runtime helpers shared by training and evaluation: seeding, device, mixed precision, W&B."""

import contextlib
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import wandb

from latent_crew.config import SegmentationConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def resolve_device(name: str) -> torch.device:
    """Fails loudly instead of silently training on the CPU."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("hardware.device is cuda, but CUDA is not available (CPU-only torch build or no GPU access)")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("hardware.device is mps, but MPS is not available")
    return torch.device(name)


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """bfloat16 mixed precision on CUDA (no GradScaler needed, unlike float16); full precision elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def init_wandb(config: SegmentationConfig, run_name: str, extra_tags: List[str] = (), run_id: str | None = None,
               group: str | None = None) -> "wandb.Run":
    """Every run carries the task label (e.g. 2d_segmentation) as job type and first tag.

    With run_id an existing run is continued (HPO resumes the same configuration in several steps).
    """
    log = config.logging
    out = Path("outputs")  # local W&B files go to outputs/wandb (gitignored, mounted in Docker)
    out.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=log.project,
        entity=log.entity,
        mode=log.mode,
        name=run_name,
        job_type=config.experiment.task,
        tags=config.wandb_tags + [t for t in extra_tags if t not in config.wandb_tags],
        config=config.to_dict(),
        dir=out,
        id=run_id,
        resume="allow" if run_id else None,
        group=group,
    )
