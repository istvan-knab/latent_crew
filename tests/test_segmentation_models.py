"""Smoke tests: every config-selectable decoder builds and runs with CNN and transformer encoders."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from latent_crew.config import SegmentationConfig
from latent_crew.models.segmentation import (
    DECODERS,
    build_segmentation_model,
    param_groups,
    set_encoder_trainable,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "segmentation.yaml"
ENCODERS = ("mobilenet_v2", "mit_b0", "tu-convnext_tiny")


def load_raw() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec(decoder: str, encoder: str, in_channels: int) -> dict:
    return {
        "decoder": decoder,
        "encoder": encoder,
        "encoder_weights": None,
        "in_channels": in_channels,
        "num_classes": 37,
    }


@pytest.mark.parametrize("in_channels", (3, 4))
@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("decoder", DECODERS)
def test_forward_shape(decoder: str, encoder: str, in_channels: int) -> None:
    model = build_segmentation_model(spec(decoder, encoder, in_channels), pretrained=False).train()
    # Batch 2: BatchNorm in the ASPP pooling branch fails on batch 1 in train mode
    logits = model(torch.randn(2, in_channels, 128, 160))
    assert logits.shape == (2, 37, 128, 160)


def test_default_config_loads() -> None:
    config = SegmentationConfig.load_from_yaml(CONFIG_PATH)
    assert config.model_spec()["in_channels"] == 4


def test_rgb_only_gives_three_channels() -> None:
    raw = load_raw()
    raw["dataset"]["modalities"] = ["rgb"]
    assert SegmentationConfig(raw).in_channels == 3


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "decoder", "mask2former"),
        ("model", "encoder", "not_an_encoder"),
        ("model", "encoder", "tu-not_a_timm_model"),
        ("dataset", "name", "kitti"),
        ("dataset", "name", "nyu_depth_v2"),  # num_classes stays 37, NYU needs 40
        ("experiment", "task", "4d_segmentation"),
        ("logging", "mode", "sometimes"),
    ],
)
def test_invalid_names_rejected(section: str, key: str, value: str) -> None:
    raw = copy.deepcopy(load_raw())
    raw[section][key] = value
    with pytest.raises(ValueError):
        SegmentationConfig(raw)


def test_param_groups_and_freezing() -> None:
    model = build_segmentation_model(spec("deeplabv3plus", "mobilenet_v2", 4), pretrained=False)
    groups = param_groups(model, lr=0.01, encoder_lr_mult=0.1)
    assert [g["lr"] for g in groups] == pytest.approx([0.001, 0.01])
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))

    set_encoder_trainable(model, False)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())


@pytest.mark.parametrize("task", ("3d_segmentation", "2d_detection", "3d_detection"))
def test_planned_tasks_are_known_but_not_implemented(task: str) -> None:
    raw = load_raw()
    raw["experiment"]["task"] = task
    with pytest.raises(NotImplementedError):
        SegmentationConfig(raw)


def test_task_is_first_wandb_tag() -> None:
    raw = load_raw()
    raw["logging"]["tags"] = ["rgbd", "2d_segmentation"]
    assert SegmentationConfig(raw).wandb_tags == ["2d_segmentation", "rgbd"]
