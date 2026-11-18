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
    assert SegmentationConfig.from_dict(raw).in_channels == 3


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "decoder", "mask2former"),
        ("model", "encoder", "not_an_encoder"),
        ("model", "encoder", "tu-not_a_timm_model"),
        ("dataset", "name", "kitti"),
        ("dataset", "name", "nyu_depth_v2"),  # only SUN RGB-D is supported
        ("logging", "mode", "sometimes"),
    ],
)
def test_invalid_names_rejected(section: str, key: str, value: str) -> None:
    raw = copy.deepcopy(load_raw())
    raw[section][key] = value
    with pytest.raises(ValueError):
        SegmentationConfig.from_dict(raw)


def test_param_groups() -> None:
    model = build_segmentation_model(spec("deeplabv3plus", "mobilenet_v2", 4), pretrained=False)
    groups = param_groups(model, lr=0.01, encoder_lr_mult=0.1)
    assert [g["lr"] for g in groups] == pytest.approx([0.001, 0.01])
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


@pytest.mark.parametrize("path", sorted((CONFIG_PATH.parent / "experiments").glob("*.yaml")), ids=lambda p: p.stem)
def test_experiment_configs_only_change_what_they_list(path: Path) -> None:
    from latent_crew.config import read_yaml

    base, experiment = load_raw(), read_yaml(path)
    assert experiment["experiment"]["name"] != base["experiment"]["name"]  # separate checkpoint folder
    changed = {(s, k) for s in base for k in base[s] if experiment[s][k] != base[s][k]}
    assert ("experiment", "name") in changed and len(changed) <= 4  # name, tags and at most two settings
    SegmentationConfig.load_from_yaml(path)  # valid as a whole


def test_dice_ce_loss() -> None:
    from latent_crew.training.losses import DiceCELoss, build_criterion

    target = torch.tensor([[[0, 1], [1, 255]]])
    perfect = torch.nn.functional.one_hot(target.clamp(max=2), 3).permute(0, 3, 1, 2).float() * 50
    loss = DiceCELoss(ignore_index=255)
    assert loss(perfect, target).item() == pytest.approx(0.0, abs=1e-3)  # ignored pixel does not count
    assert loss(torch.zeros(1, 3, 2, 2), target).item() > 1.0
    assert isinstance(build_criterion("ce", 255, None), torch.nn.CrossEntropyLoss)
    # Label smoothing penalises even a perfect, very confident prediction
    assert build_criterion("ce", 255, None, label_smoothing=0.1)(perfect, target).item() > 0.1
