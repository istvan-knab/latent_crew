"""Builds config-selectable encoder-decoder segmentation models (segmentation_models_pytorch)."""

from typing import Any, Dict, List

import segmentation_models_pytorch as smp
import timm
from torch import nn

# Decoders selectable in the config (key = model.decoder). See docs/segmentation_models.md.
DECODERS = ("unet", "fpn", "deeplabv3", "deeplabv3plus", "pspnet", "upernet", "segformer")


def check_model_names(decoder: str, encoder: str) -> None:
    """Fails early on unknown decoder or encoder names."""
    if decoder not in DECODERS:
        raise ValueError(f"model.decoder must be one of {DECODERS}, got {decoder!r}")
    # "tu-<name>" selects any timm model as encoder (e.g. tu-convnext_tiny)
    known = timm.is_model(encoder[3:]) if encoder.startswith("tu-") else encoder in smp.encoders.get_encoder_names()
    if not known:
        raise ValueError(f"Unknown model.encoder {encoder!r}: use an smp encoder name or tu-<timm model>")


def build_segmentation_model(spec: Dict[str, Any], pretrained: bool = True) -> nn.Module:
    """Builds the model from a spec (SegmentationConfig.model_spec() or the one saved in a checkpoint).

    pretrained=False skips downloading encoder weights, e.g. when the checkpoint is loaded afterwards.
    """
    check_model_names(spec["decoder"], spec["encoder"])
    try:
        return smp.create_model(
            arch=spec["decoder"],
            encoder_name=spec["encoder"],
            encoder_weights=spec["encoder_weights"] if pretrained else None,
            in_channels=spec["in_channels"],
            classes=spec["num_classes"],
            **spec.get("decoder_kwargs", {}),
        )
    except (ValueError, NotImplementedError, AttributeError) as error:
        raise ValueError(
            f"Encoder {spec['encoder']!r} cannot be combined with decoder {spec['decoder']!r} "
            f"(in_channels={spec['in_channels']}): {error}"
        ) from error


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    """Freezes or unfreezes the encoder (used for the optional frozen-encoder warmup)."""
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def param_groups(model: nn.Module, lr: float, encoder_lr_mult: float) -> List[Dict[str, Any]]:
    """Pretrained encoder gets lr * encoder_lr_mult, the randomly initialised decoder and head get lr."""
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return [
        {"params": encoder_params, "lr": lr * encoder_lr_mult, "name": "encoder"},
        {"params": other_params, "lr": lr, "name": "decoder"},
    ]
