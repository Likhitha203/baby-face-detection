"""Shared classifier architecture and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_baby_classifier(
    num_classes: int = 2,
    freeze_blocks: int = 12,
    pretrained: bool = True,
) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    for i, layer in enumerate(model.features):
        if i < freeze_blocks:
            for p in layer.parameters():
                p.requires_grad = False
    in_features = 960
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


def load_classifier_ckpt(
    model: nn.Module,
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict:
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}
