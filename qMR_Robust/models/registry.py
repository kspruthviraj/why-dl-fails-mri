"""
Model registry — single entry-point to instantiate any architecture by name.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .resnet1d import ResNet1D
from .vit1d import ViT1D

_REGISTRY = {
    "resnet1d_18": lambda cfg: ResNet1D(
        in_channels=cfg.get("input_channels", 2),
        base_channels=cfg.get("base_channels", 64),
        n_blocks=[2, 2, 2, 2],
        hidden_dim=cfg.get("hidden_dim", 256),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
    ),
    "resnet1d_50": lambda cfg: ResNet1D(
        in_channels=cfg.get("input_channels", 2),
        base_channels=cfg.get("base_channels", 64),
        n_blocks=[3, 4, 6, 3],
        hidden_dim=cfg.get("hidden_dim", 256),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
    ),
    "vit1d": lambda cfg: ViT1D(
        in_channels=cfg.get("input_channels", 2),
        seq_len=cfg.get("seq_len", 2048),
        patch_size=cfg.get("patch_size", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_transformer_layers", 6),
        output_dim=cfg.get("output_dim", 2),
        dropout=cfg.get("dropout", 0.1),
    ),
}


def build_model(name: str, cfg: dict) -> nn.Module:
    """Instantiate a model by name using the config dict."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](cfg)
