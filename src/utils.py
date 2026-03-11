"""Shared utilities for building models from configuration."""

from __future__ import annotations

import torch.nn as nn

from src.models.encoders.vl_jepa_encoder import VLJEPAEncoder
from src.models.encoders.clip_encoder import CLIPEncoder


def build_encoder(cfg: dict) -> nn.Module:
    """Instantiate an encoder from a configuration dict.

    Args:
        cfg: Top-level config with an ``encoder`` section containing at least
            ``type`` (``"vl_jepa"`` or ``"clip"``).

    Returns:
        An encoder module.

    Raises:
        ValueError: If the encoder type is unknown.
    """
    enc = cfg["encoder"]
    common = dict(
        embed_dim=enc["embed_dim"],
        img_size=enc["img_size"],
        patch_size=enc["patch_size"],
        depth=enc["depth"],
        num_heads=enc["num_heads"],
        vocab_size=enc["vocab_size"],
        max_text_len=enc["max_text_len"],
        dropout=enc.get("dropout", 0.0),
    )
    if enc["type"] == "vl_jepa":
        return VLJEPAEncoder(
            **common,
            predictor_dim=enc.get("predictor_dim", 384),
            predictor_depth=enc.get("predictor_depth", 4),
        )
    elif enc["type"] == "clip":
        return CLIPEncoder(
            **common,
            projection_dim=enc.get("projection_dim", 512),
        )
    else:
        raise ValueError(f"Unknown encoder type: {enc['type']}")
