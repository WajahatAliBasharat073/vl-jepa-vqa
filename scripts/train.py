"""Train a VQA model.

Usage::

    python scripts/train.py --config configs/default.yaml [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import torch
import yaml

from src.models.encoders.vl_jepa_encoder import VLJEPAEncoder
from src.models.encoders.clip_encoder import CLIPEncoder
from src.models.vqa_model import VQAModel
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_encoder(cfg: dict) -> torch.nn.Module:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQA model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    encoder = build_encoder(cfg)
    model = VQAModel(
        encoder=encoder,
        num_answers=cfg["model"]["num_answers"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
    )

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    logger.info("Model initialized with encoder=%s", cfg["encoder"]["type"])
    logger.info(
        "Total parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # NOTE: Actual training requires dataset setup.  This script validates the
    # pipeline construction.  To train, add dataloader creation here using the
    # loaders from src.data.datasets.
    logger.info("Training pipeline ready.  Provide a dataset to begin training.")


if __name__ == "__main__":
    main()
