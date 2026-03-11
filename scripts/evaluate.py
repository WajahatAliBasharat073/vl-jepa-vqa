"""Evaluate a trained VQA model checkpoint.

Usage::

    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/model.pt [--device cuda]
"""

from __future__ import annotations

import argparse
import logging

import torch
import yaml

from src.models.vqa_model import VQAModel
from src.utils import build_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQA model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    encoder = build_encoder(cfg)
    model = VQAModel(
        encoder=encoder,
        num_answers=cfg["model"]["num_answers"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    logger.info("Model loaded from %s", args.checkpoint)
    logger.info(
        "Total parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # NOTE: Actual evaluation requires dataset setup.  This script validates
    # the checkpoint loading pipeline.  To evaluate, add dataloader creation
    # here using the loaders from src.data.datasets.
    logger.info("Evaluation pipeline ready.  Provide a dataset to begin evaluation.")


if __name__ == "__main__":
    main()
