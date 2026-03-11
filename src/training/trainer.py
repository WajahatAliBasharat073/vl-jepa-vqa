"""Training utilities for VQA models."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Simple training loop for VQA models.

    Args:
        model: The :class:`~src.models.vqa_model.VQAModel` to train.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning-rate scheduler.
        device: Device to train on.
        max_grad_norm: Maximum gradient norm for clipping (0 to disable).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.

        Returns:
            Average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images, input_ids, attention_mask, labels=labels)
            loss = output["loss"]
            loss.backward()

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm,
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on a dataloader.

        Returns:
            Dictionary with ``loss`` and ``accuracy``.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(images, input_ids, attention_mask, labels=labels)
            total_loss += output["loss"].item()
            preds = output["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

        n = max(len(dataloader), 1)
        return {
            "loss": total_loss / n,
            "accuracy": correct / max(total, 1),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Checkpoint loaded from %s", path)
