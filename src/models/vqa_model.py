"""VQA model that works with any :class:`BaseEncoder` implementation.

The model fuses image and text representations via a multimodal fusion module,
then classifies the fused representation over a fixed answer vocabulary.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder


class MultimodalFusion(nn.Module):
    """Fuse image and text embeddings via gated element-wise interaction."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse image and text embeddings.

        Args:
            image_emb: ``(B, embed_dim)``
            text_emb:  ``(B, embed_dim)``

        Returns:
            Fused representation ``(B, hidden_dim)``.
        """
        img = self.image_proj(image_emb)
        txt = self.text_proj(text_emb)
        gate = self.gate(torch.cat([img, txt], dim=-1))
        fused = gate * img + (1 - gate) * txt
        return self.dropout(self.norm(fused))


class VQAModel(nn.Module):
    """Visual Question Answering model.

    Combines a vision-language encoder with a multimodal fusion layer and
    a classification head over a fixed answer vocabulary.

    Args:
        encoder: A :class:`BaseEncoder` (e.g. VL-JEPA or CLIP).
        num_answers: Number of answer classes.
        hidden_dim: Hidden dimension for the fusion and classifier.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        num_answers: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.fusion = MultimodalFusion(
            encoder.embed_dim, hidden_dim, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: ``(B, 3, H, W)``
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``
            labels: Optional answer label indices ``(B,)``.

        Returns:
            Dictionary with ``logits`` and optionally ``loss``.
        """
        image_emb = self.encoder.encode_image(images)
        text_emb = self.encoder.encode_text(input_ids, attention_mask)
        fused = self.fusion(image_emb, text_emb)
        logits = self.classifier(fused)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.functional.cross_entropy(logits, labels)
        return output
