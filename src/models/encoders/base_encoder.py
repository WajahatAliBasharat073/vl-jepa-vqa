"""Abstract base class for vision encoders."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Base class for vision encoders used in VQA.

    All encoder implementations must subclass this and implement
    :meth:`encode_image` and :meth:`encode_text`.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Image embeddings of shape ``(B, embed_dim)``.
        """

    @abstractmethod
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of tokenized text sequences.

        Args:
            input_ids: Token IDs of shape ``(B, L)``.
            attention_mask: Attention mask of shape ``(B, L)``.

        Returns:
            Text embeddings of shape ``(B, embed_dim)``.
        """

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both images and text.

        Returns:
            Tuple of ``(image_embeddings, text_embeddings)``.
        """
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)
