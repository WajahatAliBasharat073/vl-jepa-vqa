"""CLIP-style encoder baseline for VQA.

Implements a contrastive vision-language encoder that learns aligned image and
text representations via a contrastive (InfoNCE) loss.  This serves as the
baseline against which VL-JEPA representations are compared.

The architecture mirrors the standard CLIP dual-encoder design:
  * Vision branch — ViT context encoder (shared building blocks).
  * Text branch — Transformer text encoder (shared building blocks).

Reference:
    CLIP (2021) — https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder
from src.models.encoders.vl_jepa_encoder import (
    PatchEmbed,
    TransformerBlock,
    _sincos_pos_embed,
)


class CLIPEncoder(BaseEncoder):
    """CLIP-style dual-encoder for contrastive vision-language alignment.

    Args:
        embed_dim: Shared embedding dimension.
        img_size: Input image size.
        patch_size: Patch size for the ViT.
        depth: Number of Transformer blocks in each branch.
        num_heads: Number of attention heads.
        vocab_size: Vocabulary size for the text encoder.
        max_text_len: Maximum text sequence length.
        projection_dim: Dimension of the contrastive projection head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 12,
        num_heads: int = 12,
        vocab_size: int = 30522,
        max_text_len: int = 77,
        projection_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim)
        # --- Vision encoder ---
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(
            _sincos_pos_embed(num_patches + 1, embed_dim), requires_grad=False,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- Text encoder ---
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_embed = nn.Parameter(
            _sincos_pos_embed(max_text_len, embed_dim), requires_grad=False,
        )
        self.text_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.text_norm = nn.LayerNorm(embed_dim)

        # --- Contrastive projection heads ---
        self.image_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    # ---- public interface --------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[: x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token  (B, embed_dim)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.text_embed(input_ids)
        L = x.shape[1]
        x = x + self.text_pos_embed[:L]
        for blk in self.text_blocks:
            x = blk(x)
        x = self.text_norm(x)
        # Use EOS / last-token pooling (typical for CLIP text encoder)
        # Approximate by taking the last non-padding token per sample
        lengths = attention_mask.sum(dim=1).long() - 1  # (B,)
        x = x[torch.arange(x.shape[0], device=x.device), lengths]
        return x  # (B, embed_dim)

    def contrastive_logits(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CLIP-style contrastive logits.

        Returns:
            ``(logits_per_image, logits_per_text)`` each of shape ``(B, B)``.
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)

        img_emb = nn.functional.normalize(self.image_proj(img_emb), dim=-1)
        txt_emb = nn.functional.normalize(self.text_proj(txt_emb), dim=-1)

        scale = self.logit_scale.exp()
        logits_per_image = scale * img_emb @ txt_emb.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
