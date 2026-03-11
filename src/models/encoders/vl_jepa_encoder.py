"""VL-JEPA encoder for predictive visual representation learning.

This module wraps a Vision Transformer backbone and a text transformer to
produce embeddings suitable for VQA.  The vision branch follows the JEPA
paradigm: a *context encoder* processes visible patches while a *predictor*
learns to predict the representations of masked patches in latent space.

At inference time only the context encoder is used (no masking), but the
predictor is retained for fine-tuning and analysis.

Reference:
    VL-JEPA (2025) — https://arxiv.org/abs/2505.13954
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.encoders.base_encoder import BaseEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sincos_pos_embed(num_patches: int, embed_dim: int) -> torch.Tensor:
    """Generate fixed 1-D sinusoidal positional embeddings."""
    position = torch.arange(num_patches).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(num_patches, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (num_patches, embed_dim)


# ---------------------------------------------------------------------------
# Vision Transformer building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# JEPA Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Lightweight predictor head for the JEPA pretraining objective.

    Given context-encoder output for visible patches and learnable mask tokens,
    the predictor uses a small Transformer to predict the target representation
    for masked patches.
    """

    def __init__(self, embed_dim: int = 768, predictor_dim: int = 384,
                 depth: int = 4, num_heads: int = 6, num_patches: int = 196):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(
            _sincos_pos_embed(num_patches, predictor_dim), requires_grad=False,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context: torch.Tensor,
        visible_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict representations for masked patch positions.

        Args:
            context: Context encoder output for visible patches ``(B, V, D)``.
            visible_indices: Integer indices of visible patches ``(B, V)``.
            mask_indices: Integer indices of masked patches ``(B, M)``.

        Returns:
            Predicted representations for masked patches ``(B, M, D)``.
        """
        B, V, _ = context.shape
        M = mask_indices.shape[1]

        x_vis = self.proj_in(context) + self._gather_pos(visible_indices)
        x_mask = self.mask_token.expand(B, M, -1) + self._gather_pos(mask_indices)

        # Concatenate visible + mask tokens and run through Transformer
        x = torch.cat([x_vis, x_mask], dim=1)  # (B, V+M, predictor_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Return only predictions for mask positions
        pred = x[:, V:, :]
        return self.proj_out(pred)

    def _gather_pos(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather positional embeddings for given patch indices."""
        # indices: (B, K) -> (B, K, predictor_dim)
        pos = self.pos_embed.unsqueeze(0).expand(indices.shape[0], -1, -1)
        idx = indices.unsqueeze(-1).expand(-1, -1, pos.shape[-1])
        return torch.gather(pos, 1, idx)


# ---------------------------------------------------------------------------
# VL-JEPA Encoder
# ---------------------------------------------------------------------------

class VLJEPAEncoder(BaseEncoder):
    """Vision-Language JEPA encoder.

    The vision branch is a ViT context encoder with an optional JEPA predictor
    for pretraining.  The text branch is a learnable Transformer encoder.

    Args:
        embed_dim: Shared embedding dimension.
        img_size: Input image size.
        patch_size: Patch size for the ViT.
        depth: Number of Transformer blocks in the vision encoder.
        num_heads: Number of attention heads.
        vocab_size: Vocabulary size for the text encoder.
        max_text_len: Maximum text sequence length.
        predictor_dim: Hidden dimension of the JEPA predictor.
        predictor_depth: Number of Transformer blocks in the predictor.
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
        predictor_dim: int = 384,
        predictor_depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim)
        # --- Vision context encoder ---
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

        # --- JEPA predictor (used during pretraining) ---
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=max(1, num_heads // 2),
            num_patches=num_patches,
        )

        # --- Text encoder ---
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_embed = nn.Parameter(
            _sincos_pos_embed(max_text_len, embed_dim), requires_grad=False,
        )
        self.text_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(max(1, depth // 3))
        ])
        self.text_norm = nn.LayerNorm(embed_dim)

    # ---- public interface --------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using the full ViT context encoder (no masking).

        Args:
            images: ``(B, 3, H, W)``

        Returns:
            CLS-token embeddings ``(B, embed_dim)``.
        """
        x = self.patch_embed(images)  # (B, N, D)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = x + self.pos_embed[: x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized text.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``

        Returns:
            Text embeddings ``(B, embed_dim)`` — mean-pooled over non-padding tokens.
        """
        x = self.text_embed(input_ids)  # (B, L, D)
        L = x.shape[1]
        x = x + self.text_pos_embed[:L]
        for blk in self.text_blocks:
            x = blk(x)
        x = self.text_norm(x)
        # Mean-pool over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x

    def encode_image_patches(
        self,
        images: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode images and return *all* patch token representations.

        When ``mask_indices`` is provided the masked patches are removed before
        the context encoder and the predictor fills in their representations.

        Args:
            images: ``(B, 3, H, W)``
            mask_indices: Optional ``(B, M)`` indices of patches to mask.

        Returns:
            Patch representations ``(B, N, embed_dim)``.
        """
        x = self.patch_embed(images)  # (B, N, D)
        B, N, D = x.shape

        if mask_indices is not None:
            # Build visible indices as the complement of mask_indices
            all_indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            visible_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
            visible_mask.scatter_(1, mask_indices, False)
            visible_indices = all_indices[visible_mask].view(B, -1)

            # Gather visible patches
            idx = visible_indices.unsqueeze(-1).expand(-1, -1, D)
            x_vis = torch.gather(x, 1, idx)
            x_vis = x_vis + self._gather_pos_patches(visible_indices)

            for blk in self.blocks:
                x_vis = blk(x_vis)
            x_vis = self.norm(x_vis)

            # Predict masked patches
            x_pred = self.predictor(x_vis, visible_indices, mask_indices)

            # Scatter back into full sequence
            out = torch.zeros(B, N, D, device=x.device)
            out.scatter_(1, visible_indices.unsqueeze(-1).expand(-1, -1, D), x_vis)
            out.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, D), x_pred)
            return out
        else:
            x = x + self.pos_embed[1: N + 1]
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    # ---- helpers -----------------------------------------------------------

    def _gather_pos_patches(self, indices: torch.Tensor) -> torch.Tensor:
        """Positional embeddings for patches (skip CLS position 0)."""
        pos = self.pos_embed[1:]  # (N, D)
        pos = pos.unsqueeze(0).expand(indices.shape[0], -1, -1)
        idx = indices.unsqueeze(-1).expand(-1, -1, pos.shape[-1])
        return torch.gather(pos, 1, idx)
