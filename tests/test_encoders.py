"""Tests for vision-language encoders."""

import torch
import pytest

from src.models.encoders.vl_jepa_encoder import VLJEPAEncoder
from src.models.encoders.clip_encoder import CLIPEncoder


# Use small dimensions for fast testing
EMBED_DIM = 64
IMG_SIZE = 32
PATCH_SIZE = 8
DEPTH = 2
NUM_HEADS = 4
VOCAB_SIZE = 256
MAX_TEXT_LEN = 16
BATCH_SIZE = 2


@pytest.fixture
def vl_jepa():
    return VLJEPAEncoder(
        embed_dim=EMBED_DIM,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN,
        predictor_dim=32,
        predictor_depth=2,
    )


@pytest.fixture
def clip_encoder():
    return CLIPEncoder(
        embed_dim=EMBED_DIM,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN,
        projection_dim=32,
    )


@pytest.fixture
def dummy_inputs():
    images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_TEXT_LEN))
    attention_mask = torch.ones(BATCH_SIZE, MAX_TEXT_LEN, dtype=torch.long)
    return images, input_ids, attention_mask


class TestVLJEPAEncoder:
    def test_encode_image_shape(self, vl_jepa, dummy_inputs):
        images, _, _ = dummy_inputs
        out = vl_jepa.encode_image(images)
        assert out.shape == (BATCH_SIZE, EMBED_DIM)

    def test_encode_text_shape(self, vl_jepa, dummy_inputs):
        _, input_ids, attention_mask = dummy_inputs
        out = vl_jepa.encode_text(input_ids, attention_mask)
        assert out.shape == (BATCH_SIZE, EMBED_DIM)

    def test_forward_returns_tuple(self, vl_jepa, dummy_inputs):
        images, input_ids, attention_mask = dummy_inputs
        img_emb, txt_emb = vl_jepa(images, input_ids, attention_mask)
        assert img_emb.shape == (BATCH_SIZE, EMBED_DIM)
        assert txt_emb.shape == (BATCH_SIZE, EMBED_DIM)

    def test_encode_image_patches_no_mask(self, vl_jepa, dummy_inputs):
        images, _, _ = dummy_inputs
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        out = vl_jepa.encode_image_patches(images)
        assert out.shape == (BATCH_SIZE, num_patches, EMBED_DIM)

    def test_encode_image_patches_with_mask(self, vl_jepa, dummy_inputs):
        images, _, _ = dummy_inputs
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        num_masked = num_patches // 4
        mask_indices = torch.stack([
            torch.randperm(num_patches)[:num_masked] for _ in range(BATCH_SIZE)
        ])
        out = vl_jepa.encode_image_patches(images, mask_indices=mask_indices)
        assert out.shape == (BATCH_SIZE, num_patches, EMBED_DIM)

    def test_embed_dim_attribute(self, vl_jepa):
        assert vl_jepa.embed_dim == EMBED_DIM


class TestCLIPEncoder:
    def test_encode_image_shape(self, clip_encoder, dummy_inputs):
        images, _, _ = dummy_inputs
        out = clip_encoder.encode_image(images)
        assert out.shape == (BATCH_SIZE, EMBED_DIM)

    def test_encode_text_shape(self, clip_encoder, dummy_inputs):
        _, input_ids, attention_mask = dummy_inputs
        out = clip_encoder.encode_text(input_ids, attention_mask)
        assert out.shape == (BATCH_SIZE, EMBED_DIM)

    def test_forward_returns_tuple(self, clip_encoder, dummy_inputs):
        images, input_ids, attention_mask = dummy_inputs
        img_emb, txt_emb = clip_encoder(images, input_ids, attention_mask)
        assert img_emb.shape == (BATCH_SIZE, EMBED_DIM)
        assert txt_emb.shape == (BATCH_SIZE, EMBED_DIM)

    def test_contrastive_logits_shape(self, clip_encoder, dummy_inputs):
        images, input_ids, attention_mask = dummy_inputs
        logits_i, logits_t = clip_encoder.contrastive_logits(
            images, input_ids, attention_mask,
        )
        assert logits_i.shape == (BATCH_SIZE, BATCH_SIZE)
        assert logits_t.shape == (BATCH_SIZE, BATCH_SIZE)

    def test_embed_dim_attribute(self, clip_encoder):
        assert clip_encoder.embed_dim == EMBED_DIM
