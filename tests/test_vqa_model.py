"""Tests for the VQA model."""

import torch
import pytest

from src.models.encoders.vl_jepa_encoder import VLJEPAEncoder
from src.models.encoders.clip_encoder import CLIPEncoder
from src.models.vqa_model import VQAModel, MultimodalFusion


EMBED_DIM = 64
IMG_SIZE = 32
PATCH_SIZE = 8
DEPTH = 2
NUM_HEADS = 4
VOCAB_SIZE = 256
MAX_TEXT_LEN = 16
NUM_ANSWERS = 10
HIDDEN_DIM = 128
BATCH_SIZE = 2


def _make_vl_jepa():
    return VLJEPAEncoder(
        embed_dim=EMBED_DIM, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        depth=DEPTH, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN, predictor_dim=32, predictor_depth=2,
    )


def _make_clip():
    return CLIPEncoder(
        embed_dim=EMBED_DIM, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        depth=DEPTH, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE,
        max_text_len=MAX_TEXT_LEN, projection_dim=32,
    )


@pytest.fixture
def dummy_inputs():
    images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_TEXT_LEN))
    attention_mask = torch.ones(BATCH_SIZE, MAX_TEXT_LEN, dtype=torch.long)
    labels = torch.randint(0, NUM_ANSWERS, (BATCH_SIZE,))
    return images, input_ids, attention_mask, labels


class TestMultimodalFusion:
    def test_output_shape(self):
        fusion = MultimodalFusion(EMBED_DIM, HIDDEN_DIM)
        img = torch.randn(BATCH_SIZE, EMBED_DIM)
        txt = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = fusion(img, txt)
        assert out.shape == (BATCH_SIZE, HIDDEN_DIM)


class TestVQAModelWithVLJEPA:
    def test_forward_logits_shape(self, dummy_inputs):
        encoder = _make_vl_jepa()
        model = VQAModel(encoder, NUM_ANSWERS, HIDDEN_DIM)
        images, input_ids, attention_mask, _ = dummy_inputs
        out = model(images, input_ids, attention_mask)
        assert out["logits"].shape == (BATCH_SIZE, NUM_ANSWERS)
        assert "loss" not in out

    def test_forward_with_labels(self, dummy_inputs):
        encoder = _make_vl_jepa()
        model = VQAModel(encoder, NUM_ANSWERS, HIDDEN_DIM)
        images, input_ids, attention_mask, labels = dummy_inputs
        out = model(images, input_ids, attention_mask, labels=labels)
        assert out["logits"].shape == (BATCH_SIZE, NUM_ANSWERS)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar

    def test_backward_pass(self, dummy_inputs):
        encoder = _make_vl_jepa()
        model = VQAModel(encoder, NUM_ANSWERS, HIDDEN_DIM)
        images, input_ids, attention_mask, labels = dummy_inputs
        out = model(images, input_ids, attention_mask, labels=labels)
        out["loss"].backward()
        # Verify that *some* parameters received gradients (the predictor
        # parameters are intentionally excluded from the VQA forward path)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestVQAModelWithCLIP:
    def test_forward_logits_shape(self, dummy_inputs):
        encoder = _make_clip()
        model = VQAModel(encoder, NUM_ANSWERS, HIDDEN_DIM)
        images, input_ids, attention_mask, _ = dummy_inputs
        out = model(images, input_ids, attention_mask)
        assert out["logits"].shape == (BATCH_SIZE, NUM_ANSWERS)

    def test_forward_with_labels(self, dummy_inputs):
        encoder = _make_clip()
        model = VQAModel(encoder, NUM_ANSWERS, HIDDEN_DIM)
        images, input_ids, attention_mask, labels = dummy_inputs
        out = model(images, input_ids, attention_mask, labels=labels)
        assert "loss" in out
        assert out["loss"].ndim == 0
