"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import vqa_accuracy, simple_accuracy, per_type_accuracy


class TestVQAAccuracy:
    def test_perfect_score(self):
        preds = ["cat", "dog"]
        gts = [["cat", "cat", "cat"], ["dog", "dog", "dog"]]
        assert vqa_accuracy(preds, gts) == 1.0

    def test_partial_agreement(self):
        preds = ["cat"]
        gts = [["cat", "dog", "cat", "bird", "cat", "fish"]]
        # count=3, min(3/3, 1) = 1.0
        assert vqa_accuracy(preds, gts) == 1.0

    def test_low_agreement(self):
        preds = ["cat"]
        gts = [["cat", "dog", "bird"]]
        # count=1, min(1/3, 1) ≈ 0.333
        assert abs(vqa_accuracy(preds, gts) - 1 / 3) < 1e-6

    def test_no_match(self):
        preds = ["cat"]
        gts = [["dog", "bird", "fish"]]
        assert vqa_accuracy(preds, gts) == 0.0

    def test_empty(self):
        assert vqa_accuracy([], []) == 0.0


class TestSimpleAccuracy:
    def test_perfect(self):
        assert simple_accuracy(["a", "b"], ["a", "b"]) == 1.0

    def test_half(self):
        assert simple_accuracy(["a", "b"], ["a", "c"]) == 0.5

    def test_none(self):
        assert simple_accuracy(["a"], ["b"]) == 0.0

    def test_empty(self):
        assert simple_accuracy([], []) == 0.0


class TestPerTypeAccuracy:
    def test_single_type(self):
        preds = ["yes", "no", "yes"]
        gts = ["yes", "yes", "yes"]
        types = ["yn", "yn", "yn"]
        result = per_type_accuracy(preds, gts, types)
        assert abs(result["yn"] - 2 / 3) < 1e-6

    def test_multiple_types(self):
        preds = ["yes", "3", "cat"]
        gts = ["yes", "3", "dog"]
        types = ["yn", "number", "other"]
        result = per_type_accuracy(preds, gts, types)
        assert result["yn"] == 1.0
        assert result["number"] == 1.0
        assert result["other"] == 0.0
