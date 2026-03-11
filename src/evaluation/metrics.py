"""Evaluation metrics for VQA."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def vqa_accuracy(predictions: Sequence[str], ground_truths: Sequence[list[str]]) -> float:
    """Compute soft VQA accuracy following the VQAv2 evaluation protocol.

    For each question, accuracy is ``min(count / 3, 1)`` where *count* is the
    number of annotators that gave the predicted answer.

    Args:
        predictions: Predicted answer strings, one per question.
        ground_truths: For each question, a list of annotator answer strings
            (typically 10 for VQAv2).

    Returns:
        Average accuracy across all questions.
    """
    if len(predictions) == 0:
        return 0.0
    total = 0.0
    for pred, gts in zip(predictions, ground_truths):
        count = sum(1 for gt in gts if gt == pred)
        total += min(count / 3.0, 1.0)
    return total / len(predictions)


def simple_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Compute exact-match accuracy.

    Args:
        predictions: Predicted answer strings.
        ground_truths: Ground-truth answer strings.

    Returns:
        Fraction of exact matches.
    """
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)


def per_type_accuracy(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    question_types: Sequence[str],
) -> dict[str, float]:
    """Compute per-question-type accuracy.

    Args:
        predictions: Predicted answer strings.
        ground_truths: Ground-truth answer strings.
        question_types: Question type label for each sample (e.g. "yes/no",
            "number", "other").

    Returns:
        Dictionary mapping question type to accuracy.
    """
    type_correct: Counter[str] = Counter()
    type_total: Counter[str] = Counter()
    for pred, gt, qt in zip(predictions, ground_truths, question_types):
        type_total[qt] += 1
        if pred == gt:
            type_correct[qt] += 1
    return {
        qt: type_correct[qt] / max(type_total[qt], 1)
        for qt in sorted(type_total)
    }
