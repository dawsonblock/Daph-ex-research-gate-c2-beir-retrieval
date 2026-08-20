"""Metrics for semantic relation extraction.

Computes precision, recall, F1 for each relation type (SUPPORT,
CONTRADICT, NEUTRAL) and macro-averaged F1.

Also produces a confusion matrix.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from hrm_adaptive_memory.executive.semantic_relations.schema import RelationType


@dataclass
class ConfusionMatrix:
    """Confusion matrix for relation classification."""
    labels: tuple[str, ...] = (
        RelationType.SUPPORT.value,
        RelationType.CONTRADICT.value,
        RelationType.NEUTRAL.value,
    )
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, predicted: str, gold: str) -> None:
        self.counts[(predicted, gold)] = self.counts.get((predicted, gold), 0) + 1

    def as_dict(self) -> dict:
        matrix = {}
        for pred in self.labels:
            matrix[pred] = {}
            for gold in self.labels:
                matrix[pred][gold] = self.counts.get((pred, gold), 0)
        return matrix


@dataclass
class RelationMetrics:
    """Precision/recall/F1 for each relation type."""
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    accuracy: float = 0.0
    n_samples: int = 0
    confusion_matrix: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "per_type": self.per_type,
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "n_samples": self.n_samples,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_relation_metrics(
    predicted: Sequence[RelationType],
    gold: Sequence[RelationType],
) -> RelationMetrics:
    """Compute precision/recall/F1 for relation extraction.

    Args:
        predicted: predicted relation types
        gold: gold relation types

    Returns:
        RelationMetrics with per-type and macro-averaged scores
    """
    assert len(predicted) == len(gold), "predicted and gold must have same length"

    labels = [RelationType.SUPPORT, RelationType.CONTRADICT, RelationType.NEUTRAL]
    cm = ConfusionMatrix()

    for pred, g in zip(predicted, gold):
        cm.add(pred.value, g.value)

    per_type: dict[str, dict[str, float]] = {}
    precisions = []
    recalls = []
    f1s = []

    n_correct = 0
    for label in labels:
        label_str = label.value
        tp = cm.counts.get((label_str, label_str), 0)
        fp = sum(
            cm.counts.get((label_str, other.value), 0)
            for other in labels if other is not label
        )
        fn = sum(
            cm.counts.get((other.value, label_str), 0)
            for other in labels if other is not label
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type[label_str] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        n_correct += tp

    n = len(predicted)
    accuracy = n_correct / n if n > 0 else 0.0

    return RelationMetrics(
        per_type=per_type,
        macro_precision=round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        macro_recall=round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        macro_f1=round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        accuracy=round(accuracy, 4),
        n_samples=n,
        confusion_matrix=cm.as_dict(),
    )
