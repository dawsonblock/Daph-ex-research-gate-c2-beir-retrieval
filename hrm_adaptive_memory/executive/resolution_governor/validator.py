"""Fail-closed leakage validator for resolution governor.

Validates that resolution frames and contexts contain no evaluator
information, oracle Q-values, gold labels, or future outcomes.
"""
from __future__ import annotations

from typing import Any

from .schema import (
    ResolutionAssistanceFrame, ResolutionContext,
    Hypothesis, EvidenceAssessment, Discriminator, AnswerCondition,
    SearchSpecification, ResolutionStep, HypothesisUpdate,
)


# Fields that must never appear in any resolution object
FORBIDDEN_PATTERNS = {
    "oracle", "gold", "ground_truth", "true_answer", "correct_answer",
    "evaluator", "q_value", "q_star", "optimal", "reward",
    "task_success", "is_correct", "latent", "expected_terminal",
    "composition_complete", "conflict_resolvable",
    "required_provenance_count",
}


def _check_no_leakage(obj: Any, path: str = "") -> None:
    """Recursively check that no field names contain forbidden patterns."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in key_lower:
                    raise ValueError(
                        f"LEAKAGE: forbidden pattern '{pattern}' in field '{key}' at {path}")
            _check_no_leakage(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_no_leakage(item, f"{path}[{i}]")
    elif isinstance(obj, str):
        # Check string values for embedded oracle references
        obj_lower = obj.lower()
        for pattern in ("oracle_value", "gold_answer", "ground_truth", "true_answer"):
            if pattern in obj_lower:
                raise ValueError(
                    f"LEAKAGE: forbidden pattern '{pattern}' in value at {path}")


def validate_resolution_frame(frame: ResolutionAssistanceFrame) -> None:
    """Validate that a resolution frame has no evaluator leakage."""
    _check_no_leakage(frame.as_dict(), "frame")


def validate_resolution_context(context: ResolutionContext) -> None:
    """Validate that a resolution context has no evaluator leakage."""
    _check_no_leakage(context.as_dict(), "context")


def validate_hypothesis(h: Hypothesis) -> None:
    """Validate a single hypothesis."""
    _check_no_leakage(h.as_dict(), "hypothesis")


def validate_evidence(e: EvidenceAssessment) -> None:
    """Validate a single evidence assessment."""
    _check_no_leakage(e.as_dict(), "evidence")
