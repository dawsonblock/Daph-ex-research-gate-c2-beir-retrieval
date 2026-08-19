"""Leakage validator for execution assistance frames.

Ensures the execution governor only uses information legitimately
available to the controller. Fails closed if any forbidden field appears.

Forbidden evaluator information:
  - oracle Q-values
  - gold labels
  - oracle success paths
  - future trajectory outcomes
  - held-out evaluator state
  - minimum optimal cost
  - counterfactual returns
  - topology metadata
  - experiment condition metadata
"""
from __future__ import annotations

from typing import Any

from hrm_adaptive_memory.executive.execution_governor.schema import (
    ExecutionAssistanceFrame,
)
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import FORBIDDEN_KEYS


# Additional forbidden keys specific to execution assistance
ASSISTANCE_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "oracle_q_value",
    "oracle_return",
    "gold_label",
    "gold_answer",
    "oracle_success",
    "oracle_path",
    "future_outcome",
    "future_trajectory",
    "held_out_state",
    "minimum_optimal_cost",
    "counterfactual_return",
    "optimal_action",
    "q_star",
    "q_pi",
    "delta_q",
    "true_advantage",
    "oracle_bottleneck",
    "oracle_verification",
})


def assert_no_evaluator_leakage(frame: ExecutionAssistanceFrame) -> None:
    """Fail-closed check: no evaluator leakage in the assistance frame.

    Scans all string fields and nested structures for forbidden keys.
    """
    d = frame.as_dict()
    _scan_for_forbidden(d, "root")


def validate_assistance_frame(frame: ExecutionAssistanceFrame) -> list[str]:
    """Validate an assistance frame and return a list of issues.

    Returns an empty list if the frame is valid.
    """
    issues: list[str] = []

    # Check for evaluator leakage
    try:
        assert_no_evaluator_leakage(frame)
    except ValueError as e:
        issues.append(f"leakage: {e}")

    # Check boundedness
    if frame.max_assisted_steps < 1:
        issues.append("max_assisted_steps must be >= 1")
    if frame.max_assisted_steps > 3:
        issues.append("max_assisted_steps must be <= 3")
    if len(frame.execution_steps) > frame.max_assisted_steps:
        issues.append("execution_steps exceeds max_assisted_steps")

    # Check required fields
    if not frame.success_conditions:
        issues.append("success_conditions is required")
    if not frame.failure_conditions:
        issues.append("failure_conditions is required")
    if not frame.objective:
        issues.append("objective is required")
    if not frame.recommended_action:
        issues.append("recommended_action is required")

    # Check no STOP scaffold (STOP is terminal, no scaffold needed)
    if frame.recommended_action == "STOP":
        issues.append("STOP should not be scaffolded")

    # Check field bounds
    if len(frame.known_evidence) > 8:
        issues.append("known_evidence must have at most 8 items")
    if len(frame.missing_information) > 8:
        issues.append("missing_information must have at most 8 items")
    if len(frame.success_conditions) > 4:
        issues.append("success_conditions must have at most 4 items")
    if len(frame.failure_conditions) > 4:
        issues.append("failure_conditions must have at most 4 items")

    return issues


def _scan_for_forbidden(obj: Any, path: str) -> None:
    """Recursively scan for forbidden keys and values."""
    all_forbidden = FORBIDDEN_KEYS | ASSISTANCE_FORBIDDEN_KEYS

    if isinstance(obj, dict):
        for key in obj:
            if key in all_forbidden:
                raise ValueError(
                    f"Assistance frame leaks evaluator metadata: "
                    f"forbidden key '{key}' at path '{path}'")
            # Also check string values for forbidden substrings
            if isinstance(key, str):
                _check_string_value(key, path)
            _scan_for_forbidden(obj[key], f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_for_forbidden(item, f"{path}[{i}]")
    elif isinstance(obj, str):
        _check_string_value(obj, path)


def _check_string_value(s: str, path: str) -> None:
    """Check if a string value contains forbidden evaluator information."""
    # Check for explicit forbidden value patterns
    forbidden_patterns = [
        "oracle_q_",
        "gold_",
        "q_star=",
        "q_pi=",
        "delta_q=",
        "true_advantage=",
        "oracle_return=",
        "counterfactual_return=",
        "minimum_optimal_cost=",
        "held_out_",
    ]
    s_lower = s.lower()
    for pattern in forbidden_patterns:
        if pattern in s_lower:
            raise ValueError(
                f"Assistance frame leaks evaluator metadata: "
                f"forbidden pattern '{pattern}' in value '{s}' at path '{path}'")
