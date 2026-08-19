"""Unit tests for I3.6 execution assistance closure.

Tests the schema, planner, serializer, validator, and identity binding.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.execution_governor.schema import (
    ExecutionAssistanceFrame,
    ExecutionStep,
    ASSISTANCE_SCHEMA,
    ASSISTANCE_VERSION,
)
from hrm_adaptive_memory.executive.execution_governor.validator import (
    assert_no_evaluator_leakage,
    validate_assistance_frame,
)
from hrm_adaptive_memory.executive.execution_governor.identity import (
    assistance_frame_sha256,
    compute_assistance_identity,
)
from hrm_adaptive_memory.executive.execution_governor.planner import (
    ExecutionGovernor,
    plan_assistance,
)


def _make_test_frame(**overrides) -> ExecutionAssistanceFrame:
    """Build a valid test frame with optional overrides."""
    defaults = dict(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="VERIFY",
        bottleneck_type="UNVERIFIED_EVIDENCE",
        bottleneck_description="verification_state=UNVERIFIED",
        objective="determine whether retrieved evidence supports the candidate answer",
        target_type="evidence_item",
        target_description="highest-confidence unverified evidence",
        known_evidence=("evidence_count=3",),
        missing_information=("verification status",),
        execution_steps=(
            ExecutionStep(
                operation="verify",
                target="top evidence",
                purpose="check verification status",
                stop_condition="verification_status changes",
            ),
        ),
        success_conditions=("verification_status becomes SUFFICIENT",),
        failure_conditions=("evidence becomes FALSIFIED",),
        next_action_on_success="ANSWER",
        next_action_on_failure="SEARCH_MORE",
        max_assisted_steps=1,
        governor_reason_code="UNVERIFIED_EVIDENCE",
        source_state_sha256="abc123",
    )
    defaults.update(overrides)
    return ExecutionAssistanceFrame(**defaults)


# T1 — schema roundtrip
def test_assistance_schema_roundtrip():
    """Frame serializes and deserializes correctly."""
    frame = _make_test_frame()
    d = frame.as_dict()
    restored = ExecutionAssistanceFrame.from_dict(d)
    assert restored == frame
    assert restored.schema == ASSISTANCE_SCHEMA
    assert restored.version == ASSISTANCE_VERSION


# T2 — schema rejects unknown fields
def test_assistance_schema_rejects_unknown_fields():
    """from_dict rejects unknown fields."""
    frame = _make_test_frame()
    d = frame.as_dict()
    d["oracle_q_value"] = 5.0  # forbidden unknown field
    with pytest.raises(ValueError, match="Unknown fields"):
        ExecutionAssistanceFrame.from_dict(d)


# T3 — no evaluator leakage
def test_assistance_no_evaluator_leakage():
    """A clean frame passes the leakage check."""
    frame = _make_test_frame()
    # Should not raise
    assert_no_evaluator_leakage(frame)


def test_assistance_detects_leakage():
    """A frame with oracle data in a field is detected."""
    frame = _make_test_frame(
        objective="determine oracle_q_value for the current state")
    with pytest.raises(ValueError, match="leaks evaluator metadata"):
        assert_no_evaluator_leakage(frame)


# T4 — assistance is deterministic (same state → same frame)
def test_assistance_is_deterministic():
    """Same frame inputs produce same SHA-256."""
    frame1 = _make_test_frame()
    frame2 = _make_test_frame()
    assert assistance_frame_sha256(frame1) == assistance_frame_sha256(frame2)


def test_assistance_different_frame_different_hash():
    """Different frames produce different SHA-256."""
    frame1 = _make_test_frame(recommended_action="VERIFY")
    frame2 = _make_test_frame(recommended_action="SEARCH_MORE")
    assert assistance_frame_sha256(frame1) != assistance_frame_sha256(frame2)


# T5 — assistance hash stable
def test_assistance_hash_stable():
    """Hash is stable across multiple calls."""
    frame = _make_test_frame()
    h1 = assistance_frame_sha256(frame)
    h2 = assistance_frame_sha256(frame)
    h3 = assistance_frame_sha256(frame)
    assert h1 == h2 == h3


# T6 — max_assisted_steps enforced
def test_max_assisted_steps_enforced():
    """max_assisted_steps must be >= 1 and <= 3."""
    with pytest.raises(ValueError, match="max_assisted_steps must be >= 1"):
        _make_test_frame(max_assisted_steps=0)
    with pytest.raises(ValueError, match="max_assisted_steps must be <= 3"):
        _make_test_frame(max_assisted_steps=4)


def test_execution_steps_exceeds_max():
    """execution_steps cannot exceed max_assisted_steps."""
    with pytest.raises(ValueError, match="execution_steps.*exceeds"):
        _make_test_frame(
            max_assisted_steps=1,
            execution_steps=(
                ExecutionStep("op1", "t1", "p1", "sc1"),
                ExecutionStep("op2", "t2", "p2", "sc2"),
            ),
        )


# T7 — success condition required
def test_success_condition_required():
    """At least one success condition is required."""
    with pytest.raises(ValueError, match="success_condition is required"):
        _make_test_frame(success_conditions=())


# T8 — failure condition required
def test_failure_condition_required():
    """At least one failure condition is required."""
    with pytest.raises(ValueError, match="failure_condition is required"):
        _make_test_frame(failure_conditions=())


# T9 — validate_assistance_frame returns issues
def test_validate_clean_frame():
    """A clean frame has no validation issues."""
    frame = _make_test_frame()
    issues = validate_assistance_frame(frame)
    assert issues == []


def test_validate_detects_stop_scaffold():
    """STOP should not be scaffolded."""
    frame = _make_test_frame(recommended_action="STOP")
    issues = validate_assistance_frame(frame)
    assert any("STOP" in i for i in issues)


# T10 — identity changes on template change
def test_identity_changes_on_template_change():
    """Identity is deterministic but changes if source files change."""
    identity1 = compute_assistance_identity(
        'experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json',
        'configs/v2b_i3_1_utility_v1.json',
        'configs/v2b_i3_policy_v1.json',
    )
    identity2 = compute_assistance_identity(
        'experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json',
        'configs/v2b_i3_1_utility_v1.json',
        'configs/v2b_i3_policy_v1.json',
    )
    # Same inputs → same identity
    assert identity1["assistance_identity_sha256"] == identity2["assistance_identity_sha256"]


# T11 — field bounds enforced
def test_known_evidence_bound():
    """known_evidence must have at most 8 items."""
    with pytest.raises(ValueError, match="known_evidence must have at most 8"):
        _make_test_frame(known_evidence=tuple(f"e{i}" for i in range(9)))


def test_success_conditions_bound():
    """success_conditions must have at most 4 items."""
    with pytest.raises(ValueError, match="success_conditions must have at most 4"):
        _make_test_frame(success_conditions=tuple(f"s{i}" for i in range(5)))


# T12 — ExecutionStep serialization
def test_execution_step_serialization():
    """ExecutionStep serializes correctly."""
    step = ExecutionStep("verify", "target", "purpose", "stop_cond")
    d = step.as_dict()
    assert d == {
        "operation": "verify",
        "target": "target",
        "purpose": "purpose",
        "stop_condition": "stop_cond",
    }
