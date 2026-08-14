"""Frozen canonical input packet for the I3.4 pinned-model controller.

The packet is the **only** model-visible representation of the controller
observation.  Its structure is identical across all conditions; masked
cognitive-state fields are replaced with canonical null values, never with
condition-specific shapes.  Condition identity never appears in the packet.

Schema identity: ``DAPH_V2B_I3_4_INPUT_PACKET_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, TemporalStatus, VerificationState)

from .metareasoning_controller import ControllerObservation

PACKET_SCHEMA = "DAPH_V2B_I3_4_INPUT_PACKET_V1"
PACKET_SCHEMA_VERSION = 1

# Canonical null sentinels for masked cognitive-state fields.
NULL_VERIFICATION_STATES: list[dict[str, Any]] = []
NULL_PROVENANCE_SUMMARIES: list[str] = []
NULL_TEMPORAL_STATUS = "UNKNOWN"
NULL_CONFLICTS: list[dict[str, Any]] = []
NULL_PRIOR_DECISIONS: list[dict[str, Any]] = []
NULL_PRIOR_OUTCOMES: list[str] = []
NULL_OBSERVATION_SIGNALS: list[str] = []


def _serialize_verification_states(
    snapshot: CognitiveStateSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "target_id": item.target_id,
            "state": item.state.value,
            "evidence_count": item.evidence_count,
            "last_verified": item.last_verified,
        }
        for item in snapshot.verification_states
    ]


def _serialize_conflicts(snapshot: CognitiveStateSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "conflict_id": item.conflict_id,
            "relation": item.relation,
            "source_lineage_count": item.source_lineage_count,
            "status": item.status,
        }
        for item in snapshot.unresolved_conflicts
    ]


def _serialize_prior_decisions(
    snapshot: CognitiveStateSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "decision_id": item.decision_id,
            "selected_action": item.selected_action,
            "reason_code": item.reason_code,
            "outcome": item.outcome,
        }
        for item in snapshot.prior_decisions
    ]


def _serialize_relevant_memories(
    snapshot: CognitiveStateSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": item.memory_id,
            "relevance_score": item.relevance_score,
            "verification_state": item.verification_state.value,
            "source_lineage_count": item.source_lineage_count,
            "evidence_count": item.evidence_count,
            "conflict_state": item.conflict_state,
            "temporal_status": item.temporal_status.value,
        }
        for item in snapshot.relevant_memories
    ]


def serialize_packet(observation: ControllerObservation) -> dict[str, Any]:
    """Serialize a controller observation into the frozen canonical packet.

    The packet structure is identical whether ``cognitive_state`` is present
    (state-aware / ablation conditions) or ``None`` (state-blind).  When the
    snapshot is ``None``, every cognitive field is replaced by its canonical
    null sentinel.  No condition name, mask, or evaluator metadata appears in
    the output.
    """
    snapshot = observation.cognitive_state
    if snapshot is None:
        cognitive_state = {
            "relevant_memories": [],
            "verification_states": NULL_VERIFICATION_STATES,
            "provenance_summaries": NULL_PROVENANCE_SUMMARIES,
            "temporal_status": NULL_TEMPORAL_STATUS,
            "unresolved_conflicts": NULL_CONFLICTS,
            "prior_decisions": NULL_PRIOR_DECISIONS,
            "prior_outcomes": NULL_PRIOR_OUTCOMES,
            "observation_signals": NULL_OBSERVATION_SIGNALS,
        }
    else:
        cognitive_state = {
            "relevant_memories": _serialize_relevant_memories(snapshot),
            "verification_states": _serialize_verification_states(snapshot),
            "provenance_summaries": list(snapshot.provenance_summaries),
            "temporal_status": snapshot.temporal_status.value,
            "unresolved_conflicts": _serialize_conflicts(snapshot),
            "prior_decisions": _serialize_prior_decisions(snapshot),
            "prior_outcomes": list(snapshot.prior_outcomes),
            "observation_signals": list(snapshot.observation_signals),
        }
    return {
        "schema": PACKET_SCHEMA,
        "schema_version": PACKET_SCHEMA_VERSION,
        "task_id": observation.task_id,
        "task_summary": observation.task_summary,
        "resource_state": dict(observation.resource_state),
        "allowed_actions": [action.value for action in observation.allowed_actions],
        "executed_actions": [action.value for action in observation.executed_actions],
        "rejected_actions": [action.value for action in observation.rejected_actions],
        "policy_feedback": [
            {
                "effect": feedback.effect,
                "resolved_action": (feedback.resolved_action.value
                                    if feedback.resolved_action is not None else None),
                "reason_class": feedback.reason_class,
            }
            for feedback in observation.policy_feedback
        ],
        "cognitive_state": cognitive_state,
    }


def packet_sha256(packet: dict[str, Any]) -> str:
    """Canonical SHA-256 of the serialized packet (sorted JSON, compact)."""
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def packet_json(packet: dict[str, Any]) -> str:
    """Canonical JSON string of the serialized packet (sorted, compact)."""
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


# Forbidden keys that would leak condition identity or evaluator metadata.
_FORBIDDEN_KEYS = frozenset({
    "condition", "mask", "condition_name", "observation_mask",
    "split", "difficulty", "topology", "oracle", "expected_terminal",
    "latent_state", "benchmark_split",
})


def assert_no_condition_leakage(packet: dict[str, Any]) -> None:
    """Fail-closed check that the packet contains no condition-identity keys."""
    _check_keys(packet)


def _check_keys(obj: Any) -> None:
    if isinstance(obj, dict):
        for key in obj:
            if key in _FORBIDDEN_KEYS:
                raise ValueError(f"packet leaks evaluator metadata: key '{key}'")
            _check_keys(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _check_keys(item)
