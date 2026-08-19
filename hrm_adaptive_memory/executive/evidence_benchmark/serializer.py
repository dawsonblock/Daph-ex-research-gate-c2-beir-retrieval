"""Serializer for evidence-bearing snapshots.

Converts an EvidenceSnapshot into a model-facing packet.
Performs fail-closed leakage validation.
"""
from __future__ import annotations

import json
from typing import Any

from .schema import EvidenceSnapshot


# Fields that must NEVER appear in an evidence packet
FORBIDDEN_FIELDS = {
    "oracle_value", "ground_truth", "gold_answer", "expected_terminal",
    "latent_state", "evaluator_label", "task_success", "q_value",
    "optimal_action", "reward", "true_answer", "correct_answer",
    "is_correct", "correct_hypothesis_id", "oracle_resolution_path",
    "verify_result",  # the hidden result of verification
}


def assert_no_evidence_leakage(packet: dict) -> None:
    """Fail-closed leakage validation for evidence packets."""
    def _check_recursive(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in FORBIDDEN_FIELDS:
                    raise ValueError(
                        f"LEAKAGE DETECTED: forbidden field '{key}' at {path}")
                _check_recursive(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _check_recursive(item, f"{path}[{i}]")

    _check_recursive(packet)


def serialize_evidence_snapshot(
    snapshot: EvidenceSnapshot,
    *,
    include_hypotheses: bool = True,
    include_evidence_relations: bool = True,
) -> dict:
    """Serialize an evidence snapshot into a model-facing packet.

    Args:
        snapshot: the evidence snapshot to serialize
        include_hypotheses: whether to include competing hypotheses
        include_evidence_relations: whether to include evidence-to-hypothesis links

    The packet includes:
      - task_id, task_summary
      - visible_evidence: proposition-level evidence items
      - hidden_evidence_count: how many items remain hidden
      - hypotheses: competing explanations (if included)
      - verified_count, supporting_count, contradicting_count
      - searched, reasoning_complete
      - resource_state
      - prior_actions, prior_outcomes
    """
    packet = {
        "schema": "DAPH_V2B_I3_7_EVIDENCE_PACKET_V1",
        "task_id": snapshot.task_id,
        "task_summary": snapshot.task_summary,
        "visible_evidence": [
            _serialize_evidence_item(e, include_relations=include_evidence_relations)
            for e in snapshot.visible_evidence
        ],
        "hidden_evidence_count": snapshot.hidden_evidence_count,
        "verified_count": snapshot.verified_count,
        "supporting_count": snapshot.supporting_count,
        "contradicting_count": snapshot.contradicting_count,
        "searched": snapshot.searched,
        "reasoning_complete": snapshot.reasoning_complete,
        "resource_state": dict(snapshot.resource_state),
        "prior_actions": list(snapshot.prior_actions),
        "prior_outcomes": list(snapshot.prior_outcomes),
    }

    if include_hypotheses:
        packet["hypotheses"] = [h.as_dict() for h in snapshot.hypotheses]

    assert_no_evidence_leakage(packet)

    return packet


def _serialize_evidence_item(
    evidence: Any,
    *,
    include_relations: bool = True,
) -> dict:
    """Serialize a single evidence item, optionally hiding hypothesis relations."""
    d = evidence.as_dict()
    if not include_relations:
        d.pop("supports", None)
        d.pop("contradicts", None)
    # Never expose verify_result — that's the hidden result of verification
    d.pop("verify_result", None)
    return d


def evidence_packet_json(packet: dict) -> str:
    """Serialize packet to JSON string for the model prompt."""
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))
