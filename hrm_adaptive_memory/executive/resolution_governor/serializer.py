"""Resolution packet serializer and leakage validation.

Serializes a ResolutionAssistanceFrame into a model-facing packet.
Performs fail-closed leakage validation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation

from .schema import ResolutionAssistanceFrame, ResolutionContext


# Fields that must NEVER appear in a resolution packet
FORBIDDEN_FIELDS = {
    "oracle_value", "ground_truth", "gold_answer", "expected_terminal",
    "latent_state", "evaluator_label", "task_success", "q_value",
    "optimal_action", "reward", "true_answer", "correct_answer",
    "is_correct", "verification_result", "composition_complete",
    "conflict_resolvable", "required_provenance_count",
}


def assert_no_evaluator_leakage(packet: dict) -> None:
    """Fail-closed leakage validation for resolution packets.

    Rejects any field that could leak evaluator information.
    """
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


def serialize_resolution_packet(
    observation: ControllerObservation,
    gov_frame: Any,
    resolution_frame: ResolutionAssistanceFrame,
    context: ResolutionContext | None = None,
    mode: str = "RESOLUTION_ASSIST",
) -> dict:
    """Serialize a resolution assistance packet for the model.

    Modes:
      - RESOLUTION_ASSIST: full resolution scaffold, model retains action control
      - RESOLUTION_ASSIST_DIRECT: governor's first action is authoritative

    The packet includes:
      - task_id, task_summary (from observation)
      - governor_recommended_action
      - candidate_hypotheses (with evidence relationships)
      - current_evidence (with hypothesis links)
      - unresolved_question
      - discriminating_evidence
      - execution_plan (with decision consequences)
      - answer_conditions (explicit hypothesis -> answer mappings)
      - defer_condition
      - search_specification (if SEARCH_MORE)
      - max_additional_actions
      - context state (if persistent context provided)
    """
    packet = {
        "schema": "DAPH_V2B_I3_6D_RESOLUTION_PACKET_V1",
        "mode": mode,
        "task_id": observation.task_id,
        "task_summary": observation.task_summary,
        "governor_recommended_action": resolution_frame.recommended_action,
        "task_goal": resolution_frame.task_goal,
        "candidate_hypotheses": [h.as_dict() for h in resolution_frame.candidate_hypotheses],
        "current_evidence": [e.as_dict() for e in resolution_frame.current_evidence],
        "unresolved_question": resolution_frame.unresolved_question,
        "discriminating_evidence": [d.as_dict() for d in resolution_frame.discriminating_evidence],
        "execution_plan": [s.as_dict() for s in resolution_frame.execution_plan],
        "answer_conditions": [a.as_dict() for a in resolution_frame.answer_conditions],
        "defer_condition": resolution_frame.defer_condition,
        "search_specification": (
            resolution_frame.search_specification.as_dict()
            if resolution_frame.search_specification else None
        ),
        "max_additional_actions": resolution_frame.max_additional_actions,
    }

    # Include persistent context if provided
    if context is not None:
        packet["resolution_context"] = {
            "context_id": context.context_id,
            "completed_steps": list(context.completed_steps),
            "pending_steps": list(context.pending_steps),
            "current_best_hypothesis": context.current_best_hypothesis,
            "termination_status": context.termination_status,
            "step_counter": context.step_counter,
            "hypothesis_updates": [u.as_dict() for u in context.hypothesis_updates],
        }

    # Validate no leakage
    assert_no_evaluator_leakage(packet)

    return packet


def packet_json(packet: dict) -> str:
    """Serialize packet to JSON string for the model prompt."""
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


def packet_sha256(packet: dict) -> str:
    """Compute SHA256 hash of the packet."""
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
