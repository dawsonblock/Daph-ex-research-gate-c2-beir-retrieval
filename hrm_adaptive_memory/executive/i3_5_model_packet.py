"""I3.5 model packet: extends I3.4 packet with governor decision frame.

The packet is the model-visible representation. It contains:
- The standard I3.4 controller observation (task, resources, actions, cognitive state)
- A governor frame with bottleneck analysis and candidate action assessments

The governor frame uses only controller-visible information.
No oracle values, latent state, or topology IDs leak into the packet.

Schema identity: ``DAPH_V2B_I3_5_INPUT_PACKET_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.cognitive_control.state import CognitiveStateSnapshot
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.model_packet import (
    serialize_packet as serialize_i3_4_packet,
    assert_no_condition_leakage,
)
from hrm_adaptive_memory.executive.governor.assessor import GovernorDecisionFrame
from hrm_adaptive_memory.executive.governor.serializer import serialize_frame_dict

PACKET_SCHEMA = "DAPH_V2B_I3_5_INPUT_PACKET_V1"
PACKET_SCHEMA_VERSION = 1


def serialize_governor_packet(
    observation: ControllerObservation,
    governor_frame: GovernorDecisionFrame,
) -> dict[str, Any]:
    """Serialize a controller observation + governor frame into the frozen packet.

    The packet structure is identical whether cognitive_state is present
    (state-aware) or None (state-blind). The governor frame is always present
    and uses only controller-visible information.
    """
    base_packet = serialize_i3_4_packet(observation)
    # Override schema to I3.5
    base_packet["schema"] = PACKET_SCHEMA
    base_packet["schema_version"] = PACKET_SCHEMA_VERSION
    # Add governor frame
    base_packet["governor"] = serialize_frame_dict(governor_frame)
    return base_packet


def governor_packet_sha256(packet: dict[str, Any]) -> str:
    """Canonical SHA-256 of the serialized packet (sorted JSON, compact)."""
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def governor_packet_json(packet: dict[str, Any]) -> str:
    """Canonical JSON string of the serialized packet (sorted, compact)."""
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


def assert_no_governor_leakage(packet: dict[str, Any]) -> None:
    """Fail-closed check that the packet contains no evaluator metadata.

    Extends the I3.4 condition-leakage check with governor-specific forbidden keys.
    """
    # Reuse I3.4 check for base packet fields
    assert_no_condition_leakage(packet)

    # Additional governor-specific forbidden keys
    governor_forbidden = frozenset({
        "governor_sha256", "action_semantics_sha256",
        "oracle_value", "latent_optimal_value",
        "observable_optimal_value", "topology_id",
        "topology_depth_band", "information_class_id",
    })

    def _check_keys(obj: Any) -> None:
        if isinstance(obj, dict):
            for key in obj:
                if key in governor_forbidden:
                    raise ValueError(
                        f"governor packet leaks evaluator metadata: key '{key}'")
                _check_keys(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _check_keys(item)

    _check_keys(packet)
