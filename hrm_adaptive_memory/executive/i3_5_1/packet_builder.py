"""Packet builder for I3.5.1 — two separate packet schemas.

BASE packet (no governor):
  {
    "schema": "DAPH_V2B_I3_5_1_BASE_PACKET_V1",
    "task": {...},
    "resources": {...},
    "allowed_actions": [...],
    "history": {...},
    "cognitive_state": {...} or null
  }

GOVERNOR packet:
  {
    "schema": "DAPH_V2B_I3_5_1_GOVERNOR_PACKET_V1",
    "task": {...},
    "resources": {...},
    "allowed_actions": [...],
    "history": {...},
    "cognitive_state": {...} or null,
    "governor": {...}
  }

Never serialize "governor": null for control.
No-governor means no governor-related structure exists anywhere.

Forbidden evaluator keys (recursive scan):
  condition, experiment_arm, governor_enabled, oracle, latent,
  optimal_action, topology_id, topology_hash, difficulty,
  information_class, expected_terminal, task_success, gold, held_out
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.model_packet import (
    serialize_packet as serialize_i3_4_packet,
    assert_no_condition_leakage,
)
from hrm_adaptive_memory.executive.governor.assessor import GovernorDecisionFrame
from hrm_adaptive_memory.executive.governor.serializer import serialize_frame_dict

BASE_PACKET_SCHEMA = "DAPH_V2B_I3_5_1_BASE_PACKET_V1"
BASE_PACKET_VERSION = 1
GOVERNOR_PACKET_SCHEMA = "DAPH_V2B_I3_5_1_GOVERNOR_PACKET_V1"
GOVERNOR_PACKET_VERSION = 1

# Forbidden evaluator keys — must never appear in any packet (recursive).
FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "condition",
    "experiment_arm",
    "governor_enabled",
    "oracle",
    "latent",
    "optimal_action",
    "topology_id",
    "topology_hash",
    "difficulty",
    "information_class",
    "expected_terminal",
    "task_success",
    "gold",
    "held_out",
    "condition_id",
    "treatment_group",
    "experiment_identity_sha256",
})


def build_base_packet(observation: ControllerObservation) -> dict[str, Any]:
    """Build a BASE packet (no governor structure at all)."""
    packet = serialize_i3_4_packet(observation)
    packet["schema"] = BASE_PACKET_SCHEMA
    packet["schema_version"] = BASE_PACKET_VERSION
    assert_no_evaluator_leakage(packet)
    return packet


def build_governor_packet(
    observation: ControllerObservation,
    governor_frame: GovernorDecisionFrame,
) -> dict[str, Any]:
    """Build a GOVERNOR packet with the governor decision frame."""
    packet = serialize_i3_4_packet(observation)
    packet["schema"] = GOVERNOR_PACKET_SCHEMA
    packet["schema_version"] = GOVERNOR_PACKET_VERSION
    packet["governor"] = serialize_frame_dict(governor_frame)
    assert_no_evaluator_leakage(packet)
    return packet


def packet_sha256(packet: dict[str, Any]) -> str:
    """Canonical SHA-256 of the serialized packet."""
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def packet_json(packet: dict[str, Any]) -> str:
    """Canonical JSON string of the serialized packet."""
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


def assert_no_evaluator_leakage(packet: dict[str, Any]) -> None:
    """Fail-closed recursive scan for forbidden evaluator keys.

    Checks both the base packet fields (via I3.4 condition leakage check)
    and the expanded forbidden key set.
    """
    # Reuse I3.4 check for base packet fields
    assert_no_condition_leakage(packet)

    def _check_keys(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key in obj:
                if key in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"Packet leaks evaluator metadata: "
                        f"forbidden key '{key}' at path '{path}'")
                _check_keys(obj[key], f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check_keys(item, f"{path}[{i}]")

    _check_keys(packet)
