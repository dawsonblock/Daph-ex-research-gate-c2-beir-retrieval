"""Assistance packet serializer for I3.6.

Builds three packet types for the I3.6 experiment arms:
  - BASE: no governor (B0 / OFF arm)
  - GOVERNOR: action-only governor packet (B1 / ACTION_ONLY arm)
  - EXECUTION_ASSIST: execution assistance packet (B2 / EXECUTION_ASSIST arm)

The EXECUTION_ASSIST packet extends the governor packet with a structured
"execution_assistance" field containing the ExecutionAssistanceFrame.

The EXECUTION_ASSIST_DIRECT packet (B3) is identical to EXECUTION_ASSIST
but the model is told the governor has chosen the first action.

All packets pass the same evaluator-leakage checks as I3.5.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import (
    build_base_packet,
    build_governor_packet,
    packet_sha256 as i3_5_packet_sha256,
    packet_json as i3_5_packet_json,
    assert_no_evaluator_leakage as i3_5_assert_no_leakage,
    FORBIDDEN_KEYS,
)
from hrm_adaptive_memory.executive.governor.assessor import GovernorDecisionFrame
from hrm_adaptive_memory.executive.execution_governor.schema import (
    ExecutionAssistanceFrame,
)


EXECUTION_ASSIST_PACKET_SCHEMA = "DAPH_V2B_I3_6_EXECUTION_ASSIST_PACKET_V1"
EXECUTION_ASSIST_PACKET_VERSION = 1
EXECUTION_ASSIST_DIRECT_PACKET_SCHEMA = "DAPH_V2B_I3_6_EXECUTION_ASSIST_DIRECT_PACKET_V1"
EXECUTION_ASSIST_DIRECT_PACKET_VERSION = 1


def serialize_assistance_frame(frame: ExecutionAssistanceFrame) -> dict[str, Any]:
    """Serialize an ExecutionAssistanceFrame to a dict for the packet."""
    return frame.as_dict()


def build_execution_assist_packet(
    observation: ControllerObservation,
    governor_frame: GovernorDecisionFrame,
    assistance_frame: ExecutionAssistanceFrame,
) -> dict[str, Any]:
    """Build an EXECUTION_ASSIST packet.

    Contains:
      - base observation fields
      - governor decision frame (action candidates)
      - execution_assistance: structured scaffold

    The model retains final decision authority.
    """
    packet = build_governor_packet(observation, governor_frame)
    packet["schema"] = EXECUTION_ASSIST_PACKET_SCHEMA
    packet["schema_version"] = EXECUTION_ASSIST_PACKET_VERSION
    packet["execution_assistance"] = serialize_assistance_frame(assistance_frame)
    _assert_no_assistance_leakage(packet)
    return packet


def build_execution_assist_direct_packet(
    observation: ControllerObservation,
    governor_frame: GovernorDecisionFrame,
    assistance_frame: ExecutionAssistanceFrame,
) -> dict[str, Any]:
    """Build an EXECUTION_ASSIST_DIRECT packet.

    Same as EXECUTION_ASSIST but signals that the governor has chosen
    the first action. The model continues with the scaffold context.

    The "governor_directive" field tells the model the first action
    is pre-selected by the governor.
    """
    packet = build_execution_assist_packet(
        observation, governor_frame, assistance_frame)
    packet["schema"] = EXECUTION_ASSIST_DIRECT_PACKET_SCHEMA
    packet["schema_version"] = EXECUTION_ASSIST_DIRECT_PACKET_VERSION
    packet["governor_directive"] = {
        "action": assistance_frame.recommended_action,
        "reason": "governor selects first action; model continues with scaffold",
    }
    _assert_no_assistance_leakage(packet)
    return packet


def serialize_assistance_packet(
    observation: ControllerObservation,
    governor_frame: GovernorDecisionFrame,
    assistance_frame: ExecutionAssistanceFrame,
    mode: str = "EXECUTION_ASSIST",
) -> dict[str, Any]:
    """Serialize the appropriate packet based on mode.

    Args:
        mode: "BASE", "ACTION_ONLY", "EXECUTION_ASSIST", or "EXECUTION_ASSIST_DIRECT"
    """
    if mode == "BASE":
        return build_base_packet(observation)
    elif mode == "ACTION_ONLY":
        return build_governor_packet(observation, governor_frame)
    elif mode == "EXECUTION_ASSIST":
        return build_execution_assist_packet(
            observation, governor_frame, assistance_frame)
    elif mode == "EXECUTION_ASSIST_DIRECT":
        return build_execution_assist_direct_packet(
            observation, governor_frame, assistance_frame)
    else:
        raise ValueError(f"Unknown packet mode: {mode}")


def packet_sha256(packet: dict[str, Any]) -> str:
    """Canonical SHA-256 of the serialized packet."""
    return i3_5_packet_sha256(packet)


def packet_json(packet: dict[str, Any]) -> str:
    """Canonical JSON string of the serialized packet."""
    return i3_5_packet_json(packet)


def _assert_no_assistance_leakage(packet: dict[str, Any]) -> None:
    """Fail-closed recursive scan for forbidden evaluator keys.

    Extends the I3.5 leakage check to cover the execution_assistance field.
    """
    i3_5_assert_no_leakage(packet)

    def _check_keys(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key in obj:
                if key in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"Assistance packet leaks evaluator metadata: "
                        f"forbidden key '{key}' at path '{path}'")
                _check_keys(obj[key], f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check_keys(item, f"{path}[{i}]")

    _check_keys(packet.get("execution_assistance", {}), "execution_assistance")
    _check_keys(packet.get("governor_directive", {}), "governor_directive")


def assert_no_evaluator_leakage(packet: dict[str, Any]) -> None:
    """Public alias for the leakage check."""
    _assert_no_assistance_leakage(packet)
