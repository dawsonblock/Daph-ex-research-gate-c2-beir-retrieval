"""Governor serializer: convert GovernorDecisionFrame to model-visible JSON.

This produces the compact JSON that gets injected into the model packet.
The model sees the frame but not the governor's internal scoring details.
"""
from __future__ import annotations

import json
from hrm_adaptive_memory.executive.governor.assessor import GovernorDecisionFrame


SERIALIZER_SCHEMA = "DAPH_V2B_I3_5_GOVERNOR_SERIALIZER_V1"
SERIALIZER_VERSION = 1


def serialize_frame(frame: GovernorDecisionFrame) -> str:
    """Serialize a GovernorDecisionFrame to a compact JSON string.

    This is what the model sees in its prompt.
    """
    packet = frame.as_model_packet()
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


def serialize_frame_dict(frame: GovernorDecisionFrame) -> dict:
    """Serialize a GovernorDecisionFrame to a dict for the model packet."""
    return frame.as_model_packet()


def frame_sha256(frame: GovernorDecisionFrame) -> str:
    """Compute a deterministic SHA-256 of the frame."""
    import hashlib
    return hashlib.sha256(serialize_frame(frame).encode()).hexdigest()
