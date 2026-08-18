"""Serialization and hashing for selective governor decisions."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .intervention_gate import InterventionDecision

SERIALIZER_SCHEMA = "DAPH_V2B_I3_5_2_GATE_SERIALIZER_V1"
SERIALIZER_VERSION = 1


def serialize_decision(decision: InterventionDecision) -> dict[str, Any]:
    """Serialize an InterventionDecision to a dictionary for receipts/logging."""
    return decision.as_dict()


def decision_sha256(decision: InterventionDecision) -> str:
    """Compute deterministic SHA-256 hash of an intervention decision."""
    canonical = json.dumps(decision.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
