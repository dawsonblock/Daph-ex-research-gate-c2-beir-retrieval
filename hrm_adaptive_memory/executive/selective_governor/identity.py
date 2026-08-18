"""Identity and configuration hashing for selective governor gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

GATE_IDENTITY_SCHEMA = "DAPH_V2B_I3_5_2_GATE_IDENTITY_V1"
GATE_IDENTITY_VERSION = 1


def compute_gate_identity(
    *,
    delta_u_threshold: float = 5.0,
    max_harm_probability: float = 0.15,
    min_confidence: float = 0.60,
    predictor_name: str = "RuleBasedInterventionPredictor",
) -> dict[str, str]:
    """Compute deterministic SHA-256 identity for the selective gate."""
    pkg_dir = Path(__file__).resolve().parent

    h_features = hashlib.sha256((pkg_dir / "features.py").read_bytes()).hexdigest()
    h_model = hashlib.sha256((pkg_dir / "model.py").read_bytes()).hexdigest()
    h_gate = hashlib.sha256((pkg_dir / "intervention_gate.py").read_bytes()).hexdigest()

    config_dict = {
        "schema": GATE_IDENTITY_SCHEMA,
        "schema_version": GATE_IDENTITY_VERSION,
        "predictor_name": predictor_name,
        "delta_u_threshold": delta_u_threshold,
        "max_harm_probability": max_harm_probability,
        "min_confidence": min_confidence,
        "features_sha256": h_features,
        "model_sha256": h_model,
        "gate_sha256": h_gate,
    }

    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    h_identity = hashlib.sha256(canonical.encode()).hexdigest()

    return {
        "gate_identity_sha256": h_identity,
        "features_sha256": h_features,
        "model_sha256": h_model,
        "gate_sha256": h_gate,
    }
