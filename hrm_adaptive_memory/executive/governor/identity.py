"""Governor identity: deterministic hash of the governor configuration.

This binds the governor to a specific frozen configuration.
Any change to action semantics, bottleneck definitions, transition model,
or scoring weights constitutes a new governor identity.
"""
from __future__ import annotations

import hashlib
import json
from hrm_adaptive_memory.executive.governor.action_semantics import FROZEN_ACTION_SEMANTICS
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor


GOVERNOR_IDENTITY_SCHEMA = "DAPH_V2B_I3_5_GOVERNOR_IDENTITY_V1"
GOVERNOR_IDENTITY_VERSION = 1


def compute_governor_identity() -> dict:
    """Compute the deterministic identity of the governor configuration."""
    # Action semantics hash
    semantics_payload = {
        action: sem.as_dict()
        for action, sem in sorted(FROZEN_ACTION_SEMANTICS.items())
    }
    semantics_hash = hashlib.sha256(
        json.dumps(semantics_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # Governor configuration
    config = {
        "schema": GOVERNOR_IDENTITY_SCHEMA,
        "version": GOVERNOR_IDENTITY_VERSION,
        "action_semantics_sha256": semantics_hash,
        "max_steps": 25,
        "scoring_weights": {
            "progress": 1.0,
            "information": 1.0,
            "cost": 0.5,
            "risk": 0.5,
            "redundancy": 1.0,
            "options": 0.3,
        },
        "ordinal_levels": ["NONE", "LOW", "MEDIUM", "HIGH"],
        "bottleneck_kinds": [
            "NO_EVIDENCE", "FALSIFIED_EVIDENCE", "UNVERIFIED_EVIDENCE",
            "STALE_INFORMATION", "UNRESOLVED_CONFLICT", "INSUFFICIENT_REASONING",
            "RESOURCE_EXHAUSTION", "REPEATED_NO_GAIN", "READY_TO_ANSWER",
            "IRREDUCIBLE_UNCERTAINTY",
        ],
        "transition_model": "deterministic_symbolic_v1",
        "redundancy_rules": {
            "0_attempts": "NONE",
            "1_attempt_not_last": "LOW",
            "1_attempt_is_last": "MEDIUM",
            "2_plus_attempts": "HIGH",
        },
    }

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "schema": GOVERNOR_IDENTITY_SCHEMA,
        "version": GOVERNOR_IDENTITY_VERSION,
        "governor_sha256": config_hash,
        "action_semantics_sha256": semantics_hash,
        "configuration": config,
    }
