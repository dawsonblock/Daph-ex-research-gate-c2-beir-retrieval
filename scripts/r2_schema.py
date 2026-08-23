#!/usr/bin/env python3
"""
R2 Canonical Dynamic Schema Construction.

Builds the JSON schema for constrained generation from the allowed action set.
Does NOT introduce any new action descriptions, reason-code hints, target hints,
or wording changes between arms.

Critical invariant:
    Schema_R2(Allowed = ACTION_VOCABULARY) == Schema_R13

When the allowed action set equals the full vocabulary, the schema must be
byte-identical to the R13 static schema (verified by Q2 qualification gate).
"""

from __future__ import annotations

import hashlib
import json

from r2_allowed_actions import ACTION_VOCABULARY


# Canonical action order (matches R13 static schema enum order).
# This ensures Schema_R2(Allowed=ACTION_VOCABULARY) == Schema_R13 byte-for-byte.
CANONICAL_ACTION_ORDER = (
    "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
    "REASON_MORE", "DEFER", "STOP",
)
_ACTION_ORDER_INDEX = {a: i for i, a in enumerate(CANONICAL_ACTION_ORDER)}


def _canonical_sort(actions: frozenset[str]) -> list[str]:
    """Sort actions by canonical R13 order (not alphabetical)."""
    return sorted(actions, key=lambda a: _ACTION_ORDER_INDEX[a])


def build_action_schema(allowed_actions: frozenset[str]) -> dict:
    """Build the JSON schema for the action proposal.

    The schema structure matches the R13 static schema exactly, except the
    action enum is constructed from allowed_actions instead of being hardcoded.

    When allowed_actions == ACTION_VOCABULARY, the output must be identical
    to the R13 static schema.
    """
    assert set(allowed_actions) <= ACTION_VOCABULARY, (
        f"allowed_actions must be subset of ACTION_VOCABULARY, "
        f"got extra: {set(allowed_actions) - ACTION_VOCABULARY}"
    )
    assert len(allowed_actions) > 0, "allowed_actions must not be empty"

    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": _canonical_sort(allowed_actions),
            },
            "reason_code": {
                "type": "string",
                "pattern": "^[A-Z][A-Z0-9_]*$",
            },
            "target_id": {"type": ["string", "null"]},
        },
        "required": ["action", "reason_code", "target_id"],
        "additionalProperties": False,
    }


def canonical_schema_json(schema: dict) -> str:
    """Serialize schema to canonical JSON (sorted keys, no extra whitespace)."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def schema_sha256(schema: dict) -> str:
    """Compute SHA256 of the canonical schema serialization."""
    return hashlib.sha256(
        canonical_schema_json(schema).encode()
    ).hexdigest()


def schema_action_enum(schema: dict) -> list[str]:
    """Extract the action enum from the schema."""
    return schema["properties"]["action"]["enum"]


def verify_schema_invariant(schema: dict, allowed_actions: frozenset[str]) -> None:
    """Assert that the schema's action enum matches the allowed action set."""
    enum_set = set(schema["properties"]["action"]["enum"])
    assert enum_set == set(allowed_actions), (
        f"Schema enum != allowed_actions: {enum_set} != {set(allowed_actions)}"
    )


# The R13 static schema for comparison (from model_backend.py)
R13_STATIC_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                "REASON_MORE", "DEFER", "STOP",
            ],
        },
        "reason_code": {
            "type": "string",
            "pattern": "^[A-Z][A-Z0-9_]*$",
        },
        "target_id": {"type": ["string", "null"]},
    },
    "required": ["action", "reason_code", "target_id"],
    "additionalProperties": False,
}


def r13_static_schema_sha256() -> str:
    """SHA256 of the R13 static schema (canonical serialization)."""
    return schema_sha256(R13_STATIC_SCHEMA)


def c0_schema_identity_check() -> tuple[bool, str, str]:
    """Verify that Schema_R2(Allowed=ACTION_VOCABULARY) == Schema_R13.

    Returns (passed, r2_sha, r13_sha).
    """
    r2_schema = build_action_schema(ACTION_VOCABULARY)
    r2_sha = schema_sha256(r2_schema)
    r13_sha = r13_static_schema_sha256()
    return (r2_sha == r13_sha, r2_sha, r13_sha)
