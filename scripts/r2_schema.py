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


# The R13 static schema for comparison (from model_backend.py).
# Kept for reference and three-way tie-out, but the authoritative check
# uses FROZEN_R13_ACTION_SCHEMA_SHA256 below.
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

# Authoritative frozen R13 schema SHA.
# This is the canonical SHA of the action schema used by the R13 confirmation
# run (computed from model_backend.py's static schema with sort_keys=True and
# separators=(",",":")). It is frozen here as an independent constant so that
# accidental modification of both R13_STATIC_SCHEMA and build_action_schema()
# together would still be caught.
FROZEN_R13_ACTION_SCHEMA_SHA256 = (
    "2208076c081272b5354fd38b02f6943f79f0e8a695638bc25625a52fb49bacca"
)


def r13_static_schema_sha256() -> str:
    """SHA256 of the R13 static schema (canonical serialization).

    This computes from the local R13_STATIC_SCHEMA copy. For the authoritative
    check, use FROZEN_R13_ACTION_SCHEMA_SHA256 instead.
    """
    return schema_sha256(R13_STATIC_SCHEMA)


def c0_schema_identity_check() -> tuple[bool, str, str]:
    """Verify that Schema_R2(Allowed=ACTION_VOCABULARY) matches the frozen R13 SHA.

    This checks against the independently frozen constant, NOT against a
    local copy of the R13 schema. This prevents accidental co-modification.

    Returns (passed, actual_r2_sha, frozen_r13_sha).
    """
    r2_schema = build_action_schema(ACTION_VOCABULARY)
    r2_sha = schema_sha256(r2_schema)
    return (r2_sha == FROZEN_R13_ACTION_SCHEMA_SHA256, r2_sha, FROZEN_R13_ACTION_SCHEMA_SHA256)


def three_way_schema_tieout() -> dict:
    """Three-way schema identity tie-out.

    Verifies:
    1. R2 full-vocab schema SHA == frozen R13 expected SHA
    2. Local R13 static schema SHA == frozen R13 expected SHA

    Both must pass. If only one passes, there's a drift in one of the
    schema definitions.

    Returns dict with all three SHAs and pass/fail status.
    """
    r2_schema = build_action_schema(ACTION_VOCABULARY)
    r2_sha = schema_sha256(r2_schema)
    local_r13_sha = r13_static_schema_sha256()
    frozen_sha = FROZEN_R13_ACTION_SCHEMA_SHA256

    return {
        "r2_full_vocab_sha": r2_sha,
        "local_r13_static_sha": local_r13_sha,
        "frozen_r13_sha": frozen_sha,
        "r2_matches_frozen": r2_sha == frozen_sha,
        "local_matches_frozen": local_r13_sha == frozen_sha,
        "all_match": r2_sha == frozen_sha and local_r13_sha == frozen_sha,
    }
