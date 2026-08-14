"""Frozen fail-closed output decoder for the I3.4 pinned-model controller.

The model must emit a single JSON object with exactly three fields:

    {"action": "<ACTION_NAME>", "reason_code": "<REASON_CODE>", "target_id": null | "<id>"}

Valid action names are exactly the seven frozen ``V2B_ACTIONS``.  The decoder
rejects malformed JSON, unknown actions, missing fields, invalid reason codes,
and malformed target fields.  Every rejection is recorded as a
``DecoderOutcome`` so the controller and runner can track malformed-output
metrics without raising at the loop level.

Schema identity: ``DAPH_V2B_I3_4_OUTPUT_SCHEMA_V1`` (frozen).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS, validate_v2b_action

from .actions import ActionProposal

OUTPUT_SCHEMA = "DAPH_V2B_I3_4_OUTPUT_SCHEMA_V1"
OUTPUT_SCHEMA_VERSION = 1

VALID_ACTION_NAMES = frozenset(action.value for action in V2B_ACTIONS)
_REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class DecoderOutcome:
    """Result of decoding one model output string."""

    proposal: ActionProposal | None
    raw_output: str
    valid: bool
    rejection_code: str | None  # None when valid
    parsed_json: dict[str, Any] | None  # None when JSON parsing failed


def _reject(raw: str, code: str, parsed: dict[str, Any] | None = None) -> DecoderOutcome:
    return DecoderOutcome(proposal=None, raw_output=raw, valid=False,
                          rejection_code=code, parsed_json=parsed)


def decode_output(raw_output: str) -> DecoderOutcome:
    """Decode a raw model output string into a validated ``ActionProposal``.

    This function never raises.  Every failure mode is returned as a
    ``DecoderOutcome`` with ``valid=False`` and a structured ``rejection_code``.
    """
    if not raw_output or not raw_output.strip():
        return _reject(raw_output, "EMPTY_OUTPUT")

    stripped = raw_output.strip()
    # Extract the first JSON object from the output.  The model may emit
    # reasoning text before or after the JSON; we only accept a single
    # well-formed JSON object.
    json_str = _extract_json(stripped)
    if json_str is None:
        return _reject(raw_output, "NO_JSON_FOUND")

    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return _reject(raw_output, "MALFORMED_JSON")

    if not isinstance(parsed, dict):
        return _reject(raw_output, "NOT_JSON_OBJECT", parsed if isinstance(parsed, dict) else None)

    # Reject extra keys to enforce a strict schema.
    allowed_keys = frozenset({"action", "reason_code", "target_id"})
    extra = set(parsed.keys()) - allowed_keys
    if extra:
        return _reject(raw_output, "EXTRA_KEYS", parsed)

    # action field
    if "action" not in parsed:
        return _reject(raw_output, "MISSING_ACTION", parsed)
    action_value = parsed["action"]
    if not isinstance(action_value, str):
        return _reject(raw_output, "ACTION_NOT_STRING", parsed)
    if action_value not in VALID_ACTION_NAMES:
        return _reject(raw_output, "UNKNOWN_ACTION", parsed)

    # reason_code field
    if "reason_code" not in parsed:
        return _reject(raw_output, "MISSING_REASON_CODE", parsed)
    reason_code = parsed["reason_code"]
    if not isinstance(reason_code, str):
        return _reject(raw_output, "REASON_CODE_NOT_STRING", parsed)
    if not _REASON_CODE_RE.match(reason_code):
        return _reject(raw_output, "INVALID_REASON_CODE", parsed)

    # target_id field (required by schema: exactly three fields)
    if "target_id" not in parsed:
        return _reject(raw_output, "MISSING_TARGET_ID", parsed)
    target_id = parsed["target_id"]
    if target_id is not None and not isinstance(target_id, str):
        return _reject(raw_output, "INVALID_TARGET_ID", parsed)
    if isinstance(target_id, str):
        if not target_id.strip():
            return _reject(raw_output, "EMPTY_TARGET_ID", parsed)
        target_id = target_id.strip()

    action = validate_v2b_action(action_value)
    proposal = ActionProposal(action=action, reason_code=reason_code, target_id=target_id)
    return DecoderOutcome(
        proposal=proposal, raw_output=raw_output, valid=True,
        rejection_code=None, parsed_json=parsed)


def _extract_json(text: str) -> str | None:
    """Extract the first balanced ``{...}`` substring from *text*."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None
