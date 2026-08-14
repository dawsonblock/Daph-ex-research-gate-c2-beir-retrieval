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


def decode_output(raw_output: str, *, strict: bool = False) -> DecoderOutcome:
    """Decode a raw model output string into a validated ``ActionProposal``.

    This function never raises.  Every failure mode is returned as a
    ``DecoderOutcome`` with ``valid=False`` and a structured ``rejection_code``.

    When ``strict`` is False (default, development mode), the decoder
    extracts candidate JSON objects from prose using brace-balanced
    substring scanning.  This is permissive and allows reasoning text
    before or after the JSON.

    When ``strict`` is True (scientific mode), the decoder requires the
    entire response to be a single JSON object with no surrounding prose.
    This is the correct mode when the backend sends
    ``response_format: {"type": "json_object"}`` to the API.
    """
    if not raw_output or not raw_output.strip():
        return _reject(raw_output, "EMPTY_OUTPUT")

    stripped = raw_output.strip()

    if strict:
        # Scientific mode: require the entire response to be valid JSON.
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return _reject(raw_output, "STRICT_MODE_NOT_PURE_JSON")
    else:
        # Development mode: extract candidate JSON objects from prose.
        candidates = _extract_json_candidates(stripped)
        if not candidates:
            return _reject(raw_output, "NO_JSON_FOUND")

        parsed = None
        for json_str in candidates:
            try:
                candidate = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
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


def _extract_json_candidates(text: str) -> list[str]:
    """Extract all balanced ``{...}`` substrings from *text*.

    The model may emit reasoning text containing braces before the actual
    JSON object.  We scan for every opening brace and collect every
    brace-balanced substring, ordered by start position.  The caller tries
    each candidate until one parses as a valid JSON object.
    """
    candidates: list[str] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
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
                        candidates.append(text[start:i + 1])
                        pos = i + 1
                        break
        else:
            # Unbalanced braces from this start; advance past it.
            pos = start + 1
    return candidates
