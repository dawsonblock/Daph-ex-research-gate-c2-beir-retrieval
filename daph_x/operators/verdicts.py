"""Verification verdict enums and parser.

This module must be strictly robust against substring aliasing.
"""
from __future__ import annotations

from enum import Enum
import re
import json


class VerificationVerdict(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class CritiqueVerdict(str, Enum):
    ERROR = "ERROR"
    NO_ERROR = "NO_ERROR"
    UNCERTAIN = "UNCERTAIN"


class FailureMode(str, Enum):
    ARITHMETIC = "ARITHMETIC"
    ALGEBRA = "ALGEBRA"
    LOGIC = "LOGIC"
    COMBINATORICS = "COMBINATORICS"
    SEQUENCE = "SEQUENCE"
    INSTRUCTION = "INSTRUCTION"
    EXTRACTION = "EXTRACTION"
    OTHER = "OTHER"
    NONE = "NONE"


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from text."""
    # Try the entire text first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object within the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def parse_verification_verdict(text: str) -> VerificationVerdict:
    """Parse a verification verdict. Never use substring tests."""
    data = _extract_json(text)
    raw = str(data.get("verdict", "")).strip().upper()

    if raw == VerificationVerdict.CORRECT:
        return VerificationVerdict.CORRECT
    if raw == VerificationVerdict.INCORRECT:
        return VerificationVerdict.INCORRECT
    if raw == VerificationVerdict.INCONCLUSIVE:
        return VerificationVerdict.INCONCLUSIVE

    return VerificationVerdict.INCONCLUSIVE


def parse_critique_verdict(text: str) -> CritiqueVerdict:
    """Parse a critique verdict. Never use substring tests."""
    data = _extract_json(text)
    raw = str(data.get("verdict", "")).strip().upper()

    if raw == CritiqueVerdict.ERROR:
        return CritiqueVerdict.ERROR
    if raw == CritiqueVerdict.NO_ERROR:
        return CritiqueVerdict.NO_ERROR
    if raw == CritiqueVerdict.UNCERTAIN:
        return CritiqueVerdict.UNCERTAIN

    return CritiqueVerdict.UNCERTAIN
