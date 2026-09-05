"""Canonical answer extraction for external operator responses.

External reasoning systems (ThinkBooster, OptiLLM, PaCoRe, etc.) may emit
full reasoning traces rather than just a final answer. The R13 evaluator
expects concise answers like "420", "5/14", "yes", "knight".

This module provides a single provider-independent reducer:

    raw response → canonical answer extractor → terminal_answer

The reasoning_trace is always the complete raw response.

Extraction strategy by answer_type:
  - numeric: extract integer or float
  - fraction: extract a/b or decimal
  - yes_no: extract yes/no
  - true_false: extract true/false
  - letter: extract single letter (A-E)
  - string: extract last quoted/boxed/final-line answer
  - default: last number or last boxed expression
"""
from __future__ import annotations

import re
from typing import Any


# Regex patterns
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_LAST_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_FRACTION_RE = re.compile(r"-?\d+\s*/\s*\d+")
_DECIMAL_RE = re.compile(r"-?\d+\.\d+")
_INTEGER_RE = re.compile(r"-?\d+")
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-E])\b")
_QUOTED_RE = re.compile(r'"([^"]+)"')
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer\s*[:=]|the\s+answer\s+is)\s*:?\s*"
    r"([-+]?\d+(?:\.\d+)?(?:/\d+)?|[A-Ea-e]|yes|no|true|false)",
    re.IGNORECASE,
)


def extract_answer(raw_text: str, answer_type: str = "default") -> str:
    """Extract a canonical terminal answer from a raw response.

    Args:
        raw_text: The full raw response from the operator.
        answer_type: The expected answer type from the task:
            "numeric", "integer", "float", "fraction",
            "yes_no", "true_false", "letter", "string", "default"

    Returns:
        The extracted canonical answer string, stripped of whitespace.
        Returns the full stripped text if no pattern matches.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Always try \\boxed{} first — it is the most explicit signal
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    # Try "final answer: X" / "answer is X" patterns
    final_match = _FINAL_ANSWER_RE.findall(text)
    if final_match:
        return final_match[-1].strip()

    # Type-specific extraction
    at = answer_type.lower().strip()

    if at in ("numeric", "integer", "int"):
        nums = _INTEGER_RE.findall(text)
        if nums:
            return nums[-1]
        # Fallback: any number
        nums = _LAST_NUMBER_RE.findall(text)
        if nums:
            return nums[-1]

    elif at == "float":
        decimals = _DECIMAL_RE.findall(text)
        if decimals:
            return decimals[-1]
        nums = _LAST_NUMBER_RE.findall(text)
        if nums:
            return nums[-1]

    elif at == "fraction":
        fracs = _FRACTION_RE.findall(text)
        if fracs:
            return fracs[-1].replace(" ", "")
        # Maybe a decimal representation
        decimals = _DECIMAL_RE.findall(text)
        if decimals:
            return decimals[-1]

    elif at == "yes_no":
        yn = _YES_NO_RE.findall(text)
        if yn:
            return yn[-1].lower()

    elif at == "true_false":
        tf = _TRUE_FALSE_RE.findall(text)
        if tf:
            return tf[-1].lower()

    elif at == "letter":
        letters = _LETTER_RE.findall(text)
        if letters:
            return letters[-1].upper()

    elif at == "string":
        # Try quoted strings
        quoted = _QUOTED_RE.findall(text)
        if quoted:
            return quoted[-1].strip()
        # Try last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            return lines[-1]

    # Default fallback: try last number
    nums = _LAST_NUMBER_RE.findall(text)
    if nums:
        return nums[-1]

    # Last resort: last non-empty line, truncated
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1][:200]

    return text[:200] if text else ""
