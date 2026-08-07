"""Shared answer verifiers.

Single source of truth for answer verification across all gates. Both the
historical V4 context study and the C4 integrated pipeline must delegate here
so that verifier semantics cannot drift between experiments.

The key normalization step is stripping HRM control tokens of the form
``<|...|>`` (e.g. ``<|box_end|>``) *before* extracting word tokens. Failing to
do so turns ``THETA-OLIVE<|box_end|>`` into ``theta olive box_end`` instead of
``theta olive``, which systematically marks correct canonical answers wrong.
"""

from __future__ import annotations

import math
import re

__all__ = ["normalize_answer", "verify_answer"]

_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


def normalize_answer(value: str) -> str:
    """Lowercase, strip HRM control tokens, and collapse whitespace."""
    stripped = _CONTROL_TOKEN_RE.sub(" ", value.lower().strip())
    return " ".join(stripped.split())


def verify_answer(verifier: str, answer: str, output: str) -> tuple[float, bool]:
    """Verify ``output`` against ``answer`` under the named verifier type.

    Parameters
    ----------
    verifier:
        One of ``"exact"``, ``"numeric"``, ``"canonical"``. Any unknown value
        falls back to exact matching (mirrors the historical fallback).
    answer:
        The gold answer string.
    output:
        The raw model output string (may contain control tokens).
    """
    if verifier == "exact":
        passed = normalize_answer(output) == normalize_answer(answer)
    elif verifier == "numeric":
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", output)
        passed = bool(numbers) and math.isclose(
            float(numbers[-1]), float(answer), rel_tol=1e-9, abs_tol=1e-9
        )
    elif verifier == "canonical":
        # Non-numeric answers (symbolic labels, enums, booleans, entity names,
        # small JSON fields). Mirrors the numeric verifier's semantics: the
        # answer must be the last candidate the output commits to, so an output
        # that lists several candidates does not pass on the strength of one.
        answer_terms = tuple(re.findall(r"\w+", normalize_answer(answer)))
        output_terms = tuple(re.findall(r"\w+", normalize_answer(output)))
        width = len(answer_terms)
        starts = [
            index for index in range(len(output_terms) - width + 1)
            if output_terms[index:index + width] == answer_terms
        ] if width else []
        passed = bool(starts) and starts[-1] + width == len(output_terms)
    else:
        # Fallback: exact semantics for unknown verifier types.
        passed = normalize_answer(output) == normalize_answer(answer)
    return float(passed), passed
