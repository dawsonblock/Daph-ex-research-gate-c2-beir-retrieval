"""C4 metrics — one authoritative source for all scoring formulas.

No other module should independently recreate quality, gap capture, or
related metric logic. Everything lives here so that:

- quality is always the protocol-defined partial-credit rule
- oracle-gap capture always uses (C4_4 - C4_0) / (C4_6 - C4_0)
- selector-gap capture always uses (C4_4 - C4_3) / (C4_5 - C4_3)
- tie-breaking is always explicit and documented

This module is frozen under C4 protocol v2.  Any change requires a new
protocol version.
"""
from __future__ import annotations

from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# 1.1  Quality — the protocol-defined partial-credit rule
# ---------------------------------------------------------------------------

def compute_quality(*, correct: bool, evidence_complete: bool) -> float:
    """Authoritative quality score.

    +-------------------+-----------+----------+
    | evidence_complete | correct   | quality  |
    +-------------------+-----------+----------+
    | True              | True      | 1.00     |
    | True              | False     | 0.50     |
    | False             | True      | 0.25     |
    | False             | False     | 0.00     |
    +-------------------+-----------+----------+

    No other module should independently recreate this logic.
    """
    if evidence_complete and correct:
        return 1.0
    if evidence_complete and not correct:
        return 0.5
    if not evidence_complete and correct:
        return 0.25
    return 0.0


def evidence_complete(required: Sequence[str], selected: Sequence[str]) -> bool:
    """True if every required evidence ID is in the selected set."""
    return set(required).issubset(set(selected))


# ---------------------------------------------------------------------------
# 1.2  Oracle-gap capture
# ---------------------------------------------------------------------------

def oracle_gap_capture(
    qualities: Mapping[str, float],
) -> float | None:
    """OGC = (Q(C4_4) - Q(C4_0)) / (Q(C4_6) - Q(C4_0)).

    Measures how much of the total oracle gap the real pipeline (C4_4)
    closes relative to the full oracle ceiling (C4_6).

    The numerator uses C4_4 (the actual mechanism), NOT C4_5 (oracle
    selector).  C4_5 is the selector-only ceiling; C4_6 is the full
    oracle ceiling.  The oracle gap is defined against C4_6.
    """
    q0 = qualities.get("C4_0")
    q4 = qualities.get("C4_4")
    q6 = qualities.get("C4_6")
    if q0 is None or q4 is None or q6 is None:
        return None
    gap = q6 - q0
    if abs(gap) < 1e-12:
        return None
    return (q4 - q0) / gap


# ---------------------------------------------------------------------------
# 1.3  Selector-gap capture
# ---------------------------------------------------------------------------

def selector_gap_capture(
    qualities: Mapping[str, float],
) -> float | None:
    """SGC = (Q(C4_4) - Q(C4_3)) / (Q(C4_5) - Q(C4_3)).

    Measures how much of the selector gap S2c closes relative to the
    oracle selector ceiling (C4_5).
    """
    q3 = qualities.get("C4_3")
    q4 = qualities.get("C4_4")
    q5 = qualities.get("C4_5")
    if q3 is None or q4 is None or q5 is None:
        return None
    gap = q5 - q3
    if abs(gap) < 1e-12:
        return None
    return (q4 - q3) / gap


# ---------------------------------------------------------------------------
# 1.4  Arm quality (mean quality over receipts)
# ---------------------------------------------------------------------------

def arm_quality(qualities: Sequence[float]) -> float:
    """Mean quality across a list of per-task quality scores."""
    return sum(qualities) / len(qualities) if qualities else 0.0
