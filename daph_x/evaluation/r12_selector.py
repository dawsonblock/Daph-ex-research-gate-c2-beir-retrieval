"""Canonical R12 MaxCal/majority selector.

R13 uses this exact selector so that baseline comparisons are stable.
Historical R12 outputs are used as golden tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from collections import Counter

from daph_x.operators.types import Candidate


@dataclass(frozen=True)
class Selection:
    answer: str
    confidence: float
    support_count: int
    n_candidates: int


def select_r12_maxcal(candidates: Sequence[Candidate]) -> Selection:
    """Canonical R12 MaxCal selector = majority vote with agreement confidence."""
    if not candidates:
        return Selection(answer="", confidence=0.0, support_count=0, n_candidates=0)

    answers = [c.answer for c in candidates]
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0]
    answer = most_common[0]
    support_count = most_common[1]
    n_candidates = len(candidates)
    confidence = support_count / n_candidates

    return Selection(
        answer=answer,
        confidence=confidence,
        support_count=support_count,
        n_candidates=n_candidates,
    )
