"""DAPH I3.4 — Epistemic Phase Ontology.

Frozen 5-phase ontology for phase-aware epistemic control.

PHASE 0 — EVIDENCE_ACQUISITION
    Evidence coverage is not yet sufficient to resolve the hypothesis set.
    Multiple hypotheses unresolved, retrieval/search still valuable.

PHASE 1 — DISCRIMINATION
    Multiple viable hypotheses remain and there is enough structure to test them.
    n_live >= 2, verification targets available.

PHASE 2 — RESOLUTION
    Narrowed to one plausible hypothesis but result not yet terminal.
    n_live == 1, SUPPORTED_BUT_UNRESOLVED.

PHASE 3 — ANSWER_READY
    Evidence supports a viable answer sufficiently for terminal response.
    READY_TO_ANSWER.

PHASE 4 — NO_VIABLE_HYPOTHESIS
    All hypotheses eliminated. Corresponds to T2.
    n_live == 0, all eliminated.

These phases are hypotheses for the value-learning stage, not hard-coded
rules. The deterministic classifier in classifier.py assigns phases
based on observable MDSG state, but the action-value estimator will
determine whether phase-conditioned action values actually differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    """Five primary epistemic phases."""

    EVIDENCE_ACQUISITION = "EVIDENCE_ACQUISITION"
    DISCRIMINATION = "DISCRIMINATION"
    RESOLUTION = "RESOLUTION"
    ANSWER_READY = "ANSWER_READY"
    NO_VIABLE_HYPOTHESIS = "NO_VIABLE_HYPOTHESIS"

    @property
    def numeric(self) -> int:
        return _PHASE_ORDER[self]

    @classmethod
    def from_numeric(cls, n: int) -> "Phase":
        return _NUMERIC_TO_PHASE[n]


_PHASE_ORDER = {
    Phase.EVIDENCE_ACQUISITION: 0,
    Phase.DISCRIMINATION: 1,
    Phase.RESOLUTION: 2,
    Phase.ANSWER_READY: 3,
    Phase.NO_VIABLE_HYPOTHESIS: 4,
}

_NUMERIC_TO_PHASE = {v: k for k, v in _PHASE_ORDER.items()}


# Expected transition graph (not strictly acyclic — retrieval can reopen).
# Used for transition consistency analysis, not for enforcement.
EXPECTED_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.EVIDENCE_ACQUISITION: frozenset({
        Phase.EVIDENCE_ACQUISITION,
        Phase.DISCRIMINATION,
        Phase.RESOLUTION,
        Phase.NO_VIABLE_HYPOTHESIS,
    }),
    Phase.DISCRIMINATION: frozenset({
        Phase.DISCRIMINATION,
        Phase.RESOLUTION,
        Phase.NO_VIABLE_HYPOTHESIS,
        Phase.EVIDENCE_ACQUISITION,  # reversal: new evidence may reopen
    }),
    Phase.RESOLUTION: frozenset({
        Phase.RESOLUTION,
        Phase.ANSWER_READY,
        Phase.NO_VIABLE_HYPOTHESIS,
        Phase.DISCRIMINATION,  # reversal: new evidence may reopen
        Phase.EVIDENCE_ACQUISITION,
    }),
    Phase.ANSWER_READY: frozenset({
        Phase.ANSWER_READY,
        Phase.RESOLUTION,  # reversal: verification may invalidate
        Phase.NO_VIABLE_HYPOTHESIS,
    }),
    Phase.NO_VIABLE_HYPOTHESIS: frozenset({
        Phase.NO_VIABLE_HYPOTHESIS,
        Phase.RESOLUTION,  # reversal: new evidence may introduce new hypothesis
        Phase.EVIDENCE_ACQUISITION,
    }),
}


def classify_transition(
    before: Phase,
    after: Phase,
) -> str:
    """Classify a phase transition as expected, allowed_reversal, or suspicious.

    The transition graph is NOT acyclic — retrieval can reopen possibilities.
    """
    if before == after:
        return "same_phase"

    if after in EXPECTED_TRANSITIONS.get(before, frozenset()):
        # Distinguish forward progress from reversal
        if after.numeric < before.numeric and before != Phase.NO_VIABLE_HYPOTHESIS:
            return "allowed_reversal"
        return "expected"

    return "suspicious"


@dataclass(frozen=True)
class EpistemicPhase:
    """Phase classification result.

    Attributes:
        phase: the assigned Phase
        confidence: classification confidence (1.0 for deterministic)
        evidence_basis: tuple of field names that determined the phase
        ambiguous: whether the classification is uncertain
    """

    phase: Phase
    confidence: float
    evidence_basis: tuple[str, ...]
    ambiguous: bool

    def as_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "confidence": self.confidence,
            "evidence_basis": list(self.evidence_basis),
            "ambiguous": self.ambiguous,
        }


# Phase-specific expected action value profiles (hypotheses, not rules).
# These describe what we EXPECT to find if the phase hypothesis is correct.
# The empirical analysis will test whether these hold.
EXPECTED_ACTION_VALUE_PROFILES: dict[Phase, dict[str, str]] = {
    Phase.EVIDENCE_ACQUISITION: {
        "RETRIEVE": "high",
        "SEARCH_MORE": "high",
        "REASON_MORE": "medium",
        "VERIFY": "context-dependent",
        "DEFER": "low",
        "ANSWER": "low",
        "STOP": "low",
    },
    Phase.DISCRIMINATION: {
        "VERIFY": "high",
        "RETRIEVE": "medium",
        "SEARCH_MORE": "medium",
        "REASON_MORE": "medium",
        "DEFER": "low",
        "ANSWER": "low",
        "STOP": "low",
    },
    Phase.RESOLUTION: {
        "VERIFY": "high",
        "REASON_MORE": "medium",
        "RETRIEVE": "medium",
        "SEARCH_MORE": "medium",
        "ANSWER": "context-dependent",
        "DEFER": "low",
        "STOP": "low",
    },
    Phase.ANSWER_READY: {
        "ANSWER": "high",
        "VERIFY": "low",
        "RETRIEVE": "low",
        "SEARCH_MORE": "low",
        "DEFER": "low",
        "STOP": "low",
        "REASON_MORE": "low",
    },
    Phase.NO_VIABLE_HYPOTHESIS: {
        "DEFER": "high",
        "VERIFY": "~0",
        "SEARCH_MORE": "context-dependent",
        "RETRIEVE": "context-dependent",
        "ANSWER": "low",
        "STOP": "medium",
        "REASON_MORE": "low",
    },
}


ALL_PHASES = tuple(Phase)
N_PHASES = len(ALL_PHASES)
