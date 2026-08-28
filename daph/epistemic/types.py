"""Typed structures for canonical epistemic topology.

Implements the data structures defined in EPISTEMIC_SEMANTICS_V1.md §5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class HypothesisState(str, Enum):
    """Canonical hypothesis classification states (EPISTEMIC_SEMANTICS_V1 §4)."""
    SUPPORTED = "SUPPORTED"        # Has SUFFICIENT support, no SUFFICIENT contradiction
    CONTRADICTED = "CONTRADICTED"  # Has SUFFICIENT contradiction (priority over support)
    WEAKENED = "WEAKENED"          # Has FALSIFIED support only, no SUFFICIENT support/contradiction
    UNTESTED = "UNTESTED"          # No verified evidence relating to it
    STALE = "STALE"               # All previously sufficient evidence is now STALE


class TerminalReadiness(str, Enum):
    """Canonical terminal readiness states (EPISTEMIC_SEMANTICS_V1 §6)."""
    ANSWER_READY = "ANSWER_READY"
    DEFER_READY = "DEFER_READY"
    CONTINUE_REQUIRED = "CONTINUE_REQUIRED"


@dataclass(frozen=True)
class HypothesisTopology:
    """Canonical hypothesis topology derived from observable evidence.

    This is the single structure that all consumers must use.
    Computed by derive_hypothesis_topology() from observable evidence only.

    Fields per EPISTEMIC_SEMANTICS_V1 §5.2.
    """
    # Per-hypothesis classification
    hypothesis_states: Mapping[str, HypothesisState]

    # Aggregate counts
    n_viable_hypotheses: int          # count of SUPPORTED
    n_eliminated_hypotheses: int      # count of CONTRADICTED
    n_untested_hypotheses: int        # count of UNTESTED
    n_weakened_hypotheses: int        # count of WEAKENED
    n_stale_hypotheses: int           # count of STALE
    n_total_hypotheses: int

    # Verified evidence topology (SUFFICIENT only, per §3.2)
    n_hyp_with_verified_support: int       # count with >=1 SUFFICIENT support
    n_hyp_with_verified_contradiction: int # count with >=1 SUFFICIENT contradiction
    n_hyp_with_mixed_verified: int         # count with both SUFFICIENT support and contradiction

    # Resolution state
    unique_supported_hypothesis: str | None  # the single SUPPORTED hypothesis ID, or None
    has_verified_unresolved_competition: bool # True if >1 hypothesis has SUFFICIENT support
    has_unique_verified_supported: bool       # True if exactly 1 hypothesis has SUFFICIENT support

    # Evidence completeness
    verification_complete: bool          # all visible evidence is SUFFICIENT or FALSIFIED
    unverified_evidence_exists: bool     # any visible evidence is UNVERIFIED
    hidden_evidence_count: int           # count of non-retrieved evidence items

    # Per-hypothesis detail (for consumers that need it)
    verified_support_by_hypothesis: Mapping[str, tuple[str, ...]]       # SUFFICIENT support evidence IDs
    verified_contradiction_by_hypothesis: Mapping[str, tuple[str, ...]]  # SUFFICIENT contradiction evidence IDs
    falsified_support_by_hypothesis: Mapping[str, tuple[str, ...]]       # FALSIFIED support evidence IDs
    falsified_contradiction_by_hypothesis: Mapping[str, tuple[str, ...]] # FALSIFIED contradiction evidence IDs
    unverified_support_by_hypothesis: Mapping[str, tuple[str, ...]]
    unverified_contradiction_by_hypothesis: Mapping[str, tuple[str, ...]]

    def as_dict(self) -> dict:
        return {
            "hypothesis_states": {k: v.value for k, v in self.hypothesis_states.items()},
            "n_viable_hypotheses": self.n_viable_hypotheses,
            "n_eliminated_hypotheses": self.n_eliminated_hypotheses,
            "n_untested_hypotheses": self.n_untested_hypotheses,
            "n_weakened_hypotheses": self.n_weakened_hypotheses,
            "n_stale_hypotheses": self.n_stale_hypotheses,
            "n_total_hypotheses": self.n_total_hypotheses,
            "n_hyp_with_verified_support": self.n_hyp_with_verified_support,
            "n_hyp_with_verified_contradiction": self.n_hyp_with_verified_contradiction,
            "n_hyp_with_mixed_verified": self.n_hyp_with_mixed_verified,
            "unique_supported_hypothesis": self.unique_supported_hypothesis,
            "has_verified_unresolved_competition": self.has_verified_unresolved_competition,
            "has_unique_verified_supported": self.has_unique_verified_supported,
            "verification_complete": self.verification_complete,
            "unverified_evidence_exists": self.unverified_evidence_exists,
            "hidden_evidence_count": self.hidden_evidence_count,
            "verified_support_by_hypothesis": dict(self.verified_support_by_hypothesis),
            "verified_contradiction_by_hypothesis": dict(self.verified_contradiction_by_hypothesis),
            "falsified_support_by_hypothesis": dict(self.falsified_support_by_hypothesis),
            "falsified_contradiction_by_hypothesis": dict(self.falsified_contradiction_by_hypothesis),
            "unverified_support_by_hypothesis": dict(self.unverified_support_by_hypothesis),
            "unverified_contradiction_by_hypothesis": dict(self.unverified_contradiction_by_hypothesis),
        }
