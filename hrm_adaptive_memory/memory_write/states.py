"""Shared vocabulary for the verified-memory write layer.

Extracted into its own module so claim_store (which appends events) and
consolidation (which derives state from them) can both depend on the same
definitions without importing each other -- these two genuinely are peers,
and a cycle between them would be a design smell rather than an accident to
work around.

TWO ORTHOGONAL AXES, split per BACKGROUND_VERIFICATION_V1 acceptance
criterion V6 (configs/background_verification_v1_design.json).

The original single ``VerificationState`` enum conflated them, which made a
perfectly reachable state inexpressible: a claim can be both SUPPORTED by
evidence and SUPERSEDED by a newer claim. It also meant ingest wrote a
verification opinion, which V1 forbids.

    LifecycleState        does the READER see this record?
                          written by ingest / supersede / retract
    VerificationStatus    what does the evidence say about it?
                          NEVER written directly -- derived purely from
                          appended verification events
"""
from __future__ import annotations

from enum import Enum


class LifecycleState(str, Enum):
    """Controls RETRIEVABILITY. Set by ingest/supersede/retract only."""
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


class VerificationStatus(str, Enum):
    """DERIVED from verification events. Never stored on a claim record --
    storing it would make it mutable state rather than a replay conclusion,
    which is exactly what the core rule forbids."""
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    FALSIFIED = "FALSIFIED"
    #: Verifiers disagree with each other. Deliberately NOT collapsed into
    #: CONTRADICTED: "the evidence contradicts the claim" and "the verifiers
    #: disagree" are different facts, and merging them destroys the second.
    INCONCLUSIVE = "INCONCLUSIVE"


class ConflictOutcome(str, Enum):
    """What ingest OBSERVED about a new claim's relationship to existing
    memory. Descriptive only -- since V1, this no longer sets any status."""
    NOVEL = "NOVEL"
    DUPLICATE = "DUPLICATE"
    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"
    SUPERSESSION = "SUPERSESSION"


#: Lifecycle states whose records the READER must not see. Retraction and
#: supersession change RETRIEVABILITY, never history -- the records remain in
#: the log.
NON_RETRIEVABLE_STATES = frozenset({LifecycleState.RETRACTED, LifecycleState.SUPERSEDED})

#: Backward-compatible read mapping for logs written BEFORE the axis split.
#: Those logs stored a single ``verification_state``; UNVERIFIED/SUPPORTED/
#: CONTRADICTED were all retrievable, so all three map to ACTIVE. This keeps
#: pre-split logs replayable and authoritative (A6), and is what makes the
#: 1M-event state hashes verifiable as unchanged across the migration.
_LEGACY_STATE_TO_LIFECYCLE = {
    "UNVERIFIED": LifecycleState.ACTIVE,
    "SUPPORTED": LifecycleState.ACTIVE,
    "CONTRADICTED": LifecycleState.ACTIVE,
    "SUPERSEDED": LifecycleState.SUPERSEDED,
    "RETRACTED": LifecycleState.RETRACTED,
}


def lifecycle_from_event(value: str) -> LifecycleState:
    """Read a lifecycle state from either the new or the legacy encoding."""
    try:
        return LifecycleState(value)
    except ValueError:
        try:
            return _LEGACY_STATE_TO_LIFECYCLE[value]
        except KeyError:
            raise ValueError(f"unrecognized lifecycle/legacy state {value!r}") from None
