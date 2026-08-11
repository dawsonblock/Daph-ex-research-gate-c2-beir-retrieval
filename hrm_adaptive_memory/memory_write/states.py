"""Shared vocabulary for the verified-memory write layer.

Extracted into its own module so claim_store (which appends events) and
consolidation (which derives state from them) can both depend on the same
definitions without importing each other -- these two genuinely are peers,
and a cycle between them would be a design smell rather than an accident to
work around.
"""
from __future__ import annotations

from enum import Enum


class VerificationState(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


class ConflictOutcome(str, Enum):
    NOVEL = "NOVEL"
    DUPLICATE = "DUPLICATE"
    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"
    SUPERSESSION = "SUPERSESSION"


#: States whose records the READER must not see. Retraction and supersession
#: change RETRIEVABILITY, never history -- the records remain in the log.
NON_RETRIEVABLE_STATES = frozenset({VerificationState.RETRACTED, VerificationState.SUPERSEDED})
