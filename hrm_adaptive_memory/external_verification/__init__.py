"""BACKGROUND_VERIFICATION_V2A: immutable external evidence, offline first.

This package intentionally depends only on the append-only memory-write API.
It does not import the certified reader, G2, selector, HRM, or executive code.
"""

from .core import (  # noqa: F401
    AcquisitionRequest, AcquisitionResult, AcquisitionStatus,
    DeterministicExactFieldVerifier, EvidenceStore, ExternalEvidenceRecord,
    ExternalVerificationWorker, LocalStructuredFixtureAcquirer, SourceLineage,
    OfficialTextAcquirer, SourceType, VerificationDecision, VerificationJob, VerificationQueue,
    derive_current_status, explain_claim,
)

