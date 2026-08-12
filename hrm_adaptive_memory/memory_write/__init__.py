"""VERIFIED_MEMORY_WRITE_V1 -- the living-memory write layer.

Per configs/verified_memory_write_v1_design.json. This package owns corpus
STATE: ingestion, canonicalization, conflict detection, provenance,
verification state, and persistence.

It does NOT own retrieval or reasoning. CERTIFIED_MEMORY_V1 remains the
frozen, unmodified reader over whatever corpus state exists at query time.
Nothing in this package imports from or mutates the certified reasoning
path; it only READS the certified entity extractor in order to verify that
what it writes is natively parseable by the reader.
"""
from .claim_store import (  # noqa: F401
    ClaimRecord, ClaimStore, ConflictOutcome, IngestResult,
    IntegrityError, NotNativelyParseableError)
from .states import (  # noqa: F401
    LifecycleState, VerificationStatus)
