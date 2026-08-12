# BACKGROUND_VERIFICATION_V2A status

## Implemented capability

V2A now provides a CPU-only, offline-testable path from a canonical claim to
a durable queue, immutable external snapshot, deterministic exact-field
decision, and append-only `VERIFICATION` event. External evidence is stored
outside claims, is content/provenance-addressed, and is resolved before an
external verification event is appended.

Disagreement stays `INCONCLUSIVE`; neither recency, confidence, source count,
nor source class chooses a winner. Evidence retraction is append-only and
removes the snapshot from the V2A *current* view while retaining the original
evidence/event for historical audit. The V1 substrate was additionally
hardened to authenticate a manifest before replay, recompute record and event
identities, reject unknown events, require structural supersession compatibility,
and exclude withdrawn in-store evidence from present-tense status derivation.

## Verification performed

- `tests/unit/test_background_verification_v2a.py` covers V2A-T1 through
  V2A-T16 using only local fixture snapshots.
- The focused V2A + V1 verification/write + integrity regression command
  passed: **92 passed**.
- The complete suite passed: **1,659 passed, 3 skipped**.
- `CERTIFIED_MEMORY_V1` was pinned and its identity remained
  `97b3b49cb95fa33b` in the V2A invariance test.

## Not a V2A promotion certificate yet

The V2A implementation is deliberately not labeled
`QUALIFIED_FOR_EXTERNAL_EVIDENCE_V2A`. The 1k-to-1m external-evidence
pressure characterization, a separately recorded network-integration smoke,
and a package-level build manifest binding archive hash to the committed source
and protocol hashes remain required before qualification.

V2A does not claim general truth verification, literature understanding,
semantic contradiction resolution, LLM fact checking, verification-aware
retrieval, or answer-quality improvement. It proves only immutable external
evidence capture and deterministic comparison for the narrowly authorized
structured-source classes.
