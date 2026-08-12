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

## Qualification result

```text
BACKGROUND_VERIFICATION_V2A
run_valid          = true
mechanism_status   = QUALIFIED
scientific_verdict = PASS
```

The qualification gates are recorded and hashed:

- External-evidence pressure ladder:
  `evidence/background_verification_v2a/pressure/pressure_full_1m.json`.
  All 1k, 10k, 50k, 100k, 250k, 500k, and 1M checkpoints reproduced the
  incremental claim-verification and evidence state hashes from genesis.
  At 1M, evidence append throughput was 2,874/s, verification-event append
  throughput was 3,350/s, and cold replay completed in 56.87 seconds.
- Permanent adversarial evidence:
  `evidence/background_verification_v2a/adversarial/adversarial.json`.
  All 10 frozen attack/recovery cases passed.
- Recorded network integration smoke:
  `evidence/background_verification_v2a/network_smoke/network_smoke.json`.
  Two stable World Bank records completed live capture and deterministic
  verification, then reproduced offline from captured bytes with zero network
  calls. Both records share one publisher, and no independent-publisher
  corroboration claim is made.
- Complete repository suite:
  `evidence/background_verification_v2a/tests/full_suite.json`:
  **1,659 passed, 3 skipped, 0 failed**.

The root `BUILD_MANIFEST.json` and its packaged copy bind the exact source
commit/tree, protocol and schema hashes, C4 H100 lock, frozen
`CERTIFIED_MEMORY_V1` identity, and the four qualification receipt hashes.

## Bounded claim

DAPH can acquire immutable external evidence, preserve source lineage,
deterministically verify supported exact-field claim classes, survive
retries/replay/tampering, and maintain auditable current verification state.

V2A does not claim general truth verification, literature understanding,
semantic contradiction resolution, LLM fact checking, verification-aware
retrieval, or answer-quality improvement. It proves only immutable external
evidence capture and deterministic comparison for the narrowly authorized
structured-source classes.
