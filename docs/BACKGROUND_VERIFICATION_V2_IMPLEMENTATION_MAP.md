# BACKGROUND_VERIFICATION_V2A implementation map

## Pre-change anchors

The originally audited downloaded checkout had no `.git` directory. Its
supplied matching remote branch resolved read-only via `git ls-remote` to
`fc4a454f67d79163b95867e54dd1443401169e71`; that value is frozen in the V2A
design as a remote-branch anchor. The repository default HEAD at lookup was
`dd3f0fd9fc68fc769868919a9fd2f93dbab53d83`.

| Anchor | Value |
| --- | --- |
| `CERTIFIED_MEMORY_V1` live identity | `97b3b49cb95fa33b` |
| remote branch `c4-v2_1-prompt-order-conformance` | `fc4a454f67d79163b95867e54dd1443401169e71` |
| `configs/certified_memory_v1.json` SHA-256 | `8b714fac36f0c044ff29d064bfd6dc48354d9c80d4e465c2407faaaa534bbc18` |
| V1 background design SHA-256 | `b7ef0706a58c5d3f8d829417f02a15ab17729329f129ef14a20fee959d38827b` |
| persistent-memory baseline SHA-256 | `8b838b578299954d6338e5c592c7043d9d01334e188f0d51f5329a336291f9b9` |
| baseline 100k / 500k / 1m state hashes | `f2a8ff09…`, `bea65499…`, `a1b15359…` |
| full-suite collection before V2A | 1649 tests collected |

The pre-change full-suite command was `python -m pytest -q` with experiment
tracking disabled. V2A uses CPU-only offline fixtures; it does not require the
network or GPU.

## Exact reuse boundaries

| Concern | Existing reusable code | V2A decision |
| --- | --- | --- |
| Claim model | `memory_write.claim_store.ClaimRecord` | Claims are inputs only; never changed. |
| Lifecycle/status vocabulary | `memory_write.states.LifecycleState`, `VerificationStatus` | V2A emits existing verification results and retains V1 derivation. |
| Verification-event log | `ClaimStore.append_verification`, `VerificationEvent`, `derive_status` | Extend event provenance fields compatibly; do not alter V1 derivation. |
| Replay/idempotency | `ClaimStore._stream_events`, `_apply_event`, `_verification_ids` | Replay fails closed on unknown/tampered events; retry-safe jobs use deterministic identities. |
| Claim durability | `ClaimStore._append`, `_atomic_write`, `publish_snapshot` | Evidence gets a parallel fsync/atomic-snapshot store. |
| State hashing | `ConsolidatedState.state_hash`, `ClaimStore.log_sha256` | Evidence state uses the same canonical JSON/SHA-256 convention. |
| V1 worker | `DeterministicMemoryConsistencyChecker` | Hardened to require different source IDs for corroboration and to avoid requeueing current claims; V2A worker remains separate. |
| Certified-reader pin | `experiment_integrity.certified_memory.assert_certified_memory_v1_unchanged` | Called by V2A invariance tests; no external module imports reader/G2 code. |
| Test style | `tests/unit/test_background_verification.py` | New V2A acceptance tests use local snapshots only. |

## New V2A boundary

`hrm_adaptive_memory.external_verification` owns immutable captured evidence,
source taxonomy/lineage, typed acquisition, deterministic exact-field checks,
and a durable queue. It may depend on `memory_write` for the append-only claim
and verification-event APIs. It must not import C4, HRM, G2, selectors, or
executive-policy modules.

The authoritative external evidence remains separate from claims. A URL is
only acquisition metadata; raw captured bytes and hashes are the evidence.
