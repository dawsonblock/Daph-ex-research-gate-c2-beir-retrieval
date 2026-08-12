# BACKGROUND_VERIFICATION_V2A status

## Current disposition

```text
IMPLEMENTED          yes
CORRECTNESS TESTED   yes (focused V2A suite)
EXACT BUILD QUALIFIED no
PRODUCTION CLAIM     no

run_valid          = false
mechanism_status   = IMPLEMENTED_REQUIRES_COMMIT_BOUND_REQUALIFICATION
scientific_verdict = VOID_PENDING_REQUALIFICATION
```

The previous exact-build `QUALIFIED / PASS` declaration is void. Its four
qualification receipts were produced from different commits and trees, while
the old package builder checked only that each receipt contained
`run_valid=true`. Receipt hashes authenticated the files that were packaged;
they did not prove that those files qualified the packaged source.

The historical results remain useful engineering evidence, not qualification
evidence for the exact packaged build:

- 1M-event pressure/replay: commit `aa5aac06687241ab49276f74aed5a108b44c7114`
- 10-case adversarial run: commit `078923a8b072fb1a72101723154b29d73f9214e3`
- recorded live/offline network smoke: commit `b06f181e30d2159f4e379220a236b004e3749bf8`
- 1,659-pass full-suite receipt: commit `6cabd45129ccb3c76ed328132078ea1e0ceb847e`

The focused local V2A suite independently reproduced as 9 passing tests during
the archive audit. The stored full-suite result is historical because it was
not produced from the same source identity and its environment was not
reproducible from the former `dev` extra alone.

## Requalification contract

Every new pressure, adversarial, network-smoke, and full-suite receipt must
carry the same `BACKGROUND_VERIFICATION_V2A_QUALIFICATION_IDENTITY_V1`. That
identity binds:

- source commit and source tree;
- protocol, external-verification runtime, qualification validator, claim
  store, and verification-event hashes;
- test-corpus hash;
- exact qualification dependency-lock hash;
- Python/interpreter identity and installed dependency versions.

The package builder recomputes this identity from the claimed Git commit and
current qualification environment, requires all four receipts to match it,
checks gate-specific contents, and fails closed on missing or mixed identities.
It packages the exact common source commit and overlays only the validated
receipts plus the generated manifest. The environment can be installed with
`python -m pip install -e '.[qualification]'` or the frozen
`configs/v2a_qualification_requirements.lock`.

## Implemented capability and bounded language

V2A provides a CPU-only, offline-testable path from a canonical claim to a
durable queue, immutable external snapshot, deterministic exact-field decision,
and append-only `VERIFICATION` event. Disagreement stays `INCONCLUSIVE` and
evidence retraction changes current derived state without erasing history.

The implemented mechanism can preserve declared or deterministically assigned
source-lineage identifiers. It does not discover hidden syndication. Repeated
execution, crash-boundary recovery, and replay are idempotent; V2A does not yet
implement retry scheduling, backoff, or retry budgets. Generic HTTP capture is
an integration adapter, not a production authority decision or a production
network-security boundary.

V2A does not claim general truth verification, literature understanding,
semantic contradiction resolution, LLM fact checking, verification-aware
retrieval, answer-quality improvement, or production readiness.
