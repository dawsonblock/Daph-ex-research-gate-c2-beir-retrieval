# Semantica donor integration: bounded cognitive-control slice

## Status

```text
IMPLEMENTED        yes
UNIT TESTED        yes
INTEGRATED WITH EXECUTIVE no
SCIENTIFICALLY QUALIFIED no
PRODUCTION READY   no
```

This is post-qualification development, governed by
`DAPH_COGNITIVE_CONTROL_V2B_QUALIFICATION_IDENTITY_V1` in
[`configs/cognitive_control_v2b_design.json`](../configs/cognitive_control_v2b_design.json).
It is separate from the qualified V2A external-verification identity. The
current branch must not be represented as a byte-identical V2A-qualified
source tree merely because its V2A core still matches the V2A manifest.

The current V2B-infrastructure focused development check is **31 passed, 0
failed** across authority/comparator, peer-bound-network, checkpoint,
action/schema, adversarial, and cognitive-control tests. It is a local
development result, not a V2A replacement receipt or a V2B scientific
qualification.

The first V2B infrastructure milestone additionally introduces a lifecycle-bound
authority registry, relation-bound typed comparators, peer-bound HTTPS capture,
and externally trusted Ed25519 checkpoint contracts. Truth-bearing acquisition
requires a registry frozen for an experiment or qualification, validates every
redirect and final URI against its authority definition, executes only the
verified extractor module/symbol named by that definition, and records an
authority attestation. Typed verification rejects an authoritative-looking
record unless that attestation matches the currently frozen registry. A
checkpoint is verified only against an external signer/key registry; it cannot
trust a public key carried in its own payload. These are implementation
prerequisites only: no authority endpoint, typed relation, checkpoint,
controller, benchmark, or receipt is scientifically qualified by their
presence.

The donor archive `semantica-main 2.zip` has SHA-256
`54abd52728f488ca687a318ec8361d70250ff5b6019a80f69218a7296d652b1b` and is
MIT licensed by Hawksight AI. DAPH adapts only the selected concepts; it does
not vendor the Semantica package or its dependency surface.

## Implemented now

- Complete-payload, identity-bound, operation-bound, schema-bound,
  previous-event-hash-bound provenance events.
- Provenance attribution, derivation, version, activity, bundle, and
  invalidation fields.
- Bi-temporal facts with separate valid time and recorded/known time.
- Explicit unresolved conflict events over simultaneously applicable facts.
  No majority, recency, credibility, or confidence voting exists.
- Structured executive decision and outcome records with parent-decision
  causal ancestry.
- A finite positive-Datalog engine and deterministic policy gate with
  `ALLOW`, `DENY`, and `REQUIRE` effects. Incompatible required actions deny
  with `POLICY_CONFLICT`; compatible requirements coalesce. Rules are
  validated at gate construction, so an unknown action or malformed rule never
  becomes a runtime action-selection failure.
- An append-only canonical event log; all graphs and indexes are replayed
  materialized views. Cognitive-control V2 stores timezone-aware timestamps in
  canonical UTC RFC3339 form and rejects lifecycle events that predate their
  causes.

The timestamp and outcome-lifecycle hardening changes the cognitive event
schema from `DAPH_COGNITIVE_CONTROL_V1` to
`DAPH_COGNITIVE_CONTROL_V2`. V1 event logs are rejected rather than silently
rewritten, because an automatic rewrite could invent missing timestamps or
alter historical event hashes. This is acceptable while the subsystem is
explicitly development-only; any future migration must be separate,
deterministic, and evidence-preserving.

## Deliberately untouched

- C2/C4 retrieval, dense search, RRF, selectors, and certified evidence.
- V2A's external-verification result semantics.
- Semantica ingestion, parsing, connectors, REST/MCP services,
  visualization, vector stores, Node2Vec, and automatic conflict resolution.
- Learned executive routing and verification-aware retrieval.

This slice establishes storage and deterministic control primitives. It does
not show that using them improves answers or executive decisions. That needs
the separate frozen V2B protocol, environment-bound receipts, an independently
verified package, and a restricted initial action space before any learned
executive is introduced.
