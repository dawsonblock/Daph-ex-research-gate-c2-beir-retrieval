# Semantica donor integration: bounded cognitive-control slice

## Status

```text
IMPLEMENTED        yes
UNIT TESTED        yes
INTEGRATED WITH EXECUTIVE no
SCIENTIFICALLY QUALIFIED no
PRODUCTION READY   no
```

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
  `ALLOW`, `DENY`, and `REQUIRE` effects.
- An append-only canonical event log; all graphs and indexes are replayed
  materialized views.

## Deliberately untouched

- C2/C4 retrieval, dense search, RRF, selectors, and certified evidence.
- V2A's external-verification result semantics.
- Semantica ingestion, parsing, connectors, REST/MCP services,
  visualization, vector stores, Node2Vec, and automatic conflict resolution.
- Learned executive routing and verification-aware retrieval.

This slice establishes storage and deterministic control primitives. It does
not show that using them improves answers or executive decisions. That needs a
separate frozen integration protocol and experimental receipts after V2A's
commit-bound requalification is repaired.
