# V2B-I3.3.1 Benchmark Integrity Pass

V2B-I3.3.1 preserves the I3.3 seven-action environment and performs the final
benchmark-integrity pass after the I3.2.2 methodology freeze. It does not add a
model controller and is not a scientific V2B result.

The concrete corpus contains 750 immutable task instances: 300 development,
150 validation, 100 instance-held-out, 50 surface-held-out, and 150
structure-held-out. It covers verification, temporal validity,
provenance lineage, conflict, decision history, composition, irreducible
partial observability, state-irrelevant controls, and budget-conditioned
pairs. Structure-held-out exact semantic signatures are disjoint from
development; surface-held-out templates are also disjoint. Concrete JSON is
authoritative; the deterministic generator and operational seed
are provenance for the frozen bytes rather than an instruction to regenerate
held-out evaluation data.

Each task commits both coarse and exact semantic-structure identities derived
from latent state, budgets, policy-relevant state, deterministic transition
effects, terminal semantics, and the cognitive channel. IDs, surface text,
entities, split names, and intended labels are excluded. Generator intent is
non-authoritative: the exact latent oracle defines the tied or singleton
optimal-action set and the oracle-confirmed balance report must show complete
agreement before freeze.

All hash-bearing JSON uses one strict RFC-8259 writer (`allow_nan=False`). A
task with no successful terminal path records `successful_path_exists=false`
and `minimum_remaining_cost=null`; it never serializes `Infinity`. Fast tests
verify cache identities and representative recomputation, while exhaustive
750-task latent and seven-condition sequential regeneration remains a separate
qualification gate.

The benchmark uses the I3.2.2 protocol's seven actions and utility semantics.
Each controller condition changes only the observation mask. Oracle cache
records bind latent and sequential observable ground truth without enabling a
model-controller claim.

Status: **FROZEN BENCHMARK INTEGRITY EVIDENCE, NOT AN EXECUTIVE RESULT**.
