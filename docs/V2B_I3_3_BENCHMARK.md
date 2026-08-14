# V2B-I3.3.2 Scientific Split Hardening

V2B-I3.3.2 is the final benchmark-engineering milestone before a pinned model
executive. It preserves the I3.2.2 protocol, seven-action vocabulary, exact
latent oracle, and sequential information-state oracle. It adds no model,
critic, sub-agent, skill system, or new executive action.

The frozen corpus contains 750 immutable instances:

| Split | Tasks | Scientific role |
|---|---:|---|
| Development | 300 | Prompt/controller development |
| Validation | 150 | Pre-test selection and threshold calibration |
| Held-out instance | 100 | New instances of familiar control problems |
| Held-out surface | 50 | Unseen task-summary realizations |
| Held-out structure | 150 | Unseen executable metareasoning topology |

## Three structure identities

Each private task retains coarse and exact semantic identities for diagnostic
stratification. I3.3.2 adds a stronger `transition_topology_sha256` derived
from the reachable proposal, policy-resolution, transition-connectivity, and
terminal-result graph.

The topology identity deliberately excludes task IDs, split names, surface
text, entity names, generator indices, cognitive-channel labels, state labels,
and budget-profile names. Resource limits affect topology only through the
executable graph they produce.

The final isolation invariant is:

```text
T(HELD_OUT_STRUCTURE) ∩
  (T(DEVELOPMENT) ∪ T(VALIDATION)) = ∅
```

Instance-held-out may share topology with development by design. Surface-
held-out shares control semantics but uses a disjoint frozen template pool.
The three splits therefore support different, non-interchangeable claims.

## Real multistep programs

Validation and structural-held-out tasks use deterministic conditional action
effects to express staged programs. Examples include retrieval followed by
verification, verification followed by reasoning, search followed by
verification, and failed or misordered operations that poison the current
control path. The runtime and both oracles execute the same frozen conditional
effect semantics.

Tasks are characterized by minimum optimal trajectory depth, maximum relevant
depth, decision branch points, and policy interventions. The held-out
structure split includes depth-four-or-greater optimal trajectories rather
than relying on decoy metadata for novelty.

## Decision difficulty

Difficulty is evaluator-only and comes from the exact latent-oracle Q margin:

```text
normalized margin =
  (best Q − best non-tied alternative Q) /
  (correct-answer reward − incorrect-answer reward)
```

The frozen bands are:

- `HARD`: `0 < margin < 0.005`
- `MEDIUM`: `0.005 ≤ margin < 0.10`
- `EASY`: `margin ≥ 0.10`
- `TIE`: multiple exactly optimal actions

No difficulty label, Q value, topology identity, split role, latent state, or
oracle output appears in a controller packet.

## Artifact and qualification boundary

All hash-bearing JSON uses the shared strict RFC-8259 serializer with
`allow_nan=False`. The benchmark closure binds the private corpus, public
packets, split manifests, topology allocation, semantic/topology reports,
policy, utility, budgets, observation masks, latent oracle set, and all seven
sequential observable-oracle sets.

The identity is:

```text
DAPH_V2B_I3_3_2_SCIENTIFIC_SPLIT_IDENTITY_V1
```

Its claim boundary is intentionally narrow:

> **FROZEN SCIENTIFIC BENCHMARK; NO MODEL EXECUTIVE RESULT**

Exhaustive cache regeneration remains an explicit qualification test rather
than part of the fast unit suite. Once this identity is frozen, benchmark
mathematics and held-out data must not be tuned around model behavior.
