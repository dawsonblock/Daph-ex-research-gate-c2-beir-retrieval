# V2B-I3: Metareasoning Experimental Validity

## Status

```text
IMPLEMENTED                 yes
DEVELOPMENT PROTOCOL        yes
LEARNED EXECUTIVE           no
SCIENTIFICALLY QUALIFIED    no
PRODUCTION READY            no
```

V2B-I3 replaces neither V2A nor the historical V2B-I2 harness. It is a
development-only deterministic metareasoning environment that removes I2's
experimental-design confounds before any model controller is introduced.

The comparison is deliberately narrow:

| Condition | Controller | Cognitive-state input | Shared substrate |
| --- | --- | --- | --- |
| `STATE_BLIND_CONTROLLER` | `v2b_i3_matched_metareasoning_controller_v1` | Masked | Benchmark, action schema, budgets, policy gate, executor, utility weights |
| `STATE_AWARE_CONTROLLER` | Same controller and parameters | Bounded snapshot exposed | Same |

The state-blind condition is **not policy-free**. Both conditions remain
subject to the same hidden deterministic policy safety substrate. The isolated
independent variable is controller visibility of the bounded cognitive
snapshot.

## Latent versus observable state

Every task has a latent environment state used only for transition dynamics,
terminal scoring, and the exact oracle. It includes the terminal criterion and
is never passed to either controller. The state-aware controller instead sees
only a bounded projection: verification status, temporal status, provenance
count, unresolved conflict, composition progress, prior decisions/outcomes,
and remaining resources. It does not receive `expected_terminal`, transition
maps, utility weights, oracle output, `reasoning_required`, or an
`evidence_sufficient` oracle label.

The benchmark uses `TIGHT`, `STANDARD`, and `GENEROUS` resource profiles plus
ambiguous task pairs with the same task summary. One pair also has identical
initial observable cognitive state but different hidden action transitions, so
neither controller can infer the better evidence action until it observes an
action result. Resource state is therefore a causal input to the optimal
action rather than passive bookkeeping.

## Execution semantics

The loop records all three stages independently:

```text
proposed action → policy-resolved action → execution status → executed action|null
```

`REQUIRE` may substitute the explicitly required action. `DENY` records a
cost-free rejected proposal and returns control to the controller for a new
legal proposal; it never implicitly converts into `DEFER`.

Every executed action records pre/post state hashes and semantic deltas. Tool
usefulness is credited only when that action produces a decision-relevant
state improvement, not merely because the eventual task succeeds.

## Oracle and metrics

The finite deterministic environment has an exact dynamic-programming oracle
under the same action costs, policy constraints, and task-specific budget.
The oracle is evaluation-only. It provides optimal utility and action/trajectory
regret, not controller input.

Reported metrics include task success, unsupported assertions, correct
deferrals, policy/resource rejection counts, real premature-stop and
failure-to-stop rates, state-delta tool usefulness, action regret, trajectory
regret, and normalized executive regret:

```text
sum(oracle_utility - realized_utility) / (sum(abs(oracle_utility)) + 1e-9)
```

Run `scripts/run_v2b_i3_development.py` only from a clean committed checkout.
It writes a development-only receipt and logs the receipt identity plus both
conditions' metrics to LitLogger. This does not create a qualification claim.
