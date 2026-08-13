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

I2 is preserved at Git tag `v2b-i2-development-baseline`:

```text
commit = ef23b7ee2982842f05f058cb502155ab7f385fdd
tree   = 006d6518260009237b75471c7a655357b06a7b5f
```

I3 is an additional experimental layer. It does not rewrite the V2A
qualification boundary or the V2B-I1/I2 trust stack.

The comparison is deliberately narrow:

| Condition | Controller | Cognitive-state input | Shared substrate |
| --- | --- | --- | --- |
| `STATE_BLIND_CONTROLLER` | `v2b_i3_matched_metareasoning_controller_v1` | Masked | Benchmark, action schema, budgets, policy gate, executor, utility weights |
| `STATE_AWARE_CONTROLLER` | Same controller and parameters | Bounded snapshot exposed | Same |

The state-blind condition is **not policy-free**. Both conditions remain
subject to the same hidden deterministic policy safety substrate. The isolated
independent variable is controller visibility of the bounded cognitive
snapshot.

The frozen mask artifact also defines `NO_VERIFICATION`, `NO_PROVENANCE`,
`NO_TEMPORAL`, `NO_CONFLICT`, and `NO_HISTORY` ablations. They reuse the exact
same controller, policy, budgets, action costs, and environment.

## Latent versus observable state

Every task has a latent environment state used only for transition dynamics,
terminal scoring, and the exact oracle. It lives in the private environment
artifact; controller packets are a separately stored, allow-listed artifact.
The terminal criterion is never passed to either controller. The state-aware controller instead sees
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

The private corpus is split into `development`, `validation`, and `held_out`
partitions. This small corpus is a protocol fixture, not the eventual
100–300/100/200+ qualification-scale benchmark; the held-out partition is
still executed and receipted separately so that future scale-up preserves the
same split boundary.

## Execution semantics

The loop records all three stages independently:

```text
proposed action → policy-resolved action → execution status → executed action|null
```

`REQUIRE` may substitute the explicitly required action. `DENY` records a
cost-free rejected proposal and returns control to the controller for a new
legal proposal; it never implicitly converts into `DEFER`.

At most three policy rejections are permitted per task. Crossing that bound
terminates the trajectory with `POLICY_REJECTION_LIMIT`; no rejected action is
misreported as executed.

`ANSWER`, `DEFER`, and `STOP` remain separate terminal semantics: answer
asserts a result, defer returns insufficient evidence, and stop terminates an
internal task without asserting an answer.

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
conditions and ablations' metrics to LitLogger. It also writes one JSONL
trajectory receipt per condition with observation/state hashes, resources,
actions, terminal result, utility, and regret. This does not create a
qualification claim.
