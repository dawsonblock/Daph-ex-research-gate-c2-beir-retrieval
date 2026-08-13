# V2B-I3.1: Oracle Efficiency and Information-Bound Correctness

## Status

```text
IMPLEMENTED                 yes
DEVELOPMENT METHODOLOGY     yes
SCIENTIFICALLY QUALIFIED    no
LEARNED EXECUTIVE           no
PRODUCTION READY            no
```

I3.1 is a separate development milestone layered on top of the immutable I3
baseline.  It does not amend V2A, V2B-I1, V2B-I2, or the historical I3 result.

```text
I3 baseline commit = b2febc37a0dac167772b49d7f439abd36e6ba1c0
I3 baseline tree   = afd21ab275c68508c15f2ad33999d125c62ec4eb
I3 baseline tag    = v2b-i3-development-baseline
```

The recorded baseline hashes live in
`configs/v2b_i3_1_baseline.json`.

## What changes in I3.1

I3.1 introduces no action and no new cognitive subsystem.  The action space
remains fixed:

```text
ANSWER  RETRIEVE  VERIFY  SEARCH_MORE  REASON_MORE  DEFER  STOP
```

It separates the rich runtime audit state from the minimal Markov oracle state:

```text
RuntimeState (audit/resource instrumentation)
                 │ canonicalize
                 ▼
OracleState (only transition/utility-relevant fields)
                 │ forward reachability
                 ▼
OraclePolicyTable (one table per latent task + budget)
```

`elapsed_ms` is not an oracle-key dimension because under the current frozen
action table it is derivable from the remaining consumable resources; monetary
cost is omitted because it is identically zero.  Every nonterminal transition
strictly consumes an executive step, and table construction rejects a
zero-cost control cycle or state-space limit breach.

The table is built once and cached across all observation masks.  Action regret
is then a direct dictionary lookup rather than another recursive oracle solve.
Tables serialize with their task, budget, policy, utility, action-cost, and
implementation identities plus state/transition complexity measurements.

## Information-bound evaluation

The latent oracle is an environmental upper bound, not a fair direct target
for a partially informed controller.  I3.1 also constructs one observable
oracle per frozen observation mask.  It groups identical **opening** controller
packets and, with a frozen uniform prior over the resulting latent states,
chooses the shared proposal with maximum expected latent Q-value.

```text
latent optimum V_L*
        │
        │ information lost by observation mask
        ▼
observable optimum V_O*
        │
        │ controller decision error
        ▼
controller trajectory utility V^π
```

The receipt reports:

```text
InformationGap = V_L* - V_O*
DecisionGap    = V_O* - V^π
TotalRegret    = V_L* - V^π
```

and normalized information/decision/total versions.  `InformationGap` is a
property of the representation; `DecisionGap` is the controller's loss given
the information it was allowed to see.  This prevents hidden-state uncertainty
from being mislabeled as poor executive decision-making.

## Privacy and replay boundaries

I3.1 uses an opaque public `instance_id` in controller-visible packets.  The
private environment task ID, latent terminal label, transition map, utility,
and oracle output never enter the controller observation.  The intentionally
aliased hidden-transition pair shares the same public instance ID and packet.
Leakage tests reject oracle/latent labels and assert that a private task ID
cannot appear in the controller packet.

Each trajectory receipt binds the latent table hash, observable table hash,
mask, controller revision, policy, utility, and budget.  Replay re-executes
recorded actions against the deterministic environment and requires exact
state/resource/utility parity.

## Running the development protocol

From a clean committed checkout:

```bash
python scripts/run_v2b_i3_1_development.py \
  --output /path/to/empty-output \
  --run-id v2b-i3-1-development
```

The runner serializes latent and observable oracle tables, trajectory receipts,
and an aggregate development-only receipt.  It also logs every run to
LitLogger.  None of these artifacts represents a scientific V2B claim.

The future qualification identity is
`DAPH_V2B_I3_1_ORACLE_EFFICIENCY_IDENTITY_V1`; it fails closed on the current
development configuration.  I3.1 exit requires deterministic table/replay and
runtime-policy-cost parity, no packet leakage, bounded state-space construction,
and budget-sensitive coverage before corpus scaling or any model controller.
