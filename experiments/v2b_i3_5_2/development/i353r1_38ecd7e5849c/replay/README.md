# I3.5.3-r2.1 Replay Artifacts

This directory contains offline replay results for the I3.5.3-r1
SELECTIVE_QPIB_BASE_FIRST experiment.

## Criterion configurations

| Slug | Threshold | Margin | Purpose |
|---|---|---|---|
| `tau5_margin5` | 5.0 | 5.0 | Frozen production/development gate criterion |
| `tau0_margin0` | 0.0 | 0.0 | Permissive diagnostic only — **not** a candidate production policy |

## File naming convention

Each criterion produces:
- `gate_evaluations_<slug>.jsonl` — per-step evaluation records (full precision)
- `gate_evaluation_summary_<slug>.json` — aggregate statistics + provenance

## Important semantic rules

1. **Historical experiment identity ≠ replay identity.** The replay identity
   (`replay_identity_sha256`) binds the offline reanalysis inputs. It does not
   modify the original live experiment identity.

2. **Full precision is used internally.** Display fields (`*_display`) are
   rounded to 4 decimal places for human readability. Scientific calculations
   use the full-precision values.

3. **The `tau0_margin0` configuration is diagnostic only.** It shows what the
   gate *would* do if the criterion were permissive. It is not a candidate
   production policy and must not be used for validation.

4. **At `tau0_margin0`, the hard invariant holds:** `approved == pred_positive`.
   This is asserted during replay execution.
