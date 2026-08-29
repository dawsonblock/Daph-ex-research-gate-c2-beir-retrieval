# I3.30R3 — Baseline State Freeze

**Date: 2026-08-29**
**Parent commit: `a714319`**
**Branch: `i3.30r3-authority-isolation`**

## Purpose

This document records the exact state of all frozen artifacts at the start of
I3.30R3. No Q retraining, certificate modification, threshold change, or
benchmark modification is permitted before the primary AUTH-vs-SHADOW result.

## Frozen artifact SHAs

| Artifact | SHA-256 (prefix) |
|----------|------------------|
| Q_V3R2_A model | `face8cbd3e2df9cf...` |
| V3R2 feature schema | `fcb99458da01268f...` |
| Q_V1 model | `d90d72dab250ba7c...` |
| V1 feature schema | `9722343e8d87b264...` |
| Canonical topology | `f0299cd6ef27ce8f...` |
| Canonical V3 features | `409f0777ff3d27db...` |
| Authority policy V2 | `340dd37323bd15bb...` |
| Authority policy V3 | `85af10990cf73508...` |
| Authority __init__ | `e5f850deb267318b...` |
| I3.30 frozen runner | `db8196f68c11c5a4...` |
| I3.29 runner | `fa10abc92db8ee63...` |
| Utility config | `e5c6d34acc9cc73a...` |
| I3.29 generator | `eb4c44b0d6cd2160...` |
| I3.30 D5 generator | `28133cbcab5ca0c7...` |
| Offline gates | `aedaf56bd90413a3...` |

## Frozen constants

- Authority threshold: `5.0`
- Near-optimal epsilon: `3.0`
- V3 frozen rule: `A2AD_V3_POSITIVE_CERTIFICATE`
- V2 frozen rule: `A2AD_V2`
- Benchmark seed: `9817`
- Task count: `185` (D1: 35, D2: 35, D3: 45, D4: 35, D5: 35)

## I3.30R2 diagnostic result (frozen)

- V1 success: 97/185 = 52.43%
- V3 success: 106/185 = 57.30%
- Rescues: 15
- Breaks: 6
- Mean paired ΔU: +6.8622
- V3 hard authority events: 81 (78 ANSWER, 3 DEFER)
- Observed hard-terminal-wrong: 0
- Status: REJECTED_FOR_PROMOTION, VALID_FOR_DIAGNOSIS

## Known defects (from I3.30R2 diagnostic)

### Problem A — Q/advisory regression (5 breaks)
- d1_0004, d1_0010, d2_0003, d3_0022, d3_0038
- Fix target: Q_V3R2 training/support/calibration
- NOT certificate relaxation

### Problem B — possible authority undercoverage (1 break)
- d5_0026
- Fix target: determine whether certificate SHOULD have fired
- Requires D5 state-level causal truth audit before any change

## Freeze constraints

1. **Q_V3R2-A is frozen.** No retraining before the primary AUTH-vs-SHADOW result.
2. **V3 certificate logic is frozen.** No threshold or certificate condition changes.
3. **Benchmark is frozen.** Same 185 tasks, seed 9817.
4. **The only new code is the shadow arm, normalized receipts, counterfactual
   replay, and treatment-purity tests.** No changes to Q prediction, feature
   extraction, topology, certificate evaluation, or the executor.
5. **V1 remains the confirmed champion.** V3R2-A is not promoted by this experiment.
