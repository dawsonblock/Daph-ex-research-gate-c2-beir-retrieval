# C4 Development Rescore Report — evaluator_v2

## Background

The original C4 development run (evaluator_v1) used a local verifier that failed
to strip HRM control tokens (e.g. `<|box_end|>`) before comparing answers. This
systematically marked correct canonical/symbolic outputs wrong:

- `THETA-OLIVE<|box_end|>` was normalized to `theta olive box_end` instead of
  `theta olive`, causing every canonical answer to fail.

The stored HRM generations are valid — only the evaluator annotation was wrong.
This rescore applies the corrected shared verifier
(`hrm_adaptive_memory.evaluation.verifiers.verify_answer`) to the existing
receipts without any HRM rerun.

## Corrected Results

### Quality scores (protocol metric: 0.0 / 0.25 / 0.5 / 1.0)

| Arm  | Q_v1    | Q_v2    | Delta   |
|------|---------|---------|---------|
| C4-0 | 0.0583  | 0.1625  | —       |
| C4-1 | 0.0333  | 0.1708  | +0.0083 |
| C4-2 | 0.0500  | 0.2125  | +0.0500 |
| C4-3 | 0.0500  | 0.2125  | +0.0500 |
| C4-4 | 0.0583  | 0.3417  | +0.1792 |
| C4-5 | 0.2333  | 0.7917  | +0.6292 |
| C4-6 | 0.2417  | 0.9542  | +0.7125 |

### Binary correct rates (for cross-reference with external audit)

| Arm  | Correct_v1 | Correct_v2 | Rate_v1 | Rate_v2 |
|------|------------|------------|---------|---------|
| C4-0 | 7/120      | 20/120     | 0.0583  | 0.1667  |
| C4-1 | 4/120      | 19/120     | 0.0333  | 0.1583  |
| C4-2 | 6/120      | 20/120     | 0.0500  | 0.1667  |
| C4-3 | 6/120      | 20/120     | 0.0500  | 0.1667  |
| C4-4 | 7/120      | 31/120     | 0.0583  | 0.2583  |
| C4-5 | 28/120     | 106/120    | 0.2333  | 0.8833  |
| C4-6 | 29/120     | 109/120    | 0.2417  | 0.9083  |

### By verifier type (correct counts)

| Arm  | numeric v1→v2 | canonical v1→v2 |
|------|---------------|-----------------|
| C4-0 | 7/30 → 7/30   | 0/90 → 13/90    |
| C4-4 | 7/30 → 7/30   | 0/90 → 24/90    |
| C4-5 | 28/30 → 28/30 | 0/90 → 78/90    |
| C4-6 | 29/30 → 29/30 | 0/90 → 80/90    |

Numeric verifier was unaffected (it extracts numbers, which ignores control
tokens). The entire delta comes from canonical verifier fixes.

### By entity regime (quality scores)

| Arm  | canonical v2 | abbreviation v2 |
|------|-------------|-----------------|
| C4-0 | 0.2542      | 0.0708          |
| C4-4 | 0.2750      | 0.4083          |
| C4-5 | 0.9083      | 0.6750          |
| C4-6 | 0.9583      | 0.9500          |

## Protocol Criterion Assessment (corrected)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | C4-4 quality > C4-0 by +0.15 | **PASS** (delta = +0.1792) |
| 2 | No canonical/abbreviation regression > 0.05 | **PASS** (both improve) |
| 3 | Alias and description both improve | **UNOBSERVABLE** (development has 0 alias/description tasks) |
| 4 | FalseResolutionRate <= 0.02 | Pending measurement |
| 5 | Oracle selector gap materially reduced | Pending bootstrap |
| 6 | Oracle-leak validation | **PASS** (runtime payloads unchanged) |
| 7 | Budgets fixed | **PASS** (runtime payloads unchanged) |
| 8 | Bootstrap lower bound > 0 | Pending computation |
| 9 | No post-hoc changes | **PASS** (verifier fix is bug fix, not config change) |

## Critical Notes

### Quality metric vs. binary accuracy

The C4 protocol defines "quality" as the partial-credit metric
(0.0/0.25/0.5/1.0), not binary accuracy. With the corrected verifier:

- **Quality delta** C4-4 vs C4-0 = **+0.1792** → exceeds +0.15 threshold
- **Binary accuracy delta** C4-4 vs C4-0 = **+0.0917** → does not exceed +0.15

The external audit used binary accuracy, which is a stricter metric than the
protocol's quality score. Both metrics are reported here for transparency.

### Unresolved implementation defects

Two implementation-conformance defects identified by the audit remain unfixed
at this rescore stage:

1. **Canonical EXACT path missing**: The identity stage never produces
   `status="EXACT"` for canonical subjects, so S2c is disabled on all 60
   canonical development tasks. C4-4's gain comes entirely from the 60
   abbreviation tasks.

2. **Subject+bridge+relation not implemented**: C4-1 renders queries without
   a bridge term, so it does not reproduce the qualified iterative retrieval
   mechanism.

These must be fixed before the corrected scores can be considered the final
C4 mechanism verdict.

### Classification

- Original run: `C4_DEVELOPMENT_RUN_VOID_FOR_EVALUATOR_BUG`
- Rescored run: `C4_DEVELOPMENT_RESCORED_EVALUATOR_V2`
- Status: `MECHANISM_SIGNAL_BUT_IMPLEMENTATION_CONFORMANCE_DEFECTS_UNRESOLVED`
