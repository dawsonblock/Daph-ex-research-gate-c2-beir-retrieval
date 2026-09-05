# R13-A v2.3 Replicated Three-Seed Qualification Result

## Integrity

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| (checkpoint, operator, replicate) cells | 1,350 | 1,350 | PASS |
| Averaged (checkpoint, operator) records | 450 | 450 | PASS |
| Complete checkpoints (5 operators each) | 90 | 90 | PASS |

All three seeds (42, 123, 2024) completed successfully for all 90 checkpoints × 5 operators.

## Q4 — Operator causal utility (replicated, 1,350 raw events)

| Operator | All-state rescue | Conditional rescue | Break | Mean tokens |
|----------|-----------------|-------------------|-------|-------------|
| **VERIFY_TARGETED** | **3.7%** | **7.2%** | 1.1% | 490 |
| SAMPLE_STANDARD | 1.5% | 2.9% | 0.4% | 407 |
| CRITIQUE_RETRY | 1.1% | 2.2% | 0.4% | 434 |
| SAMPLE_DIVERSE | 0.7% | 1.4% | 0.0% | 491 |

## Q5 — Three ceilings (replicated, 90 complete checkpoints)

**v0 (λ=0 best continuation): VERIFY_TARGETED**

### Heterogeneous oracle action distribution (λ = 0)

| Action | Count | Share |
|--------|-------|-------|
| STOP | 81 | 90.0% |
| VERIFY_TARGETED | 5 | 5.6% |
| SAMPLE_STANDARD | 2 | 2.2% |
| CRITIQUE_RETRY | 2 | 2.2% |
| SAMPLE_DIVERSE | 0 | 0.0% |

### Ceiling comparison

| λ | J_het | J_bin_best (VERIFY) | Δ mean | UCB95 | LCB5 | Eligibility |
|---|-------|---------------------|--------|-------|------|-------------|
| 0.000 | 0.5333 | 0.5259 | +0.0074 | +0.0157 | +0.0000 | **FAIL** |
| 0.002 | 0.5332 | 0.5258 | +0.0072 | +0.0155 | +0.0000 | **FAIL** |
| 0.005 | 0.5330 | 0.5256 | +0.0073 | +0.0156 | +0.0000 | **FAIL** |
| 0.010 | 0.5327 | 0.5253 | +0.0074 | +0.0158 | +0.0000 | **FAIL** |
| 0.020 | 0.5321 | 0.5248 | +0.0072 | +0.0156 | +0.0000 | **FAIL** |
| 0.050 | 0.5303 | 0.5230 | +0.0072 | +0.0156 | +0.0001 | **FAIL** |
| 0.100 | 0.5273 | 0.5201 | +0.0071 | +0.0154 | +0.0002 | **FAIL** |
| 0.200 | 0.5213 | 0.5144 | +0.0068 | +0.0146 | +0.0005 | **FAIL** |
| 0.500 | 0.5037 | 0.4984 | +0.0039 | +0.0100 | +0.0003 | **FAIL** |

## Eligibility gate (Addendum 3)

**Rule**: `UCB_95(J_het - J_bin_best) < 0.005` for every λ in the frozen grid.

**Result**: **FAIL at all 9 λ values.**

The UCB95 ranges from +0.0100 (λ=0.5) to +0.0158 (λ=0.01), all well above the 0.005 materiality threshold.

## Scientific conclusion

> **R13-B NOT ELIGIBLE**: Replicated three-seed evidence does not establish equivalence between heterogeneous routing and the best fixed-continuation binary architecture.

The binary STOP/CONTINUE reduction with a single fixed continuation is **not** sufficient to capture the value of the heterogeneous oracle. The mean gap is +0.7 percentage points at low λ and +0.4 at high λ, and the 95% upper bound never drops below 1.0 percentage points.

## What the evidence supports

The heterogeneous oracle action distribution reveals a hierarchical structure:

1. **STOP dominates**: 81/90 states (90%) should stop.
2. **Among continuation states**: VERIFY_TARGETED wins 5/9, SAMPLE_STANDARD wins 2/9, CRITIQUE_RETRY wins 2/9.
3. **SAMPLE_DIVERSE is dead**: 0 oracle selections across all 90 checkpoints and all 9 λ values.

This points toward a **hierarchical controller**:

```
                    RuntimeState
                         │
                         ▼
                 STOP vs ESCALATE
                    /         \
                 STOP          │
                               ▼
                     continuation choice
                     /       |       \
                 VERIFY   STANDARD   CRITIQUE
```

Not a flat five-class router, and not a binary STOP/CONTINUE with one fixed continuation.

## Audit trail

- `bec4716`: seed-42 evidence boundary (v2.1)
- `ed2c42a`: R13-B preregistration
- `c0220c6`: R13-B Addendum 1
- `beabab6`: R13-A v2.2 erratum + R13-B Addendum 2
- `777bef6`: R13-A v2.3 analyzer + R13-B Addendum 3
- `4c2068c`: R13-A v2.3.1 bootstrap fix
- `2d615ab`: R13-A v2.3.2 fail-closed integrity
- This commit: R13-A v2.3 replicated qualification result
