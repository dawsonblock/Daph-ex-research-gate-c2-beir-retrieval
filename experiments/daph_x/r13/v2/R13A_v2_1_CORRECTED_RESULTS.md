# R13-A v2.1 Corrected Preliminary Results

## What was corrected

1. **SAMPLE_STANDARD and SAMPLE_DIVERSE terminal answers** now use the canonical R12 selector over the combined prefix + generated candidates, not the first generated answer.
2. **Cost accounting** no longer double-counts completion tokens.
3. **Q5 per-λ action distributions** are reset for each λ.
4. **Bootstrap** now clusters by `task_id`, not checkpoint.
5. **Three ceilings** compared:
   - Heterogeneous oracle
   - Binary STOP/CONTINUE oracle
   - Best fixed operator

## Data

- 403 completed executions, 400 from matched complete checkpoints.
- One replicate (seed 42) only.

## Q4 — Operator causal utility (corrected)

| Operator | All-state rescue | Conditional rescue | Break | Mean tokens |
|----------|-----------------|-------------------|-------|-------------|
| STOP | — | — | — | 0 |
| **VERIFY_TARGETED** | **5.0%** | **9.1%** | 1.2% | 498 |
| CRITIQUE_RETRY | 2.5% | 4.5% | 0.0% | 436 |
| SAMPLE_DIVERSE | 1.2% | 2.3% | 0.0% | 142 |
| SAMPLE_STANDARD | 1.2% | 2.3% | 0.0% | 89 |

### Key Q4 finding

The original 6.3% / 11.1% SAMPLE_STANDARD rescue rate was inflated by scoring the first new candidate. After correct MaxCal scoring, **SAMPLE_STANDARD's rescue rate drops to 1.2% / 2.3%** — the lowest of the continuation operators.

**VERIFY_TARGETED is the strongest fixed continuation** on both all-state (5.0%) and conditional (9.1%) rescue, with zero breaks on baseline-wrong states.

**Q4 gate status**: PENDING / NOT PASSED. No non-standard operator beats SAMPLE_STANDARD by the preregistered material margin on a small sample, but the absolute pattern suggests VERIFY_TARGETED is the only non-standard action with potentially useful value.

## Q5 — Three ceiling comparison (corrected)

| λ | Heterogeneous U | Binary STOP/CONTINUE U | Best fixed | Δ het vs best fixed | Δ het vs binary |
|---|----------------|----------------------|-----------|-------------------|-----------------|
| 0.000 | 0.5250 | 0.5250 | VERIFY_TARGETED (0.4875) | +0.0375 | **0.0000** |
| 0.010 | 0.5246 | 0.5246 | VERIFY_TARGETED (0.4825) | +0.0421 | **0.0000** |
| 0.050 | 0.5229 | 0.5229 | VERIFY_TARGETED (0.4626) | +0.0603 | **0.0000** |
| 0.100 | 0.5208 | 0.5208 | SAMPLE_STANDARD (0.4536) | +0.0671 | **0.0000** |
| 0.200 | 0.5165 | 0.5165 | STOP (0.4500) | +0.0665 | **0.0000** |

### Heterogeneous action distribution

- STOP: 74 / 80 (92.5%)
- VERIFY_TARGETED: 3 / 80
- CRITIQUE_RETRY: 2 / 80
- SAMPLE_STANDARD: 1 / 80

### Critical Q5 finding

**Heterogeneous oracle utility equals binary STOP/CONTINUE oracle utility for every λ.**

That is:

$$
J_{\text{heterogeneous}} = J_{\text{binary STOP/CONTINUE}}
$$

This means there is **no value to selecting among multiple continuation operators**. The entire value of the oracle is in deciding **STOP vs CONTINUE**, and when continuing, the best single action is usually enough.

### Three ceilings summary

| Ceiling | λ=0 U | λ=0.1 U |
|---------|-------|---------|
| Heterogeneous oracle | 0.5250 | 0.5208 |
| Binary STOP/CONTINUE | 0.5250 | 0.5208 |
| Best fixed operator | 0.4875 | 0.4536 |

At low cost penalty, the best fixed is `VERIFY_TARGETED` everywhere.
At high cost penalty, the best fixed is `STOP` everywhere.

The gap between heterogeneous and best fixed is **+0.0375 to +0.0671**.
The gap between heterogeneous and binary is **0**.

## Bootstrap CIs (clustered by task_id)

| Operator | All-state rescue% | P5% | P95% |
|----------|------------------|-----|------|
| VERIFY_TARGETED | 5.0% | 1.2% | 9.2% |
| CRITIQUE_RETRY | 2.5% | 0.0% | 5.3% |
| SAMPLE_DIVERSE | 1.2% | 0.0% | 3.8% |
| SAMPLE_STANDARD | 1.2% | 0.0% | 3.8% |

## Interpretation

The corrected R13-A v2.1 evidence does **not** justify a multiclass DAPH-X executive over heterogeneous cognitive actions.

What the evidence supports is a much simpler architecture:

```
RuntimeState
    │
    ▼
STOP / ESCALATE
   │       │
 STOP     ▼
      continuation type
        /          \
 STANDARD          VERIFY
```

But because the binary STOP/CONTINUE oracle already matches the heterogeneous oracle, even the "escalate" branch may not need operator selection. The right continuation is typically just the best single continuation operator for the budget:
- Low compute budget → `STOP`
- Moderate budget → `VERIFY_TARGETED`
- If `VERIFY_TARGETED` is too expensive → `SAMPLE_STANDARD`

## Gate status

- **R13-Q4**: **PENDING / NOT PASSED**. No operator materially beats the others on the full data, though `VERIFY_TARGETED` shows the only promising absolute signal.
- **R13-Q5**: **PENDING**. The heterogeneous oracle headroom over best fixed is +3.75 to +6.71 utility points, but the **heterogeneous oracle equals the binary oracle**. Therefore the value is in the STOP/CONTINUE decision, not in action selection. The stated Q5 criterion (heterogeneous beats best fixed by 0.5% accuracy or 5% compute) is marginally met in utility, but the architectural implication is that a binary policy suffices.

## Next step

Before any policy training, finish the full 450/450 tournament and run replicates 123 and 2024. Then recompute the three ceilings using:

$$
\hat Q(s,a) = \frac{1}{R} \sum_{r=1}^{R} Y_{s,a,r}
$$

If the replicated three-seed analysis still shows heterogeneous = binary, the R13 engineering conclusion should be:

> Build a lightweight binary STOP/CONTINUE decision with a single continuation operator, not a multiclass executive.
