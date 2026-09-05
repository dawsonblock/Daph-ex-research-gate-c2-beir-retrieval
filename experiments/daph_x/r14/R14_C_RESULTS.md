# R14-C Results: Replicated Qualification (Corrected)

**Commit:** 90ff784 (data), b6e6ff1 (initial analysis), this commit (corrections)
**Date:** 2026-09-05
**Protocol:** R14_C_PROTOCOL.md (preregistered)
**Corrections:** See R15_A_ADDENDUM_1.md §1-2

## Integrity

- 810 action cells = 90 checkpoints × 3 operators × 3 seeds ✓
- All cells present, all status=SUCCESS
- Results identical across seeds (temp=0.0 greedy decoding)

## Statistical note: seeds are not independent replicates

The pooled 270-row McNemar is pseudo-replication. The correct table is the unique 90-checkpoint table:

|  | COT correct | COT wrong |
|---|---|---|
| STOP correct | 43 | 1 |
| STOP wrong | 38 | 8 |

McNemar: b=38, c=1, stat=33.23 (p<<0.001).

Task-clustered bootstrap (81 unique tasks, 1000 resamples):

| Comparison | Mean diff | 95% CI |
|---|---|---|
| COT_REFLECT - STOP | +0.432 | [+0.321, +0.543] |
| RE2 - STOP | +0.099 | [-0.025, +0.222] |
| COT_REFLECT - RE2 | +0.333 | [+0.222, +0.444] |

## Accuracy results (identical at all 3 seeds)

| Operator | Accuracy | N correct | Mean lat (s) | Median (s) | P90 (s) | P95 (s) |
|---|---|---|---|---|---|---|
| STOP | 0.489 | 44/90 | 0.000 | 0.000 | 0.000 | 0.000 |
| OPT_RE2 | 0.556 | 50/90 | 1.172 | 0.345 | 5.136 | 7.095 |
| OPT_COT_REFLECT | 0.900 | 81/90 | 7.603 | 7.303 | 11.188 | 12.306 |

Latency uses 3-run mean: L̄(s,a) = (1/3) Σ_r L(s,a,r).

## Oracle headroom: +3.3pp

| Policy | Accuracy | Mean lat (s) |
|---|---|---|
| Always STOP | 0.489 | 0.000 |
| Always RE2 | 0.556 | 1.172 |
| Always COT | 0.900 | 7.603 |
| STOP→COT oracle | 0.911 | 4.676 |
| STOP→RE2 oracle | 0.711 | 0.800 |
| 3-way oracle | 0.933 | 2.449 |

- 3-way oracle: 93.3% vs best fixed 90.0% = **+3.3pp**
- STOP→COT oracle: 91.1% at 4.676s = **38.5% latency reduction** vs always-COT
- 3-way oracle: 93.3% at 2.449s = **67.8% latency reduction** vs always-COT

## λ-scan (3-run mean latency)

| λ | J_oracle | J_best_fixed | Best fixed | Headroom |
|---|---|---|---|---|
| 0.000 | 0.933 | 0.900 | COT | +0.033 |
| 0.010 | 0.909 | 0.824 | COT | +0.085 |
| 0.050 | 0.811 | 0.520 | COT | +0.291 |
| 0.054 | 0.801 | 0.492 | RE2 | +0.309 |
| 0.057 | 0.794 | 0.489 | STOP | +0.305 |
| 0.100 | 0.708 | 0.489 | STOP | +0.219 |
| 1.000 | 0.584 | 0.489 | STOP | +0.095 |

RE2 is best fixed in a narrow interval around λ≈0.054. Routing headroom is positive at all λ values.

## Deployment view

| Target | Best policy | Accuracy | Mean lat (s) |
|---|---|---|---|
| ≥85% | 3-way oracle | 0.933 | 2.449 |
| ≥88% | 3-way oracle | 0.933 | 2.449 |
| ≥90% | 3-way oracle | 0.933 | 2.449 |

## Threshold frontier (DEV EVIDENCE ONLY)

Corrected: direction fixed for entropy/uncertainty, equivalence-class splits, full Pareto frontier.

Best dev Pareto points (all at ~86.7-87.8% accuracy):
- p_top1 < 0.8333: 86.7% at 5.522s (57/90 escalated)
- entropy > 0.4506: 86.7% at 5.522s (57/90 escalated)
- uncertainty > 0.0: 87.8% at 6.627s (75/90 escalated)

None reach the STOP→COT oracle (91.1% at 4.676s). The gap is the opportunity for a learned router.

3/46 STOP-wrong states have p_top1=1.0 (unanimous but wrong) — a floor on achievable STOP-keeping.

## STOP/COT contingency table

|  | COT correct | COT wrong |
|---|---|---|
| STOP correct | 43 | 1 |
| STOP wrong | 38 | 8 |

COT hurts STOP on only 1/90. COT rescues 38/90 STOP failures. Both fail on 8/90.

## Cost limitation

Wall_ms only. True token/GPU compute is NOT instrumented.

## Conclusion

R14-C establishes that a perfect STOP→COT oracle saves 38.5% latency at +1.1pp accuracy. The 3-way oracle saves 67.8% at +3.3pp. Simple thresholds capture some signal but cannot reach oracle performance. R15-A tests whether a learned linear router can close this gap on 419 held-out R12 tasks.
