# R14-C Results: Replicated Qualification

**Commit:** 90ff784 (data), this commit (analysis corrections)
**Date:** 2026-09-05
**Protocol:** R14_C_PROTOCOL.md (preregistered before running seeds 123, 2024)
**Provenance:** R14_C_PROVENANCE.json

## Integrity

- 810 action cells = 90 checkpoints × 3 operators × 3 seeds ✓
- All cells present, all status=SUCCESS
- STOP results are identical across seeds (deterministic, no inference)
- COT_REFLECT and RE2 results are identical across seeds (temp=0.0 greedy decoding)

## Statistical correction: seeds are not independent replicates

Because llama-server runs with `--temp 0.0` (greedy decoding), all three seeds produce identical answers. The 3-seed "replication" verifies runner reproducibility and absence of transient service errors, **not** stochastic replication.

**The pooled 270-row McNemar (b=114, c=3, stat=103.42) is pseudo-replication** — it triples the same 90 paired observations. It must not be cited as stronger evidence than the single-seed result.

The correct paired table is the unique 90-checkpoint table:

|  | COT correct | COT wrong |
|---|---|---|
| STOP correct | 43 | 1 |
| STOP wrong | 38 | 8 |

McNemar: b=38, c=1, stat=33.23 (p<<0.001). This was already significant in B2.

The task-clustered bootstrap (81 unique tasks, 1000 resamples) is the appropriate qualification statistic:

| Comparison | Mean diff | 95% CI |
|---|---|---|
| COT_REFLECT - STOP | +0.432 | [+0.321, +0.543] |
| RE2 - STOP | +0.099 | [-0.025, +0.222] |
| COT_REFLECT - RE2 | +0.333 | [+0.222, +0.444] |

## Accuracy results (identical at all 3 seeds)

| Operator | Accuracy | N correct | Mean lat (s) | Median (s) | P90 (s) | P95 (s) |
|---|---|---|---|---|---|---|
| STOP | 0.489 | 44/90 | 0.0 | 0.0 | 0.0 | 0.0 |
| OPT_RE2 | 0.556 | 50/90 | 1.24 | 0.33 | 5.1 | 7.1 |
| OPT_COT_REFLECT | 0.900 | 81/90 | 7.55 | 7.29 | 11.2 | 12.3 |

## Arithmetic correction: oracle headroom is +3.3pp

3-way oracle: 84/90 = 93.3%
Best fixed (COT): 81/90 = 90.0%
Headroom: 93.3% - 90.0% = **+3.3pp** (not 3.0pp)

## Question 1: Does COT-reflection replicate?

**YES.** 0.900 at all 3 seeds. Task-clustered bootstrap CI for (COT - STOP) = [+0.321, +0.543]. McNemar (unique 90): b=38, c=1, stat=33.23 (p<<0.001).

## Question 2: Is RE2 useful?

**MARGINAL.** RE2 - STOP CI = [-0.025, +0.222] crosses zero. Not statistically significant. But RE2 is on the Pareto frontier (1.24s vs 7.55s for COT).

## Question 3: Pareto frontier

**STOP → RE2 → COT_REFLECT** (by mean latency). RE2 is not dominated.

## Question 4: Oracle headroom

| Policy | Accuracy | Mean lat (s) |
|---|---|---|
| Always STOP | 0.489 | 0.00 |
| Always RE2 | 0.556 | 1.24 |
| Always COT | 0.900 | 7.55 |
| STOP→COT oracle | 0.911 | 4.64 |
| STOP→RE2 oracle | 0.711 | 0.83 |
| 3-way oracle | 0.933 | 2.45 |

3-way oracle: 93.3% vs best fixed 90.0% = **+3.3pp** headroom.
STOP→COT oracle: 91.1% at 4.64s (vs 7.55s for always-COT) = **38.5% latency reduction** at +1.1pp accuracy.

Note: the 3-way oracle achieves lower latency than STOP→COT because it uses RE2 as an intermediate step (1.24s) when STOP is wrong but RE2 is correct, avoiding COT's 7.55s in those cases.

## Question 5 (PRIMARY): λ-scan — routing headroom

J_λ(s,a) = Q̄(s,a) - λ_L · L̄(s,a), where Q̄ = correctness, L̄ = wall seconds.

| λ | J_oracle | J_best_fixed | Best fixed | Headroom |
|---|---|---|---|---|
| 0.000 | 0.933 | 0.900 | COT | +0.033 |
| 0.001 | 0.931 | 0.892 | COT | +0.038 |
| 0.005 | 0.921 | 0.862 | COT | +0.059 |
| 0.010 | 0.909 | 0.825 | COT | +0.084 |
| 0.020 | 0.884 | 0.749 | COT | +0.135 |
| 0.050 | 0.811 | 0.522 | COT | +0.289 |
| 0.100 | 0.707 | 0.489 | STOP | +0.218 |
| 0.200 | 0.652 | 0.489 | STOP | +0.163 |
| 0.500 | 0.614 | 0.489 | STOP | +0.125 |
| 1.000 | 0.572 | 0.489 | STOP | +0.083 |

**Key findings:**
- At λ=0 (pure accuracy): headroom = +3.3pp
- At λ=0.01-0.05 (moderate latency penalty): headroom grows to +8-29pp because the oracle avoids COT's latency when cheaper actions suffice
- At λ=0.1 (high latency penalty): best-fixed switches from COT to STOP, but oracle still gains +21.8pp by using COT selectively
- **Routing headroom is positive at all λ values tested**

## Deployment view: minimum latency at target accuracy

| Target | Best policy | Accuracy | Mean lat (s) |
|---|---|---|---|
| ≥85% | 3-way oracle | 0.933 | 2.45 |
| ≥88% | 3-way oracle | 0.933 | 2.45 |
| ≥90% | 3-way oracle | 0.933 | 2.45 |

No fixed policy achieves ≥85% except always-COT (90% at 7.55s). The 3-way oracle achieves all three targets at 2.45s — **67.5% latency reduction** vs always-COT.

## Simple threshold baselines (DEVELOPMENT EVIDENCE ONLY)

**Warning:** These thresholds were tuned on the same 90 checkpoints used for evaluation. They are development evidence, not confirmation performance. R15-A will evaluate on new tasks.

### p_top1 threshold → escalate to COT if p_top1 < threshold

| Threshold | N escalated | Accuracy | Mean lat (s) |
|---|---|---|---|
| 0.0 (never escalate) | 0 | 0.489 | 0.00 |
| 0.5 | 8 | 0.556 | 0.78 |
| 0.6 | 45 | 0.822 | 4.53 |
| 0.7 | 49 | 0.844 | 4.87 |
| 0.8 | 57 | 0.867 | 5.49 |
| 1.0 (always escalate) | 58 | 0.867 | 5.54 |

Best simple threshold: p_top1 < 0.8 → 86.7% accuracy at 5.49s (27% latency saving vs always-COT, but -3.3pp accuracy).

### STOP correctness vs observable features

| Feature | STOP correct (n=44) | STOP wrong (n=46) | Separation |
|---|---|---|---|
| p_top1 | mean=0.871, min=0.333 | mean=0.525, min=0.250 | Moderate overlap |
| agreement_rate | mean=0.871 | mean=0.525 | Same as p_top1 |
| entropy | mean=0.258, max=1.330 | mean=0.864, max=1.561 | Moderate overlap |
| margin | mean=0.771 | mean=0.190 | Good separation but overlap |

**Critical finding:** 3/46 STOP-wrong states have p_top1=1.0 (unanimous but wrong). 29/44 STOP-correct states have p_top1=1.0. So even perfect thresholding on p_top1 cannot separate all STOP-correct from STOP-wrong states. A learned model using multiple features may do better, but the 3 unanimous-wrong states are likely unresolvable from observable features alone.

### What the threshold analysis tells us

1. Simple thresholds capture signal: p_top1=0.8 threshold achieves 86.7% at 5.49s
2. But they can't reach the oracle: 86.7% vs 91.1% (STOP→COT oracle) or 93.3% (3-way)
3. The gap between simple thresholds and oracle is the value a learned router could add
4. The 3 unanimous-wrong states set a floor on achievable STOP-keeping accuracy

## STOP/COT contingency table

|  | COT correct | COT wrong |
|---|---|---|
| STOP correct | 43 | 1 |
| STOP wrong | 38 | 8 |

- COT hurts STOP on only **1/90** checkpoints
- COT rescues **38/90** STOP failures
- Both fail on 8/90 (need a stronger operator or different approach)

## Cost limitation

Wall_ms is observed end-to-end latency. True token/GPU compute is NOT instrumented. R14-C qualifies accuracy vs observed wall latency, not vs true compute.

## Conclusion

R14-C establishes three things:

1. **COT_REFLECT is very strong** (90% vs 49% for STOP)
2. **COT_REFLECT is expensive** (7.55s vs 0s for STOP)
3. **Many states don't need COT** (38/90 STOP-wrong states are rescued, but 43/90 STOP-correct states don't need escalation)

The routing headroom is positive at all λ values. The STOP→COT oracle saves 38.5% latency at +1.1pp accuracy. The 3-way oracle saves 67.5% latency at +3.3pp accuracy.

**Simple thresholds on observable features capture some signal but cannot reach oracle performance.** The gap between best simple threshold (86.7% at 5.49s) and STOP→COT oracle (91.1% at 4.64s) is the opportunity for a learned router.

The next experiment (R15-A) tests whether a learned model can close this gap on **new tasks**.
