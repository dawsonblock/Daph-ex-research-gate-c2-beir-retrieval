# R14-C Preregistration Protocol

**Frozen at:** 2026-09-05T22:00:00Z
**Status:** PREREGISTERED — questions and analysis plan fixed before running additional seeds.

## Background

R14-B2 (commit 8c11c85) screened 90 checkpoints × 4 operators (STOP, OPT_COT_REFLECT, OPT_RE2, OPT_SC_LOW) at seed 42. Results:

| Operator | Accuracy | Mean wall (s) | Median wall (s) |
|---|---|---|---|
| STOP | 0.489 | 0 | 0 |
| OPT_RE2 | 0.556 | 1.23 | 0.37 |
| OPT_COT_REFLECT | 0.900 | 7.70 | 7.29 |
| OPT_SC_LOW | 0.644 | 25.39 | 1.98 |

McNemar (discordant pairs, p<0.05 if >3.84):
- COT_REFLECT vs STOP: b=38, c=1, stat=33.23 (highly significant)
- RE2 vs STOP: b=20, c=14, stat=0.74 (not significant)
- SC vs STOP: b=27, c=13, stat=4.22 (significant)

**Pareto frontier (accuracy/mean-latency):** STOP → RE2 → COT_REFLECT

SC is dominated by COT_REFLECT (lower accuracy, higher mean latency). SC is retired from R14-C.

## STOP/COT contingency table (seed 42)

|  | COT correct | COT wrong |
|---|---|---|
| STOP correct | 43 | 1 |
| STOP wrong | 38 | 8 |

- Always STOP: 44/90 = 48.9%
- Always COT: 81/90 = 90.0%
- Perfect STOP-vs-COT oracle: 82/90 = 91.1%
- Oracle needs COT on only 46/90 = 51.1% of states
- Estimated oracle latency: 0.511 × 7.7s ≈ 3.94s vs 7.7s for always-COT
- ~49% latency reduction at slightly higher accuracy

## Operators

| Operator | Status | Rationale |
|---|---|---|
| STOP | INCLUDED | Free baseline, current state answer |
| OPT_RE2 | INCLUDED | Low-cost intermediate tier on Pareto frontier |
| OPT_COT_REFLECT | INCLUDED | High-accuracy strategy, dominant |
| OPT_SC_LOW | RETIRED | Dominated by COT_REFLECT |
| OPT_PLANSEARCH_LOW | RETIRED | 10% accuracy at 106s mean latency in isolated screening. Outputs are Python code blocks, not direct answers. Dominated by STOP and COT_REFLECT. |

## Design

```
90 frozen checkpoints (from R13-A v2 corpus)
×
{STOP, OPT_RE2, OPT_COT_REFLECT}
×
seeds {42, 123, 2024}
=
810 action cells
```

## Preregistered questions

1. **Does COT-reflection replicate its large accuracy advantage?**
   - H1: COT_REFLECT accuracy > STOP accuracy at each seed
   - Test: task-clustered McNemar or bootstrap CI on accuracy difference
   - Replication criterion: significant at p<0.05 at ≥2 of 3 seeds

2. **Is RE2 actually a useful low-cost intermediate tier?**
   - H2: RE2 accuracy > STOP accuracy (task-clustered)
   - H2b: RE2 accuracy < COT_REFLECT accuracy
   - Utility criterion: RE2 is useful if it is on the Pareto frontier at realistic latency penalties

3. **What is the replicated accuracy/latency Pareto frontier?**
   - Compute mean and median wall time per operator per seed
   - Report the frontier using mean latency (deployment-oriented) and median latency

4. **What is the oracle headroom over the best fixed action?**
   - Oracle: per-checkpoint best of {STOP, RE2, COT_REFLECT}
   - Best fixed: highest-accuracy fixed operator
   - Headroom = oracle accuracy - best fixed accuracy
   - Report with task-clustered bootstrap CI

5. **PRIMARY: What compute/latency can a state-dependent policy save at matched accuracy?**
   - For each λ: J_λ(s,a) = Q̄(s,a) - λ_L · L̄(s,a)
   - J_oracle(λ) = (1/N) Σ_s max_a J_λ(s,a)
   - J_best-fixed(λ) = max_a (1/N) Σ_s J_λ(s,a)
   - Routing headroom = J_oracle(λ) - J_best-fixed(λ)
   - Scan λ over realistic latency penalties
   - **This is the DAPH-X existence test.**

## Deployment-oriented view

Also report minimum average latency subject to accuracy ≥ target:

| Target | Always COT | Simple threshold → COT | DAPH-X oracle |
|---|---|---|---|
| 85% | ? | ? | ? |
| 88% | ? | ? | ? |
| 90% | ? | ? | ? |

The simple threshold baseline: invoke COT_REFLECT only when STOP uncertainty exceeds a threshold (e.g. low agreement rate, high entropy). This is the minimum viable executive.

## Analysis requirements

1. **Task-clustered:** 81 unique task IDs across 90 checkpoints. All CIs and tests must account for task clustering (bootstrap over tasks, or mixed effects).

2. **Latency distribution:** Report mean, median, p90, p95 for each operator. SC showed mean 25.4s but median 2.0s — mean alone is misleading.

3. **Cost limitation:** Wall_ms is observed end-to-end latency. True token/GPU compute is NOT instrumented. R14-C qualifies accuracy vs wall latency only.

4. **Provenance:** Frozen in R14_C_PROVENANCE.json. Includes GGUF SHA-256, llama.cpp version/build, OptiLLM version/source hash, hardware, invocation flags.

5. **No data dredging:** Questions are fixed before running seeds 123 and 2024. The contingency table from seed 42 is reported here as prior, not as R14-C result.

## ThinkBooster

ThinkBooster is deferred to R14-D or a later operator-admission experiment. It cannot run on macOS (vllm dependency). Do not block R14-C qualification on ThinkBooster.
