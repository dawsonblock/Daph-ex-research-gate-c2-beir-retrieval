# R14-C Results: Replicated Qualification

**Commit:** (this commit)
**Date:** 2026-09-05
**Protocol:** R14_C_PROTOCOL.md (preregistered before running seeds 123, 2024)
**Provenance:** R14_C_PROVENANCE.json

## Integrity

- 810 action cells = 90 checkpoints × 3 operators × 3 seeds ✓
- All cells present, all status=SUCCESS
- STOP results are identical across seeds (deterministic, no inference)
- COT_REFLECT and RE2 results are identical across seeds (temp=0.0)

## Replication: COT_REFLECT accuracy is exactly 0.900 at all 3 seeds

| Operator | Seed 42 | Seed 123 | Seed 2024 | Pooled |
|---|---|---|---|---|
| STOP | 0.489 | 0.489 | 0.489 | 0.489 |
| OPT_RE2 | 0.556 | 0.556 | 0.556 | 0.556 |
| OPT_COT_REFLECT | 0.900 | 0.900 | 0.900 | 0.900 |

The identical results across seeds are expected: llama-server runs with `--temp 0.0` (greedy decoding), so the only stochasticity is in OptiLLM's internal sampling for RE2 and COT_REFLECT. Both strategies appear deterministic at temp=0.0.

**Caveat:** The 3-seed "replication" is not a true stochastic replication because the base model is greedy. The seeds primarily verify that the runner is reproducible and that no transient service errors corrupted individual cells. True stochastic replication would require temp>0 in the base model.

## Question 1: Does COT-reflection replicate its large accuracy advantage?

**YES.** COT_REFLECT accuracy is 0.900 at all 3 seeds. Task-clustered bootstrap CI for (COT_REFLECT - STOP) is [+0.321, +0.543], which does not cross zero. McNemar pooled: b=114, c=3, stat=103.42 (p<<0.001).

## Question 2: Is RE2 a useful low-cost intermediate tier?

**MARGINAL.** RE2 accuracy is 0.556 vs STOP 0.489. Task-clustered bootstrap CI for (RE2 - STOP) is [-0.025, +0.222], which crosses zero. RE2 is NOT statistically significantly better than STOP.

However, RE2 is on the Pareto frontier because it is much cheaper than COT_REFLECT (mean 1.1s vs 7.6s). Under a sufficiently high latency penalty, RE2 could be the optimal fixed action for states where STOP is uncertain but COT is too expensive.

## Question 3: Replicated Pareto frontier

Using mean latency:

| Operator | Accuracy | Mean latency (s) | Median (s) | P90 (s) | P95 (s) |
|---|---|---|---|---|---|
| STOP | 0.489 | 0.0 | 0.0 | 0.0 | 0.0 |
| OPT_RE2 | 0.556 | 1.1 | 0.3 | 5.1 | 7.1 |
| OPT_COT_REFLECT | 0.900 | 7.6 | 7.3 | 11.2 | 12.3 |

**Pareto frontier (accuracy / mean latency):** STOP → RE2 → COT_REFLECT

RE2 is not dominated because it costs 7× less than COT_REFLECT while providing +6.7pp over STOP.

## Question 4: Oracle headroom over best fixed action

| Seed | 3-way Oracle | Best Fixed (COT) | Headroom |
|---|---|---|---|
| 42 | 0.933 (84/90) | 0.900 (81/90) | +3.0pp |
| 123 | 0.933 (84/90) | 0.900 (81/90) | +3.0pp |
| 2024 | 0.933 (84/90) | 0.900 (81/90) | +3.0pp |

The 3-way oracle picks the best of {STOP, RE2, COT} per checkpoint. It achieves 93.3% vs 90.0% for always-COT. That is +3.0pp of headroom.

## Question 5 (PRIMARY): Routing headroom — latency saved at matched accuracy

**STOP→COT oracle:** If the executive invokes COT only when STOP is wrong:
- Accuracy: 0.911 (slightly higher than always-COT's 0.900)
- Expected latency: 3.89s (vs 7.60s for always-COT)
- **Latency saving: 48.9%**
- **Accuracy gain: +1.1pp**

This is the strongest evidence for DAPH-X: a state-dependent policy that predicts when STOP is wrong could halve the cost of CoT-reflection while slightly improving accuracy.

**3-way oracle vs always-COT:**
- Accuracy: 0.933 vs 0.900 (+3.3pp)
- This is the upper bound on what a perfect 3-way router could achieve.

## STOP/COT contingency table (identical at all 3 seeds)

|  | COT correct | COT wrong |
|---|---|---|
| STOP correct | 43 | 1 |
| STOP wrong | 38 | 8 |

- COT hurts STOP on only 1/90 checkpoints
- COT rescues STOP on 38/90 checkpoints
- Both wrong on 8/90 (these need a stronger operator or different approach)

## Cost limitation

Wall_ms is observed end-to-end latency. True token/GPU compute is NOT instrumented. The latency distribution for RE2 is skewed (mean 1.1s, median 0.3s, p90 5.1s), so mean alone is misleading for deployment planning.

## Conclusion

The project question has shifted from "Can DAPH-X make Qwen reason better?" to "Can DAPH-X recognize when the cheap answer is already sufficient and avoid paying for CoT-reflection unnecessarily?"

The answer is: **there is substantial headroom for this.** A perfect STOP-vs-COT oracle achieves 91.1% accuracy at 49% lower latency than always-COT. A perfect 3-way oracle achieves 93.3% accuracy (+3.3pp over best fixed).

The next step is to learn a state-dependent policy that predicts when to invoke COT_REFLECT, using the observable features in the checkpoint state (agreement rate, entropy, margin, trajectory shape).
