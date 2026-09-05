# R15-A Preregistration: Predictive Escalation Qualification

**Frozen at:** 2026-09-05T23:00:00Z
**Status:** PREREGISTERED — questions, design, and success criteria fixed before running any R15-A inference.

## Background

R14-C established:
- COT_REFLECT achieves 90.0% accuracy vs STOP's 48.9%
- COT_REFLECT costs 7.55s mean latency vs 0s for STOP
- A perfect STOP→COT oracle achieves 91.1% at 4.64s (38.5% latency reduction)
- A 3-way oracle achieves 93.3% at 2.45s (67.5% latency reduction)
- Simple thresholds on observable features capture some signal (86.7% at 5.49s) but cannot reach oracle performance
- 3/46 STOP-wrong states have p_top1=1.0 (unanimous but wrong) — a floor on achievable STOP-keeping

R14-C used 90 checkpoints from 81 unique tasks. The threshold analysis was development evidence only (tuned on the same checkpoints used for evaluation).

## Primary question

**Can frozen observable RuntimeState features decide STOP versus COT_REFLECT while preserving most of COT's 90% accuracy and materially reducing latency?**

This is the DAPH-X existence test: can a learned router capture the ~39-49% latency-saving opportunity that the oracle demonstrates?

## Design

### Data split

- **Development set (training + tuning):** R13-A v2 checkpoints (90 checkpoints, 81 unique tasks)
  - Already have STOP and COT_REFLECT outcomes from R14-C
  - Used for model training and threshold selection
  - NOT used for confirmation performance reporting

- **Confirmation set (evaluation):** New checkpoints from 419 tasks not in R13
  - Generate checkpoints using the same R13 pipeline (R12 corpus, k values, stratified sampling)
  - Run STOP and COT_REFLECT on all confirmation checkpoints
  - Report confirmation performance ONLY on this set

### Confirmation checkpoint generation

- Source: R12 enriched corpus (`experiments/daph_x/r12/r12_enriched_corpus.jsonl`)
- 419 tasks not in R13, each with 12 pre-generated candidates
- k values: {2, 4, 6} (same as R13)
- Stratified sampling: same strata as R13 (k × p_top1 > 0.6)
- Target: ~90-150 confirmation checkpoints (stratified sample)
- Feature computation: same `compute_observable_features` as R13
- Selector: same `select_r12_maxcal` as R13

### Operators

- STOP: use current state answer (free)
- OPT_COT_REFLECT: CoT reflection via OptiLLM (expensive, ~7.6s)

RE2 is deferred to a secondary extension. The binary STOP-vs-COT decision is where almost all the economic value appears.

### Models to benchmark (in order)

1. **Always COT** — 90% accuracy, ~7.6s (upper bound on accuracy, upper bound on cost)
2. **Always STOP** — 48.9% accuracy, 0s (lower bound on accuracy, lower bound on cost)
3. **Single uncertainty threshold → COT** — if feature < threshold, escalate
   - Features to try: p_top1, agreement_rate, entropy, margin
   - Threshold selected on development set
4. **Logistic regression predicting P(STOP wrong)** — escalate if P > 0.5
   - Features: all observable RuntimeState features
5. **Logistic regression predicting ΔJ_COT(s)** — escalate if ΔJ > 0
   - Target: Q(s,COT) - Q(s,STOP) - λ · L(s,COT)
   - λ selected on development set
6. **Only if necessary: small nonlinear model** (e.g., decision tree, gradient boosting)
   - Only if logistic regression fails to meet Bronze tier

No neural router. The frozen feature set is small enough that linear models should suffice.

### Frozen feature set

```
k
p_top1
p_top2
margin
entropy
n_unique_answers
agreement_rate
uncertainty_current
uncertainty_delta
uncertainty_ema          (if available; may need computation)
margin_delta
answer_changed
stable_prefix_count
difficulty               (categorical: easy/medium/hard)
category                 (categorical: math/logic/combinatorics/sequence)
```

### Success criteria

| Tier | Accuracy | Latency saving vs always-COT |
|---|---|---|
| Bronze | ≥ 89.5% | ≥ 25% |
| Silver | ≥ 89.5% | ≥ 35% |
| Gold | ≥ 89.5% | ≥ 45% |

The oracle says ~39-49% is theoretically available. Recovering 25-35% on unseen tasks would already be meaningful.

### Evaluation protocol

1. Train models on development set (R13 checkpoints, 90 checkpoints)
2. Select hyperparameters (thresholds, λ) on development set
3. Generate confirmation checkpoints from 419 new tasks
4. Run STOP and COT_REFLECT on all confirmation checkpoints
5. Apply trained models to confirmation checkpoints
6. Report accuracy and mean latency for each model
7. Report oracle upper bounds on confirmation set

### Statistical requirements

- Task-clustered bootstrap CIs (confirmation set will have ~80-120 unique tasks)
- Report accuracy, mean latency, median latency, p90 latency
- Report confusion matrix: STOP-kept-correct, STOP-kept-wrong, COT-escalated-correct, COT-escalated-wrong
- Report the fraction of checkpoints escalated to COT

### What this experiment does NOT test

- RE2 as an intermediate tier (deferred to R15-B)
- Three-way routing (STOP → RE2 → COT)
- Neural routers
- Online learning or adaptation
- Cost in tokens/GPU compute (wall_ms only)

### What would falsify the DAPH-X thesis

If no model (including nonlinear) can achieve Bronze tier (≥89.5% accuracy at ≥25% latency saving) on the confirmation set, then:
- The observable features do not contain enough signal to predict STOP correctness
- The 3 unanimous-wrong states and the overlap between STOP-correct and STOP-wrong feature distributions are fundamental
- Just running COT_REFLECT all the time is the better engineering choice

This would be a negative result but still scientifically valuable.

## Provenance

Same as R14-C (R14_C_PROVENANCE.json):
- llama-server v9760 (6ee0f6579), temp=0.0, ctx=4096
- OptiLLM 0.3.22, approach=auto per-request
- Qwen2.5-7B-Instruct Q4_K_M GGUF
- Apple M2 Pro, 16GB

## Cost limitation

Same as R14-C: wall_ms only, not token/GPU compute.
