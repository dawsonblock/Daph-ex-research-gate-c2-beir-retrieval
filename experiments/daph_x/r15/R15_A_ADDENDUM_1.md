# R15-A Addendum 1: Methodological Corrections

**Frozen at:** 2026-09-05T23:30:00Z
**Status:** This addendum supersedes conflicting text in R15_A_PROTOCOL.md. All corrections are frozen before any R15-A COT inference.

## Summary of corrections

1. R14-C latency aggregation uses 3-run mean L̄(s,a) = (1/3) Σ_r L(s,a,r)
2. Threshold analysis corrected: direction, equivalence classes, full frontier
3. Confirmation corpus frozen: all 419 held-out R12 tasks, one checkpoint each, deterministic K
4. "Held-out R12 tasks" — not "new tasks" or "entirely new tasks"
5. Success gate changed to confirmation-relative: A_router ≥ A_COT - 0.005
6. Latency tiers rebased on oracle-headroom recovery
7. Model definitions fixed: logistic for P(STOP wrong), ridge for ΔJ. ONE champion frozen before confirmation. No nonlinear clause.
8. uncertainty_ema removed from feature set (not in frozen R13 schema)

## 1. R14-C corrected latency (3-run mean)

Using L̄(s,a) = (1/3) Σ_r L(s,a,r) across seeds 42, 123, 2024:

| Policy | Accuracy | Mean lat (s) |
|---|---|---|
| STOP | 0.489 | 0.000 |
| RE2 | 0.556 | 1.172 |
| COT_REFLECT | 0.900 | 7.603 |
| STOP→COT oracle | 0.911 | 4.676 |
| 3-way oracle | 0.933 | 2.449 |

Latency savings vs always-COT:
- STOP→COT oracle: 38.5%
- 3-way oracle: 67.8%

Corrected λ-scan headroom: +0.0333 at λ=0, +0.0849 at 0.01, +0.2910 at 0.05, +0.2188 at 0.1, +0.0954 at 1.0. RE2 is best fixed in a narrow interval around λ≈0.054.

## 2. Threshold analysis corrected

Corrections:
- Entropy and uncertainty_current: escalate if value is ABOVE threshold (high uncertainty → escalate). Previous script escalated lowest values for all features.
- Threshold splits use complete equivalence classes of identical values. No arbitrary splitting of tied values.
- Full Pareto frontier reported for each feature, not a single "best" threshold.

Best dev Pareto points (DEV EVIDENCE ONLY):
- p_top1 < 0.8333: 86.7% at 5.522s (57 escalated)
- entropy > 0.4506: 86.7% at 5.522s (57 escalated)
- uncertainty > -0.0000: 87.8% at 6.627s (75 escalated)

None of these reach the STOP→COT oracle (91.1% at 4.676s). The gap is the opportunity for a learned router.

## 3. Confirmation corpus frozen

**File:** `experiments/daph_x/r15/r15_a_confirmation_manifest.jsonl`
**SHA-256:** 63420f48e2012b157f3e3d3bee98410804119a8f8dba26876eb3edbc9c580e6f

- 419 held-out R12 tasks (not in R13 checkpoints)
- One checkpoint per task (maximizes independent task count)
- K assigned deterministically: hash(task_id) mod 3 → {2, 4, 6}
- K distribution: k=2: 138, k=4: 143, k=6: 138
- Categories: math=273, logic=67, sequence=44, combinatorics=35
- Difficulties: easy=133, medium=170, hard=116
- Answer types: int=296, string=87, float=36
- Same feature computation as R13 (compute_observable_features)
- Same selector as R13 (select_r12_maxcal)
- Same corpus (R12 enriched corpus, SHA-256 in manifest)

This is a **within-corpus confirmation test**, not an external-distribution confirmation. The tasks and their pre-generated candidates already existed in R12. They are new to R13/R14 routing evaluation. If R15-A succeeds, a later test should use a genuinely external benchmark.

## 4. Terminology

Use "held-out R12 tasks" throughout. Do not call these "new tasks" or "entirely new tasks."

## 5. Success gate: confirmation-relative accuracy

The primary accuracy gate is:

    A_router ≥ A_COT - 0.005

where A_COT is the confirmation-set accuracy of always-COT_REFLECT.

This is a non-inferiority gate with 0.5pp margin. It does not assume COT will score exactly 90% on the confirmation distribution.

The descriptive target of ≥89.5% absolute accuracy is kept as secondary information but is NOT the pass/fail gate.

## 6. Latency tiers: oracle-headroom recovery

The binary STOP→COT oracle saves 38.5% of always-COT latency on the development set. The latency tiers measure what fraction of this available saving the router recovers on the confirmation set:

    R_L = (1 - L_router / L_COT) / (1 - L_oracle / L_COT)

where L_oracle is the confirmation-set STOP→COT oracle latency.

| Tier | Oracle-headroom recovery | Approx. absolute saving (if oracle ≈ 38.5%) |
|---|---|---|
| Bronze | ≥ 50% | ≥ 19.3% |
| Silver | ≥ 75% | ≥ 28.9% |
| Gold | ≥ 90% | ≥ 34.7% |

Both the accuracy non-inferiority gate AND the latency tier must be met.

## 7. Model definitions and champion selection

### Models to train on development set (81 unique tasks, grouped CV)

**Model A: Single threshold**
- Feature candidates: p_top1, agreement_rate, entropy, margin, uncertainty_current
- Direction: feature-specific (low confidence → escalate for p_top1/margin/agreement_rate; high uncertainty → escalate for entropy/uncertainty)
- Threshold grid: all unique values in development set
- Selection: task-grouped 5-fold CV, optimize for J_λ at λ=0.01 (moderate latency penalty)

**Model B: Logistic regression for P(STOP wrong | s)**
- Features: all frozen observable features (see §8)
- Preprocessing: standardize continuous features, one-hot encode categorical (difficulty, category)
- Regularization: L2, grid C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
- Selection: task-grouped 5-fold CV, optimize AUC
- Decision: escalate if P(STOP wrong) > threshold, threshold selected on dev set to maximize J_λ at λ=0.01

**Model C: Ridge regression for ΔJ_COT(s)**
- Target: ΔJ_COT(s) = Q_COT(s) - Q_STOP(s) - λ · L_COT(s)
- λ grid: {0.005, 0.01, 0.02}
- Features: same as Model B
- Regularization: L2, grid α ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
- Selection: task-grouped 5-fold CV, optimize MSE
- Decision: escalate if ΔJ_COT(s) > 0

### Champion selection

1. Train all three models on development set with task-grouped 5-fold CV
2. Select the model with highest mean CV J_λ at λ=0.01 as the champion
3. Refit the champion on ALL development data (no held-out fold)
4. Freeze: coefficients, preprocessing parameters, threshold, model hash
5. Commit the frozen champion to the repository
6. ONLY THEN run COT_REFLECT on the 419 confirmation checkpoints
7. Apply the frozen champion to confirmation features
8. Report confirmation performance

**No model selection after seeing confirmation outcomes.** If the champion fails, record the failure. A nonlinear R15-B can be preregistered against another untouched confirmation set.

### Nonlinear models removed

The "nonlinear model only if necessary" clause is removed. If the frozen linear champion fails R15-A, that is the result. A nonlinear R15-B is a separate experiment with a separate confirmation set.

## 8. Frozen feature set

```
k                      (float)
p_top1                 (float)
p_top2                 (float)
margin                 (float)
entropy                (float)
n_unique_answers       (float)
agreement_rate         (float)
uncertainty_current    (float)
uncertainty_delta      (float)
margin_delta           (float)
answer_changed         (float)
stable_prefix_count    (float)
difficulty             (categorical: easy/medium/hard)
category               (categorical: math/logic/combinatorics/sequence)
```

**uncertainty_ema is REMOVED.** It is not present in the frozen R13 observable_features schema. Do not compute it retroactively. Do not add new features after seeing predictive performance.

## 9. Evaluation protocol (unchanged from protocol, restated for clarity)

1. Train models on development set (R13 checkpoints, 90 checkpoints, 81 unique tasks)
2. Select champion via task-grouped 5-fold CV
3. Refit champion on all development data
4. Freeze champion (coefficients, preprocessing, threshold, hash)
5. Run STOP and COT_REFLECT on all 419 confirmation checkpoints
6. Apply frozen champion to confirmation features
7. Report:
   - Accuracy: A_router, A_COT, A_STOP
   - Latency: mean, median, p90, p95 for router, always-COT, always-STOP
   - Oracle: STOP→COT oracle on confirmation set
   - Confusion: STOP-kept-correct, STOP-kept-wrong, COT-escalated-correct, COT-escalated-wrong
   - Fraction escalated
   - R_L: oracle-headroom recovery
   - Task-clustered bootstrap CI for A_router - A_COT
8. Pass/fail:
   - PASS if A_router ≥ A_COT - 0.005 AND R_L ≥ 0.50 (Bronze)
   - Report tier: Bronze (≥50%), Silver (≥75%), Gold (≥90%)

## 10. What would falsify the DAPH-X thesis

If the frozen linear champion cannot achieve Bronze tier (A_router ≥ A_COT - 0.005 AND R_L ≥ 50%) on the confirmation set, then:
- The observable features do not contain enough signal to predict STOP correctness on held-out tasks
- Just running COT_REFLECT all the time is the better engineering choice
- The R14-C oracle headroom is real but not capturable from available state features

This would be a negative result but scientifically valuable. It would not be retried with a nonlinear model on the same confirmation set.
