# R12 — Decision-Aligned Adaptive Compute

## Scientific Claim

**Primary claim:**

> DAPH-X maintains fixed-budget accuracy while using materially less reasoning compute.

**Principal comparator:** MaxCal@8 (68.4% accuracy, K=8).

**Non-inferiority margin:** δ_acc = 1 percentage point.

DAPH-X passes accuracy preservation if:

    Acc_DAPH ≥ Acc_MaxCal@8 - 1pp

**Compute saving requirement:**

    E[K]_DAPH ≤ 6

**Stretch target:**

    E[K] ≤ 5

## Central Hypothesis

> DAPH-X can allocate reasoning compute selectively and achieve a better
> accuracy-cost frontier than fixed-compute and simple uncertainty policies.

## Current Best Observed Points

| System | Accuracy | E[K] |
|--------|----------|------|
| MaxCal@8 | 68.4% | 8.0 |
| DAPH-X value_v_t010 | 68.4% | 6.4 |
| Oracle lookahead6 | 68.8% | 2.4 |
| Oracle@12 | 77.6% | 12.0 |

The oracle already answers "can adaptive compute work?" — yes.
The remaining problem is: can the learned policy identify high-value states more precisely?

## Training Target

The fundamental R12 training label is the counterfactual action advantage:

    ΔQ_K = Q(s_K, GENERATE) - Q(s_K, STOP)

where:

    Q(s_K, STOP) = U(a_K)
    Q(s_K, GENERATE) = U(a_{K+2}) - λ_C * C_{+2}

The continuous value target is:

    Y_K = ΔU_K - λ_C * C_{+2}

A cost-sensitive classification label is also acceptable:

    Y_K = 1 if ΔU_K > λ_C * C_{+2} else 0

## Feature Policy

**Compact-first:** Start with a minimal feature set and add families only if
they improve J on held-out development data (not AUROC).

**Initial feature set (R12-base):**

1. `p_top1` — current top calibrated probability
2. `margin` — p_top1 - p_top2
3. `answer_entropy` — H(A)
4. `stability` — 1 if MaxCal choice unchanged from K-2 to K
5. `dp_top1` — p_top1,K - p_top1,K-2
6. `candidate_disagreement` — fraction of candidates with different answer

**Feature addition rule:**

For every added feature family, require ALL of:

    ΔJ > 0  (on held-out dev)
    ΔE[K] < 0  OR  ΔAccuracy ≥ 0
    ΔAccuracy ≥ -0.5pp

Do not select features by AUROC. Select by J, Accuracy, and E[K].

## Cost Definition

For early development: candidate count K.

For scientific compute-efficiency claim:

    C = w_t * Tokens + w_l * Latency + w_g * GPUSeconds

Report tokens, generation calls, wall-clock latency, and GPU seconds
where available.

## Threshold Selection

Select thresholds only on development/calibration data.

Grid: τ ∈ {-0.05, -0.025, 0, 0.01, 0.025, 0.05, 0.10, 0.20}

Choose τ maximizing J_λ subject to the accuracy non-inferiority constraint.

**Do not touch the threshold after confirmation begins.**

## Risk Control

Train a separate break-risk head:

    p_B = P(ΔU < 0 | s, GENERATE)

Decision rule:

    GENERATE ⟺ LCB(ΔQ) > 0 AND p_B < ρ

Sweep: ρ ∈ {0.01, 0.025, 0.05, 0.10}

Do not optimize for zero observed breaks at all costs; that can collapse
recall to zero. Measure the tradeoff.

## Calibration

Estimate LCB_α(ΔQ) = ΔQ_hat - q_α.

Build explicit tests that verify nominal calibration levels.
If requesting 90% coverage, the calibration test must demonstrate
approximately 90% coverage on held-out exchangeable calibration data.

For OOD evaluation, report coverage rather than assuming transfer.

## Baselines

| System | Adaptive? | Learned? |
|--------|-----------|----------|
| MaxCal@2 | No | No |
| MaxCal@4 | No | No |
| MaxCal@6 | No | No |
| MaxCal@8 | No | No |
| MaxCal@12 | No | No |
| Random adaptive | Yes | No |
| p_top1 threshold | Yes | No |
| entropy threshold | Yes | No |
| margin threshold | Yes | No |
| stability heuristic | Yes | No |
| DAPH-X simple value | Yes | Yes |
| DAPH-X value+risk | Yes | Yes |
| Oracle adaptive | Yes | Oracle |

The random adaptive policy must match E[K]_DAPH so that DAPH-X cannot
win merely because it uses a different average budget.

## Pareto Frontier

For each policy produce (E[K], Accuracy).

DAPH-X succeeds when it moves the frontier upward or leftward.

Examples of meaningful improvements:
- MaxCal@8=(8.0, 68.4%) vs DAPH-X=(5.0, 68.4%)
- DAPH-X=(6.0, 70%)

Examples of uninteresting results:
- DAPH-X=(7.8, 68.5%)

## Oracle Regret

Define wasted compute:

    W_K = E[K_learned - K_oracle]

Current: W_K ≈ 4.0.

Define action regret:

    R_Q = E[Q(s, a*) - Q(s, â)]

Strong R12 target: reduce W_K from 4.0 to 2.0 while preserving accuracy.

## Intervention Forensics

For every checkpoint, classify the learned action:

- **Rescue:** GENERATE and ΔU > 0
- **Break:** GENERATE and ΔU < 0
- **Waste:** GENERATE and ΔU = 0
- **Correct stop:** STOP and ΔU ≤ 0
- **Missed rescue:** STOP and ΔU > 0

Track:
- RescueRecall
- BreakRate
- WasteRate
- MissedRescueRate
- CorrectStopRate

## Oracle Decision Boundary Study

Inspect states where Oracle=STOP vs Oracle=GENERATE.

Compare distributions of confidence, entropy, stability, verification
disagreement, candidate diversity, answer type, and task family.

Goal: determine whether oracle decision boundary is observable.

If oracle-positive states are separable → learnable policy problem.
If indistinguishable until after future candidates → state representation
lacks sufficient predictive information.

## Statistical Design

Paired evaluation. For every task:

    d_i = U_i^DAPH - U_i^baseline

Bootstrap tasks, not candidate rows.

Report paired 95% confidence intervals for accuracy difference AND
compute difference:

    ΔAccuracy = -0.2pp, 95% CI [-0.8, +0.5]
    ΔK = -2.6, 95% CI [-3.0, -2.2]

## Qualification Gates

| Gate | Requirement |
|------|-------------|
| Q1 Evaluator integrity | All normalization tests pass |
| Q2 Reproducibility | Manifest and hashes verify |
| Q3 No leakage | Task-group split checks pass |
| Q4 Accuracy | Within 1pp of MaxCal@8 |
| Q5 Compute | At least 20% lower E[K] than MaxCal@8 |
| Q6 Adaptive baseline | Beats random matched-budget policy |
| Q7 Simple baseline | Beats or Pareto-dominates best uncertainty heuristic |
| Q8 Calibration | Prespecified calibration target passes |
| Q9 Risk | Break rate below predefined tolerance |
| Q10 Statistical support | CI supports compute reduction, no material accuracy degradation |
| Q11 OOD | Advantage survives structural/mechanism holdout |
| Q12 Provenance | Clean release manifest and verifier |

**Promotion target:**

    E[K] ≤ 5 while matching MaxCal@8 accuracy

## Hard Stop Conditions

Stop and classify the result as negative if any of:

1. Simple p_top1 < τ dominates learned DAPH-X
2. OracleAdaptive loses its advantage on larger dataset
3. DAPH-X only performs well after threshold tuning on test data
4. Learned value model improves AUROC but repeatedly worsens J
5. Mechanism-OOD collapses completely

## Corpus Requirements

- N ≥ 500 reasoning tasks (prefer 800-1000)
- K_max = 12 candidates per task (K_max = 16 for a subset)
- Mixed difficulty: 40%-80% initial policy accuracy across task families
- Domains: arithmetic, algebra, logic, constraint reasoning, number theory,
  combinatorics, probability, symbolic sequences, word problems, multi-step deduction
- Split on structural task families for OOD, not just random IDs

## Splits

- Train: 60% of tasks
- Calibration: 15% of tasks
- Development: 10% of tasks
- Confirmation (test): 15% of tasks

No task appears in more than one split.
Splits are determined by task family for OOD testing.

## Execution Order

1. Freeze 555aae9 ✓
2. Add release verifier ✓
3. Define R12 protocol ✓
4. Expand reasoning dataset to ≥500 tasks
5. Generate 12 candidates/task
6. Build checkpoint-level counterfactual records
7. Compute Oracle, MaxCal, and simple adaptive baselines
8. Verify oracle adaptive advantage survives at scale
9. Train minimal value model
10. Add trajectory features one family at a time
11. Train separate break-risk head
12. Calibrate ΔQ uncertainty
13. Select policy thresholds on development only
14. Freeze everything
15. Run untouched confirmation
16. Run mechanism-OOD
17. Produce Pareto frontiers and paired CIs
18. Promote only if predefined gates pass
19. Only then expand into CRITIQUE, VERIFY, multiple generation strategies
20. Test model transfer

## Near-Term Target

    Accuracy ≥ 67.4% AND E[K] ≤ 5.0

against MaxCal@8 = 68.4%, K=8.

## Stretch Target

    Accuracy ≈ 68.4% at E[K] ≤ 5

## Ultimate Stretch

    Accuracy > MaxCal@8 while E[K] < 6
