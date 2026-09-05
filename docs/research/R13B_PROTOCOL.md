# R13-B Protocol — Value-of-Compute Binary Controller

## Preregistration

This protocol is frozen **before** inspecting the completed three-seed R13-A v2.1 results. Execution is conditional on the replicated R13-A conclusion. No model is trained until the replication confirms the eligibility gate.

## Eligibility gate

R13-B proceeds **only if** the replicated three-seed R13-A v2.1 result satisfies:

1. Heterogeneous oracle provides **no material advantage** over binary STOP/CONTINUE oracle.
2. One fixed continuation operator dominates the non-STOP actions (expected: `VERIFY_TARGETED`).

If the three-seed evidence instead shows genuine continuation-action heterogeneity (e.g., `SAMPLE_STANDARD` and `VERIFY_TARGETED` each win stable, large subsets), halt R13-B and revisit the action set without changing this preregistered protocol.

## Research question

Can DAPH-X learn a binary value-of-compute controller that decides, from observable state alone, whether the expected utility of continuing exceeds the expected utility of stopping?

Formally, learn:

```
ΔJ_λ(s) = [Q(s, V) - Q(s, S)] - λ [C(s, V) - C(s, S)]
```

where:
- `V` = the single frozen continuation operator (expected: `VERIFY_TARGETED`).
- `S` = `STOP`.
- `Q(s, a)` = terminal correctness (0 or 1).
- `C(s, a)` = token cost of action `a`.
- `λ` = cost penalty.

Runtime rule:

```
CONTINUE  ⇔  ΔĴ_λ(s) > 0
STOP      ⇔  ΔĴ_λ(s) ≤ 0
```

## Continuation operator

- The continuation is a **single fixed operator**, not a learned per-state selection.
- It is chosen from the replicated R13-A evidence after averaging over seeds 42, 123, and 2024.
- Expected choice: `VERIFY_TARGETED`.
- If `VERIFY_TARGETED` is not the dominant continuation, the operator with the highest mean `Q(s,a) - λ C(s,a)` at the chosen budget `λ` will be selected, and that choice will be documented and frozen before any training.

## Outcome construction

For each checkpoint and action, compute the **mean outcome across the three seeds** before constructing `ΔV` or `ΔJ_λ`:

```
Q̄(s, a) = (1/3) Σ_r 1[terminal_answer_{s,a,r} is correct]
C̄(s, a) = (1/3) Σ_r tokens_{s,a,r}
```

Then:

```
ΔV(s)    = Q̄(s, V) - Q̄(s, S)
ΔC(s)    = C̄(s, V) - C̄(s, S)
ΔJ_λ(s)  = ΔV(s) - λ ΔC(s)
```

Two targets may be analyzed, but the **primary** is `ΔJ_λ(s)`. A secondary classification target `P(VERIFY helps | s)` (rescue probability) may be reported but must not drive the final policy.

## Features

Only the **already-frozen observable state features** may be used:

- `k`
- `p_top1`
- `p_top2`
- `margin`
- `entropy`
- `n_unique_answers`
- `agreement_rate`
- `uncertainty_current`
- `uncertainty_delta`
- `uncertainty_ema`
- `margin_delta`
- `answer_changed`
- `stable_prefix_count`
- `difficulty`
- `category`

No `correct_answer`, `maxcal_correct`, `candidate_correct`, or other oracle labels may enter the feature set. The `RuntimeState` boundary from R13-A v2.1 remains in force.

## Train / cal / dev / confirmation split

- Split **grouped by `task_id`**, not by checkpoint.
- A task appearing at K=2, 4, 6 belongs entirely to one split.
- Proposed split:
  - 60% train (≈49 tasks)
  - 15% calibration (≈12 tasks)
  - 15% development (≈12 tasks)
  - 10% confirmation (≈8 tasks)
- All operator definitions, `λ` search grid, feature schema, model classes, and threshold-selection code are frozen **before** any confirmation evaluation.

## Candidate models

Initial model set is deliberately small:

1. **Constant policy** — always `STOP`, always `CONTINUE`, or always the continuation.
2. **Confidence threshold** — `CONTINUE` if `p_top1 < τ`.
3. **Entropy threshold** — `CONTINUE` if `entropy > τ`.
4. **Temporal-uncertainty threshold** — `CONTINUE` if `uncertainty_delta` or `uncertainty_ema` exceeds `τ`.
5. **Logistic regression** — predicts `ΔJ_λ(s) > 0` from observable features.
6. **Regularized linear value regression** — predicts `ΔJ_λ(s)` from observable features (ridge/Huber).

No neural networks, gradient-boosted ensembles, or deep models are permitted for the first R13-B. No feature engineering beyond the frozen list.

## Primary comparison

The learned binary controller is compared against the **best simple threshold controller**, not merely `always-STOP`.

The simple threshold controller is selected on the calibration/development set from the confidence, entropy, and temporal-uncertainty threshold baselines, choosing the action (STOP/CONTINUE) that maximizes `J_λ`.

Secondary comparisons:
- `always-STOP`
- `always-VERIFY`
- `oracle` (upper bound on the same data, used only for diagnostic purposes)

## λ and threshold selection

- `λ` is selected by solving for a target compute budget `B` on the development set, or by grid search over a small frozen set.
- Threshold `τ` for simple baselines and the sign threshold for `ΔĴ_λ` are selected on development data only.
- After selection, the policy is frozen and evaluated on the untouched confirmation set.

## Evaluation metrics

For each policy at the selected `λ`:

- Accuracy
- Average tokens
- Average calls
- Utility `J_λ = accuracy - λ × tokens / 1000`
- Rescue, break, waste rates
- P(STOP) and P(CONTINUE)
- Δ to `always-STOP` and to best simple threshold
- Task-grouped 95% confidence intervals (bootstrap)

## Success gate

R13-B qualifies only if one of the following holds on the **unseen confirmation** set:

1. A learned policy improves `J_λ` over the best simple threshold controller with a positive task-clustered 95% lower confidence bound, **or**
2. A learned policy achieves at least 5% lower compute at matched accuracy versus the best simple threshold.

If no learned policy beats a simple threshold, the conclusion is:

> DAPH-X does not require a learned controller. A simple uncertainty/confidence threshold with a fixed continuation suffices.

## Scientific conclusion if R13-B succeeds

A value-of-compute binary controller can be learned from observable state and improves over simple thresholds. The DAPH-X executive then takes this form:

```
RuntimeState
    │
    ▼
observable state features
    │
    ▼
value-of-compute ΔĴ_λ(state)
    │
  ┌─┴─┐
 ≤0   >0
  │    │
 STOP  VERIFY_TARGETED
```

## Scientific conclusion if R13-B fails

If the learned binary controller cannot beat a simple threshold, the R13 engineering answer is:

> Use a simple uncertainty/confidence threshold and a fixed `VERIFY_TARGETED` continuation. Do not add a learned executive.

## No modifications during replication

This protocol is frozen. No operator implementation, threshold, action set, feature schema, or analysis rule is modified while the R13-A v2.1 replicates 123 and 2024 run. Any deviation invalidates the replication boundary.
