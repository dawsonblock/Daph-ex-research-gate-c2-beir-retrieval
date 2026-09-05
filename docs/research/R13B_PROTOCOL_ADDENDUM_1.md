# R13-B Protocol Addendum 1

## Blind clarification, pre-replication

This addendum is committed **before** inspecting the completed seeds 123 and 2024 of the R13-A v2.1 replication. The original `R13B_PROTOCOL.md` (commit `ed2c42a`) is preserved unchanged. This addendum freezes additional degrees of freedom that were underspecified in the preregistration.

## 1. Cost convention (unit freeze)

The cost of an action is defined as:

```
c(s, a) = tokens(s, a) / 1000
```

Then the utility of an action is:

```
J_λ(s, a) = Q(s, a) - λ c(s, a)
```

and the value-of-compute target is:

```
ΔJ_λ(s) = [Q(s, V) - Q(s, S)] - λ [c(s, V) - c(s, S)]
```

λ is the **utility penalty per 1,000 tokens**. All reported `J_λ` and `ΔJ_λ` values use this convention. The previously used `tokens / 1000` form is identical; this addendum makes the `c` variable explicit.

## 2. Continuation selection and λ selection ordering

The following order is frozen:

1. **Select the fixed continuation operator V using λ = 0.**
   - Compute mean `Q(s, a)` for each continuation operator over the replicated three-seed R13-A evidence.
   - Choose the operator with the highest mean `Q`.
   - Tie-break by lower mean token cost `c`.
   - Tie-break by lower mean number of model calls.
   - Final tie-break by operator id lexical order.
2. **Freeze V.** It does not change during R13-B.
3. **Then and only then** tune the operating `λ` (or budget target) on R13-B development data.

This removes the circularity between continuation selection and λ selection.

## 3. R13-A eligibility gate (numerical)

R13-B proceeds only if all of the following hold on the **three-seed replicated** R13-A evidence:

### 3.1 Heterogeneous vs binary oracle

For every λ in the fixed R13-B grid (see §5.1):

```
J_het - J_bin < 0.005
```

where `J_het` is the mean per-state utility of the best heterogeneous oracle and `J_bin` is the mean per-state utility of the binary STOP/CONTINUE oracle.

Additionally, the task-clustered 95% two-sided confidence interval for `J_het - J_bin` must not exclude zero on the positive side.

### 3.2 Continuation dominance

The selected continuation V must:

1. Have the highest replicated mean `Q(s, a)` among continuation operators at λ = 0.
2. No rival continuation must show a statistically supported gain above a 0.5 percentage-point materiality threshold when it is the selected continuation under the heterogeneous oracle. Specifically, for each rival operator `a'`:
   - Let `S_{a'}` be the set of states where `a'` is the best non-STOP action under the heterogeneous oracle.
   - The number of such states must be small enough that the task-clustered 95% CI for `mean_a'[Q] - mean_V[Q]` is not entirely above 0.005.

If both conditions are not met, R13-B is **not executed**. The protocol remains frozen for documentation.

## 4. Confirmation design (independent holdout)

The 90 R13-A checkpoints (81 unique `task_id`s) are used **only** for R13-B train/calibration/development.

A **separate R13-B confirmation set** is constructed from tasks that were **not in the R13-A task set**. This confirmation set is generated using the same pipeline and checkpoint procedures as R13-A v2.1, but with new task IDs.

The confirmation set remains **untouched** until:
1. V is frozen from R13-A.
2. The R13-B controller is fully trained and frozen on train/cal/dev.
3. All hyperparameters and thresholds are frozen.

Only then is the confirmation set scored.

This makes the scientific progression:

```
R13-A replicated evidence → freeze V
    ↓
R13-B train / cal / dev
    ↓
freeze controller
    ↓
NEW untouched R13-B confirmation
```

This is stronger than a split of the same 81 tasks.

## 5. Frozen R13-B model and search degrees of freedom

### 5.1 λ grid (operating)

The operating λ is chosen from:

```
{0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
```

Alternatively, the budget target `B` is chosen from:

```
{50, 100, 200, 400, 800, 1200} tokens average per checkpoint
```

If the budget formulation is used, `λ` is solved via bisection to hit `B` on development data.

### 5.2 Threshold grids

Simple threshold controllers search over:

- Confidence threshold: `p_top1 ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` (CONTINUE if below).
- Entropy threshold: `entropy ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.3}` (CONTINUE if above).
- Temporal uncertainty threshold: `uncertainty_ema ∈ {0.05, 0.1, 0.2, 0.3, 0.5}` (CONTINUE if above).
- `uncertainty_delta` threshold: `{-0.2, -0.1, 0.0, 0.05, 0.1, 0.2}` (CONTINUE if above).

For each threshold, choose the action (STOP/CONTINUE) that maximizes `J_λ` on development.

### 5.3 Model classes and hyperparameters

1. **Constant policy** — `always-STOP`, `always-CONTINUE`.
2. **Best threshold** — selected from §5.2.
3. **Logistic regression** — `scikit-learn` `LogisticRegression` with `solver='lbfgs'`, `max_iter=1000`, `C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}`.
4. **Regularized linear value regression** — `scikit-learn` `Ridge` with `alpha ∈ {0.001, 0.01, 0.1, 1.0, 10.0}`.
   - Huber regression is **excluded** from R13-B to reduce model-selection variance. Only Ridge is used.
5. **Elastic net** — `not used`.
6. **Trees / forests / GBM / neural nets** — `not used`.

### 5.4 Feature preprocessing

- Continuous features are **standardized** to zero mean and unit variance using statistics from the training set.
- `difficulty` and `category` are **one-hot encoded** from the training set.
- No feature interactions.
- No feature selection beyond the frozen list in `R13B_PROTOCOL.md`.

### 5.5 Splits and random seeds

- Train / cal / dev / confirmation split by `task_id`.
- Split seed: `2025`.
- Proportions: 60% train, 15% cal, 15% dev, 10% confirmation.
- Model fitting seeds: `42` for logistic regression, `123` for Ridge.
- Bootstrap seed: `99`.
- Bootstrap replicates: `10,000`.
- Resampling unit: `task_id`.

### 5.6 Confidence intervals

- Task-clustered **two-sided 95%** percentile bootstrap intervals are reported as the default.
- The success gate uses the **lower 95% one-sided bound** (equivalent to the 5th percentile of the two-sided interval) for the `ΔJ_λ` comparison.

### 5.7 Tie-breaking

For model predictions that produce exactly zero (e.g., `ΔĴ_λ(s) = 0`), the rule is **STOP**. The continuation decision requires `ΔĴ_λ(s) > 0`.

For hyperparameter selection, tie-break by lower `C` (more regularization), then lower `alpha`.

For threshold selection, tie-break by fewer `CONTINUE` decisions (more conservative), then lower threshold value.

## 6. Matched-accuracy gate (numerical)

The secondary success criterion is:

```
A_learned ≥ A_baseline - 0.005
```

and

```
C_learned ≤ 0.95 C_baseline
```

where `A` is confirmation accuracy and `C` is mean cost `c(s,a)` in tokens-per-1000.

The primary success criterion remains: improvement in `J_λ` over the best simple threshold with a positive task-clustered 95% lower confidence bound on the confirmation set.

## 7. Stop condition

If the three-seed replication does not pass the R13-A eligibility gate (§3), R13-B is not executed. The protocol is not revised; it is simply not run.

If R13-B is executed but no model passes the success gate (§6 and primary criterion), the conclusion is:

> A learned value-of-compute controller does not improve over a simple uncertainty or confidence threshold. DAPH-X should use a simple threshold with the frozen fixed continuation.

## 8. Audit trail

- `R13B_PROTOCOL.md`: original preregistration, commit `ed2c42a`, unchanged.
- `R13B_PROTOCOL_ADDENDUM_1.md`: this file, frozen before replication inspection.

No further addenda are expected unless the user commits them explicitly. The next commit in this sequence should be the three-seed R13-A v2.1 replicated analysis, followed by the conditional R13-B execution or the R13-B cancellation notice.
