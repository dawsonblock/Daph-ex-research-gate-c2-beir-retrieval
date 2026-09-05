# R13-C Protocol Addendum 1

## Blind clarification to R13C_PROTOCOL.md

This addendum is committed **after** the R13-C protocol (`6933b57`) and **before** any descriptive analysis or model training. It freezes four remaining degrees of freedom that could materially affect the R13-C result.

## 1. Confirmation experiment specification

The R13-C confirmation set is frozen as follows:

- **Task source**: New task IDs only, not in the R13-A task set.
- **Task count**: `≥30` unique tasks. Target `40--50` tasks.
- **Checkpoints per task**: `K = {2, 4, 6}` for each task (same K-stratification as R13-A).
- **Total confirmation checkpoints**: `≥90` (30 tasks × 3 K-values).
- **Actions executed at every confirmation checkpoint**: All 4 actions (`STOP`, `VERIFY_TARGETED`, `SAMPLE_STANDARD`, `CRITIQUE_RETRY`).
- **Replicates per action**: 3 frozen inference seeds: `{42, 123, 2024}`.
- **Total confirmation cells**: `≥ 90 × 4 × 3 = 1080` execution cells.
- **Completeness rule**: A confirmation checkpoint is used only if all `4 actions × 3 seeds = 12` receipts are present. Any checkpoint with incomplete data is dropped under the fail-closed rule (same as Addendum 3 for R13-A).
- **Checkpoint generation**: Same two-stage pipeline and v2.3 checkpoint format as R13-A v2 (`scripts/run_r13_freeze_checkpoints_v2.py`), but with new task IDs.

## 2. Frozen simple hierarchical baseline search space

`J_simple_hierarchical` is defined as the best simple two-stage policy from a frozen set of threshold rules and grids. A two-stage policy is:

```
if feature_f(s) < τ:
    STOP
else:
    VERIFY_TARGETED
```

The candidate features and threshold grids are:

| Feature `f` | Decision rule | Threshold grid `τ` |
|-------------|--------------|-------------------|
| `p_top1` | `STOP if p_top1 ≥ τ` | `{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` |
| `margin` | `STOP if margin ≥ τ` | `{0.1, 0.2, 0.3, 0.4, 0.5}` |
| `entropy` | `STOP if entropy ≤ τ` | `{0.1, 0.3, 0.5, 0.7, 1.0, 1.3}` |
| `uncertainty_current` | `STOP if uncertainty_current ≤ τ` | `{0.05, 0.1, 0.2, 0.3, 0.5}` |
| `uncertainty_delta` | `STOP if uncertainty_delta ≤ τ` | `{-0.2, -0.1, 0.0, 0.05, 0.1, 0.2}` |
| `uncertainty_ema` | `STOP if uncertainty_ema ≤ τ` | `{0.05, 0.1, 0.2, 0.3, 0.5}` |
| `agreement_rate` | `STOP if agreement_rate ≥ τ` | `{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` |
| `stable_prefix_count` | `STOP if stable_prefix_count ≥ τ` | `{1, 2, 3, 4, 5, 6}` |

Selection: on the development set, compute `J_λ` for every `(f, τ)` pair and choose the pair that maximizes `J_λ`. Tie-break: fewer CONTINUE decisions (more conservative), then lower threshold value.

The R13-B protocol (`ed2c42a`) defined a similar threshold policy set. R13-C uses this frozen set for the simple hierarchical baseline. The `ΔĴ` formulation is the learned policy to be compared against the best `(f, τ)` pair.

## 3. Frozen λ selection

λ selection uses **only the λ grid**, not a budget formulation.

### λ grid

```
{0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
```

### Primary λ

The primary operating λ is frozen as:

```
λ_primary = 0.005
```

This corresponds to a utility penalty of 0.005 per 1,000 tokens (≈ 5 utility points per 1M tokens, i.e., a modest compute penalty).

### Secondary λ values

For sensitivity analysis, all λ values in the grid are also reported, but the success gate is evaluated at `λ_primary` only.

This avoids post-hoc λ selection. The chosen λ is frozen before any confirmation evaluation.

## 4. Oracle-recovery denominator safeguard

The secondary success criterion (oracle-gap recovery fraction) is defined as:

```
R_simple = (J_learned - J_simple_hierarchical) / (J_het_oracle - J_simple_hierarchical)
```

### Safeguard

If on the confirmation set:

```
J_het_oracle - J_simple_hierarchical < 0.005
```

then the secondary criterion is reported as **NOT_EVALUABLE**, not PASS. The experiment cannot distinguish a meaningful recovery from a negligible one when the oracle headroom is too small.

### Strengthened secondary criterion

For the secondary criterion to PASS, both conditions must hold:

1. `J_het_oracle - J_simple_hierarchical ≥ 0.005` (denominator safeguard).
2. `R_simple ≥ 0.25`.
3. Task-clustered 95% bootstrap LCB of `J_learned - J_simple_hierarchical > 0` (one-sided, 10,000 replicates, seed 99, grouped by confirmation task_id).
4. Break rate of the learned policy does not exceed the simple hierarchical break rate by more than 1 percentage point on confirmation.

## 5. Diagnostic metric

In addition to the secondary criterion, report:

```
R_verify = (J_learned - J_fixed_VERIFY) / (J_het_oracle - J_fixed_VERIFY)
```

This answers the auxiliary question: how much value does the learned action selection recover beyond the fixed VERIFY policy? It is reported diagnostically, not as a gate.

## 6. Prohibition on descriptive-analysis-driven changes

The descriptive analysis of the 9 continuation states against the 81 STOP states is performed **after** this addendum is frozen and **before** any model training. No feature, hyperparameter, threshold, λ, action space, or model class is changed based on the descriptive result.

The descriptive analysis is purely diagnostic. It determines whether the heterogeneous oracle headroom has plausible observable structure. If no structure is visible, the correct conclusion is:

> The oracle headroom is real but not learnable from the current feature set. R13-C should fail gracefully with `R_simple` close to zero and the primary gate not passing.

This is a valid scientific result and does not indicate an implementation failure.

## 7. Audit trail

- `R13C_PROTOCOL.md`: original R13-C preregistration, commit `6933b57`.
- `R13C_PROTOCOL_ADDENDUM_1.md`: this document, freezing confirmation design, baseline search space, λ selection, and recovery denominator safeguard.

No R13-A evidence or R13-B protocols are modified. No model is trained. No descriptive analysis has been performed as of this commit.
