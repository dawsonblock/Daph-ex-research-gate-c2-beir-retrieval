# R13-C Protocol — Action-Value Regression for Hierarchical Cognitive Control

## Preregistration

This protocol is committed **after** the R13-A v2.3 replicated qualification result (`86b898f`) and **before** any model training or descriptive analysis of the continuing-state structure. No model is trained until this protocol is complete.

## Research question

Can observable runtime state predict action-specific incremental value well enough to recover materially more of the heterogeneous oracle than a fixed-VERIFY escalation policy?

Formally, for each surviving continuation `a ∈ {V, S, C}`, learn:

```
ΔJ_a(s) = [Q̄(s, a) - Q̄(s, STOP)] - λ [c̄(s, a) - c̄(s, STOP)]
```

and then decide:

```
a*(s) = argmax_a ΔĴ_a(s)     if max_a ΔĴ_a(s) > 0
a*(s) = STOP                  otherwise
```

## Action space

The surviving continuation actions are:

- `V` = `VERIFY_TARGETED`
- `S` = `SAMPLE_STANDARD`
- `C` = `CRITIQUE_RETRY`

**`SAMPLE_DIVERSE` is formally retired** from R13-C. It had zero oracle selections across all 90 checkpoints and all 9 λ values in the replicated R13-A evidence. Its R13-A evidence remains in the audit trail at `experiments/daph_x/r13/v1/`.

## Architecture

```
                         RuntimeState
                              │
                              ▼
              estimate continuation advantages
                 ΔĴV      ΔĴS      ΔĴC
                    \        |        /
                     \       |       /
                      max predicted value
                             │
                  ┌──────────┴──────────┐
                  │                     │
               max ≤ 0                max > 0
                  │                     │
                 STOP              argmax action
                                    /    |    \
                                 VERIFY STD CRITIQUE
```

This is **not** a flat four-class classifier. Each checkpoint contributes a training target for every action, yielding 90 state-level training examples per action—not merely 9 continuation-class labels.

## Outcome construction

For each checkpoint `s` and each action `a`:

```
ΔV(s, a) = Q̄(s, a) - Q̄(s, STOP)
Δc(s, a) = c̄(s, a) - c̄(s, STOP)
ΔJ_λ(s, a) = ΔV(s, a) - λ Δc(s, a)
```

where `Q̄` and `c̄` are the replicated means from R13-A v2.3 (seeds 42, 123, 2024).

The training target for each `(s, a)` pair is `ΔJ_λ(s, a)` at the chosen operating λ.

## Features

Only the **already-frozen observable state features** from R13-A v2.1 may be used:

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

No oracle labels, no `correct_answer`, no `maxcal_correct`, no `candidate_correct` may enter the feature set.

## Model classes

Three independent regression models, one per action:

1. **Constant policy** — predict the training-set mean `ΔJ_a` for every state.
2. **Ridge regression** — `scikit-learn` `Ridge` with `alpha ∈ {0.001, 0.01, 0.1, 1.0, 10.0}`.
3. **Logistic regression** (secondary, for `P(ΔJ_a > 0)` only) — `scikit-learn` `LogisticRegression` with `C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}`.

No Ridge+interaction terms, no Huber, no elastic net, no trees, no forests, no GBM, no neural networks.

## Feature preprocessing

- Continuous features standardized to zero mean and unit variance using train-set statistics.
- `difficulty` and `category` one-hot encoded from the training set.
- No feature interactions. No feature selection beyond the frozen list.

## Split design

- **R13-A tasks** (81 unique `task_id`s from the replicated tournament) are used **only** for train / calibration / development.
- **R13-C confirmation** is a **separate, newly generated set** of checkpoints from tasks **not** in the R13-A task set. It uses the same two-stage pipeline and v2.3 checkpoint format.
- The internal R13-A split proportions are:
  - Train: 66.67%
  - Calibration: 16.67%
  - Development: 16.67%
  - Confirmation: **0% from R13-A; external only**
- Split is grouped by `task_id`. Split seed: `2025`.

## λ selection

- The operating λ is chosen on the development set by solving for a target budget `B` in `{200, 400, 600}` tokens average per checkpoint, or by evaluating the frozen λ grid `{0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}`.
- If budget formulation is used, λ is solved via bisection to hit B on development data.
- The chosen λ is frozen before confirmation evaluation.

## Four pre-registered ceilings

For every evaluation, compute:

| Ceiling | Definition |
|---------|------------|
| `J_STOP` | Always STOP. `J_λ = Q(S) - λ c(S) = Q(S)` (cost 0). |
| `J_fixed_VERIFY` | Always STOP→VERIFY. `J_λ = Q(V) - λ c(V)`. |
| `J_simple_hierarchical` | Best simple threshold to decide escalation, then always VERIFY when escalating. E.g., `CONTINUE if entropy > τ`, tuned on development. |
| `J_heterogeneous_oracle` | Per-state `max_a J_λ(s,a)` over the three actions + STOP. Upper bound. |

**`J_simple_hierarchical` is the primary baseline.** R13-C must beat it, not merely `J_STOP`.

## Success gate

R13-C qualifies on the **unseen R13-C confirmation set** if at least one of the following holds:

### Primary (superiority)

A learned action-value regression policy improves `J_λ` over `J_simple_hierarchical` with a positive task-clustered 95% **lower** confidence bound (10,000 bootstrap replicates, seed 99, one-sided).

### Secondary (oracle recovery)

A learned action-value regression policy recovers at least **25% of the heterogeneous oracle headroom** over `J_fixed_VERIFY`:

```
(J_learned - J_fixed_VERIFY) / (J_het_oracle - J_fixed_VERIFY) ≥ 0.25
```

and does not materially increase the break rate relative to the simple hierarchical baseline (break rate within 1 percentage point).

## Descriptive analysis (before training)

Before any model fitting, analyze the 9 continuation states (VERIFY winners: 5, STANDARD winners: 2, CRITIQUE winners: 2) against the 81 STOP states using the frozen features. Report:

- Distribution of each frozen feature by oracle action group.
- Whether the three continuation groups are visually or numerically distinguishable from STOP states.
- Whether SAMPLE_DIVERSE winners (zero) are a meaningful class or noise.

This is descriptive only. No features are selected or dropped based on this analysis. The goal is to establish whether the heterogeneous oracle headroom has any plausible observable structure.

## What R13-C is NOT

- **Not** a flat four-class softmax classifier over {STOP, V, S, C}.
- **Not** a neural router.
- **Not** an ensemble of boosted trees.
- **Not** a two-stage pipeline with a learned STOP classifier followed by a learned continuation classifier.
- **Not** a re-run of any R13-A inference.

## Scientific conclusion if R13-C succeeds

A small action-value regression can predict action-specific incremental value from observable state and recover a material fraction of heterogeneous oracle headroom. DAPH-X then takes the hierarchical form shown in the architecture diagram.

## Scientific conclusion if R13-C fails

If no model beats `J_simple_hierarchical` on the confirmation set, the R13 engineering answer is:

> Use a simple uncertainty or confidence threshold with the fixed `VERIFY_TARGETED` continuation. Do not add action-specific value regression.

## Audit trail

- `R13A_v2_3_REPLICATED_RESULT.md`: R13-A qualification result, commit `86b898f`.
- `R13B_PROTOCOL.md`: R13-B preregistration, commit `ed2c42a`.
- `R13B_PROTOCOL_ADDENDUM_1.md`: Addendum 1, commit `c0220c6`.
- `R13B_PROTOCOL_ADDENDUM_2.md`: Addendum 2, commit `beabab6`.
- `R13B_PROTOCOL_ADDENDUM_3.md`: Addendum 3, commit `777bef6`.
- `R13C_PROTOCOL.md`: this document, frozen before any training.
