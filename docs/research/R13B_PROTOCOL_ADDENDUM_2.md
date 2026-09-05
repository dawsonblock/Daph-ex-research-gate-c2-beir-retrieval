# R13-B Protocol Addendum 2

## Blind correction, still pre-replication

This addendum is committed **after** discovering and correcting two analysis flaws in `scripts/analyze_r13a_v2_1.py`, but **before** inspecting the completed seeds 123 and 2024. The original `R13B_PROTOCOL.md` (`ed2c42a`) and `R13B_PROTOCOL_ADDENDUM_1.md` (`c0220c6`) are preserved unchanged. This addendum overrides specific parts of Addendum 1.

## 1. What was wrong in the prior analysis

### 1.1 Tautological binary ceiling

The previous `J_bin` was defined as:

```
best_bin = max(J_STOP, max(J_continuation))
```

which is identical to:

```
best_het = max(J_all)
```

by construction. This forced `J_het - J_bin = 0` regardless of whether action heterogeneity existed.

### 1.2 Inconsistent cost token decoding

The v2 operators stored `cost["tokens"]` with two different conventions:

- `SAMPLE_STANDARD` v2 and `SAMPLE_DIVERSE` v2: `tokens` = prompt tokens; `completion_tokens` = completion tokens.
- `CRITIQUE_RETRY` v2 and `VERIFY_TARGETED` v2: `tokens` = total (prompt + completion); `completion_tokens` = completion only.
- `STOP`: zero.

The v2.1 analysis added `tokens + completion_tokens` for all operators, double-counting CRITIQUE_RETRY and VERIFY_TARGETED.

## 2. Corrected definitions

### 2.1 Normalized cost

For R13-A v2.2 analysis and R13-B, the normalized token cost is:

```
c(s, a) = tokens_normalized(s, a) / 1000
```

where `tokens_normalized` is decoded from the historical receipt by:

| Operator | Version | `tokens_normalized` |
|----------|---------|---------------------|
| `STOP` | any | 0 |
| `SAMPLE_STANDARD` | `2` | `cost["tokens"] + cost["completion_tokens"]` |
| `SAMPLE_DIVERSE` | `2` | `cost["tokens"] + cost["completion_tokens"]` |
| `CRITIQUE_RETRY` | `2` | `cost["tokens"]` |
| `VERIFY_TARGETED` | `2` | `cost["tokens"]` |

If other operator versions appear, their decoder must be explicit and not assumed.

### 2.2 Corrected three ceilings

For each state `s` and action `a`:

```
J_λ(s, a) = Q(s, a) - λ c(s, a)
```

**Heterogeneous oracle**:

```
J_het(λ) = (1/N) Σ_s max_a J_λ(s, a)
```

**Best fixed binary**:

```
v_*(λ) = argmax_{v ≠ STOP} (1/N) Σ_s max[J_λ(s, STOP), J_λ(s, v)]
J_bin_best(λ) = (1/N) Σ_s max[J_λ(s, STOP), J_λ(s, v_*(λ))]
```

**λ=0 fixed binary** (the one R13-B will freeze):

```
v_0 = argmax_{v ≠ STOP} (1/N) Σ_s Q(s, v)
J_bin_v0(λ) = (1/N) Σ_s max[J_λ(s, STOP), J_λ(s, v_0)]
```

Tie-breaking for `v_0`: lower mean token cost, then fewer mean model calls, then lexical operator id.

## 3. Updated R13-A eligibility gate

R13-B proceeds **only if**:

```
UCB_95(J_het(λ) - J_bin_best(λ)) < 0.005
```

at the reference λ (preferred: λ = 0, and also at the planned R13-B operating λ if different).

This is a **one-sided upper bound** from task-clustered bootstrap. A non-significant positive mean does not establish equivalence; only a 95% upper bound below the materiality threshold does.

If the UCB is ≥ 0.005, the data do **not** rule out meaningful heterogeneous routing value, and the binary-only R13-B architecture is not automatically justified.

## 4. Remove the rival-action winner's-curse test

Addendum 1 §3.2 included a rival-action test that conditioned on states where a rival action was selected by the heterogeneous oracle. That is a winner's-curse/selection-bias test and is removed.

The `J_het - J_bin_best` ceiling comparison is the only statistical eligibility gate. Per-operator winning-state counts may still be reported descriptively, but they do not enter a hypothesis test.

## 5. Continuation selection and λ selection order (clarified)

1. **Compute the reference continuation** `v_0` using λ = 0 on the replicated R13-A evidence:
   
   ```
   v_0 = argmax_{v ≠ STOP} mean_s Q(s, v)
   ```

2. **Freeze `v_0`**. It does not change during R13-B.

3. **Only then** select the operating `λ` for R13-B on the R13-B development data, using `v_0` as the frozen continuation.

The R13-B target is:

```
ΔJ_λ(s) = [Q(s, v_0) - Q(s, STOP)] - λ [c(s, v_0) - c(s, STOP)]
```

## 6. Split design correction

### 6.1 What was contradictory

Addendum 1 §4 said all 81 R13-A unique `task_id`s are used only for train/cal/dev and that confirmation is external. Addendum 1 §5.5 still specified a 60/15/15/10 split over those same tasks. This is a contradiction.

### 6.2 Corrected split

- **R13-A tasks** (the 81 unique `task_id`s from the replicated tournament) are used **only** for R13-B **train / calibration / development**.
- **R13-B confirmation** is a **separate, newly generated set** of checkpoints from tasks that are **not** in the R13-A task set.
- The internal R13-A split proportions are renormalized to:
  - Train: 66.67%
  - Calibration: 16.67%
  - Development: 16.67%
  - Confirmation: **0% from R13-A; external only**
- Split is grouped by `task_id`.
- Split seed: `2025`.

### 6.3 New confirmation set

The R13-B confirmation set is generated using the same two-stage pipeline and v2.2 checkpoint format as R13-A v2.1, but from new tasks. It remains untouched until:

1. `v_0` is frozen from R13-A.
2. The R13-B controller is fully trained and frozen on R13-A train/cal/dev.
3. All hyperparameters, thresholds, and `λ` are frozen.
4. Only then is the confirmation set scored.

## 7. Updated success gate

Primary success criterion (unchanged in spirit, but with the corrected `v_0` and independent confirmation):

A learned binary value-of-compute controller must improve `J_λ` over the best simple threshold on the **unseen R13-B confirmation set** with a positive task-clustered 95% **lower** confidence bound.

Secondary success criterion (unchanged):

```
A_learned ≥ A_baseline - 0.005
C_learned ≤ 0.95 C_baseline
```

where `A` is confirmation accuracy and `C` is mean `c(s,a)`.

## 8. What this means for the current seed-42 analysis

The v2.2 seed-42 analysis is provisional and **not** the basis for R13-B eligibility. It showed:

- `J_het - J_bin_best` mean ≈ +0.022 at λ = 0.
- `UCB_95(J_het - J_bin_best)` ≈ +0.037.
- This does **not** pass the 0.005 equivalence threshold.

R13-B eligibility is **pending** the three-seed replicated analysis. If the replicated `UCB_95` also fails to drop below 0.005, R13-B is not executed as a binary-only architecture.

## 9. Audit trail

- `R13B_PROTOCOL.md`: original preregistration, `ed2c42a`, unchanged.
- `R13B_PROTOCOL_ADDENDUM_1.md`: first blind addendum, `c0220c6`, preserved for transparency; superseded where this document conflicts.
- `R13B_PROTOCOL_ADDENDUM_2.md`: this file, `new commit`, correcting the binary ceiling, cost decoder, eligibility gate, and split design.
- `experiments/daph_x/r13/v2/R13A_v2_2_ERRATUM.md`: seed-42 v2.2 corrected results.
- `scripts/analyze_r13a_v2_2.py`: v2.2 analyzer.

No raw execution receipts were modified. No model was trained. Seeds 123 and 2024 are still uninspected.
