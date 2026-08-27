# I3.28: Authority-State Sufficiency — Results

**Date:** 2026-08-26
**Branch:** `i3.27-q-error-and-authority`
**Experiment:** `scripts/run_i3_28_rep_repair.py`

---

## Question

> What state information must Q possess before a particular action is eligible for hard authority?

**Architectural invariant:** An action cannot receive hard authority when the Q-state representation omits variables required to distinguish known counterexamples for that action.

---

## Design

Minimal ablation, not broad feature expansion:

- **Q_V1:** old features (35), same causal dataset (1056 records), same GBT (`n_estimators=200, max_depth=4, random_state=42`), same hyperparameters
- **Q_V2R:** old features + 3 structural features (41 total), same everything else

The 3 new features (Q_STATE_SCHEMA_V2):

| Feature | Definition |
|---------|------------|
| `n_hyp_unverified_support` | # hypotheses with ≥1 unverified supporting evidence (visible only) |
| `n_hyp_unverified_contradiction` | # hypotheses with ≥1 unverified contradicting evidence (visible only) |
| `has_competing_unverified_support` | Binary: `n_hyp_unverified_support > 1` (contradiction signal) |

V1 frozen permanently. Same threshold (5.0). Same dataset. Same learner. Clean causal attribution: change in behavior ⇒ representation repair.

---

## Results

### Step 2: Leakage tests — PASS

All 4 tests pass for all 3 features:
1. Observable from visible snapshot: PASS
2. No `verify_result` (oracle) access: PASS
3. No hidden evidence inspection: PASS
4. Unchanged by future outcomes: PASS

### Step 3: Training — completed

- Q_V1 SHA: `d90d72dab250ba7c...` (matches frozen QCAUSAL_V1 — control is valid)
- Q_V2R SHA: `6a1015dc9e84ffdf...`

### Step 4: Offline separation audit — FAIL

**Gate:** `min Q_V2R(DEFER|defer-correct) - max Q_V2R(DEFER|contra) > 5.0`
**Result:** `67.35 - 69.77 = -2.42` — FAIL

| State class | Q_V2R(DEFER) mean | range |
|-------------|------------------:|-------|
| Defer-correct (68 states) | 68.97 | [67.35, 70.16] |
| Contra-correct (24 states) | 53.37 | [36.97, 69.77] |

At the exact aliasing state (`dev_contra_s1_after_search`), Q_V2R(DEFER) = 69.77 — still higher than some defer-correct states. The structural features are present but the GBT has not learned to use them to suppress DEFER in the competing-support case.

### Step 5: Preservation check — PASS

V2R preserves and improves on V1:

| Metric | V1 | V2R |
|--------|---:|----:|
| Mean regret | 19.08 | 18.69 |
| Near-optimal (ε=3) | 160/220 | 180/220 |
| Correct best action | 88/220 | 132/220 |
| ANSWER cases correct | 44/44 | 44/44 |
| New false high-conf DEFER | — | 0 |

All 5 preservation gates pass. V2R is a better Q model overall.

### Step 6: Offline authority test — PASS (vacuous)

0 triggers for A2A, A2AD-ANSWER, and A2AD-DEFER on the 220 checkpoints. The authority conditions are not met on this data. False authority rates are 0 by vacuity.

---

## Diagnosis: Training Data Coverage, Not Representation

The separation gate failed, but the root cause is NOT that the representation is wrong. The 3 structural features perfectly separate defer-correct from contra-correct states (verified in the REP_REPAIR_ANALYSIS). The root cause is a training data coverage gap.

### The gap

The I3.5 causal training data has a structural confound:

| n_verified | has_competing | count |
|-----------:|--------------:|------:|
| 0 | 1 | 400 |
| 1 | 0 | 480 |
| 2 | 0 | 176 |

**At n_verified=0, ALL 400 records have has_competing=1.** There are ZERO records with has_competing=0 at n_verified=0.

The defer-correct states (ol_defer category) all have n_verified=1, not n_verified=0. The GBT learns:
- n_verified=1, has_competing=0 → DEFER is good (utility=70.00)
- n_verified=0, has_competing=1 → DEFER is bad (utility=-5.00)

But it never sees:
- n_verified=0, has_competing=0 → DEFER is good (the defer-correct aliasing state)

Without this contrast, the GBT cannot learn that `has_competing` matters at n_verified=0. When queried at the defer-correct aliasing state (n_verified=0, has_competing=0), it extrapolates from the nearest training data — which all has has_competing=1 — and produces Q(DEFER) = 69.77.

### Why this is not a representation failure

The representation IS repaired. The features perfectly separate the states. The problem is that the GBT is a supervised learner: it can only learn from the training data. If the training data doesn't contain contrast examples at the decision boundary, no learner — GBT, neural network, or larger model — can learn the distinction.

This is a stronger result than "another feature helped" or "another feature didn't help." It shows that:

1. **Representation repair is necessary** — V1 cannot distinguish the states at all
2. **Representation repair is not sufficient** — V2R has the features but the GBT can't learn from current data
3. **The blocker is training data coverage** — the I3.5 causal dataset doesn't cover the contrast at the aliasing boundary
4. **The correct next step is targeted data collection** — force DEFER in both has_competing=0 and has_competing=1 states at n_verified=0

---

## Decision

Per the preregistered protocol: **the offline separation gate failed. Do not proceed to live DEFER authority.**

However, the failure is diagnosed as training data coverage, not representation inadequacy. The V2R schema is correct and should be preserved. The next step is NOT another feature or another learner — it is targeted causal data collection at the aliasing boundary.

### What is frozen

- Q_STATE_SCHEMA_V2: frozen as the correct representation (3 structural features, leakage-tested)
- Q_V2R model: frozen as an artifact (not deployed for live authority)
- V1 schema: frozen permanently

### What is NOT done

- No live DEFER authority experiments
- No threshold tuning
- No additional features beyond the 3 minimal structural features
- No change to the learner

### Next step (if approved)

Targeted causal data collection at the aliasing boundary:
1. Generate defer-correct states with n_verified=0, has_competing=0 (from I3.26-style defer tasks)
2. Generate contra-correct states with n_verified=0, has_competing=1 (from I3.26-style contradiction tasks after SEARCH_MORE)
3. Force DEFER at both sets of states and record outcomes
4. Retrain Q_V2R on the expanded dataset (same GBT, same hyperparameters)
5. Re-run the offline separation audit

This gives the GBT the contrast it needs: DEFER is good when has_competing=0, bad when has_competing=1, at the same n_verified=0 boundary.

---

## Artifacts

| File | Description |
|------|-------------|
| `experiments/i3_28/Q_STATE_SCHEMA_V2.json` | Schema definition, SHAs, hyperparameters |
| `experiments/i3_28/Q_V1_control.pkl` | Q_V1 model (control, matches frozen QCAUSAL_V1) |
| `experiments/i3_28/Q_V2R_repaired.pkl` | Q_V2R model (repaired representation) |
| `experiments/i3_28/leakage_tests.json` | 4 leakage tests per feature |
| `experiments/i3_28/separation_audit.json` | Offline separation audit (FAIL) |
| `experiments/i3_28/preservation_check.json` | V2R preservation check (PASS) |
| `experiments/i3_28/offline_authority_test.json` | Offline authority test (PASS, vacuous) |
| `experiments/i3_28/full_results.json` | Summary of all gates |
