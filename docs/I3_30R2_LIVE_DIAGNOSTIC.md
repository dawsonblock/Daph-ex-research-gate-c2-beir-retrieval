# I3.30R2 LIVE DEVELOPMENT — DIAGNOSTIC REPORT

**Status: REJECTED / DIAGNOSTIC**
**Date: 2026-08-29**
**Run: Colab T4, 370 trajectories, 24.2 min**
**Commit: 5d60bde**

## Provenance

This is a **development live run**, not a preregistered confirmation run.

- **Runner**: Patched on Colab to use V3R2-A artifacts (fixes committed to git as `6ae399e`, `44528fc`, `5d60bde`)
- **Model**: Qwen2.5-7B-Instruct Q4_K_M from DhruvalLabs mirror
  - Colab SHA: `35f9f55b0c7cdd52...`
  - Local SHA: `65b8fcd92af6b4fe...`
  - **NOT byte-identical**. 5-packet behavioral smoke test passed. This is a backend compatibility smoke test, not model equivalence. Insufficient for a confirmation experiment, especially near action boundaries.
- **Three bugs were fixed during this run** (not part of the original freeze):
  1. Runner used buggy `compute_v3_features` instead of `compute_v3_features_canonical`
  2. Executor's `_check_defer_success` didn't check `answer_action` of supported hypothesis
  3. DEFER certificate didn't check continuation admissibility

## Gate Results: 12/12 PASS

All 12 pre-registered gates pass numerically. **This does not validate the intended mechanism.** The gates measure aggregate outcomes, not isolated causal contributions of the authority mechanism.

## Authority Event Accounting

### V1 authority (confirmed champion)
- Mode: `A2A_hard_select` (84 events)
- **All 84 force ANSWER** — V1 is ANSWER-only hard authority
- V1 forces ANSWER on:
  - D4: 33 (correct — D4 expected terminal is ANSWER)
  - D5: 35 (V1 succeeds on all 35 — see D5 discussion below)
  - D2: 8 (**V1 forces ANSWER where expected terminal is DEFER → all 8 fail**)
  - D3: 8 (mixed)

### V3 authority (V3R2-A candidate)
- Modes: `A2AD_hard_ANSWER` (78), `A2AD_hard_DEFER` (3)
- V3 forces ANSWER on:
  - D4: 31 (correct)
  - D5: 31
  - D3: 16
- V3 forces DEFER on:
  - D2: 3

### Event semantics

The 81 V3 hard authority events are genuine terminal overrides (certificate passed, Q gap exceeded threshold, action forced). The `authority_mode` field was not recorded in the event log (shows `?`), but cross-referencing with trajectory `hard_force_count` confirms all 81 events correspond to `A2AD_hard_*` modes in the authority log.

**However, `hard_auth_breaks = 0` only means no forced action resulted in `TERMINAL_WRONG`.** It does not establish that every intervention was beneficial. A forced action that matches what the model would have done anyway provides no causal value. A forced terminal action that succeeds might still yield lower utility than a continuation that would have resolved more evidence. The current logging does not record the model's proposed action at authority events, so forced-vs-shadow counterfactuals cannot be computed from this run.

## D5: Task Terminal vs Decision-State Truth

D5 tasks have `expected_terminal=ANSWER`. The study documentation describes D5 as "CONTINUE — ambiguous post-verification state." These are not necessarily contradictory.

A task can legitimately be:

```
initial state:
    CONTINUE / VERIFY required (competing verified support)
after discriminator verification:
    ANSWER_READY (unique supported hypothesis)
eventual expected_terminal:
    ANSWER
```

**`expected_terminal=ANSWER` does not by itself imply that the D5 decision state is ANSWER_READY.** Whether the targeted D5 state is CONTINUE-required must be established from the causal action values and topology at that state, not inferred from the task-level terminal outcome.

The diagnostic from the previous version of this report incorrectly concluded "D5 is NOT a CONTINUE-correct stratum." That inference was too strong. The correct statement is:

> D5's task-level terminal outcome is ANSWER. Whether the targeted D5 decision state is CONTINUE-required must be established from the causal action values and topology at that state rather than inferred from `expected_terminal`.

This distinction matters because it determines whether V1's forced ANSWER on D5 is semantically correct or merely exploiting the evaluator.

## First-Divergence Analysis: 6 Breaks

### Break classification

| Task | Stratum | Bucket | V1 | V3 |
|------|---------|--------|-----|-----|
| d1_0004 | D1 | BOTH_ADVISORY | DEFER→SUCCESS | REASON_MORE→FAIL |
| d1_0010 | D1 | BOTH_ADVISORY | DEFER→SUCCESS | REASON_MORE→FAIL |
| d2_0003 | D2 | BOTH_ADVISORY | DEFER→SUCCESS | REASON_MORE→FAIL |
| d3_0022 | D3 | BOTH_ADVISORY | VERIFY,VERIFY,ANSWER→SUCCESS | VERIFY,RETRIEVE,VERIFY→STEP_LIMIT |
| d3_0038 | D3 | BOTH_ADVISORY | VERIFY,VERIFY,ANSWER→SUCCESS | VERIFY,RETRIEVE,VERIFY→STEP_LIMIT |
| d5_0026 | D5 | V1_HARD_V3_ADVISORY | VERIFY,ANSWER(forced)→SUCCESS | VERIFY,REASON_MORE→FAIL |

### Defect decomposition

**Problem A — Q/advisory regression (5 breaks)**

Tasks: d1_0004, d1_0010, d2_0003, d3_0022, d3_0038

All five are BOTH_ADVISORY: V3R2-A's Q model produces different Q values → different epsilon/refined set → LLM makes a different (worse) choice. No hard authority involved at the divergence point.

- D1 breaks: V3 refined set is REASON_MORE (21/35) vs V1's DEFER (31/35). V3R2-A overvalues REASON_MORE in resource-exhausted states.
- D2 break: V3's refined set includes both DEFER and REASON_MORE; LLM chooses REASON_MORE.
- D3 breaks: V3 chooses RETRIEVE at step 1 where V1 chooses VERIFY, leading to step exhaustion.

**Fix target: Q_V3R2 training/support/calibration. NOT certificate relaxation.** D1 has no certificate firing at all, so relaxing certificates cannot fix d1_0004 or d1_0010.

**Problem B — possible authority undercoverage (1 break)**

Task: d5_0026

V1 forces ANSWER at step 1 (Q gap = 12.79, A2A_hard_select). V3 stays advisory (certificate or Q gap failed), LLM chooses REASON_MORE → fails.

**Fix target: determine whether the certificate SHOULD have fired using causal state truth before changing anything.** The following must be answered for the divergence state s in d5_0026:

1. What is the canonical topology at s?
2. Is `ANSWER_READY(s)` actually true?
3. What is `Q*(s, ANSWER)` vs `Q*(s, REASON_MORE)` vs `Q*(s, VERIFY)`?
4. Would forced ANSWER succeed at s?
5. Would advisory REASON_MORE fail at s?
6. Did the certificate fail structurally, or did the Q gap fail threshold?

The repair depends on the answer:

- If `certificate = true` but `Q gap < threshold` → confidence/threshold issue
- If `certificate = false` but canonical state really is `ANSWER_READY` → certificate recall bug
- If `certificate = false` and state is still unresolved → V1's forced ANSWER may exploit the evaluator, and V3 may be semantically more correct despite losing the task

**Do not relax the certificate until this diagnosis is complete.**

## Rescue Mechanism: 15 Rescues

### Rescue classification

| Bucket | Count | Mechanism |
|--------|-------|-----------|
| V3_HARD_RESCUE | 10 | V3 certificate forces ANSWER, V1 didn't force (or model chose wrong) |
| BOTH_HARD_DIFFERENT_ACTION | 3 | V1 forced ANSWER (wrong), V3 forced DEFER (correct) — D2 |
| V1_HARD_V3_ADVISORY | 2 | V1 forced ANSWER (wrong), V3 advisory → model chose DEFER (correct) — D2 |

### Causal attribution limitation

The 10 V3_HARD_RESCUE events are **associated with** successful rescue trajectories, but the incremental causal contribution of the authority intervention has **not been isolated**.

The comparison changes multiple things simultaneously:

```
V1:  Q_V1     + V1 epsilon guidance     + V1 authority
V3:  Q_V3R2   + V3 epsilon guidance     + V3 certificate authority
```

One rescue example demonstrates the confounding:

```
D3_0006:
  V1:  RETRIEVE → VERIFY → VERIFY → fail
  V3:  VERIFY → VERIFY → ANSWER(hard) → success
```

The trajectory diverged at **step 0** (RETRIEVE vs VERIFY), before the V3 ANSWER certificate fired at step 2. The actual causal path could be:

```
different Q model
    → better initial VERIFY
    → better epistemic state
    → certificate fires
    → ANSWER
```

The certificate participates in the successful mechanism, but it is not isolated from the Q-model and advisory differences that also changed.

**Current evidence supports:**

> V3 certificate authority is associated with successful rescue trajectories, but its incremental causal contribution has not yet been isolated from Q-model and advisory guidance differences.

## D1 Regression: V1 8/35 → V3 6/35

D1 has no verify budget → no verified evidence → no certificates can fire. D1 is purely advisory.

- V1 refined set at step 0: DEFER (31/35 tasks)
- V3 refined set at step 0: DEFER (14/35), REASON_MORE (21/35)

V3R2-A's Q model ranks REASON_MORE higher than DEFER on many D1 tasks. This changes the epsilon set, and the LLM chooses REASON_MORE instead of DEFER. Since D1 has no useful continuations, REASON_MORE wastes steps and fails.

**Root cause**: V3R2-A was trained on data where DEFER was correct primarily after verification. On D1 (no verification possible), the model doesn't confidently recommend DEFER. This is a Q-model training gap, separate from the certificate mechanism.

## Causal Picture

The intended mechanism:
```
canonical topology → positive terminal certificate → correct hard ANSWER/DEFER → rescues
```

The actual picture (confounded):
```
V3R2-A Q model
      ↓
  different Q values
      ↓
  ┌──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
different epsilon/refined set    certificate fires
      ↓                                  ↓
LLM makes different choices      hard ANSWER/DEFER force
      ↓                                  ↓
  ┌─────┐                         ┌─────┐
  │     │                         │     │
  ▼     ▼                         ▼     ▼
D1:    D3/D5:                  D3: 10 rescues    D2: 3 rescues
worse  mixed                   (associated,     (associated,
       results                  not isolated)    not isolated)
```

## What Has Been Demonstrated

**Demonstrated:**
- Canonical V3 authority can execute live
- ANSWER and DEFER certificates fire in real trajectories
- 81 certificate-qualified interventions with zero observed terminal-wrong outcomes
- Overall V3 behavioral improvement over V1 (+9 success, +6.86 mean paired ΔU)
- Lower premature termination (V3: 5, V1: 6)
- Clear D2/D3 behavior improvements

**NOT yet demonstrated:**
- Isolated causal benefit of adaptive authority (confounded with Q-model differences)
- Structural generalization of authority
- Exact-model confirmation (different GGUF SHA)
- That D5's decision state is ANSWER_READY rather than CONTINUE-required
- That relaxing certificates is safe
- That V3 should replace V1

## I3.30R3: Authority-Isolation Experiment

The next phase should isolate the causal effect of adaptive authority, not attempt another repair cycle.

### Three-arm design

| Arm | Q model | Guidance | Hard terminal authority |
|-----|---------|----------|------------------------|
| V1 | Q_V1 | V1 epsilon | ANSWER-only V1 |
| V3-SHADOW | Q_V3R2 | V3R2 epsilon | OFF (certificates evaluated/logged, never override) |
| V3-A | Q_V3R2 | V3R2 epsilon | V3 certificates (hard override) |

### Decomposition

```
V3-SHADOW vs V1     → effect of new state representation / Q / advisory interface
V3-A vs V3-SHADOW   → incremental adaptive-authority value (the isolated ATE)
```

The key equation:

```
ATE_authority = E[U | Q_V3R2, A_hard] - E[U | Q_V3R2, A_shadow]
```

### Counterfactual event logging

At every potential authority state, record:

- `state_sha`
- LLM proposed action
- V3 epsilon set
- Q values for all legal actions
- certificate evaluation result
- forced action (if any)
- counterfactual shadow action (what would have happened without force)
- counterfactual immediate utility
- counterfactual rollout utility

Each authority intervention can then be classified as:

| Classification | Definition |
|---------------|------------|
| RESCUE | Force succeeds, shadow would have failed |
| BENEFICIAL_NONRESCUE | Force succeeds, shadow would have succeeded with lower utility |
| NEUTRAL | Force and shadow produce same outcome |
| HARMFUL_NONBREAK | Force produces lower utility than shadow would have |
| BREAK | Force fails, shadow would have succeeded |

### Phase plan

1. **Re-audit D5 state-level causal truth** — For d5_0026 and representative D5 tasks, compute Q* and topology at the decision state. Determine whether the state is ANSWER_READY or CONTINUE-required.
2. **Freeze V3R2 exactly as it is.** Do not retrain.
3. **Add V3-SHADOW arm** — same Q, same I2, same prompts, same certificates evaluated, no hard override.
4. **Run V3-SHADOW vs V3-A** on the existing diagnostic tasks first.
5. **At every potential authority state, perform paired counterfactual rollout** — force / don't force.
6. **Compute**: authority rescue rate, authority break rate, authority ΔU, authority precision, authority coverage, certificate recall, certificate false-positive rate.
7. **Only then decide** whether the next change belongs in Q, certificate, threshold, or benchmark.

## V1 Remains the Confirmed Champion

V3R2-A has passed an important development milestone. The certificate mechanism fires in live trajectories and is associated with successful outcomes. But the causal benefit of adaptive authority has not been isolated from Q-model and advisory differences, and the D5 decision-state truth has not been established. V1 remains confirmed. V3R2-A is not promoted.
