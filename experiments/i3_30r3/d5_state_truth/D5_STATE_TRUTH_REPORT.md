# I3.30R3 Phase 8: D5 State-Level Causal Truth Audit

**Date: 2026-08-29**
**Branch: `i3.30r3-authority-isolation`**

## Executive Summary

The D5 state-truth audit confirms:

1. **D5 initial state IS CONTINUE_REQUIRED** (all 35 tasks) — the user's semantic correction is confirmed. `expected_terminal=ANSWER` is the eventual task terminal, not the initial decision-state truth.
2. **After VERIFY(discriminator), D5 becomes ANSWER_READY** (all 35 tasks) — the transition is real and consistent.
3. **For d5_0026, the V3 certificate PASSES at the answer-ready state** — there is no certificate recall bug and no Q-gap threshold failure.
4. **The d5_0026 break is a trajectory-level divergence in VERIFY target selection**, not a certificate or authority issue.

## D5 Task Structure

Each D5 task has:
- E1: SUFFICIENT + supports(H1) — verified support for correct hypothesis
- E2: SUFFICIENT + supports(H2) — verified support for competing hypothesis
- E3: UNVERIFIED + contradicts(H2) — discriminator that can eliminate H2
- E4/E5 (optional): UNVERIFIED + supports(H1 or H2) — non-discriminating evidence
- H1: answer_action = ANSWER (correct)
- H2: answer_action = DEFER (wrong)
- expected_terminal = ANSWER
- oracle = ("VERIFY:E3", "ANSWER")

## State Transition Audit (all 35 tasks)

| State | Terminal Readiness | is_answer_ready | has_verified_unresolved_competition |
|-------|-------------------|-----------------|-------------------------------------|
| s0 (initial) | CONTINUE_REQUIRED | False | True |
| s1 (after VERIFY E3) | ANSWER_READY | True | False |

All 35 tasks show the same pattern:
- s0: competing verified support → CONTINUE_REQUIRED
- s1: H2 eliminated by discriminator → H1 uniquely supported → ANSWER_READY

## d5_0026 Deep Dive

### State 0 (initial)

```
Terminal readiness: CONTINUE_REQUIRED
is_answer_ready: False
has_verified_unresolved_competition: True
valid_verify_targets: ['E3', 'E4', 'E5']

Q_V1:  DEFER=69.13, REASON_MORE=62.30, STOP=56.57, VERIFY=56.54, ANSWER=8.13
Q_V3R: REASON_MORE=75.56, VERIFY=69.89, DEFER=23.59, STOP=-14.33, ANSWER=-116.74

V1 would force: False (Q argmax is DEFER, not ANSWER)
V3 would force: False (certificate fails: no unique verified support)

ANSWER at s0: FAILS (state is not answer_ready)
VERIFY at s0: non-terminal (correct — CONTINUE is required)
```

### State 1 (after VERIFY discriminator E3)

```
Terminal readiness: ANSWER_READY
is_answer_ready: True
unique_supported_hypothesis: H1
has_verified_unresolved_competition: False

Q_V1:  ANSWER=82.45, DEFER=69.66, REASON_MORE=55.42, STOP=55.04
Q_V3R: ANSWER=99.78, REASON_MORE=78.59, DEFER=-13.13, STOP=-38.31

V1 would force: True (ANSWER, gap=12.79 >= 5.0, sole near-optimal)
V3 would force: True (certificate passes, gap=21.20 >= 5.0, sole near-optimal)
V3 certificate passed: True
V3 certificate type: unique_verified_support_answer

ANSWER at s1: SUCCEEDS
```

### Critical finding: VERIFY target matters

| VERIFY target | Resulting state | is_answer_ready | ANSWER succeeds |
|---------------|----------------|-----------------|-----------------|
| E3 (discriminator) | H2 eliminated | True | True |
| E4 (supports H2) | H2 still supported | False | False |
| E5 (supports H1) | Both still supported | False | False |

Only VERIFY(E3) resolves the competition. VERIFY(E4) or VERIFY(E5) leaves
the state with competing verified support.

## Diagnostic Answers for d5_0026

**Q1: Canonical topology at divergence state (s1, after VERIFY)?**
- n_supported: 1 (H1)
- n_contradicted: 1 (H2)
- unique_supported_hypothesis: H1
- has_verified_unresolved_competition: False

**Q2: Is ANSWER_READY(s1) actually true?**
- Yes. is_answer_ready = True, terminal_readiness = ANSWER_READY

**Q3: Q values at s1?**
- Q_V1: ANSWER=82.45, DEFER=69.66, REASON_MORE=55.42
- Q_V3R: ANSWER=99.78, REASON_MORE=78.59, DEFER=-13.13
- Both rank ANSWER highest with sufficient gap

**Q4: Would forced ANSWER succeed at s1?**
- Yes. Simulated ANSWER returns TASK_SUCCESS.

**Q5: Would advisory REASON_MORE fail at s1?**
- REASON_MORE is non-terminal. It doesn't immediately fail, but it wastes
  a step and doesn't resolve the state. If the trajectory then runs out of
  steps or the LLM never returns to ANSWER, it fails.

**Q6: Did the certificate fail structurally, or did the Q gap fail threshold?**
- Neither. At the simulated s1 state (after VERIFY E3):
  - Certificate passes: True
  - Q gap: 21.20 (well above 5.0 threshold)
  - V3 would force: True
  - V3 forced action: ANSWER

## Root Cause of the d5_0026 Break

The certificate mechanism is **correct at the state-truth level**. The
d5_0026 break is NOT caused by:

- Certificate recall failure (certificate passes at answer-ready state)
- Q-gap threshold failure (Q gap = 21.20 >> 5.0)
- Certificate being too restrictive (it fires correctly when the state is answer-ready)
- V1 exploiting the evaluator (V1's ANSWER is semantically correct at s1)

The break is caused by **trajectory-level divergence in VERIFY target selection
at step 0**:

- V1's LLM verified E3 (the discriminator) → state became answer_ready → V1 forced ANSWER → success
- V3's LLM verified E4 or E5 (non-discriminator) → state remained competing → V3 certificate correctly abstained → LLM chose REASON_MORE → failure

Both arms chose VERIFY at step 0, but the LLM selected different evidence
targets. The Q values and guidance differ between V1 and V3 at s0, which
may influence the LLM's target selection through the guidance fields.

## Classification

This is **Problem A (Q/advisory regression)**, not Problem B (authority undercoverage):

- The certificate mechanism works correctly
- The break is caused by different LLM behavior under different Q/guidance
- Fix target: Q_V3R2 training/support/calibration or guidance design
- NOT certificate relaxation

## Implications for I3.30R3

1. **D5 is correctly designed** — CONTINUE_REQUIRED at s0, ANSWER_READY at s1
2. **The V3 certificate is semantically correct** — fires at answer-ready, abstains at continue-required
3. **The d5_0026 break will appear in the I3.30R3 three-arm study** as a V3-SHADOW vs V3-AUTH difference of zero (both arms would fail if the LLM verifies the wrong target) and a V3-SHADOW vs V1 difference (V1's LLM happened to choose the right target)
4. **The authority-isolation experiment will correctly attribute this break** to Q/advisory differences, not to authority
5. **No certificate change is needed** — the certificate is working as designed

## Files

- `experiments/i3_30r3/d5_state_truth/d5_state_audit.json` — full per-task audit
- `experiments/i3_30r3/d5_state_truth/d5_readiness_summary.json` — summary
- `scripts/audit_d5_state_truth.py` — audit script
