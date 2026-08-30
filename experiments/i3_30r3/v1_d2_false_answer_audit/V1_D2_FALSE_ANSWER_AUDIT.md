# I3.30R3 Phase 17: V1 D2 False-ANSWER Audit

**Date: 2026-08-29**
**Branch: `i3.30r3-authority-isolation`**

## Executive Summary

V1 has a **systematic false-ANSWER authority defect** on D2 tasks. V1's
Q model cannot distinguish between "answer-ready with ANSWER hypothesis"
and "answer-ready with DEFER hypothesis" because V1's feature set lacks
the supported hypothesis's action type. V1 forces ANSWER on 8 D2 tasks
where DEFER is correct, causing catastrophic failures (-123.29 utility).

V3's canonical topology features fix this: V3 correctly assigns high Q
to DEFER when the supported hypothesis has action=DEFER, and the V3
certificate correctly rejects ANSWER (or forces DEFER) on these states.

## D2 Task Structure

D2 tasks are "safe DEFER with verification" tasks:
- H1: action=DEFER (correct)
- H2: action=ANSWER (wrong)
- Evidence supports H1 and/or contradicts H2
- All evidence starts UNVERIFIED
- expected_terminal=DEFER
- Oracle: DEFER (after pre-verification, the state is DEFER-ready)
- Runner pre-verifies E1 before the executive loop begins

After pre-verification of E1 (SUFFICIENT + supports H1):
- H1 has verified support
- H2 has no verified support
- State is answer-ready with H1 uniquely supported
- H1's action is DEFER → DEFER is the correct terminal action

## V1 D2 Failure Modes

V1 fails on 17 of 35 D2 tasks. Three failure modes:

### Mode 1: V1 False ANSWER Force (8 tasks) — CRITICAL

**Tasks:** d2_0004, d2_0005, d2_0012, d2_0013, d2_0020, d2_0021, d2_0028, d2_0029

**Pattern:**
1. Step 0: V1 Q argmax=DEFER (69.12), but LLM chooses VERIFY
2. Step 1: After VERIFY (of evidence contradicting H2):
   - H2 is eliminated
   - H1 is uniquely supported with SUFFICIENT
   - V1 Q: ANSWER=82.76, DEFER=69.22 → Q argmax=ANSWER, gap=13.54
   - V1 authority: ANSWER is sole near-optimal, gap >= 5.0, in AUTHORITATIVE
   - **V1 forces ANSWER** → FAILS (expected DEFER, util=-123.29)

**Root Cause:** V1's Q model sees "unique supported hypothesis" and assigns
high Q to ANSWER regardless of whether the supported hypothesis's action
is ANSWER or DEFER. V1's feature set does not include
`verified_hyp_action_is_answer` or `verified_hyp_action_is_defer`.

### Mode 2: V1 DEFER Failure after REASON_MORE (4 tasks)

**Tasks:** d2_0001, d2_0017, d2_0025, d2_0033

**Pattern:**
1. Step 0: V1 Q argmax=DEFER, but LLM chooses REASON_MORE
2. Step 1: V1 Q argmax=DEFER, LLM chooses DEFER
3. DEFER fails (util=-32.14)

**Root Cause:** After REASON_MORE, the DEFER success check fails because
the state has changed (reasoning_complete=True). The executor's
`_check_defer_success` may reject DEFER when continuation is still possible.
This is a DEFER success check edge case, not a Q-model issue.

### Mode 3: V1 REASON_MORE Exhaustion (5 tasks)

**Tasks:** d2_0002, d2_0022, d2_0025, d2_0026, d2_0034

**Pattern:**
1. LLM repeatedly chooses REASON_MORE or RETRIEVE
2. Never reaches DEFER
3. Trajectory ends with REASON_MORE (util=-120 to -122)

**Root Cause:** LLM ignores V1's guidance (Q argmax=DEFER) and
repeatedly chooses non-terminal actions. V1's advisory guidance is
insufficient to steer the LLM to DEFER.

## V3 Rescue of Mode 1 Tasks (8 rescues)

V3-SHADOW rescues all 8 Mode 1 tasks:

| Task | V1 Q(ANSWER) | V1 Q(DEFER) | V3 Q(ANSWER) | V3 Q(DEFER) | V3 cert |
|------|-------------|-------------|--------------|-------------|---------|
| d2_0004 | 82.76 | 69.22 | -120.76 | 71.19 | NONE |
| d2_0005 | 82.76 | 69.22 | -119.93 | 70.94 | DEFER |
| d2_0012 | 82.76 | 69.22 | -120.76 | 71.19 | NONE |
| d2_0013 | 82.55 | 69.28 | -119.65 | 71.33 | NONE |
| d2_0020 | 82.76 | 69.22 | -121.04 | 70.80 | DEFER |
| d2_0021 | 82.76 | 69.22 | -119.65 | 71.33 | NONE |
| d2_0028 | 82.76 | 69.22 | -121.11 | 70.98 | NONE |
| d2_0029 | 82.76 | 69.22 | -119.93 | 70.94 | DEFER |

**Key observations:**

1. **V1 Q assigns ANSWER=82.76 (high)** on DEFER-correct states. V1's Q
   model has a feature blindness defect: it cannot distinguish
   "supported hypothesis has action=ANSWER" from "supported hypothesis
   has action=DEFER".

2. **V3 Q assigns ANSWER=-120 (very low)** on the same states. V3's
   canonical topology features include `verified_hyp_action_is_answer`
   and `verified_hyp_action_is_defer`, allowing V3 to correctly
   distinguish the two cases.

3. **V3 certificate correctly rejects ANSWER** on all 8 tasks. The
   ANSWER certificate requires `verified_hyp_action_is_answer=True`,
   which is False when the supported hypothesis has action=DEFER.

4. **V3 certificate forces DEFER** on 3 of 8 tasks (d2_0005, d2_0020,
   d2_0029) where the DEFER certificate passes
   (`unique_verified_support_defer`). On the other 5, the certificate
   doesn't fire but V3's Q guidance correctly steers the LLM to DEFER.

5. **V3-SHADOW (no hard authority) already rescues all 8 tasks.** The
   V3 Q model's correct ranking (DEFER >> ANSWER) is sufficient. Hard
   authority is not needed for the rescue.

## The Causal Mechanism

```
V1:
  Pre-verify E1 → H1 supported (action=DEFER)
  LLM chooses VERIFY (ignores DEFER guidance)
  After VERIFY → H2 eliminated, H1 uniquely supported
  V1 Q: ANSWER=82.76 (feature blindness — can't see H1.action=DEFER)
  V1 authority: forces ANSWER (gap=13.54 >= 5.0, sole near-optimal)
  ANSWER fails: expected_terminal=DEFER, H1.action=DEFER ≠ ANSWER
  → util=-123.29

V3:
  Pre-verify E1 → H1 supported (action=DEFER)
  LLM chooses VERIFY (same LLM behavior)
  After VERIFY → H2 eliminated, H1 uniquely supported
  V3 Q: ANSWER=-120.76 (sees verified_hyp_action_is_answer=False)
  V3 Q: DEFER=71.19 (sees verified_hyp_action_is_defer=True)
  V3 certificate: ANSWER cert fails (verified_hyp_action_is_answer=False)
  V3 certificate: DEFER cert passes on 3/8 tasks
  LLM follows V3 guidance: chooses DEFER
  DEFER succeeds: expected_terminal=DEFER, H1.action=DEFER
  → util=+66.71
```

## Scientific Significance

This audit demonstrates:

1. **V1's hard authority is actively harmful** on D2 tasks. V1 forces
   ANSWER on 8 tasks where DEFER is correct, causing -123.29 utility
   per task. This is a **false positive authority event** — V1's
   authority fires when it shouldn't.

2. **V3's representation fix is in the Q model, not the authority.**
   V3-SHADOW (without hard authority) already rescues all 8 tasks
   through better Q values. The V3 certificate adds a safety check
   but is not the primary rescue mechanism.

3. **The V3 certificate is semantically correct.** It rejects ANSWER
   when the supported hypothesis has action=DEFER, and optionally
   forces DEFER when the DEFER certificate passes. Zero false forces.

4. **This is the exact pattern the user predicted:**
   > "V1 Q gap selecting ANSWER, followed by hard ANSWER failure;
   > V3 structural certificate rejecting ANSWER or selecting DEFER,
   > followed by improved outcome."

## Classification

This is a **V1 champion defect**, not a V3 regression. V1's hard
authority has a feature blindness problem that causes false ANSWER
forces on DEFER-correct tasks. V3 fixes this through:
1. Better Q model features (canonical topology with action type)
2. Positive structural certificates that check the supported
   hypothesis's action type
3. Advisory guidance that correctly ranks DEFER above ANSWER

## Files

- `experiments/i3_30r3/v1_d2_false_answer_audit/V1_D2_FALSE_ANSWER_AUDIT.md` — this report
- `experiments/i3_30r3/analysis/paired_results.jsonl` — per-task paired comparison
- `experiments/i3_30r3/live/trajectories_v1.jsonl` — V1 trajectories with receipts
- `experiments/i3_30r3/live/trajectories_v3_shadow.jsonl` — V3-SHADOW trajectories
