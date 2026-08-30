# I3.30R3 Phase 16: D1 Q-Error Audit

**Date: 2026-08-29**
**Branch: `i3.30r3-authority-isolation`**

## Executive Summary

The 2 D1 breaks (d1_0004, d1_0012) are caused by a **V3 Q-model calibration
defect**: V3's Q model systematically undervalues DEFER and overvalues
REASON_MORE on D1 tasks. This is a Q training/calibration issue, not a
certificate or authority issue.

## D1 Task Structure

D1 tasks are "safe DEFER" tasks:
- H1: action=DEFER (correct)
- H2: action=ANSWER (wrong)
- Evidence supports H1 and/or contradicts H2
- expected_terminal=DEFER
- Oracle: DEFER (immediate, no verification needed)

## The 2 D1 Breaks

### d1_0004 and d1_0012 (identical Q values — same task structure)

| Arm | Q(DEFER) | Q(REASON_MORE) | Q argmax | Action | Outcome |
|-----|----------|----------------|----------|--------|---------|
| V1 | 69.42 | 62.26 | DEFER | DEFER | SUCCESS (+69.89) |
| V3-SHADOW | 44.16 | 49.01 | REASON_MORE | REASON_MORE | FAIL (-120.0) |
| V3-AUTH | 44.16 | 49.01 | REASON_MORE | REASON_MORE | FAIL (-120.0) |

### Root Cause

V3's Q model assigns:
- Q(REASON_MORE) = 49.01 > Q(DEFER) = 44.16

V1's Q model assigns:
- Q(DEFER) = 69.42 > Q(REASON_MORE) = 62.26

The V3 Q model ranks REASON_MORE above DEFER on a D1 task where DEFER is
the correct immediate terminal action. The LLM follows V3's guidance and
chooses REASON_MORE, which is non-terminal. The trajectory then runs out
of steps or the LLM never returns to DEFER.

### Why V3's Q Model is Wrong Here

D1 tasks have a simple structure: the correct action is DEFER immediately.
No verification or reasoning is needed. The V3 Q model was trained on a
richer feature set (53 features including canonical topology features)
that may overfit to patterns where REASON_MORE has value. On D1 tasks
where the state is already DEFER-ready, the V3 Q model incorrectly
assigns value to REASON_MORE.

V1's Q model, with its simpler feature set, correctly identifies DEFER
as the highest-value action.

### Why Authority Cannot Fix This

- No certificate fires on D1 tasks at step 0
- The state is DEFER-ready but the V3 DEFER certificate requires:
  - verified hypothesis action = DEFER, OR
  - eliminated hypotheses with at most one viable, OR
  - legacy resource-exhausted/no-support conditions
- The V3 Q model's ranking (REASON_MORE > DEFER) means DEFER is not
  the sole near-optimal action, so even if the certificate passed,
  the Q gap threshold would not be met for DEFER
- This is a Q/advisory defect, not a certificate precision defect

## Classification

**Problem A: Q/advisory regression**

- Fix target: Q_V3R2 retraining or calibration
- NOT certificate relaxation
- NOT authority threshold changes
- NOT benchmark relabeling

## Comparison with I3.30R2

In I3.30R2, 5 D1 breaks were identified (d1_0004, d1_0010, d2_0003,
d3_0022, d3_0038). In I3.30R3, only 2 D1 breaks appear (d1_0004, d1_0012).
The reduction may be due to:
- Fixed V3 feature computation (canonical V3 features)
- Fixed DEFER success check
- Different LLM backend behavior (Metal vs Colab)

The remaining 2 breaks are consistent Q-model calibration issues where
V3 systematically overvalues REASON_MORE on simple DEFER-ready tasks.

## Recommended Fix

Create a separate Q_V3R3 candidate (not modifying V3R2 within I3.30R3)
that addresses the D1 calibration issue by:
1. Adding D1-specific training examples where DEFER is immediately correct
2. Penalizing REASON_MORE when the state is already terminal-ready
3. Ensuring the canonical topology features `verified_hyp_action_is_defer`
   are properly weighted

Do NOT relax certificate conditions or authority thresholds to compensate
for this Q-model defect.
