# R12.9 Production Trajectory Qualification Report

## Smoke Test Artifacts
- 32 trajectories (n_per_cell=1, seed=42)
- 2 arms: A1_INFERRED, R1_INFERRED
- Q3 retrieval only
- Backend: Gemma 3 12B IT Q4_0 (SHA: 2ad4c9ce431a2d5b...)
- Server: ctx=32768, parallel=4, effective slot=8192

---

## R12.9A: False-T2 Forensic Analysis

### Finding: 2 false T2 activations on MATCHED_NEG_IMMEDIATE

Both matched-negative-immediate tasks (i3_15c_0008, i3_15c_0009) triggered T2
on the R1 arm, despite being designed as T2-negative controls.

### Root Cause: TWO concurrent defects

#### Defect 1: TASK_GENERATOR_BUG — `verify_result` not overridden

`_gen_matched_t2_negative_immediate()` creates E2 by:
```python
ev2 = replace(ev2_verified,
              verification_state=VerificationState.UNVERIFIED,
              supports=(),
              contradicts=())
```

It changes `verification_state`, `supports`, and `contradicts`, but does NOT
change `verify_result`. E2 inherits `verify_result="SUFFICIENT"` from the
conflict-evidence template.

In contrast, `_gen_matched_t2_negative_late()` correctly sets:
```python
ev2 = EvidenceItem(..., verify_result="MISSING", ...)
```

**Effect**: After VERIFY(E2), E2 becomes SUFFICIENT (not MISSING).
This makes E2's inferred CONTRADICT relation actionable for elimination.

**Counterfactual test**: With `verify_result="MISSING"`, after VERIFY(E2),
E2 becomes MISSING → H2 is NOT eliminated → T2 does NOT fire. Confirmed.

#### Defect 2: SEMANTIC_FALSE_CONTRADICTION — extractor infers CONTRADICT for NEUTRAL evidence

E2's proposition text (e.g., "the message queue is processing messages with
no backlog and all consumers are active") contains "current" status keywords.

H2's proposition ("the message queue is currently unavailable or unconfirmed")
has hypothesis orientation "stale" (because it mentions "unavailable/unconfirmed").

The DeterministicRelationExtractor applies Rule 2 (TEMPORAL_MISMATCH):
- Evidence claims "current" (entailment verb + current status keywords)
- Hypothesis wants "stale"
- → CONTRADICT

But the gold relation is NEUTRAL because E2 is supposed to be an irrelevant/
neutral evidence item that doesn't bear on either hypothesis.

**This is a legitimate semantic extractor limitation**, not an implementation
bug. The extractor cannot distinguish "evidence about the same subject that
happens to support the opposite conclusion" from "evidence that is neutral/
irrelevant to the hypothesis."

### Causal Chain

```
Task generator: E2.verify_result = SUFFICIENT (should be MISSING)
        ↓
Model executes VERIFY(E2)
        ↓
E2.verification_state → SUFFICIENT
        ↓
Extractor: E2 text has "current" keywords, H2 wants "stale"
        ↓
Inferred relation: E2→H2 = CONTRADICT (gold: NEUTRAL)
        ↓
_classify_from_snapshot: H2 = ELIMINATED
        ↓
All hypotheses eliminated → T2 fires
        ↓
False T2 on matched-negative control
```

### Classification

| Task | Stratum | Classification |
|------|---------|---------------|
| i3_15c_0008 | MATCHED_NEG_IMMEDIATE | TASK_GENERATOR_BUG + SEMANTIC_FALSE_CONTRADICTION |
| i3_15c_0009 | MATCHED_NEG_IMMEDIATE | TASK_GENERATOR_BUG + SEMANTIC_FALSE_CONTRADICTION |

### Required Repair

**Repair 1 (mechanical)**: Fix `_gen_matched_t2_negative_immediate()` to set
`verify_result="MISSING"` on E2, matching the pattern used by
`_gen_matched_t2_negative_late()`.

**Repair 2 (NOT a repair)**: The semantic false contradiction is a legitimate
end-to-end failure of the deterministic extractor. It should NOT be "fixed"
before R13. Instead, R13 analysis must separate:
- `FalseT2_semantic`: false T2 caused by extractor errors
- `FalseT2_structural`: false T2 caused by implementation bugs

After Repair 1, the matched-negative-immediate tasks should no longer trigger
T2 because E2 will become MISSING after verification (not SUFFICIENT), so the
inferred CONTRADICT relation won't be actionable.

---

## R12.9B: VERIFY-Loop Forensic Analysis

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total VERIFY calls | 192 |
| Useful VERIFY (state changed) | 8 (4.2%) |
| No-op VERIFY (no state change) | 184 (95.8%) |
| Repeated consecutive target | 22 (11.5%) |
| Already-verified target | 30 (15.6%) |
| Trajectories with VERIFY | 32 |
| Mean VERIFY per trajectory | 6.0 |
| Verify-loop step-limit terminations | 32 (100%) |

### Key Observations

1. **ALL 32 trajectories terminated via RESOURCE_EXHAUSTED** after 6 VERIFY
   calls (max_verification_calls=5, plus 1 additional step).

2. **95.8% of VERIFY calls are no-ops** — they do not change the MDSG
   decision state or hypothesis elimination status.

3. **15.6% of VERIFY calls target already-verified evidence** — the model
   re-verifies evidence that has already reached a terminal verification state.

4. **The model verifies corpus passages, not just task evidence** — targets
   include IDs like DE001, DM024, DH027, DCF007, etc. These are retrieved
   corpus passages, not the task's E1/E2 evidence items.

5. **R1 trajectories show slightly more useful VERIFY calls** (8/96 = 8.3%)
   than A1 trajectories (0/96 = 0%), suggesting the M3 representation may
   help the model make more informative verification choices.

### Case Analysis

**Case 1 — Illegal repeated verification (already-verified target)**:
- i3_15c_0001 A1: VERIFY(DE004) × 6, with 5 already-verified
- i3_15c_0007 A1: VERIFY(DE015) × 5, with 4 already-verified
- i3_15c_0010 R1: VERIFY(E2) × 3, with 3 already-verified
- i3_15c_0013 R1: VERIFY(DE007) × 2, with 1 already-verified

This indicates the environment permits VERIFY on terminally-verified evidence.
This is an **action-validity bug** — the affordance should exclude
already-verified targets.

**Case 2 — Legal but useless verification**:
- Most no-op VERIFY calls fall here: the model verifies different evidence
  items, but none change the MDSG state because the evidence doesn't bear
  on the hypotheses (or the relation is NEUTRAL).

**Case 3 — Genuinely useful verification**:
- i3_15c_0002 R1: 1 useful VERIFY (likely E2 → state change)
- i3_15c_0004 R1: 2 useful VERIFYs
- i3_15c_0005 R1: 2 useful VERIFYs
- i3_15c_0008 R1: 1 useful VERIFY (E2 → false T2 trigger)
- i3_15c_0009 R1: 1 useful VERIFY (E2 → false T2 trigger)

### Required Repair

**Repair 3 (mechanical)**: Add `valid_verify_targets()` function as the single
source of truth for legal VERIFY targets. Exclude evidence that is already
terminally verified (SUFFICIENT, FALSIFIED, or MISSING).

**NOT a repair**: The model's tendency to verify irrelevant corpus passages
is policy behavior. The model sees retrieved evidence and chooses to verify
it. This is what R13 measures. Do not change the action space or prompt.

---

## R12.9C: Required Mechanical Repairs

### Repair 1: Fix matched_neg_immediate verify_result

File: `hrm_adaptive_memory/executive/semantic_relations/i3_15c_task_generator.py`
Function: `_gen_matched_t2_negative_immediate`

Change E2 to set `verify_result="MISSING"` instead of inheriting "SUFFICIENT"
from the conflict evidence template.

### Repair 2: Add valid_verify_targets()

Create a canonical function that returns only legal VERIFY targets:
- Evidence must be UNVERIFIED
- Evidence must be verifiable
- Evidence must not be already terminally verified

### Repair 3: Freeze runtime context configuration

Add to frozen runtime identity:
- `server_ctx_size = 32768`
- `parallel_slots = 4`
- `effective_ctx_per_slot = 8192`
- `max_tokens = 128`

### Repair 4: Add context-capacity preflight

Before any experiment, serialize all possible initial production packets,
measure token count, and verify:
`max_packet_tokens + 128 ≤ 0.8 × 8192 = 6553`

### NOT Repairs (do not change)

- A1/M3 prompts
- Action definitions
- DEFER semantics
- T2 definition
- Utility
- Budgets
- Benchmark outcome labels
- DeterministicRelationExtractor rules
- Gemma model
- Sampling parameters
- Q3 retrieval ranking
- Statistical thresholds
