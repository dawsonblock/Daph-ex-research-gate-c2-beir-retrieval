# I3.30R3 ATTEMPT 2: Corrected Authority Isolation — Analysis

**Date: 2026-08-29**
**Branch: `i3.30r3-authority-isolation`**
**Commit: `a338808`**
**Run: Local (Metal), Qwen2.5-7B-Instruct Q4_K_M GGUF**
**GGUF SHA256: `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`**
**Trajectories: 555/555 completed, 0 errors**

## Executive Summary

ATTEMPT 1 was invalid: both V3 arms narrowed `schema_actions` to the
singleton forced action before LLM generation, constraining the decoder.
ATE=0 was an artifact of constrained decoding.

ATTEMPT 2 fixes the treatment boundary. Both V3 arms now see the full
legal action set. Treatment is applied only after decoding.

**The corrected primary finding:**

> **ATE_authority = +15.57, 95% CI [8.90, 23.17], 18 rescues, 0 breaks.**
> **When the LLM is genuinely free to disagree, hard authority causally helps.**

The LLM disagrees with the certificate in 38 of 90 certificate-positive
events (42.2% disagreement rate). Hard authority overrides those
disagreements, producing 18 rescues with 0 breaks.

## Results

### Primary Comparison: V3-AUTH vs V3-SHADOW

| Metric | ATTEMPT 1 (contaminated) | ATTEMPT 2 (corrected) |
|--------|--------------------------|----------------------|
| ATE_authority | 0.0000 | **+15.57** |
| 95% CI | [0.00, 0.00] | **[8.90, 23.17]** |
| Rescues | 0 | **18** |
| Breaks | 0 | **0** |
| Effective intervention rate | 0.00% | **12.46%** |
| LLM agreement with certificate | 100% (forced) | **57.78%** (genuine) |

### Secondary Comparison: V3-SHADOW vs V1

| Metric | ATTEMPT 1 (contaminated) | ATTEMPT 2 (corrected) |
|--------|--------------------------|----------------------|
| ΔU(SHADOW - V1) | +18.03 | **+2.46** |
| 95% CI | [9.97, 26.73] | **[-6.32, 11.53]** |
| Rescues | 24 | **15** |
| Breaks | 2 | **11** |

The secondary comparison collapsed from +18.03 to +2.46 (CI includes 0).
In ATTEMPT 1, V3-SHADOW benefited from hidden pre-generation authority
(decoder constraint). In ATTEMPT 2, V3-SHADOW is truly unconstrained,
and the representation effect alone is much smaller and not statistically
significant.

### Authority Event Classification

| Classification | Count |
|---------------|-------|
| RESCUE | 30 |
| BREAK | 0 |
| BENEFICIAL_NONRESCUE | 42 |
| HARMFUL_NONBREAK | 0 |
| NEUTRAL | 52 |

### Stratum Breakdown

| Stratum | V1 | SHADOW | HARD | Authority effect | Representation effect |
|---------|-----|--------|------|-----------------|----------------------|
| D1 | 28.57% | 22.86% | 22.86% | 0.00% | -5.71% (Q regression) |
| D2 | 54.29% | 77.14% | 77.14% | 0.00% | +22.86% (Q fix) |
| D3 | 13.33% | 17.78% | **48.89%** | **+31.11%** | +4.44% |
| D4 | 100% | 100% | 100% | 0.00% | 0.00% |
| D5 | 100% | 88.57% | **100%** | **+11.43%** | -11.43% (LLM regression) |

### Aggregate Success Rates

| Arm | Success | Total | Rate | Mean Utility |
|-----|---------|-------|------|--------------|
| V1 | 105 | 185 | 56.76% | 13.10 |
| V3-SHADOW | 109 | 185 | 58.92% | 15.56 |
| V3-HARD | 127 | 185 | 68.65% | 31.13 |

## Gate Evaluation: 11 passed, 1 failed

| Gate | Result | Value |
|------|--------|-------|
| G1 treatment_purity | PASS | 0 mismatches in 90 paired events |
| G2 authority_breaks | PASS | 0 |
| G3 false_answer_authority | PASS | 0 |
| G4 false_defer_authority | PASS | 0 |
| G5 authority_effect | PASS | +15.57 >= 0 |
| G6 rescues_gt_breaks | PASS | 18 > 0 |
| G7 answer_coverage | PASS | 38 effective ANSWER interventions |
| G8 defer_coverage | FAIL | 0 effective DEFER interventions |
| G9 semantic_consistency | PASS | 0 D5 disagreements |
| G10 reliability | PASS | 0 errors |
| G11 artifact_identity | PASS | 0 mismatches |
| G12 event_receipts | PASS | 100% complete |

G8 fails because no DEFER certificate fired with an LLM disagreement —
the LLM always chose DEFER when the DEFER certificate passed (3/3 events,
all neutral). This is a property of the current LLM behavior, not a defect.

## D3 Authority Rescues (14 tasks)

**This is the main authority effect.** All 14 D3 rescues follow the same
pattern:

1. After VERIFY, the state becomes ANSWER_READY with unique verified support
2. The V3 certificate fires: `unique_verified_support_answer`, forced=ANSWER
3. The LLM (free to choose) chooses REASON_MORE instead of ANSWER
4. SHADOW: executes REASON_MORE → fails (never reaches ANSWER)
5. HARD: forces ANSWER → succeeds (+96.7 utility)

Example (d3_0002):
```
SHADOW: VERIFY → REASON_MORE → fail (-5.8)
  step 1: cert=unique_verified_support_answer would_force=True
    forced=ANSWER llm=REASON_MORE exec=REASON_MORE force=False

HARD:   VERIFY → ANSWER → success (+96.7)
  step 1: cert=unique_verified_support_answer would_force=True
    forced=ANSWER llm=REASON_MORE exec=ANSWER force=True action_changed=True
```

The LLM systematically avoids ANSWER in favor of REASON_MORE on D3 tasks
even when the state is answer-ready. The certificate correctly identifies
the answer-ready state and forces ANSWER.

5 of these 14 D3 rescues are also SHADOW-vs-V1 breaks: V1 succeeds on
them (V1's ANSWER-only hard authority forces ANSWER), SHADOW fails (LLM
chooses REASON_MORE), HARD rescues (certificate forces ANSWER).

## D5 Authority Rescues (4 tasks)

4 D5 tasks (d5_0001, d5_0002, d5_0013, d5_0017) follow this pattern:

1. After VERIFY(discriminator), state becomes ANSWER_READY
2. The V3 certificate fires: `unique_verified_support_answer`, forced=ANSWER
3. The LLM chooses VERIFY (wrong target) or REASON_MORE instead of ANSWER
4. SHADOW: executes the LLM's choice → fails
5. HARD: forces ANSWER → succeeds (+96.7)

These 4 are also SHADOW-vs-V1 breaks: V1 succeeds (V1's ANSWER-only
authority forces ANSWER at the right time), SHADOW fails (LLM doesn't
choose ANSWER), HARD rescues.

## D2 Representation Effect (8 rescues, no authority effect)

8 D2 tasks are rescued by V3's representation over V1, with no authority
effect (SHADOW = HARD = 27/35 on D2):

- V1: After VERIFY, Q argmax=ANSWER (82.76) → V1 forces ANSWER → FAILS
  (expected DEFER, H1.action=DEFER)
- V3: After VERIFY, Q argmax=DEFER (71.19) → LLM chooses DEFER → SUCCEEDS
- Certificate: NONE for 5/8 tasks, `unique_verified_support_defer` for 3/8
  (but LLM already chose DEFER, so no force needed)

This is a **pure representation effect**: V3's Q model has
`verified_hyp_action_is_defer` feature and correctly ranks DEFER > ANSWER.
V1's Q model lacks this feature and incorrectly ranks ANSWER > DEFER.

## D1 Q Regression (2 breaks, unchanged from ATTEMPT 1)

d1_0004 and d1_0012: V3 Q ranks REASON_MORE (49.01) > DEFER (44.16) on
DEFER-ready states. V1 Q correctly ranks DEFER (69.42) > REASON_MORE
(62.26). No certificate fires on D1, so authority cannot repair this.

Fix target: Q_V3R3 retraining with D1-specific examples where DEFER is
immediately causally optimal (resource exhausted + no useful continuation).

## Causal Interpretation

### What the corrected experiment proves

1. **Hard authority is causally beneficial.** ATE=+15.57, 18 rescues,
   0 breaks. The certificate correctly identifies answer-ready states
   and forces ANSWER when the LLM chooses otherwise.

2. **The LLM systematically under-answers.** In 38/90 certificate-positive
   events (42.2%), the LLM chooses REASON_MORE or VERIFY instead of
   ANSWER even when the state is answer-ready. This is the mechanism
   by which authority helps.

3. **The V3 representation effect is real but smaller than ATTEMPT 1
   suggested.** The +2.46 ΔU(SHADOW-V1) with CI including 0 means the
   representation alone is not statistically significant. The ATTEMPT 1
   finding of +18.03 was inflated by hidden decoder constraint.

4. **V3-SHADOW has both rescues and breaks over V1.** The representation
   helps on D2 (Q fix for DEFER-correct states) but hurts on D3 and D5
   (LLM under-answers without authority to correct it).

5. **V3-HARD dominates both V1 and V3-SHADOW.** V3-HARD (68.65%) beats
   V1 (56.76%) by +11.89pp and V3-SHADOW (58.92%) by +9.73pp. The
   combination of V3 representation + hard authority is the best
   configuration.

### What the corrected experiment does NOT prove

1. It does not prove that V3-SHADOW alone is better than V1. The
   secondary comparison is not statistically significant (CI includes 0).

2. It does not prove that the V3 representation is necessary for
   authority to work. A separate experiment would need to test V1 +
   V3 certificate vs V1 + V1 authority.

3. It does not prove that the certificate is optimal. The certificate
   fires correctly in this run (0 false forces), but the 0 DEFER
   interventions (G8 fail) suggest the DEFER certificate may be
   under-powered or the LLM happens to agree on DEFER.

### Decision tree branch

Per the preregistered decision tree:

> **AUTH > SHADOW: authority provides measurable benefit.**

This is the strongest branch. Hard authority is not redundant — it
provides a statistically significant causal improvement of +15.57 utility
with 18 rescues and 0 breaks.

## Comparison: ATTEMPT 1 vs ATTEMPT 2

| Dimension | ATTEMPT 1 (contaminated) | ATTEMPT 2 (corrected) |
|-----------|--------------------------|----------------------|
| Schema for V3 arms | Narrowed to forced action | Full legal action set |
| LLM freedom | Constrained to forced action | Free to choose |
| LLM agreement | 100% (forced by decoder) | 57.78% (genuine) |
| ATE_authority | 0.0000 | +15.57 |
| Rescues | 0 | 18 |
| Breaks | 0 | 0 |
| ΔU(SHADOW-V1) | +18.03 | +2.46 |
| SHADOW-V1 CI | [9.97, 26.73] | [-6.32, 11.53] |
| SHADOW-V1 breaks | 2 | 11 |
| Gates passed | 8 | 11 |

The collapse of the secondary comparison from +18.03 to +2.46 confirms
that ATTEMPT 1's V3-SHADOW was contaminated. The hidden decoder constraint
was doing significant work that was attributed to the "representation."

## Files

- `experiments/i3_30r3/live/` — ATTEMPT 2 trajectory and event files
- `experiments/i3_30r3/analysis/authority_analysis.json` — full metrics
- `experiments/i3_30r3/analysis/gate_evaluation.json` — 12 gate results
- `experiments/i3_30r3/analysis/authority_counterfactuals.jsonl` — per-event
- `experiments/i3_30r3/analysis/paired_results.jsonl` — per-task paired
- `experiments/i3_30r3/attempt1_invalid/` — preserved ATTEMPT 1 data
- `experiments/i3_30r3/I3_30R3_RESULTS.md` — auto-generated from analysis
