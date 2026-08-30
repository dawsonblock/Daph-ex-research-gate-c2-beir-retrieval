# I3.30R3 ATTEMPT 2: Corrected Authority Isolation — Analysis

**Date: 2026-08-29**
**Branch: `i3.30r3-authority-isolation`**
**Commit: `a338808` (run), post-run fixes applied**
**Run: Local (Metal), Qwen2.5-7B-Instruct Q4_K_M GGUF**
**GGUF SHA256: `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`**
**Trajectories: 555/555 completed, 0 errors**

## Status: DEVELOPMENT CAUSAL PASS — NOT CONFIRMATION

This is a development study on a benchmark that has been repeatedly
inspected across I3.29-I3.30R3. The bootstrap interval describes
uncertainty over this empirical task sample, not a generalization bound
to unseen tasks. An untouched structural confirmation is required before
promotion.

## Recommended Claim Wording

The strongest claim supported by this archive:

> On the 185-task I3.30R3 development benchmark, holding the V3R2-A
> executive, Q model, advisory guidance, prompts, legal actions and
> decoder treatment constant, enabling certificate-gated hard ANSWER
> authority increased success from 58.9% to 68.6% and mean paired
> utility by 15.57 [bootstrap 95% CI 8.90, 23.17], with 18 paired
> rescues and zero observed paired breaks. The result has not yet been
> confirmed on an untouched structural benchmark, and DEFER hard
> authority lacks effective intervention coverage.

## Executive Summary

ATTEMPT 1 was invalid: both V3 arms narrowed `schema_actions` to the
singleton forced action before LLM generation, constraining the decoder.
ATE=0 was an artifact of constrained decoding.

ATTEMPT 2 fixes the treatment boundary. Both V3 arms now see the full
legal action set. Treatment is applied only after decoding.

**The corrected primary finding (task-level paired):**

> **ATE_authority = +15.57, 95% CI [8.90, 23.17], 18 task-level paired rescues, 0 task-level paired breaks.**

The LLM disagrees with the certificate in 38 of 90 certificate-positive
events (42.2% disagreement rate). Hard authority overrides those
disagreements, producing 18 task-level paired rescues with 0 breaks.

**Important distinction:** The 18 rescues are task-level paired causal
counts. The event-level classifications (30 "rescues", 42
"beneficial_nonrescues", 52 "neutral") are trajectory-associated
certificate-event classifications, NOT event-level causal effects.
A single trajectory can contain multiple certificate-positive events,
so event counts exceed task-level counts. The causal headline is the
task-level paired comparison.

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

### Trajectory-Associated Certificate-Event Classifications

**WARNING: These are NOT event-level causal effects.** A single
trajectory can contain multiple certificate-positive events, so event
counts exceed task-level counts. The causal headline is the task-level
paired comparison (18 rescues, 0 breaks above).

| Classification | Count | Meaning |
|---------------|-------|---------|
| rescue | 30 | trajectory-associated (NOT independent causal rescues) |
| break | 0 | trajectory-associated |
| beneficial_nonrescue | 42 | trajectory-associated |
| harmful_nonbreak | 0 | trajectory-associated |
| neutral | 52 | trajectory-associated |

The 30 event-level "rescues" correspond to 18 task-level paired rescues.
The difference (12 events) is because some rescued trajectories contain
multiple certificate-positive events that all inherit the same
trajectory-level success classification.

Until full state-fork-and-rollout replay exists, these event
classifications should be called "trajectory-associated certificate-event
classifications," not "event-level causal effects."

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

## Gate Evaluation: 10 passed, 2 failed

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
| G9 semantic_consistency | PASS | 0 (D5 + cross-stratum) |
| G10 reliability | PASS | 0 errors |
| G11 artifact_identity | FAIL | 2 mismatches (runner + evaluator SHA) |
| G12 event_receipts | PASS | 100% complete |

### G8 (defer_coverage) — informative, not a defect

G8 fails because no DEFER certificate fired with an LLM disagreement —
the LLM always chose DEFER when the DEFER certificate passed (3/3 events,
all neutral). This is a property of the current LLM behavior, not a
defect. Do not weaken G8 to make the run 12/12. The failure is informative.

Classification:
- ANSWER_AUTHORITY: DEVELOPMENT_CAUSAL_PASS
- DEFER_CERTIFICATE: NO_FALSE_FORCE_OBSERVED
- DEFER_AUTHORITY_EFFECT: UNTESTED / NO_EFFECTIVE_COVERAGE

### G11 (artifact_identity) — provenance mismatch, disclosed

G11 fails because the frozen manifest was created with the original
runner/evaluator SHAs, but both were modified after the run:
- runner: Fix 2 (write-once manifest logic)
- evaluator: Fix 4 (event rename) + Fix 5 (G9 strengthening)

The primary numerical result was independently reproduced byte-for-byte
with the newer evaluator. The full provenance chain is disclosed in
`experiments/i3_30r3/analysis/ANALYSIS_MANIFEST.json`. The frozen
manifest is now write-once and cannot be overwritten. The next
confirmation run will produce a fresh manifest with matching SHAs.

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

4. It does not prove authority is safe in general. 0 observed breaks
   with 38 effective interventions gives a rule-of-three upper bound
   of approximately 3/38 ≈ 7.9% break rate. "Authority is proven safe"
   is not supported; "0 breaks observed in 38 interventions" is.

5. It does not prove generalization to unseen tasks. The bootstrap
   interval [8.90, 23.17] describes uncertainty over this empirical
   task sample, not a generalization bound. This is a development
   benchmark that has been repeatedly inspected.

6. It does not establish DEFER authority effectiveness. The 3 DEFER
   certificate events all had LLM agreement, so DEFER authority has
   no effective intervention coverage.

### Decision tree branch

Per the preregistered decision tree:

> **AUTH > SHADOW: authority provides measurable benefit.**

This is the strongest branch. Hard ANSWER authority is not redundant —
it provides a statistically significant causal improvement of +15.57
utility with 18 task-level paired rescues and 0 task-level paired breaks.

### Causal decomposition

```
V1           56.76% success
       representation/Q change
                ↓
V3-SHADOW    58.92%  (+2.16pp, CI includes 0, not significant)
       hard ANSWER authority
                ↓
V3-HARD      68.65%  (+9.73pp, CI [8.90, 23.17], significant)
```

The first step is small and uncertain. The second step is large.
Most of the observed improvement comes from authority, not from V3
Q/advisory improvements alone.

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
| Gates passed | 8 | 10 (post-fix) |

The collapse of the secondary comparison from +18.03 to +2.46 confirms
that ATTEMPT 1's V3-SHADOW was contaminated. The hidden decoder constraint
was doing significant work that was attributed to the "representation."

## Scientific Status

| Component | Status |
|-----------|--------|
| DAPH V1 | CONFIRMED HISTORICAL CHAMPION |
| V3R2 representation | DEVELOPMENT MIXED (D2 fix, D1 regression, not independently superior) |
| V3 ANSWER authority | DEVELOPMENT CAUSAL PASS |
| V3 DEFER authority | NO EFFECTIVE CAUSAL COVERAGE |
| I3.30R3 ATTEMPT 1 | INVALID (treatment contamination) |
| I3.30R3 ATTEMPT 2 | VALID DEVELOPMENT STUDY, NOT CONFIRMATION |
| V3 PROMOTION | NOT YET |

## Confirmation Design

Do not alter Q_V3R2-A, epsilon, authority threshold, certificate logic,
or prompt before confirmation. Use V3-SHADOW and V3-HARD as primary arms.

Generate 300-500 fresh tasks from structural configurations not present
in the development benchmark.

Primary confirmatory hypothesis:
- H1: E[U_HARD - U_SHADOW] > 0

Required gates for promotion:
- CI_95%,lower(ΔU) > 0
- paired breaks = 0 observed
- false ANSWER authority = 0
- terminal authority on CONTINUE_REQUIRED = 0
- effective ANSWER interventions > 0
- artifact mismatch = 0
- semantic disagreement = 0
- runtime errors = 0

Keep DEFER authority explicitly outside the confirmation claim unless
the new benchmark generates actual DEFER disagreements.

## Post-Fix Provenance

Five fixes were applied after the ATTEMPT 2 run based on independent audit:

1. **Fix 1**: Frozen runner/isolation/evaluator/checkpoint/restore/GGUF/grammar
   SHAs added to preregistration (not just runtime manifest)
2. **Fix 2**: frozen_manifest.json is now genuinely write-once and fail-closed
   (any mismatch aborts; execution cannot overwrite the identity document)
3. **Fix 3**: Post-run analysis manifest (`ANALYSIS_MANIFEST.json`) discloses
   the full provenance chain (trajectory-producing runner SHA vs current
   evaluator SHA)
4. **Fix 4**: Event-level classifications renamed to "trajectory-associated
   certificate-event classifications" (not causal effects)
5. **Fix 5**: G9 strengthened to check cross-stratum certificate/executor
   agreement, not just D5 initial-state semantics

G11 now correctly FAILS (2 mismatches: runner + evaluator SHA) because
the frozen manifest predates the fixes. This is honest. The next
confirmation run will produce a fresh manifest with matching SHAs.

## Files

- `experiments/i3_30r3/live/` — ATTEMPT 2 trajectory and event files
- `experiments/i3_30r3/analysis/authority_analysis.json` — full metrics
- `experiments/i3_30r3/analysis/gate_evaluation.json` — 12 gate results
- `experiments/i3_30r3/analysis/authority_counterfactuals.jsonl` — per-event
- `experiments/i3_30r3/analysis/paired_results.jsonl` — per-task paired
- `experiments/i3_30r3/analysis/ANALYSIS_MANIFEST.json` — post-run provenance
- `experiments/i3_30r3/attempt1_invalid/` — preserved ATTEMPT 1 data
- `experiments/i3_30r3/I3_30R3_RESULTS.md` — auto-generated from analysis
- `experiments/i3_30r3/I3_30R3_PREREGISTRATION.json` — updated with full freeze
