# I3.30R3: Authority Isolation Study — Results

**Date: 2026-08-29**
**Branch: `i3.30r3-authority-isolation`**
**Commit: `abf9b6c`**
**Run: Local (Metal), Qwen2.5-7B-Instruct Q4_K_M GGUF**
**GGUF SHA256: `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`**

## Executive Summary

The I3.30R3 authority isolation study completed 555 trajectories (185 tasks × 3 arms) with zero runtime errors. The primary finding is:

> **Adaptive hard authority is causally redundant given the V3 advisory layer.**
> **ATE_authority = 0.0000 (exactly zero), 0 rescues, 0 breaks.**

In all 90 certificate-positive events, the LLM had already chosen the same
action that the certificate would have forced. The hard override never
changed a single action.

The secondary finding confirms the V3 representation/Q/advisory layer
provides substantial improvement over V1:

> **ΔU(SHADOW - V1) = +18.03, 95% CI [9.97, 26.73], 24 rescues, 2 breaks.**

## Results

### Primary Comparison: V3-AUTH vs V3-SHADOW

| Metric | Value |
|--------|-------|
| ATE_authority | 0.0000 |
| 95% CI | [0.0000, 0.0000] |
| n | 185 |
| Rescues | 0 |
| Breaks | 0 |
| Both success | 127 |
| Both fail | 58 |

**Interpretation:** The hard authority override has zero causal effect.
The V3 advisory guidance (Q values, epsilon near-optimal set, progress
tie-break, confidence) is sufficient to steer the LLM to the certificate's
action in 100% of certificate-positive states.

### Secondary Comparison: V3-SHADOW vs V1

| Metric | Value |
|--------|-------|
| ΔU(SHADOW - V1) | +18.03 |
| 95% CI | [+9.97, +26.73] |
| n | 185 |
| Rescues | 24 |
| Breaks | 2 |

**Interpretation:** The V3 representation/Q/advisory layer provides a
statistically significant improvement over V1. The improvement is
concentrated in D2 (+22.86% success rate) and D3 (+35.56% success rate).

### Authority Event Classification

| Classification | Count |
|---------------|-------|
| RESCUE | 0 |
| BREAK | 0 |
| BENEFICIAL_NONRESCUE | 0 |
| HARMFUL_NONBREAK | 0 |
| NEUTRAL | 90 |

All 90 certificate-positive events are NEUTRAL — the LLM chose the same
action the certificate would have forced.

### Authority Rates

| Rate | Value |
|------|-------|
| Certificate coverage | 29.51% (90/305 total steps) |
| Force rate | 29.51% (90/305) |
| Effective intervention rate | 0.00% (0/305) |

Certificate coverage = force rate because the hard arm applies force
whenever the certificate passes. Effective intervention rate = 0 because
the LLM always agrees with the certificate.

### Certificate Types

| Certificate Type | Count |
|-----------------|-------|
| unique_verified_support_answer | 87 |
| unique_verified_support_defer | 3 |

### Stratum Breakdown

| Stratum | V1 | SHADOW | AUTH | Δ(SHADOW-V1) |
|---------|-----|--------|------|-------------|
| D1 | 28.57% | 22.86% | 22.86% | -5.71% |
| D2 | 54.29% | 77.14% | 77.14% | +22.86% |
| D3 | 13.33% | 48.89% | 48.89% | +35.56% |
| D4 | 100.00% | 100.00% | 100.00% | 0.00% |
| D5 | 100.00% | 100.00% | 100.00% | 0.00% |

### Aggregate Success Rates

| Arm | Success | Total | Rate |
|-----|---------|-------|------|
| V1 | 97 | 185 | 52.43% |
| V3-SHADOW | 121 | 185 | 65.41% |
| V3-AUTH | 121 | 185 | 65.41% |

V1 success rate (52.43%) matches the I3.30R2 diagnostic exactly.
V3 success rate (65.41%) is higher than I3.30R2 (57.30%) — likely due to
backend/implementation differences (Metal vs Colab GGUF, fixed V3 feature
computation, fixed DEFER success check).

## Gate Evaluation

| Gate | Name | Result | Value |
|------|------|--------|-------|
| G1 | treatment_purity | PASS | 25/25 tests |
| G2 | authority_breaks | PASS | 0 |
| G3 | false_answer_authority | PASS | 0 |
| G4 | false_defer_authority | PASS | 0 |
| G5 | authority_effect | PASS | 0.0 (≥ 0) |
| G6 | rescues_gt_breaks | FAIL | 0 = 0 |
| G7 | answer_coverage | FAIL | 0 effective |
| G8 | defer_coverage | FAIL | 0 effective |
| G9 | semantic_consistency | PENDING | D5 audit done |
| G10 | reliability | PASS | 0 errors |
| G11 | artifact_identity | PASS | 0 mismatches |
| G12 | event_receipts | PASS | 100% complete |

**8 passed, 3 failed, 1 pending.**

The 3 failures (G6, G7, G8) are all consequences of the zero effective
intervention rate — they ask for rescues > breaks and > 0 effective
interventions, which cannot happen when the LLM always agrees with the
certificate. These are not defects; they are the experimental result.

## D5 Results

All 35 D5 tasks succeeded under all three arms (100% success rate).
This includes d5_0026, which was a break in I3.30R2. In this run, the
LLM chose the correct VERIFY target (E3 discriminator) under all arms.

The D5 state-truth audit (Phase 8) confirmed that D5's initial state is
CONTINUE_REQUIRED and the certificate correctly abstains. The certificate
fires only after VERIFY(E3) when the state becomes ANSWER_READY.

## D1 Breaks (Problem A: Q/Advisory Regressions)

The 2 SHADOW-vs-V1 breaks are D1 tasks:

| Task | V1 | V3-SHADOW | V3-AUTH | Root Cause |
|------|-----|-----------|---------|------------|
| d1_0004 | DEFER (success) | REASON_MORE (fail) | REASON_MORE (fail) | V3 Q ranks REASON_MORE > DEFER |
| d1_0012 | DEFER (success) | REASON_MORE (fail) | REASON_MORE (fail) | V3 Q ranks REASON_MORE > DEFER |

These are Q/advisory regressions, not authority issues. V1 correctly
identifies DEFER as the best action, while V3's Q model ranks
REASON_MORE higher. No certificate fires on D1 tasks (the state is
not answer-ready), so authority cannot repair these.

Fix target: Q_V3R2 training/calibration, not certificate relaxation.

## D2 Rescues (8 tasks)

V3 rescues 8 D2 tasks where V1 fails. In all cases, V1 chooses DEFER
(utility -123.3) while V3-SHADOW succeeds (utility +66.7). The V3
representation correctly identifies the answer-ready state where V1
does not.

## D3 Rescues (16 tasks)

V3 rescues 16 D3 tasks where V1 fails. V1 fails with various utilities
(-5.8 to -125.3) while V3-SHADOW succeeds (utility +96.7). The V3
representation/Q model provides substantially better guidance on
multi-hypothesis verification tasks.

## Scientific Conclusion

### Decision Tree Branch

Per the preregistered decision tree:

> **AUTH ≈ SHADOW: authority mostly redundant.**

The hard authority override is causally redundant given the V3 advisory
layer. The LLM, when given V3's Q-guided near-optimal action set and
confidence information, always chooses the same action the certificate
would have forced.

### Implications

1. **V3R2-A's value comes from the representation/Q/advisory layer, not
   from hard authority.** The +18.03 utility delta over V1 is entirely
   attributable to the V3 state representation, Q model, epsilon guidance,
   and progress tie-breaking — not to the hard override.

2. **The certificate mechanism is semantically correct** (confirmed by
   Phase 8 D5 audit) but practically redundant when the advisory layer
   is effective. The certificate fires correctly at answer-ready states,
   but the LLM already knows what to do.

3. **Hard authority would only add value if the LLM disagreed with the
   certificate.** This could happen with:
   - A weaker LLM that ignores guidance
   - A stronger Q model that identifies correct actions the LLM wouldn't
   - Adversarial or corrupted LLM behavior
   - Tasks where the correct action is counterintuitive

4. **V1 remains the confirmed champion for authority purposes.** V3's
   improvement is in the representation layer, not the authority layer.
   V3-SHADOW (without hard authority) already beats V1.

5. **The 2 D1 breaks are Q/advisory regressions** that should be
   addressed through Q model improvement, not certificate changes.

### Next Steps

Per the preregistered plan:

1. **Freeze the authority policy.** Hard authority is validated as
   safe (zero breaks, zero false forces) but redundant. No certificate
   changes needed.

2. **Address D1 Q/advisory regressions separately.** The 2 D1 breaks
   (d1_0004, d1_0012) are Q model issues where V3 ranks REASON_MORE
   above DEFER. Fix target: Q_V3R2 retraining or calibration.

3. **Consider whether hard authority is worth retaining.** It adds zero
   value in this configuration but provides a safety net for weaker LLMs
   or adversarial conditions. The cost is implementation complexity.

4. **If pursuing V3 promotion, the case rests on the representation
   layer (V3-SHADOW), not authority.** V3-SHADOW beats V1 by +18.03
   utility with 24 rescues and 2 breaks.

## Run Provenance

| Field | Value |
|-------|-------|
| Commit | abf9b6c958e5 |
| Branch | i3.30r3-authority-isolation |
| GGUF SHA256 | 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423 |
| Platform | macOS (Metal) |
| Python | 3.12.0 |
| Errors | 0 |
| Trajectories | 555/555 completed |
| Authority events | 90 (SHADOW) + 90 (HARD) |

## Files

- `experiments/i3_30r3/live/` — trajectory and event files
- `experiments/i3_30r3/analysis/authority_analysis.json` — full metrics
- `experiments/i3_30r3/analysis/gate_evaluation.json` — 12 gate results
- `experiments/i3_30r3/analysis/authority_counterfactuals.jsonl` — per-event
- `experiments/i3_30r3/analysis/paired_results.jsonl` — per-task paired
- `experiments/i3_30r3/d5_state_truth/` — D5 state-truth audit
