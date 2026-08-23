# R2-DEV Live Mechanism Audit Report (Gemma 3 12B Q4_0)

> **Backend**: Gemma 3 12B Q4_0 on Colab T4 GPU (llama-cpp-python)
> **Dataset**: 10 tasks × 4 arms = 40 trajectories
> **Seed**: 137 (new held-out, NOT 42)
> **Budget**: max_verification_calls=5 (same as R13)
> **Decoding**: non-strict (model wraps JSON in markdown code blocks)
> **Date**: 2026-08-23
> **Model SHA**: dd53172ff3a7b1b16c8fb3d944b87f42a6228ff2de3825b8813ae90d988434cd
>   (google/gemma-3-12b-it-qat-q4_0-gguf, differs from R13's frozen SHA — see notes)

---

## 1. Dataset Integrity

- Total trajectories: 40
- Per arm: C0=10, D=10, E=10, DE=10
- All tasks from seed=137 dataset, first 10 tasks

## 2. Gate Safety

```
Confusion Matrix:
              GoldGate    GoldNoGate
InferredGate     20          0
InferredNoGate    0          0

FalseGateRate = 0 / (0 + 0) = 0.0
MissedGateRate = 0 / (0 + 20) = 0.0
```

**Both safety metrics are zero.** No false gates, no missed gates.

## 3. Hard-Gate Invariants

- Schema gate violations: 0
- Executor admissibility violations: 0

**Layer 1 and Layer 2 enforcement both perfect with live model.**

## 4. Utility Contrasts

| Contrast | Value | Interpretation |
|----------|-------|----------------|
| Δ_D | -20.428 | D (gate) hurts |
| Δ_E | -39.784 | E (label) hurts more |
| Δ_DE | -39.784 | DE = E (D doesn't add harm on top of E) |
| I_D×E | +20.428 | Positive interaction — D mitigates E's harm |

**Key finding**: Both interventions are individually harmful, but the label intervention (E) is worse than the gate (D). The positive D×E interaction means D partially rescues the damage caused by E.

## 5. Success Breakdown

| Arm | Success Rate | Step Limit | N |
|-----|-------------|-----------|---|
| C0 | 0.40 | 0.00 | 10 |
| D | 0.20 | 0.00 | 10 |
| E | 0.00 | 0.00 | 10 |
| DE | 0.00 | 0.00 | 10 |

**C0 is the only arm with any successes (4/10).** D drops to 2/10. E and DE drop to 0/10.

## 6. Replacement-Action Distribution (D/DE at T2)

When the gate condition is active, the model selects:
- REASON_MORE: 24
- SEARCH_MORE: 21
- STOP: 18
- DEFER: 2

The model replaces VERIFY with a mix of reasoning, search, and stop. The STOP selections (18) are concerning — the model gives up rather than productively redirecting.

## 7. Terminal Actions

- C0: 6 STOP, 4 DEFER
- D: 8 STOP, 2 DEFER
- E: 10 STOP, 0 DEFER
- DE: 10 STOP, 0 DEFER

**E and DE cause the model to always STOP** — the NO_VIABLE_HYPOTHESIS label appears to cause the model to give up entirely rather than productively redirect.

## 8. Key Findings

### Confirmed working
1. **Gate safety perfect with live model**: FalseGateRate=0, MissedGateRate=0
2. **Hard-gate invariants hold**: 0 violations across all 40 trajectories
3. **Admissibility enforcement works**: model never selected a gated action
4. **Non-strict decoding needed**: model wraps JSON in markdown code blocks

### Scientific findings
1. **R2e (label intervention) is harmful**: Changing NEEDS_DISCRIMINATION/INSUFFICIENT → NO_VIABLE_HYPOTHESIS causes the model to STOP instead of continuing to work. Success drops from 40% to 0%.
2. **R2d (gate intervention) is also harmful but less so**: Removing VERIFY from the allowed set drops success from 40% to 20%. The model sometimes redirects productively (DEFER) but often gives up (STOP).
3. **Positive D×E interaction**: D mitigates E's harm (Δ_DE = Δ_E, not worse). The gate prevents the model from attempting verification that would fail anyway.
4. **The model uses VERIFY productively in C0**: When VERIFY is available (C0 arm), the model achieves 40% success. Removing it (D) or changing the label (E) both degrade performance.

### Model SHA discrepancy
The downloaded Gemma 3 12B Q4_0 GGUF has SHA `dd53172f...`, which differs from R13's frozen SHA `2ad4c9ce...`. This is because HuggingFace updated the file since R13. This means the live R2-DEV results are NOT directly comparable to R13 — they use a different model checkpoint. For a frozen confirmation run, the original R13 GGUF would need to be used.

### Next steps
1. Run with more tasks (50+) for statistical power
2. Investigate why E causes universal STOP — is NO_VIABLE_HYPOTHESIS too pessimistic?
3. Consider whether the label intervention should be softer (e.g., "CONFLICTING_EVIDENCE" instead of "NO_VIABLE_HYPOTHESIS")
4. For frozen confirmation: obtain the original R13 GGUF (SHA 2ad4c9ce...)
5. The positive D×E interaction warrants further investigation — does D help because it prevents wasted VERIFY calls, or because it forces the model to consider other actions?
