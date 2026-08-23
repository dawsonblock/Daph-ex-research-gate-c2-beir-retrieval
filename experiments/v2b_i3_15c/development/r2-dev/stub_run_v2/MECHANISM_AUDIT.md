# R2-DEV Mechanism Audit Report (Stub Backend v2)

> **Backend**: R2StubBackend (deterministic, no live model)
> **Dataset**: 50 tasks × 4 arms = 200 trajectories
> **Seed**: 137 (new held-out, NOT 42)
> **Budget**: max_verification_calls=5 (same as R13, not default 3)
> **R2e fix**: Relabels INSUFFICIENT/NEEDS_DISCRIMINATION/NEEDS_EVIDENCE/SUPPORTED_BUT_UNRESOLVED → NO_VIABLE_HYPOTHESIS at T2

---

## 1. Dataset Integrity

- Total trajectories: 200
- Per arm: C0=50, D=50, E=50, DE=50

## 2. Gate Safety

```
Confusion Matrix:
              GoldGate    GoldNoGate
InferredGate     80          0
InferredNoGate    0         20

FalseGateRate = 0 / (0 + 20) = 0.0
MissedGateRate = 0 / (0 + 80) = 0.0
```

**Both safety metrics are zero.**

## 3. Hard-Gate Invariants

- Schema gate violations: 0
- Executor admissibility violations: 0

## 4. R2e Label Intervention

**Working correctly**: At T2, E and DE arms relabel the decision state:
- `decision_state_internal`: INSUFFICIENT (or other internal label)
- `decision_state_exposed`: NO_VIABLE_HYPOTHESIS

50/50 E arm T2 calls had label changed to NO_VIABLE_HYPOTHESIS.

## 5. R2d Gate Condition

**Gate condition fires correctly at T2**: All D/DE T2 calls have `verify_gate_condition_active=True`.

**VERIFY removal**: `verify_removed_by_epistemic_gate=False` in all cases because VERIFY is never legal at T2 with the stub backend. The stub always tries VERIFY first (priority order), exhausting the verification budget before T2 fires. Additionally, at T2_IMMEDIATE (step 0), there's no visible evidence to verify, so `can_verify=False`.

This is a property of the stub's action selection, not a gate logic bug. The live Gemma model may follow paths where VERIFY is still legal at T2 (as R13's forensic audit found: 228/228 T2 states had valid VERIFY targets).

## 6. Utility Contrasts

All zero (Δ_D=0, Δ_E=0, Δ_DE=0, I_D×E=0) because the stub backend is deterministic and selects the same actions regardless of arm.

## 7. Key Findings

### Confirmed working
1. R2d gate logic: FalseGateRate=0, MissedGateRate=0
2. Hard-gate invariants: 0 violations
3. R2e label intervention: INSUFFICIENT → NO_VIABLE_HYPOTHESIS at T2
4. Per-call receipts: full provenance with all required fields
5. C0 schema identity: no confound from dynamic schema
6. Budget fix: max_verification_calls=5 matches R13

### Important for live model run
1. **VERIFY never legal at T2 with stub**: The stub exhausts verification budget before T2. The live model may preserve VERIFY legality at T2 by following different action patterns.
2. **Decision state at T2 is INSUFFICIENT**: Not NEEDS_DISCRIMINATION. The R2e fix now handles this correctly.
3. **Stub cannot test efficacy**: Only the live Gemma model can test whether the gate changes behavior.

### Next steps
1. Run with live Gemma backend (max_tokens=128, temperature=0.0, seed=42)
2. Check whether VERIFY is legal at T2 with the live model's action patterns
3. If VERIFY is legal at T2, the gate will have a behavioral effect to measure
4. If not, the dataset may need tasks where T2 fires before verification budget exhausts
