# R2-DEV Mechanism Audit Report (Stub Backend)

> **Backend**: R2StubBackend (deterministic, no live model)
> **Dataset**: 50 tasks × 4 arms = 200 trajectories
> **Seed**: 137 (new held-out, NOT 42)
> **Date**: R2-DEV stub run
> **Commit**: 3c83346

---

## 1. Dataset Integrity

- Total trajectories: 200
- Per arm: C0=50, D=50, E=50, DE=50
- Strata covered: T2_IMMEDIATE, T2_LATE_1, T2_LATE_2, T2_LATE_3_NONTRIGGER, MATCHED_NEG_IMMEDIATE
- Full dataset (800 tasks) has 12 strata; this run used first 50 tasks

## 2. Arm Isolation

All arms share 50 common tasks. Expected diffs were found in allowed_actions (D/DE) and decision_state_exposed (E/DE).

## 3. Gate Safety

```
Confusion Matrix:
              GoldGate    GoldNoGate
InferredGate     80          0
InferredNoGate    0         20

FalseGateRate = 0 / (0 + 20) = 0.0
MissedGateRate = 0 / (0 + 80) = 0.0
```

**Both safety metrics are zero.** The R2d gate logic is perfect on structural-gold cases.

## 4. Hard-Gate Invariants

- Schema gate violations: 0
- Executor admissibility violations: 0

**Layer 1 and Layer 2 enforcement both perfect.**

## 5. Per-Call Receipt Verification

### D arm at T2 (430 calls):
- `verify_gate_condition_active`: True
- `verify_removed_by_epistemic_gate`: False
- `verify_gate_reason`: ALL_HYPOTHESES_ELIMINATED
- VERIFY in legal: False (budget exhausted)
- VERIFY in allowed: False
- `admissibility_assertion_passed`: True

**Finding**: At T2, VERIFY is already illegal (verification budget exhausted). The gate condition fires but doesn't actually remove VERIFY from the allowed set — it was already removed by legality. This is the correct `verify_gate_condition_active vs verify_removed_by_epistemic_gate` distinction.

### C0 arm at T2 (430 calls):
- `verify_gate_condition_active`: False
- VERIFY in allowed: False (also budget-exhausted)

### E arm at T2 (430 calls):
- `decision_state_internal`: INSUFFICIENT
- `decision_state_exposed`: INSUFFICIENT
- VERIFY in allowed: False

**Finding**: The decision state at T2 is classified as INSUFFICIENT, not NEEDS_DISCRIMINATION. The R2e label intervention only changes NEEDS_DISCRIMINATION → NO_VIABLE_HYPOTHESIS, so it does not fire when the state is INSUFFICIENT. This is an important finding for the live model run: the R2e intervention may need to also handle INSUFFICIENT → NO_VIABLE_HYPOTHESIS at T2, or the T2/decision-state mapping needs reconciliation.

### DE arm at T2 (430 calls):
- `decision_state_internal`: INSUFFICIENT
- `decision_state_exposed`: INSUFFICIENT (no label change, same as E)
- `verify_gate_condition_active`: True
- VERIFY in allowed: False

## 6. Replacement-Action Distribution (D/DE at T2)

When the gate condition is active, the stub backend selects:
- REASON_MORE: 320 times
- RETRIEVE: 300 times
- SEARCH_MORE: 240 times

No DEFER, ANSWER, or STOP selected by stub (it follows priority order).

## 7. Loop Metrics

| Action | MaxRun (mean) | RepeatedActionRate (mean) |
|--------|---------------|---------------------------|
| RETRIEVE | 3.8 | 0.70 |
| REASON_MORE | 4.0 | 0.74 |
| SEARCH | 0.0 | 0.00 |
| VERIFY | 1.0 | 0.125 |

The stub backend loops on RETRIEVE/REASON_MORE until step limit. This is expected for a deterministic stub — the real test is whether the live model escapes these loops.

## 8. Loop Migration

All arms: LoopMigrationRate = 1.0 (all T2 trajectories hit step limit with non-VERIFY terminal action).

This is expected for the stub backend — it never terminates voluntarily. The live model test will show whether D/DE actually escape the loop or merely relocate it.

## 9. Utility Contrasts

| Contrast | Value |
|----------|-------|
| Δ_D | 0.0 |
| Δ_E | 0.0 |
| Δ_DE | 0.0 |
| I_D×E | 0.0 |

All zero because the stub backend is deterministic and selects the same actions regardless of arm (VERIFY is already illegal at T2, so the gate doesn't change the stub's behavior).

## 10. Success/Break

All arms: 0% success, 80% step-limit, 0% DEFER, 0% ANSWER.

Expected for stub backend — it never reaches a terminal action.

---

## Key Findings

### Confirmed working
1. **R2d gate logic is perfect**: FalseGateRate=0, MissedGateRate=0
2. **Hard-gate invariants hold**: 0 schema violations, 0 executor violations
3. **Gate condition vs removal distinction works**: at T2 with budget exhausted, condition fires but removal doesn't (VERIFY already illegal)
4. **Per-call receipts contain full provenance**: all required fields logged
5. **C0 schema identity preserved**: no confound from dynamic schema

### Important for live model run
1. **Decision state at T2 is INSUFFICIENT, not NEEDS_DISCRIMINATION**: The R2e label intervention (NEEDS_DISCRIMINATION → NO_VIABLE_HYPOTHESIS) does not fire. This needs investigation — either:
   - R2e should also relabel INSUFFICIENT → NO_VIABLE_HYPOTHESIS at T2
   - Or the T2/decision-state mapping needs reconciliation
2. **VERIFY is already illegal at T2 due to budget exhaustion**: The epistemic gate doesn't actually change the allowed set at T2 in these tasks. The live model test may need tasks where VERIFY is still legal at T2 to test the gate's behavioral effect.
3. **Stub backend cannot test efficacy**: The deterministic stub selects the same actions regardless of arm. Only the live Gemma model can test whether the gate changes behavior.

### Next steps
1. Investigate the INSUFFICIENT vs NEEDS_DISCRIMINATION issue at T2
2. Check whether any tasks have VERIFY still legal at T2 (where the gate would actually remove it)
3. Run with live Gemma backend for efficacy testing
4. The stub run validates the harness mechanics; the live run tests the scientific question
