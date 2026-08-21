# I3.15c Nemotron Pilot — Protocol

## Pre-registered Contrasts

- Delta_T2+ = U(R1) - U(A1) | T2-eligible (IMMEDIATE + LATE)
- Delta_T2_immediate = U(R1) - U(A1) | T2_CONFLICT_IMMEDIATE
- Delta_T2_late = U(R1) - U(A1) | T2_CONFLICT_LATE
- Delta_DEFER- = U(R1) - U(A1) | DEFER_CONTROL
- Delta_ANSWER = U(R1) - U(A1) | ANSWER_CONTROL
- I_phase = Delta_T2+ - Delta_DEFER-

## Desired Signature

- Delta_T2+ > 0
- I_phase > 0
- Delta_DEFER- ~ 0
- Delta_ANSWER ~ 0
- False T2 on controls = 0

## Statistical Method

Paired bootstrap CI over tasks (B=2000).

## Structural Qualification

- T2_CONFLICT_IMMEDIATE: T2 fires at initial state = 100%
- T2_CONFLICT_LATE: T2 at initial state = 0%, T2 after gold transition = 100%
- DEFER_CONTROL: T2 at gold state = 0%
- ANSWER_CONTROL: T2 at gold state = 0%

## Classification

DEVELOPMENT / CROSS-MODEL PILOT

This result is NOT confirmatory because:
1. Model identity (Nemotron) differs from frozen LOCAL_POLICY_V2 (Liquid LFM2.5)
2. Prompt was modified (PROMPT_V2) after the original freeze
3. Backend response variance not characterized
4. A1/R1 arm equivalence on controls not audited
