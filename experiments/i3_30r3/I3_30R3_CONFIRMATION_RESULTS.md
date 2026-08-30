# I3.30R3 CONFIRMATION: Fresh In-Family Structural Replication — Results

**Date: 2026-08-30**
**Branch: `i3.30r3-authority-isolation`**
**Confirmed source: git tag `v3r2-confirmed` (commit `e924908`)**
**Run: Local (Metal), Qwen2.5-7B-Instruct Q4_K_M GGUF**
**GGUF SHA256: `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`**
**Trajectories: 800/800 completed, 0 errors**

## Status: FRESH IN-FAMILY REPLICATION PASS

The primary confirmatory hypothesis is confirmed:

> **H1: E[U_HARD - U_SHADOW] > 0 on fresh in-family replication benchmark**

### Important qualification

This is a **fresh in-family structural replication**, not a structural-OOD
confirmation. The D1-D5 stratum architecture is preserved from development.
The benchmark uses a fresh seed, fresh task IDs, fresh natural-language
domain templates, and different resource-budget configurations, but the
underlying epistemic topology family is the same.

A genuine structural-OOD test requires states with novel feature signatures
not present in development. Such a pool has been built separately at
`experiments/i3_30r3/structural_ood/` but has not yet been run.

## Primary Result

| Metric | Development (ATTEMPT 2) | Confirmation |
|--------|------------------------|-------------|
| ATE_authority | +15.57 | **+18.24** |
| 95% CI | [8.90, 23.17] | **[13.11, 23.84]** |
| CI lower bound | 8.90 | **13.11** |
| n | 185 | **400** |
| Rescues | 18 | **44** |
| Breaks | 0 | **0** |
| Both success | 109 | 259 |
| Both fail | 58 | 97 |

**Promotion criterion: CI_95%,lower(ΔU) > 0 → 13.11 > 0 → PASS**

The effect is stronger on the confirmation benchmark than on development.
The CI is tighter because of the larger sample (400 vs 185 tasks).

## Stratum Breakdown

| Stratum | SHADOW | HARD | Hard effect | 
|---------|--------|------|-------------|
| D1 | 23.75% | 23.75% | 0.00% |
| D2 | 71.25% | 80.00% | **+8.75%** |
| D3 | 31.25% | 75.00% | **+43.75%** |
| D4 | 100.00% | 100.00% | 0.00% |
| D5 | 97.50% | 100.00% | **+2.50%** |

D3 remains the dominant authority effect (+43.75% vs +31.11% in development).
D2 now shows an authority effect (+8.75%) that was absent in development.
D5 shows a smaller effect (+2.50% vs +11.43%).

## Authority Event Classification (trajectory-associated, NOT event-level causal)

| Classification | Count |
|---------------|-------|
| rescue | 75 |
| break | 0 |
| beneficial_nonrescue | 63 |
| harmful_nonbreak | 0 |
| neutral | 109 |

**WARNING: These are trajectory-associated classifications, NOT event-level causal effects.**
The causal headline is the 44 task-level paired rescues.

## Authority Rates

| Rate | Value |
|------|-------|
| Certificate coverage | 0.2422 |
| Force rate | 0.2422 |
| Effective intervention rate | 0.0939 |

## DEFER Authority: Now Tested

The confirmation benchmark generated 7 effective DEFER interventions
(vs 0 in development). G8 now PASSES.

| Metric | Development | Confirmation |
|--------|-------------|-------------|
| Effective ANSWER interventions | 38 | 62 |
| Effective DEFER interventions | 0 | **7** |
| G8 defer_coverage | FAIL | **PASS** |

The DEFER certificate fired with LLM disagreement in 7 events, and all 7
were rescues (0 breaks). This is the first evidence that DEFER hard
authority is causally beneficial, though the sample is small.

## Gate Evaluation: 11 passed, 1 failed

| Gate | Result | Value |
|------|--------|-------|
| G1 treatment_purity | PASS | 0 mismatches in 178 paired events |
| G2 authority_breaks | PASS | 0 |
| G3 false_answer_authority | PASS | 0 |
| G4 false_defer_authority | PASS | 0 |
| G5 authority_effect | PASS | +18.24 (CI lower 13.11 > 0) |
| G6 rescues_gt_breaks | PASS | 44 > 0 |
| G7 answer_coverage | PASS | 62 effective ANSWER interventions |
| G8 defer_coverage | PASS | 7 effective DEFER interventions |
| G9 semantic_consistency | PASS | 0 disagreements |
| G10 reliability | PASS | 0 errors |
| G11 artifact_identity | FAIL | 3 mismatches (expected — see below) |
| G12 event_receipts | PASS | 100% complete |

### G11 explanation

G11 fails because the evaluator checks against the development
preregistration, but the confirmation run uses a different runner and
generator. The confirmation manifest (`frozen_manifest.json`) is
self-consistent. The 3 mismatches are:
- runner_sha256 (confirmation runner ≠ development runner)
- i3_29_generator_sha256 (not used in confirmation — uses confirmation generator)
- i3_30_d5_generator_sha256 (not used in confirmation — uses confirmation generator)

This is a known limitation of using the development evaluator for
confirmation. The confirmation manifest itself is write-once and
verified.

## Confirmation Benchmark

- **Seed**: 43291 (distinct from development seed 9817)
- **Tasks**: 400 (80 per stratum × 5 strata)
- **Domain templates**: 12 NEW clinical domains (distinct from development's 12)
- **Budget configurations**: Fresh combinations with different step/verify/reason ranges
- **Task ID overlap with development**: 0
- **Domain overlap with development**: 0
- **Benchmark SHA256**: fd55dad5498df00b...

## Scientific Conclusion

The development result is **replicated** on a fresh in-family benchmark:

1. **Hard ANSWER authority is causally beneficial.** ATE=+18.24, CI [13.11, 23.84],
   44 rescues, 0 breaks. The CI lower bound (13.11) exceeds 0.

2. **Hard DEFER authority shows initial evidence of benefit.** 7 effective DEFER
   interventions, all rescues, 0 breaks. Sample is small but positive.

3. **The LLM systematically under-answers on unseen tasks.** 69/178 certificate-positive
   events (38.8%) produced LLM disagreement, all corrected by hard authority.

4. **0 breaks observed across 69 effective interventions.** Rule-of-three bound:
   3/69 ≈ 4.3% upper bound on break rate.

5. **The effect replicates within the same epistemic topology family.** The
   confirmation benchmark uses fresh domain templates, new budget configurations,
   and a fresh seed, but preserves the D1-D5 stratum architecture. This is a
   fresh in-family replication, not a structural-OOD test. Structural-OOD
   generalization has not yet been tested.

## Recommended Claim Wording (Updated)

> On a 400-task fresh in-family replication benchmark generated from the same
> D1-D5 epistemic task family but with a different seed (43291), new clinical
> surface domains, new task instances, and different resource-budget
> configurations, holding the V3R2-A executive, Q model, advisory guidance,
> prompts, legal actions and decoder treatment constant, enabling
> certificate-gated hard ANSWER authority increased success from 64.75% to
> 75.75% and mean paired utility by 18.24 [bootstrap 95% CI 13.11, 23.84],
> with 44 paired rescues and zero observed paired breaks. Hard DEFER authority
> showed initial positive evidence (7 effective interventions, all rescues,
> 0 breaks). The result replicates the development finding in-family.
> Structural-OOD and cross-model generalization remain untested.

## Scientific Status

```
I3.30R3 ATTEMPT 2          VALID DEVELOPMENT CAUSAL RESULT
I3.30R3 CONFIRMATION       VALID FRESH IN-FAMILY REPLICATION
                           NOT TRUE STRUCTURAL-OOD CONFIRMATION
ANSWER AUTHORITY           REPLICATED POSITIVE EFFECT
DEFER AUTHORITY            INITIAL POSITIVE EVIDENCE, n=7 effective interventions
STRUCTURAL GENERALIZATION  NOT YET ESTABLISHED
CROSS-MODEL GENERALIZATION NOT ESTABLISHED
```

## Provenance Note

The confirmed V3R2 source tree is preserved at git tag `v3r2-confirmed`
(commit `e924908`). The current working source tree contains V3R3
development modifications to `policy_v3.py` and `restore.py` that are
NOT part of the confirmed executive. To verify bundle self-consistency,
run `python scripts/verify_confirmation_bundle.py`.

## Files

- `experiments/i3_30r3/confirmation/` — confirmation trajectory and event files
- `experiments/i3_30r3/confirmation/frozen_manifest.json` — write-once manifest
- `experiments/i3_30r3/confirmation_analysis/authority_analysis.json` — full metrics
- `experiments/i3_30r3/confirmation_analysis/gate_evaluation.json` — 12 gate results
- `experiments/i3_30r3/confirmation_analysis/paired_results.jsonl` — per-task paired
- `hrm_adaptive_memory/executive/evidence_benchmark/i3_30r3_confirmation_generator.py` — generator
- `scripts/run_i3_30r3_confirmation.py` — confirmation runner
