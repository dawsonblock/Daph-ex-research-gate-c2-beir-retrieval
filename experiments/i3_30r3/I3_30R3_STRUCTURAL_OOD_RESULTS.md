# I3.30R3 STRUCTURAL-OOD: First Genuine Structural Generalization Test — Results

**Date: 2026-08-30**
**Branch: `i3.30r3-authority-isolation`**
**Confirmed executive: git tag `v3r2-confirmed` (commit `e924908`)**
**Run: Local (Metal), Qwen2.5-7B-Instruct Q4_K_M GGUF**
**Trajectories: 240/240 completed, 0 errors**

## Status: STRUCTURAL-OOD PASS

The structural-OOD hypothesis is confirmed:

> **H1: E[U_HARD - U_SHADOW] > 0 on structurally novel tasks with 0% development overlap**

## Primary Result

| Metric | In-Family Confirmation | Structural-OOD |
|--------|----------------------|----------------|
| Tasks | 400 | **120** |
| SHADOW success | 64.75% | **33.33%** |
| HARD success | 75.75% | **83.33%** |
| Absolute improvement | 11.00 pp | **50.00 pp** |
| ATE (ΔU) | +18.24 | **+63.26** |
| 95% CI | [13.11, 23.84] | **[51.75, 74.45]** |
| Rescues | 44 | **60** |
| Breaks | 0 | **0** |
| Sign test p | ~1e-13 | **8.67e-19** |

**Promotion criterion: CI_95%,lower(ΔU) > 0 → 51.75 > 0 → PASS**

## What Makes This Different

This is NOT a fresh in-family replication. The 120-task OOD pool was built with:

1. **Explicit feature-signature exclusion**: 129 development structural signatures were computed. Every candidate was rejected if its 14-field V3 structural signature matched any development signature.

2. **Novel structural configurations**:
   - 4-6 hypotheses (development used 2-3)
   - 5-way elimination (development had max 2)
   - 4-way competing verified support
   - 6-hypothesis all-unverified topology
   - Mixed verified/unverified with search available

3. **Nearest-neighbor distance verification**: All 120 accepted tasks have nearest-neighbor distance >= 4.68 in standardized feature space (threshold 3.0).

4. **0% structural overlap** with development.

## By Category

| Category | N | SHADOW | HARD | Rescues | Breaks |
|----------|---|--------|------|---------|--------|
| ood_4hyp_mixed | 20 | 0 | 0 | 0 | 0 |
| ood_5hyp_all_unverified | 20 | 0 | 20 | 20 | 0 |
| ood_6hyp_partial_verify | 20 | 0 | 20 | 20 | 0 |
| ood_4hyp_all_verified_unique | 20 | 20 | 20 | 0 | 0 |
| ood_4hyp_competing_verified_defer | 20 | 20 | 20 | 0 | 0 |
| ood_5hyp_3elim_unique | 20 | 0 | 20 | 20 | 0 |

### Key observations:

1. **3 categories (60 tasks) are pure rescues**: The model fails completely without authority (0% SHADOW) but authority recovers to 100% (HARD). These are 5-hypothesis and 6-hypothesis tasks with partial verification — structures the model has never seen.

2. **2 categories (40 tasks) are both-succeed**: Both SHADOW and HARD handle 4-hypothesis tasks with all-verified unique support and 4-hypothesis competing-verified DEFER. These are closer to development structures.

3. **1 category (20 tasks) is both-fail**: 4-hypothesis mixed verified/unverified with search — neither arm succeeds. The model can't handle the mixed state and authority can't certify ANSWER or DEFER.

4. **Zero breaks across all 120 tasks and 60 effective interventions.**

## Scientific Significance

This is the first experiment in the DAPH project that tests structural generalization rather than in-family replication. The result is stronger than the in-family confirmation:

- **Larger effect size**: +63.26 ΔU vs +18.24
- **Larger improvement**: 50 pp vs 11 pp
- **More rescues per task**: 60/120 = 50% vs 44/400 = 11%
- **Zero breaks**: maintained in both experiments

The authority mechanism generalizes to genuinely novel epistemic topologies that the model has never encountered. The model fails catastrophically on novel structures (33% SHADOW success), but hard authority recovers most of them (83% HARD success).

## Recommended Claim Wording

> On a 120-task structural-OOD benchmark with 0% structural overlap with
> development (verified by 14-field V3 structural signature exclusion and
> nearest-neighbor distance >= 4.68 in standardized feature space), holding
> the confirmed V3R2-A executive constant, certificate-gated hard authority
> increased success from 33.33% to 83.33% and mean paired utility by 63.26
> [bootstrap 95% CI 51.75, 74.45], with 60 paired rescues and zero observed
> paired breaks (sign test p = 8.67e-19). The authority mechanism generalizes
> to novel epistemic topologies with 4-6 hypotheses and 5-way elimination
> patterns not present in development.

## Limitations

1. **Single model backend**: Only Qwen2.5-7B-Instruct tested.
2. **One task family**: Evidence-based medical reasoning. Generalization to other domains (coding, retrieval QA) is untested.
3. **One category both-fails**: The 4-hyp mixed verified/unverified with search category (20 tasks) fails in both arms. This is a known limitation of the current certificate set.
4. **OOD pool is novel but not exhaustive**: There are many possible structural configurations not tested.

## Files

- `experiments/i3_30r3/structural_ood/ood_pool.json` — 120-task OOD pool
- `experiments/i3_30r3/structural_ood/development_signatures.json` — 129 dev signatures
- `experiments/i3_30r3/structural_ood_run/trajectories_v3_shadow.jsonl` — 120 SHADOW trajectories
- `experiments/i3_30r3/structural_ood_run/trajectories_v3_hard.jsonl` — 120 HARD trajectories
- `experiments/i3_30r3/structural_ood_run/frozen_manifest.json` — frozen manifest
- `experiments/i3_30r3/structural_ood_run/results.json` — computed results
