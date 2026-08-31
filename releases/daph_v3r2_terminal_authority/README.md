# DAPH V3R2 Terminal Authority — FROZEN CONFIRMED BASELINE

**Release ID:** daph_v3r2_terminal_authority
**Source tag:** v3r2-confirmed
**Dirty worktree:** False
**Files:** 64 (all SHA256 validated)
**Status:** FROZEN BASELINE — do not modify

This is the immutable baseline for all future DAPH-X experiments.
Every DAPH-X result must answer: "Is this actually better than the
confirmed baseline?" — not just "Does this version work?"

## Key Results (historical, 120-task pool)

| Metric | Value |
|--------|-------|
| SHADOW success | 40/120 (33.33%) |
| HARD success | 100/120 (83.33%) |
| ATE (ΔU) | +63.26 |
| Rescues | 60 |
| Breaks | 0 |
| Sign test p | 8.67e-19 |
| Mechanism | Certificate-driven (Q = burden reduction) |
| Ablation | Q-only = CERT-only = Q+CERT |

## Benchmark Validity

20 OOD_4HYP_MIXED tasks in the original 120-task pool have INVALID
oracle paths (benchmark construction failure, not certificate recall gap).
Templates have been fixed and all 8 templates now pass canonical oracle-path
validation (G_B1-G_B4). The 140-task pool is the corrected version.

On 100 valid OOD tasks: SHADOW=40%, HARD=100%, 60 rescues, 0 breaks.
(Forensic reanalysis, not new confirmation.)

## Qualification Gates

14/15 original gates PASS. G9 (semantic conformance) = FAIL because the
original 120-task pool contained invalid oracle paths. G_B1-G_B4 PASS
with canonical topology validator. G9 will pass only after a new R4
pool is generated, validated, and run.

## Verification

```bash
python scripts/verify_release.py
python scripts/validate_benchmark_oracles.py
```

## GGUF Model

Qwen2.5-7B-Instruct Q4_K_M — NOT included in bundle (SHA256 in manifest).

## Claim Level

Level 2 — structural task OOD (behavioral pass, mechanism certificate-driven)
NOT Level 3 — force-state OOD (d_F min NN = 1.19, only 33% ≥ 3.0)

## Promotion Status

NOT PROMOTED — G9 fails, benchmark templates need regeneration and rerun.
This baseline is preserved for comparison. DAPH-X development proceeds
on a separate branch.

## Frozen Components

The following are frozen and must not change in any V3R2 comparison:
- Q_V3R2 model (Q_V3R2_A.pkl)
- Q_V3R2 feature schema (SCHEMA)
- V3 feature extractor
- Authority policy V3 (policy_v3.py)
- Authority isolation layer
- Evidence executor
- Allowed-actions logic
- Grammar
- LLM backend wrapper
- Utility config
- Task generator (fixed templates)
- GGUF SHA256
- Python package versions
- Git commit/tree
