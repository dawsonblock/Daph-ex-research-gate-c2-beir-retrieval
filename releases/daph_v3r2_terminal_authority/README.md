# DAPH V3R2 Terminal Authority — Confirmed Release

**Release ID:** daph_v3r2_terminal_authority
**Source tag:** v3r2-confirmed
**Dirty worktree:** False
**Files:** 61 (all SHA256 validated)

## Key Results

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

## Benchmark Validity Note

20 OOD_4HYP_MIXED tasks have INVALID oracle paths (benchmark construction
failure, not certificate recall gap). Templates have been fixed and all 8
templates now pass oracle-path semantic validation (G_B1-G_B4).

On 100 valid OOD tasks: SHADOW=40%, HARD=100%, 60 rescues, 0 breaks.
(Forensic reanalysis, not new confirmation.)

## Qualification Gates

14/15 PASS. G9 (semantic conformance) = FAIL/PARTIAL due to benchmark
oracle-path invalidity in original templates.

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
