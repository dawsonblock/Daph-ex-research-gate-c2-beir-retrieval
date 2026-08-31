# DAPH V3R2 Terminal Authority — Confirmed Release

**Release ID:** daph_v3r2_terminal_authority
**Source commit:** dfdd955fb7e9c2db378ed91bb7c8927a7825721b
**Source tag:** v3r2-confirmed
**Dirty worktree:** False
**Created:** 2026-08-31T02:31:35.807182+00:00

## Contents

- `executive/` — All executive source files (hashed)
- `models/` — Q models, schemas, utility config
- `benchmark/` — OOD pool, development signatures, novelty report
- `raw/` — Raw trajectories (SHADOW, HARD, ablations)
- `analysis/` — Forensic audits, distance stratification, both-fail diagnostic
- `scripts/` — Verification and reproduction scripts

## Key Results

| Metric | Value |
|--------|-------|
| SHADOW success | 40/120 (33.33%) |
| HARD success | 100/120 (83.33%) |
| ATE (ΔU) | +63.26 |
| Rescues | 60 |
| Breaks | 0 |
| Sign test p | 8.67e-19 |
| Mechanism | Certificate-driven |
| Ablation | Q-only = CERT-only = Q+CERT |

## GGUF Model

The Qwen2.5-7B-Instruct Q4_K_M GGUF file is NOT included in this bundle.
Expected SHA256: `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`

## Verification

```bash
python scripts/verify_release.py
```

All file hashes are recorded in `RELEASE_MANIFEST.json`.
Every SHA256 is exactly 64 lowercase hex characters.

## Claim Level

Level 2 — structural task OOD (behavioral pass, mechanism certificate-driven)

## Promotion Status

NOT PROMOTED — pending full Q-input novelty closure and force-state OOD proof.
