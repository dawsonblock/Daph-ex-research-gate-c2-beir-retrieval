# I3.15c Nemotron Pilot — Closure Snapshot

## Classification

**DEVELOPMENT / CROSS-MODEL PILOT**

Not confirmation. The model/backend (Nemotron via OpenRouter) differs from the frozen LOCAL_POLICY_V2 (Liquid LFM2.5). Results are development evidence only.

## Observed Results (preserved without reinterpretation)

| Contrast | Mean | 95% CI | n | Status |
|---|---|---|---|---|
| Delta_T2+ | +9.95 | [-7.88, +27.60] | 40 | POSITIVE_POINT_ESTIMATE |
| Delta_T2_immediate | -4.35 | [-29.45, +19.72] | 20 | NEGATIVE_POINT_ESTIMATE |
| Delta_T2_late | +24.25 | [-0.14, +48.17] | 20 | POSITIVE_POINT_ESTIMATE |
| Delta_DEFER- | -5.47 | [-15.29, +1.06] | 20 | NEGATIVE_POINT_ESTIMATE |
| Delta_ANSWER | 0.00 | [0, 0] | 20 | NO_DIFFERENCE_DETECTED |
| I_phase | +15.42 | — | — | POSITIVE_POINT_ESTIMATE |
| False_T2_controls | 0 | — | 40 | CONFIRMED_ZERO |

## Key Observations

1. Delta_T2+ is positive but CI includes zero — not statistically confirmed.
2. Delta_T2_late is strongly positive (+24.25) and CI barely includes zero (-0.14) — suggestive.
3. Delta_T2_immediate is slightly negative — immediate conflict does not benefit from R1.
4. False T2 on controls = 0 — R1 never triggers T2 on DEFER or ANSWER controls.
5. Delta_ANSWER = 0 — no spurious effect on ANSWER controls.
6. Delta_DEFER- = -5.47 — control equivalence NOT established.
7. R1 T2 triggered on 19/20 immediate and 17/20 late tasks.
8. Step-limit rate: R1 20% vs A1 32.5% on T2+ (R1 hits step limit less).

## Caveats

- Model identity differs from frozen LOCAL_POLICY_V2.
- Prompt was modified (PROMPT_V2 adds decision procedure and VERIFY-before-DEFER rule).
- Nemotron is a reasoning model; reasoning appears in content field and is not controlled.
- Backend response variance not characterized.
- A1/R1 arm equivalence on controls not audited.
- Statistical reporting does not yet include median, wins/losses/ties, or effect size.
- I_phase CI not computed via direct bootstrap.

## Run Parameters

- n_per_cell: 10
- retrieval: Q3_RERANKED only
- arms: A1_INFERRED, R1_INFERRED
- total trajectories: 160
- seed: 42
- max_tokens: 8192
- n_workers: 8
- backend: OpenRouter (nvidia/nemotron-3.5-lightning:free)

## Artifacts

- `MANIFEST.sha256.json` — full artifact hashes and frozen identities
- `results.jsonl` — 160 trajectory records
- `analysis.json` — computed contrasts
- `mechanism_receipts.jsonl` — 80 R1 trajectory receipts
- `environment.json` — environment and git state
