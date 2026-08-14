# Confirmation #1 — VALID NEGATIVE CONFIRMATION

**Verdict: FAIL.** The fixed-k C5/S2 integrated mechanism (J1 = frozen_rrf + S2)
failed its prospective fresh-confirmation criteria.

- `RUN_VALID = true` — the run itself has no certification defect: crossover
  parity 0 violations, prompt-binding failures 0, packet budget respected,
  determinism precondition passed, 500/500 receipts written.
- `SCIENTIFIC_PASS = false` — the mechanism did not meet its frozen criteria.

This is **not** inconclusive, **not** a near-pass, **not** an infrastructure
failure, and **not** a selector failure alone.

## Provenance: the verdict predated the Q observation

Two hard structural gates failed in the **selection-only dry pass**, before any
HRM generation existed:

| gate | value | requirement |
|---|---|---|
| J1 EXACT_bridged answer retention | **0.2364** | ≥ 0.60 |
| J1 bridge retention | **0.2522** vs J0 0.2957 (−0.0435) | ≥ J0 − 0.01 |

Committed as `e302737` *before* the GPU run started. The downstream Q numbers
were therefore never available to rescue the failed structural criteria.

**Threshold changes: none. Post-hoc rescue: none.**

## Results (500 tasks, fresh split, 2500 generations)

| arm | fusion | selector | Q | correct |
|---|---|---|---|---|
| J0 | frozen_rrf | S0 | 0.1665 | 0.1140 |
| J1 | frozen_rrf | S2 | 0.2385 | 0.1880 |
| J2 | frozen_rrf | oracle | 0.4205 | 0.4920 |
| J3 | — | oracle evidence | 0.9410 | 0.8820 |
| J1r | R1 | S2 | 0.2415 | 0.1960 |

Pooled primary `Q(J1) − Q(J0) = +0.0720` against the frozen **+0.15** bar — a
real but insufficient effect. Grouped LCBs were positive (family +0.0280,
cluster +0.0445), so the mechanism helps; it does not help enough, and it broke
two structural gates.

## Development → confirmation

| metric | development | confirmation |
|---|---|---|
| candidate CES | 75.8% | **41.2%** |
| J1 EXACT_bridged retention | 67.6% | **23.6%** |
| J1 bridge retention vs J0 | +4.4pp | **−4.4pp** |
| primary ΔQ | +0.1750 | +0.0720 |

## Why it failed (B3-B stop gate, `OUTCOME_A_POSITIVE`)

Conditioning on availability separates retrieval from selector.

**Availability `P(role in candidate pool)`** — the collapse:

| role | available |
|---|---|
| identity | 1.0000 |
| terminal | 0.5340 |
| temporal_current | 0.4500 |
| **bridge** | **0.3286** |
| complete required set | 0.4120 |

**Conditional retention `P(selected | available)`** — S2 is *not* the problem:

| role | J0 | J1 | Δ |
|---|---|---|---|
| identity | 1.0000 | 1.0000 | 0.0000 |
| terminal | 0.2547 | 0.3895 | **+0.1348** |
| temporal_current | 0.1778 | 0.5111 | **+0.3333** |
| bridge | 0.2957 | 0.2522 | **−0.0435** |

**Utility given complete availability** (n=206): Q 0.3956 → 0.5728,
**ΔQ +0.1772**, grouped LCB **+0.0977**, selected CES 0.5485 → 0.7233.

So S2 conditional on availability reproduces its development effect almost
exactly (+0.1772 here vs +0.1750 on development). The mechanism did not fail
because the selector stopped working; it failed because **bridge evidence
reaches the candidate pool only 32.9% of the time at this corpus scale**.

One honest caveat: the bridge conditional-retention delta of **−0.0435** is a
genuine negative signal and sits just inside the −0.05 material boundary. It
passed, but marginally, and it is the one place S2 degrades rather than improves
a role. Worth watching if candidate pools grow.

## Status

- `C5_FIXED_K_S2` = FAILED_FRESH_CONFIRMATION
- `S2_SELECTOR` = DEVELOPMENT_MECHANISM_SUCCESS, GENERALIZATION_NOT_ESTABLISHED
  (works conditional on availability; untested at scale with adequate pools)
- `FIXED_K50` = SCALE_ROBUSTNESS_FAILED
- `CURRENT_BOTTLENECK` = candidate_availability_under_corpus_scaling
- `NEXT_GATE` = B3_CANDIDATE_BUDGET_SCALING
- Confirmation #1 split = **CONSUMED**
