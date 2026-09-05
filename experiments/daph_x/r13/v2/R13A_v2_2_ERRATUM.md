# R13-A v2.2 Erratum — Seed-42 Corrected Analysis

## What changed

This is a **blind erratum** committed before inspecting seeds 123 and 2024. It corrects two analysis flaws in R13-A v2.1:

1. **Binary ceiling was tautological**. The previous `J_bin` was defined as `max(J_STOP, max(J_continuation))`, which equals `J_het` by definition. The corrected `J_bin_best` is the best globally fixed continuation paired with STOP per state.
2. **Cost decoding was inconsistent**. Different v2 operators stored `cost["tokens"]` differently. The v2.2 analyzer normalizes by operator version:
   - `SAMPLE_STANDARD`, `SAMPLE_DIVERSE`: total = `tokens + completion_tokens`.
   - `CRITIQUE_RETRY`, `VERIFY_TARGETED`: total = `tokens`.
   - `STOP`: 0.

No operator implementations were changed. No executions were rerun. The raw 450 seed-42 execution receipts are identical; only the analysis decoder changed.

## Data

- 450 successful executions (90 checkpoints × 5 operators) for seed 42.

## Q4 — Operator causal utility (v2.2)

| Operator | All-state rescue | Conditional rescue | Break | Mean tokens |
|----------|-----------------|-------------------|-------|-------------|
| **VERIFY_TARGETED** | **5.6%** | **10.9%** | 1.1% | 488 |
| CRITIQUE_RETRY | 2.2% | 4.3% | 0.0% | 436 |
| SAMPLE_DIVERSE | 1.1% | 2.2% | 0.0% | 492 |
| SAMPLE_STANDARD | 1.1% | 2.2% | 0.0% | 405 |

## Q5 — Three ceilings (v2.2)

| λ | Heterogeneous U | Best fixed binary U | Δ het vs bin-best | UCB95 of Δ |
|---|----------------|--------------------|------------------|-----------|
| 0.000 | 0.5667 | 0.5444 | +0.0222 | +0.0377 |
| 0.005 | 0.5664 | 0.5443 | +0.0221 | +0.0369 |
| 0.010 | 0.5662 | 0.5441 | +0.0221 | +0.0368 |
| 0.020 | 0.5657 | 0.5437 | +0.0219 | +0.0372 |
| 0.050 | 0.5642 | 0.5427 | +0.0215 | +0.0358 |
| 0.100 | 0.5617 | 0.5409 | +0.0208 | +0.0346 |
| 0.200 | 0.5566 | 0.5374 | +0.0193 | +0.0321 |

- **Best fixed binary continuation**: `VERIFY_TARGETED` at all λ.
- **V0 fixed binary continuation**: `VERIFY_TARGETED` (same because it is the λ=0 best continuation).
- Heterogeneous oracle distribution: `STOP` 83/90, `VERIFY_TARGETED` 4/90, `CRITIQUE_RETRY` 2/90, `SAMPLE_STANDARD` 1/90.

## Corrected interpretation

The seed-42 data show a **mean utility gain of +0.022 to +0.022 from heterogeneous action selection over the best fixed binary policy**.

The **task-clustered 95% UCB of that gain is +0.032 to +0.038**, which is well above the 0.005 materiality threshold.

Therefore, **the seed-42 data do not establish that multiclass routing is unnecessary**. The apparent +2.2% mean gain is not statistically ruled out as large enough to matter.

The correct scientific statement is now:

> Seed-42 operator results are available, but the binary-vs-heterogeneous architectural conclusion is pending corrected fixed-continuation oracle analysis on the full three-seed replicated data.

## Implications for R13-B

R13-B eligibility is now **conditional on the three-seed replication** showing:

- `UCB_95(J_het - J_bin_best) < 0.005` at the reference λ.
- `VERIFY_TARGETED` remains the λ=0 best fixed continuation.

If the three-seed data also fail the equivalence criterion, the R13-B direction (binary STOP/CONTINUE with a fixed continuation) is not the only defensible architecture, and a lightweight multiclass router may still be considered. If the three-seed data pass the equivalence criterion, R13-B proceeds as preregistered.

## Files

- `scripts/analyze_r13a_v2_2.py`: corrected analyzer.
- `docs/research/R13B_PROTOCOL_ADDENDUM_2.md`: updated R13-B preregistration.

The following prior commits are preserved unchanged:
- `bec4716`: original seed-42 evidence boundary.
- `ed2c42a`: R13-B preregistration.
- `c0220c6`: R13-B Addendum 1.
