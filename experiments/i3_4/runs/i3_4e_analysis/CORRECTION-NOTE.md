# I3.4E-ANALYSIS-CORRECTION-001

## Issue

The frozen Phase B report (`01_permutation_distribution.json`) compared Phase A
P2/B0 means (computed over all 80 Phase A tasks) against the Phase B PS
distribution (computed over 30 Phase B tasks). Those are different task samples,
making the percentile comparison invalid.

## Old (incorrect) values

| Statistic | Old value | Source |
|-----------|-----------|--------|
| P2 percentile | 50.0% | Phase A 80-task mean (42.43) vs Phase B 30-task PS distribution |
| B0 percentile | 18.75% | Phase A 80-task mean (37.57) vs Phase B 30-task PS distribution |

## Corrected values (same 30 Phase B tasks)

| Statistic | Corrected value |
|-----------|----------------|
| P2 mean on Phase B tasks | 47.74 |
| B0 mean on Phase B tasks | 41.19 |
| P0 mean on Phase B tasks | 38.15 |
| P2 percentile | 75.0% (12/16 PS below) |
| B0 percentile | 43.8% (7/16 PS below) |
| PS > P2 | 4/16 (25%) |
| PS > B0 | 9/16 (56%) |
| median(PS - P2) | -5.11 |
| mean(PS - P2) | -4.12 |
| median(PS - B0) | +1.44 |
| mean(PS - B0) | +2.43 |

## Corrected interpretation

- **P2 performs above most frozen permutations** (75th percentile on matched tasks).
  B1 is not useless — it is better than most random mappings on the same tasks.
- **B0 performs around the middle** (43.8th percentile). The global DEFER-biased
  prior is a strong simple baseline.
- **Some fixed mappings still outperform P2** (4/16), including PS05 which
  substantially outperforms everything.
- **P2 does not significantly beat B0** (CI includes 0 on both Phase A and
  Phase B tasks).
- **B1 is useful but not sufficiently state-specific.** The phase-conditioned
  values add something over most random mappings, but do not reliably beat
  the simpler B0 global prior.

## Effect on scientific disposition

This correction makes P2/B1 look somewhat better, not worse. The original
report understated P2's position by comparing across different task samples.
The corrected analysis supports the conclusion that the architecture is worth
continuing, but the action prior needs finer state resolution than phase alone.

## Files

- `correction/phase_b_same_task_distribution.json` — distribution stats on same 30 tasks
- `correction/phase_b_corrected_percentiles.json` — corrected percentiles and beat counts
- Original report preserved unchanged at `01_permutation_distribution.json`
