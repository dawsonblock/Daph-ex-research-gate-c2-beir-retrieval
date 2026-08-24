# I3.4E Findings — Frozen Baseline for I3.5

## Experiment summary

I3.4e is a control-decomposition and frozen-permutation experiment that tests
what part of the P2 intervention is responsible for observed behavior.

- **Phase A**: 80 tasks × 7 arms = 560 trajectories
- **Phase B**: 30 tasks × 16 PS arms = 480 trajectories
- **Total**: 1040 trajectories, 0 failures, 0 collisions

## Phase A — Control decomposition

| Arm | Mean Utility | Success | Description |
|-----|-------------|---------|-------------|
| P2  | 42.43       | 0.88    | Phase + B1 values (full intervention) |
| B0  | 37.57       | 0.85    | Global action prior (DEFER-biased) |
| DEFER | 30.73     | 0.79    | Frozen DEFER heuristic |
| PV  | 27.20       | 0.78    | Phase + numeric values, no ranking |
| CONST | 22.72     | 0.75    | Uniform values |
| P0  | 21.85       | 0.75    | Baseline (no phase, no values) |
| PR  | 22.10       | 0.75    | Phase + ranking, no numeric values |

### Key contrasts

- ΔU_P2 = +20.58, CI=[+8.25, +34.32] — excludes 0
- ΔU_B0 = +15.72, CI=[+4.55, +28.70] — excludes 0
- P2-B0 = +4.86, CI=[-4.37, +14.38] — **includes 0**
- CONST and PR barely above P0 (~+0.86 and ~+0.24)

## Phase B — Frozen permutation screen (corrected percentiles)

**Corrected** (same 30 Phase B tasks, per CORRECTION-001):

| Statistic | Value |
|-----------|-------|
| P2 mean on Phase B tasks | 47.74 |
| B0 mean on Phase B tasks | 41.19 |
| P2 percentile in PS distribution | 75.0% (12/16 below) |
| B0 percentile in PS distribution | 43.8% (7/16 below) |
| PS arms beating P2 | 4/16 (25%) |
| PS arms beating B0 | 9/16 (56%) |
| median(PS - P2) | -5.11 |
| median(PS - B0) | +1.44 |

### Permutation distribution

- Mean of means: 43.62, SD: 8.84
- Range: [32.56, 64.73]
- PS05 is the top outlier (64.73, 100% success on one_live)

### one_live stratum (5 tasks)

- P0: 0% success, all ANSWER
- P2: 40% success, 40% DEFER
- B0: 20% success, 20% DEFER
- PS05: 100% success, 100% DEFER (rescued all 5)
- PS08: 80% success, 80% DEFER (rescued 4/5)
- 10/16 PS arms rescued 0 tasks (behaved like P0)

### Structural regression

- R² = 0.20 (weak)
- DEFER rank vs utility: r = -0.27 (lower DEFER rank → higher utility)
- VERIFY rank vs utility: r = +0.21

## Proven

1. **P0 < B0**: The global DEFER-biased prior significantly improves over no guidance.
2. **P0 < P2**: The full phase-conditioned intervention significantly improves over no guidance.
3. **P2 does not significantly outperform B0**: CI includes 0 on both Phase A and Phase B tasks.
4. **Action prior content matters**: DEFER and B0 capture a large share of P2's gain.
5. **Packet structure alone contributes very little**: CONST and PR are nearly baseline.
6. **Some frozen alternative priors outperform B1/P2**: 4/16 PS arms beat P2 on matched tasks.
7. **P2 sits at the 75th percentile** of the PS distribution on matched tasks — B1 is better than most random mappings but not special.

## Not proven

1. **B1 is a good causal action-value function**: B1 estimates Q(phase, a) from observational data. The causal structure of the R2 transitions does not establish that B1 captures the true Q*(s, a).
2. **Phase conditioning adds reliable value beyond B0**: The P2-B0 CI includes 0. The phase-conditioned values may add marginal value, but it is not statistically convincing.
3. **Random mappings generally help**: The PS distribution is wide. Some mappings help a lot (PS05), some hurt a lot (PS13, PS14). One successful mapping does not prove arbitrary rankings help.
4. **PS05 represents an intelligent policy**: PS05 succeeds by encoding a strong DEFER preference that happens to be correct for the one_live stratum. This is a heuristic exploit, not state-sensitive intelligence.

## Architecture assessment

The current architecture works:

```
state representation works → policy-conditioning channel works → action prior matters → B1 is useful but not sufficiently state-specific
```

The missing capability is:

```
fine-grained, causal discrimination between actions inside the same epistemic phase
```

B1 estimates Q(phase, a) but the target is Q(s, a). Two states in the same phase
may require different actions. The one_live stratum demonstrates this: some
one_live states should DEFER, others should VERIFY, RETRIEVE, or SEARCH_MORE.

## Implications for I3.5

1. The benchmark must be hardened so that no single-action heuristic can dominate.
2. Causal action data must be collected by forcing actions from checkpoints, not by observing policy-selected actions.
3. The state representation must be richer than phase alone.
4. The model ladder should start simple (B0, B1, linear, GBT) before neural networks.
5. The primary gate is Q_φ > B0, not Q_φ > P2.
