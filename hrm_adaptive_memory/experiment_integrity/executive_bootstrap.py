"""Bootstrap LCB for ExecutiveOpportunity = U(E3) - max(U(E0), U(E1)).

scripts.diagnose_c5_confirmation_stopgate.grouped_lcb bootstraps the LCB of a
simple per-task MEAN (paired deltas resampled by group). ExecutiveOpportunity
is not that: it is a difference between one per-task-averaged term
(mean(max(Q_E0, Q_E1))) and the MAX of two other per-task-averaged terms
(max(mean(Q_E0), mean(Q_E1))). Because of that max(), the statistic is
nonlinear in the per-task outcomes and grouped_lcb's single-value-per-task
interface cannot express it -- each bootstrap replicate needs the resampled
Q_E0 and Q_E1 arrays together, not one flattened value.

This module generalizes grouped_lcb's exact resampling convention (resample
GROUPS with replacement, not tasks; same default seed and iteration count)
to that compound statistic, so the two remain methodologically consistent
even though grouped_lcb itself cannot be reused directly.
"""
from __future__ import annotations

import random
from collections import defaultdict


def grouped_lcb_executive_opportunity(
    triples: list[tuple[str, float, float]], iterations: int = 2000, seed: int = 12345,
) -> float | None:
    """LCB (2.5th percentile) of ExecutiveOpportunity = mean(max(q0,q1)) -
    max(mean(q0), mean(q1)), resampling GROUPS (e.g. family) with replacement.

    ``triples`` is a list of (group_key, Q(A0), Q(A1)) for every task, exactly
    the shape the executive-opportunity runner already has on hand per task
    (no extra pipeline calls needed to compute this).
    """
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key, q0, q1 in triples:
        groups[key].append((q0, q1))
    keys = sorted(groups)
    if not keys:
        return None

    rng = random.Random(seed)
    opportunities: list[float] = []
    for _ in range(iterations):
        picked = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [pair for g in picked for pair in g]
        if not flat:
            continue
        oracle_mean = sum(max(q0, q1) for q0, q1 in flat) / len(flat)
        e0_mean = sum(q0 for q0, _ in flat) / len(flat)
        e1_mean = sum(q1 for _, q1 in flat) / len(flat)
        opportunities.append(oracle_mean - max(e0_mean, e1_mean))

    if not opportunities:
        return None
    opportunities.sort()
    return round(opportunities[int(0.025 * len(opportunities))], 4)
