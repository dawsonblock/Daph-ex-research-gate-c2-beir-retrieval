"""Fixes a real bug in run_g1_typed_path.py / run_g2_graph_traversal.py: both
picked their reported ``decision_at_M`` by ranking utility/closure across ALL
arms, without first restricting to arms that pass their own hard safety gates.
That let a bridge-safety-violating M be reported as "the decision" merely
because it had the best closure number.

The fix, per configs/gate_g2_v2_path_completion.json:decision_at_M_hard_gate_fix::
filter to eligible arms FIRST, rank second. If nothing is eligible, the answer
is NONE, never "pick the least-bad ineligible arm."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, TypeVar

K = TypeVar("K")


@dataclass(frozen=True)
class GatedDecision:
    key: K | None
    eligible_keys: tuple[K, ...]
    reason: str


def select_eligible_decision(
    candidates: Mapping[K, bool],
    ranking_metric: Mapping[K, float | None],
) -> GatedDecision:
    """``candidates`` maps each arm key to whether it passed ALL of that arm's
    hard safety gates. ``ranking_metric`` maps each key to the value used to
    rank among eligible arms (higher is better; None is treated as worst).
    Only keys with candidates[key] is True are ever eligible for selection."""
    eligible = tuple(k for k, passed in candidates.items() if passed)
    if not eligible:
        return GatedDecision(key=None, eligible_keys=(),
                             reason="no arm passed all hard safety gates")
    best = max(eligible, key=lambda k: (ranking_metric.get(k) is not None,
                                        ranking_metric.get(k) if ranking_metric.get(k) is not None else float("-inf")))
    return GatedDecision(key=best, eligible_keys=eligible, reason="ranked among eligible arms")
