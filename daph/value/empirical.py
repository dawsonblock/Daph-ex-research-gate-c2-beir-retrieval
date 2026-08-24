"""DAPH I3.4 — Empirical action-value tables (B0 and B1).

B0: Global action mean — E[Q | a] ignoring phase.
B1: Phase × Action empirical table — E[Q | phase, a].

These are the simplest baselines. If a learned model doesn't beat B1,
don't deploy it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class GlobalActionMean:
    """B0: E[Q | a] — global action mean, ignoring phase."""

    def __init__(self):
        self._values: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def fit(self, transitions: list[dict], target_fn) -> "GlobalActionMean":
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for t in transitions:
            action = t["action"]
            target = target_fn(t)
            sums[action] += target
            counts[action] += 1
        self._values = {a: sums[a] / counts[a] for a in sums}
        self._counts = dict(counts)
        return self

    def predict(self, phase: str, action: str, features: dict) -> float:
        return self._values.get(action, 0.0)

    def predict_all(self, phase: str, legal_actions: list[str], features: dict) -> dict[str, float]:
        return {a: self.predict(phase, a, features) for a in legal_actions}

    @property
    def name(self) -> str:
        return "B0_global_action_mean"


class PhaseActionTable:
    """B1: E[Q | phase, a] — phase × action empirical table."""

    def __init__(self, *, min_samples: int = 3, fallback: "GlobalActionMean | None" = None):
        self._min_samples = min_samples
        self._fallback = fallback
        self._values: dict[tuple[str, str], float] = {}
        self._counts: dict[tuple[str, str], int] = {}

    def fit(self, transitions: list[dict], target_fn) -> "PhaseActionTable":
        sums: dict[tuple[str, str], float] = defaultdict(float)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for t in transitions:
            phase = t["phase_before"]
            action = t["action"]
            target = target_fn(t)
            sums[(phase, action)] += target
            counts[(phase, action)] += 1

        self._values = {}
        self._counts = dict(counts)
        for key, total in sums.items():
            n = counts[key]
            if n >= self._min_samples:
                self._values[key] = total / n
        return self

    def predict(self, phase: str, action: str, features: dict) -> float:
        key = (phase, action)
        if key in self._values:
            return self._values[key]
        # Fallback to global action mean if available
        if self._fallback is not None:
            return self._fallback.predict(phase, action, features)
        return 0.0

    def predict_all(self, phase: str, legal_actions: list[str], features: dict) -> dict[str, float]:
        return {a: self.predict(phase, a, features) for a in legal_actions}

    @property
    def name(self) -> str:
        return "B1_phase_action_table"

    def table(self) -> dict[str, dict[str, float]]:
        """Return the full table as {phase: {action: value}}."""
        result: dict[str, dict[str, float]] = defaultdict(dict)
        for (phase, action), value in self._values.items():
            result[phase][action] = round(value, 4)
        return dict(result)

    def sample_counts(self) -> dict[str, dict[str, int]]:
        """Return sample counts as {phase: {action: count}}."""
        result: dict[str, dict[str, int]] = defaultdict(dict)
        for (phase, action), count in self._counts.items():
            result[phase][action] = count
        return dict(result)
