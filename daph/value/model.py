"""DAPH I3.4 — Learned action-value models (B2 and B3).

B2: Linear/logistic regression on phase + features.
B3: Gradient-boosted trees (LightGBM if available, else sklearn GBT).

Each model predicts Q(phase, features, action) for a given action.
The ranking module uses these predictions to rank legal actions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from daph.value.dataset import (
    get_feature_vector, get_phase, get_action,
    FEATURE_NAMES,
)


def _encode_phase(phase: str, all_phases: tuple[str, ...]) -> list[float]:
    """One-hot encode phase."""
    return [1.0 if phase == p else 0.0 for p in all_phases]


def _encode_action(action: str, all_actions: tuple[str, ...]) -> list[float]:
    """One-hot encode action."""
    return [1.0 if action == a else 0.0 for a in all_actions]


def build_design_matrix(
    transitions: list[dict],
    all_phases: tuple[str, ...],
    all_actions: tuple[str, ...],
) -> np.ndarray:
    """Build design matrix: [phase_onehot, action_onehot, features]."""
    X = []
    for t in transitions:
        phase = get_phase(t)
        action = get_action(t)
        features = get_feature_vector(t)
        row = _encode_phase(phase, all_phases) + _encode_action(action, all_actions) + features
        X.append(row)
    return np.array(X, dtype=np.float64)


def get_all_phases(transitions: list[dict]) -> tuple[str, ...]:
    return tuple(sorted(set(get_phase(t) for t in transitions)))


def get_all_actions(transitions: list[dict]) -> tuple[str, ...]:
    return tuple(sorted(set(get_action(t) for t in transitions)))


class LinearValueModel:
    """B2: Linear regression on [phase_onehot, action_onehot, features]."""

    def __init__(self):
        self._model = None
        self._all_phases: tuple[str, ...] = ()
        self._all_actions: tuple[str, ...] = ()
        self._feature_dim: int = 0

    def fit(self, transitions: list[dict], target_fn) -> "LinearValueModel":
        from sklearn.linear_model import Ridge

        self._all_phases = get_all_phases(transitions)
        self._all_actions = get_all_actions(transitions)

        X = build_design_matrix(transitions, self._all_phases, self._all_actions)
        y = np.array([target_fn(t) for t in transitions], dtype=np.float64)

        self._feature_dim = X.shape[1]
        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)
        return self

    def predict(self, phase: str, action: str, features: dict) -> float:
        row = (
            _encode_phase(phase, self._all_phases)
            + _encode_action(action, self._all_actions)
            + [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
        )
        X = np.array([row], dtype=np.float64)
        return float(self._model.predict(X)[0])

    def predict_all(self, phase: str, legal_actions: list[str], features: dict) -> dict[str, float]:
        return {a: self.predict(phase, a, features) for a in legal_actions}

    @property
    def name(self) -> str:
        return "B2_linear_ridge"


class GBTValueModel:
    """B3: Gradient-boosted trees."""

    def __init__(self, *, n_estimators: int = 100, max_depth: int = 4):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._model = None
        self._all_phases: tuple[str, ...] = ()
        self._all_actions: tuple[str, ...] = ()

    def fit(self, transitions: list[dict], target_fn) -> "GBTValueModel":
        self._all_phases = get_all_phases(transitions)
        self._all_actions = get_all_actions(transitions)

        X = build_design_matrix(transitions, self._all_phases, self._all_actions)
        y = np.array([target_fn(t) for t in transitions], dtype=np.float64)

        try:
            from lightgbm import LGBMRegressor
            self._model = LGBMRegressor(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                verbose=-1,
                seed=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = GradientBoostingRegressor(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                random_state=42,
            )

        self._model.fit(X, y)
        return self

    def predict(self, phase: str, action: str, features: dict) -> float:
        row = (
            _encode_phase(phase, self._all_phases)
            + _encode_action(action, self._all_actions)
            + [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
        )
        X = np.array([row], dtype=np.float64)
        return float(self._model.predict(X)[0])

    def predict_all(self, phase: str, legal_actions: list[str], features: dict) -> dict[str, float]:
        return {a: self.predict(phase, a, features) for a in legal_actions}

    @property
    def name(self) -> str:
        return "B3_gradient_boosted_trees"


class RandomForestValueModel:
    """B3b: Random forest (ensemble for uncertainty estimation)."""

    def __init__(self, *, n_estimators: int = 100, max_depth: int = 6):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._model = None
        self._all_phases: tuple[str, ...] = ()
        self._all_actions: tuple[str, ...] = ()

    def fit(self, transitions: list[dict], target_fn) -> "RandomForestValueModel":
        from sklearn.ensemble import RandomForestRegressor

        self._all_phases = get_all_phases(transitions)
        self._all_actions = get_all_actions(transitions)

        X = build_design_matrix(transitions, self._all_phases, self._all_actions)
        y = np.array([target_fn(t) for t in transitions], dtype=np.float64)

        self._model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict(self, phase: str, action: str, features: dict) -> float:
        row = (
            _encode_phase(phase, self._all_phases)
            + _encode_action(action, self._all_actions)
            + [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
        )
        X = np.array([row], dtype=np.float64)
        return float(self._model.predict(X)[0])

    def predict_with_uncertainty(self, phase: str, action: str, features: dict) -> tuple[float, float]:
        """Predict mean and std across trees."""
        row = (
            _encode_phase(phase, self._all_phases)
            + _encode_action(action, self._all_actions)
            + [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
        )
        X = np.array([row], dtype=np.float64)
        predictions = [tree.predict(X)[0] for tree in self._model.estimators_]
        mean = float(np.mean(predictions))
        std = float(np.std(predictions))
        return mean, std

    def predict_all(self, phase: str, legal_actions: list[str], features: dict) -> dict[str, float]:
        return {a: self.predict(phase, a, features) for a in legal_actions}

    def predict_all_with_uncertainty(
        self, phase: str, legal_actions: list[str], features: dict
    ) -> dict[str, tuple[float, float]]:
        return {a: self.predict_with_uncertainty(phase, a, features) for a in legal_actions}

    @property
    def name(self) -> str:
        return "B3b_random_forest"
