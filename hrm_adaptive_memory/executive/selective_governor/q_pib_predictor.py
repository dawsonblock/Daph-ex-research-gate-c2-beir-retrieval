"""Q^{π_B}-based intervention gate predictor for V2B-I3.5.3.

Instead of using Q* (oracle optimal continuation value), this predictor
estimates Q^{π_B}(s, a) — the value of taking action a at state s and
continuing with the actual OFF model policy.

The estimator is trained from:
1. OFF trajectory data: Q^{π_B}(s, a_taken) = realized utility from s onward
2. I3.5.2d fork data: Q^{π_B}(s, a_G) = realized utility from governor-action fork

The gate intervenes only when:
  max_a Q^{π_B}(s, a) - Q^{π_B}(s, a_base) > threshold

This directly addresses the root cause identified in I3.5.2d:
  Q* ≠ Q^{π_model}
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from .features import InterventionFeatures, extract_features, FEATURE_NAMES
from .model import (
    BaseInterventionPredictor,
    InterventionPrediction,
)

# Extended feature names: original features + action one-hot
ACTION_NAMES = ["ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"]
EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [f"act_{a}" for a in ACTION_NAMES]

GATE_SCHEMA = "DAPH_V2B_I3_5_3_QPI_GATE_V1"
GATE_VERSION = 1


def features_action_vector(features: InterventionFeatures, action: str) -> list[float]:
    """Convert features + action into a numeric vector for the regression model."""
    vec = features.to_numeric_vector()
    action_onehot = [1.0 if action == a else 0.0 for a in ACTION_NAMES]
    return vec + action_onehot


class QPiBInterventionPredictor(BaseInterventionPredictor):
    """Predictor that estimates Q^{π_B}(s, a) and intervenes when the
    estimated policy-conditional advantage exceeds threshold.

    This replaces the Q*-based RuleBasedInterventionPredictor.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        delta_q_threshold: float = 5.0,
        max_harm_probability: float = 0.15,
        min_confidence: float = 0.60,
    ):
        self.model = model
        self.delta_q_threshold = delta_q_threshold
        self.max_harm_probability = max_harm_probability
        self.min_confidence = min_confidence
        self._fitted = model is not None

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        """Predict intervention outcome using Q^{π_B} regression model.

        Estimates Q^{π_B}(s, a) for all valid actions, finds the best action,
        and computes the advantage over the baseline action.
        """
        if not self._fitted or self.model is None:
            # Fail closed: don't intervene if model not fitted
            return InterventionPrediction(
                expected_delta_utility=-999.0,
                harm_probability=1.0,
                help_probability=0.0,
                confidence=0.0,
                reason="QPI_NOT_FITTED",
            )

        try:
            # Estimate Q^{π_B}(s, a) for all valid actions
            q_values = {}
            for action in ACTION_NAMES:
                vec = features_action_vector(features, action)
                q_pred = float(self.model.predict([vec])[0])
                q_values[action] = q_pred

            # The baseline action is the one the model would take without governor
            # We don't know it at gate time, so we use the worst-case:
            # the action with the lowest Q^{π_B} that isn't the governor's recommendation
            # Actually, we need to think about this differently.

            # The gate's job: given the state, should we intervene?
            # We intervene if there exists an action a_G such that:
            #   Q^{π_B}(s, a_G) - Q^{π_B}(s, a_base) > threshold
            # But we don't know a_base at gate time.

            # Approach: find the best action and the "default" action.
            # The default action is the one the model would likely take.
            # We can estimate this as the action with the highest Q^{π_B}
            # among "natural" actions (ANSWER, RETRIEVE, STOP).
            # The governor's recommendation would be the overall best action.

            # Simpler approach: compute the spread of Q^{π_B} values.
            # If max - min > threshold, there's a meaningful action choice.
            # The gate intervenes if the best action is significantly better
            # than the "natural" action.

            best_action = max(q_values, key=q_values.get)
            best_q = q_values[best_action]

            # Natural actions the model would take without governor
            natural_actions = ["ANSWER", "RETRIEVE", "STOP"]
            natural_q = {a: q_values[a] for a in natural_actions}
            natural_best = max(natural_q, key=natural_q.get)
            natural_best_q = natural_q[natural_best]

            delta_q = best_q - natural_best_q

            # Estimate harm probability: fraction of actions worse than natural
            worse_count = sum(1 for q in q_values.values() if q < natural_best_q - 5.0)
            harm_prob = worse_count / len(q_values)

            # Confidence based on how clear the advantage is
            confidence = min(1.0, max(0.0, delta_q / 50.0))

            should_intervene = (
                delta_q > self.delta_q_threshold
                and harm_prob < self.max_harm_probability
                and confidence >= self.min_confidence
            )

            reason = f"QPI_BEST_{best_action}_DELTA_{delta_q:.1f}"

            return InterventionPrediction(
                expected_delta_utility=delta_q,
                harm_probability=harm_prob,
                help_probability=confidence,
                confidence=confidence,
                reason=reason,
            )

        except Exception as e:
            return InterventionPrediction(
                expected_delta_utility=-999.0,
                harm_probability=1.0,
                help_probability=0.0,
                confidence=0.0,
                reason=f"QPI_ERROR:{type(e).__name__}",
            )

    def predict_q_values(self, features: InterventionFeatures) -> dict[str, float]:
        """Return Q^{π_B}(s, a) estimates for all valid actions."""
        if not self._fitted or self.model is None:
            return {}
        q_values = {}
        for action in ACTION_NAMES:
            vec = features_action_vector(features, action)
            q_values[action] = float(self.model.predict([vec])[0])
        return q_values

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "delta_q_threshold": self.delta_q_threshold,
                "max_harm_probability": self.max_harm_probability,
                "min_confidence": self.min_confidence,
                "schema": GATE_SCHEMA,
                "version": GATE_VERSION,
                "feature_names": EXTENDED_FEATURE_NAMES,
                "action_names": ACTION_NAMES,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "QPiBInterventionPredictor":
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            model=data["model"],
            delta_q_threshold=data["delta_q_threshold"],
            max_harm_probability=data["max_harm_probability"],
            min_confidence=data["min_confidence"],
        )
