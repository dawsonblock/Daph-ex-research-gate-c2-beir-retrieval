"""Pairwise advantage intervention predictor for V2B-I3.5.3-r1.

Instead of estimating Q^{π_B}(s, a) for all 7 actions, this predictor directly
estimates the pairwise advantage:

  ΔQ_π(s, a_B, a_G) = Q^{π_B}(s, a_G) - Q^{π_B}(s, a_B)

Input features: state features + a_B one-hot + a_G one-hot
Output: scalar ΔQ_π

The gate intervenes only when LCB(ΔQ_π) > threshold.

This is the correct specification:
  intervene iff Q^{π_B}(s, a_G) - Q^{π_B}(s, a_B) > 0

No guessed "natural actions." No 7-action Q estimation. No fake harm probability.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .features import InterventionFeatures, FEATURE_NAMES
from .model import BaseInterventionPredictor, InterventionPrediction

# Action names for one-hot encoding
ACTION_NAMES = ["ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"]

# Feature names: state features + a_B one-hot + a_G one-hot
PAIRWISE_FEATURE_NAMES = (
    FEATURE_NAMES
    + [f"a_base_{a}" for a in ACTION_NAMES]
    + [f"a_gov_{a}" for a in ACTION_NAMES]
)

GATE_SCHEMA = "DAPH_V2B_I3_5_3R1_PAIRWISE_GATE_V1"
GATE_VERSION = 1


def pairwise_feature_vector(
    features: InterventionFeatures,
    a_base: str,
    a_gov: str,
) -> list[float]:
    """Convert state features + (a_base, a_gov) into a numeric vector."""
    vec = features.to_numeric_vector()
    base_onehot = [1.0 if a_base == a else 0.0 for a in ACTION_NAMES]
    gov_onehot = [1.0 if a_gov == a else 0.0 for a in ACTION_NAMES]
    return vec + base_onehot + gov_onehot


class PairwiseAdvantagePredictor(BaseInterventionPredictor):
    """Predictor that directly estimates ΔQ_π(s, a_B, a_G).

    Trained from fork data where:
      ΔQ_π = U(a_G + OFF continuation) - U(a_B + OFF continuation)

    At runtime, the gate receives both a_B (from the base model call) and
    a_G (from the governor), and predicts ΔQ_π. Intervention is approved
    only if the predicted ΔQ_π exceeds threshold.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        delta_threshold: float = 5.0,
        lcb_margin: float = 5.0,
    ):
        """
        Args:
            model: Fitted regression model with predict() method.
            delta_threshold: Minimum predicted ΔQ_π to consider intervention.
            lcb_margin: Safety margin subtracted from prediction for LCB.
                        Intervention requires (predicted - lcb_margin) > delta_threshold.
        """
        self.model = model
        self.delta_threshold = delta_threshold
        self.lcb_margin = lcb_margin
        self._fitted = model is not None

    def predict_advantage(
        self,
        features: InterventionFeatures,
        a_base: str,
        a_gov: str,
    ) -> float:
        """Predict ΔQ_π(s, a_B, a_G). Returns 0.0 if not fitted."""
        if not self._fitted or self.model is None:
            return 0.0
        vec = pairwise_feature_vector(features, a_base, a_gov)
        return float(self.model.predict([vec])[0])

    def should_intervene(
        self,
        features: InterventionFeatures,
        a_base: str,
        a_gov: str,
    ) -> tuple[bool, float, str]:
        """Decide whether to intervene.

        Returns (should_intervene, predicted_delta_q_pi, reason).
        """
        if a_base == a_gov:
            return False, 0.0, "SKIP_SAME_ACTION"

        if not self._fitted or self.model is None:
            return False, 0.0, "SKIP_NOT_FITTED"

        delta_q_pi = self.predict_advantage(features, a_base, a_gov)
        lcb = delta_q_pi - self.lcb_margin

        if lcb > self.delta_threshold:
            return True, delta_q_pi, f"INTERVENE_LCB_{lcb:.1f}_THRESH_{self.delta_threshold}"
        else:
            return False, delta_q_pi, f"SKIP_LCB_{lcb:.1f}_BELOW_THRESH_{self.delta_threshold}"

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        """Compatibility method for the base predictor interface.

        NOTE: This is a fallback. The base-first gate should call
        should_intervene() directly with a_base and a_gov.
        """
        # Without a_base and a_gov, we cannot make a prediction.
        # Fail closed.
        return InterventionPrediction(
            expected_delta_utility=0.0,
            harm_probability=0.0,
            help_probability=0.0,
            confidence=0.0,
            reason="PAIRWISE_REQUIRES_ACTIONS",
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "delta_threshold": self.delta_threshold,
                "lcb_margin": self.lcb_margin,
                "schema": GATE_SCHEMA,
                "version": GATE_VERSION,
                "feature_names": PAIRWISE_FEATURE_NAMES,
                "action_names": ACTION_NAMES,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseAdvantagePredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            model=data["model"],
            delta_threshold=data["delta_threshold"],
            lcb_margin=data["lcb_margin"],
        )
