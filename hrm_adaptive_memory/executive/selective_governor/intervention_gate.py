"""Selective Governor Intervention Gate.

The gate decides whether the governor is permitted to construct a decision frame
and inject it into the model packet, or whether the model receives the clean base packet.

Conservative default = silence:
The gate defaults to SKIP (no governor) whenever there is any doubt, error,
or insufficient evidence of positive expected utility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation

from .features import InterventionFeatures, extract_features
from .model import (
    BaseInterventionPredictor,
    InterventionPrediction,
    RuleBasedInterventionPredictor,
)

GATE_SCHEMA = "DAPH_V2B_I3_5_2_INTERVENTION_GATE_V1"
GATE_VERSION = 1

# Frozen default decision thresholds (Step 4 & Step 5)
DEFAULT_DELTA_U_THRESHOLD = 5.0
DEFAULT_MAX_HARM_PROBABILITY = 0.15
DEFAULT_MIN_CONFIDENCE = 0.60


@dataclass(frozen=True)
class InterventionDecision:
    """The frozen decision emitted by the selective intervention gate."""
    intervene: bool
    expected_delta_utility: float
    harm_probability: float
    confidence: float
    reason_code: str
    feature_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": "INTERVENE" if self.intervene else "SKIP",
            "intervene": self.intervene,
            "expected_delta_utility": round(self.expected_delta_utility, 4),
            "harm_probability": round(self.harm_probability, 4),
            "confidence": round(self.confidence, 4),
            "reason_code": self.reason_code,
            "feature_summary": self.feature_summary,
        }


class SelectiveGovernorGate:
    """Intervention gate that controls whether the governor is invoked.

    Control flow:
      ControllerObservation
              │
              ▼
      Intervention Gate
              │
              ├── SKIP ───────────────► Base packet ─► Model
              │
              └── INTERVENE
                      │
                      ▼
                   Governor
                      │
                      ▼
              Governor packet ────────► Model
    """

    def __init__(
        self,
        predictor: BaseInterventionPredictor | None = None,
        *,
        delta_u_threshold: float = DEFAULT_DELTA_U_THRESHOLD,
        max_harm_probability: float = DEFAULT_MAX_HARM_PROBABILITY,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.predictor = predictor or RuleBasedInterventionPredictor()
        self.delta_u_threshold = delta_u_threshold
        self.max_harm_probability = max_harm_probability
        self.min_confidence = min_confidence

    def assess(
        self,
        observation: ControllerObservation,
        *,
        remaining_steps: int,
        prior_actions: tuple[str, ...],
        prior_outcomes: tuple[str, ...],
    ) -> InterventionDecision:
        """Evaluate controller-visible state and decide whether to intervene.

        Fails closed to SKIP on any error.
        """
        try:
            # 1. Extract strictly controller-visible features
            features = extract_features(
                observation,
                remaining_steps=remaining_steps,
                prior_actions=prior_actions,
                prior_outcomes=prior_outcomes,
            )

            # 2. Predict intervention outcome
            pred = self.predictor.predict(features)

            # 3. Apply conservative gating threshold
            should_intervene = (
                pred.expected_delta_utility > self.delta_u_threshold
                and pred.harm_probability < self.max_harm_probability
                and pred.confidence >= self.min_confidence
            )

            reason_code = (
                f"INTERVENE_APPROVED:{pred.reason}"
                if should_intervene
                else f"SKIP_PREDICTION:{pred.reason}"
            )

            return InterventionDecision(
                intervene=should_intervene,
                expected_delta_utility=pred.expected_delta_utility,
                harm_probability=pred.harm_probability,
                confidence=pred.confidence,
                reason_code=reason_code,
                feature_summary={
                    "remaining_steps": features.remaining_steps,
                    "prior_action_count": features.prior_action_count,
                    "last_action": features.last_action,
                    "verification_state": features.verification_state,
                    "temporal_status": features.temporal_status,
                    "conflict_count": features.conflict_count,
                    "repeated_no_gain": features.repeated_no_gain,
                },
            )

        except Exception as e:
            # Conservative default = silence
            return InterventionDecision(
                intervene=False,
                expected_delta_utility=-999.0,
                harm_probability=1.0,
                confidence=0.0,
                reason_code=f"SKIP_ERROR_FALLBACK:{type(e).__name__}",
                feature_summary={},
            )
