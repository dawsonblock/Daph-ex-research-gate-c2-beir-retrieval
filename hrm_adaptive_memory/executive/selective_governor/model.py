"""Intervention predictors for selective governor intervention gate.

Provides interpretable rule-based and linear/calibrated predictors that estimate
the expected utility delta and harm probability of governor intervention.

Conservative default: any unexpected or uncalibrated state predicts HARM / SKIP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .features import InterventionFeatures, FEATURE_NAMES

PREDICTOR_SCHEMA = "DAPH_V2B_I3_5_2_INTERVENTION_PREDICTOR_V1"
PREDICTOR_VERSION = 1


@dataclass(frozen=True)
class InterventionPrediction:
    """Prediction output from an intervention model."""
    expected_delta_utility: float
    harm_probability: float
    help_probability: float
    confidence: float
    reason: str


class BaseInterventionPredictor:
    """Abstract base class for intervention prediction."""

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        raise NotImplementedError


class RuleBasedInterventionPredictor(BaseInterventionPredictor):
    """Calibrated rule-based predictor derived from state-level counterfactual Q-advantage analysis.

    Key empirical findings from development state counterfactuals (758 states):
    1. Step 0 SUFFICIENT/CURRENT: Model stops, governor forces answer -> HARM (-120.0 ΔQ, harm_prob=1.0)
    2. Step 0 MISSING: Model retrieves, governor prematurely forces verify -> HARM (-50.8 ΔQ, harm_prob=0.95)
    3. Step 1 post-RETRIEVE: Model already verifies, governor agrees -> NEUTRAL (0.0 ΔQ, harm_prob=0.0)
    4. Step 2+ post-VERIFY with MISSING/FALSIFIED: Model prematurely terminates with fatal answer;
       governor prevents fatal answer by exploring -> SAFE_HELP (+83.5 ΔQ, harm_prob=0.00, help_rate=68.0%)
    5. Step 3+ post-SEARCH_MORE: Governor prevents premature termination -> SAFE_HELP (+86.8 ΔQ, harm_prob=0.11, help_rate=87.3%)
    6. Repeated no gain or low resources -> LIKELY_HARM -> SKIP
    """

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        try:
            # Rule 1: Step 0 is dangerous (STOP override or premature VERIFY) -> LIKELY_HARM -> SKIP
            if features.prior_action_count == 0:
                if (
                    features.verification_state == "SUFFICIENT"
                    and features.temporal_status == "CURRENT"
                    and features.conflict_count == 0
                ):
                    return InterventionPrediction(
                        expected_delta_utility=-120.0,
                        harm_probability=1.0,
                        help_probability=0.0,
                        confidence=1.0,
                        reason="LIKELY_HARM:STEP0_SUFFICIENT_STOP_HAZARD",
                    )
                return InterventionPrediction(
                    expected_delta_utility=-25.0,
                    harm_probability=0.90,
                    help_probability=0.02,
                    confidence=0.95,
                    reason="LIKELY_HARM:STEP0_PREMATURE_INTERVENTION_HAZARD",
                )

            # Rule 2: Step 1 (model already chooses VERIFY after RETRIEVE; 100% agreement) -> NEUTRAL -> SKIP
            if features.prior_action_count == 1 and features.last_action == "RETRIEVE":
                return InterventionPrediction(
                    expected_delta_utility=0.0,
                    harm_probability=0.0,
                    help_probability=0.0,
                    confidence=0.99,
                    reason="NEUTRAL:STEP1_RETRIEVE_AGREEMENT_NO_OP",
                )

            # Rule 3: Repeated no gain -> LIKELY_HARM -> SKIP
            if features.repeated_no_gain:
                return InterventionPrediction(
                    expected_delta_utility=-10.0,
                    harm_probability=0.90,
                    help_probability=0.0,
                    confidence=0.90,
                    reason="LIKELY_HARM:REPEATED_NO_GAIN_HAZARD",
                )

            # Rule 4: Resource exhaustion -> LIKELY_HARM -> SKIP
            if (
                features.remaining_steps <= 2
                or (features.retrieval_budget_remaining == 0 and features.verification_budget_remaining == 0)
            ):
                return InterventionPrediction(
                    expected_delta_utility=-5.0,
                    harm_probability=0.85,
                    help_probability=0.0,
                    confidence=0.85,
                    reason="LIKELY_HARM:LOW_RESOURCE_HAZARD",
                )

            # Rule 5: Step 2+ post-VERIFY with MISSING / FALSIFIED evidence
            # This is the proven positive intervention region: preventing premature failing ANSWER!
            if (
                features.prior_action_count >= 2
                and features.last_action == "VERIFY"
                and features.verification_state in ("MISSING", "FALSIFIED")
            ):
                return InterventionPrediction(
                    expected_delta_utility=+83.5,
                    harm_probability=0.00,
                    help_probability=0.68,
                    confidence=0.85,
                    reason="SAFE_HELP:POST_VERIFY_PREMATURE_TERMINATION_PREVENTION",
                )

            # Rule 6: Step 3+ post-SEARCH_MORE
            if (
                features.prior_action_count >= 3
                and features.last_action == "SEARCH_MORE"
            ):
                return InterventionPrediction(
                    expected_delta_utility=+86.8,
                    harm_probability=0.11,
                    help_probability=0.87,
                    confidence=0.80,
                    reason="SAFE_HELP:POST_SEARCH_PREMATURE_TERMINATION_PREVENTION",
                )

            # Conservative fallback: unclassified / uncertain state -> SKIP_UNCERTAIN
            return InterventionPrediction(
                expected_delta_utility=0.0,
                harm_probability=0.50,
                help_probability=0.0,
                confidence=0.40,
                reason="UNCERTAIN:CONSERVATIVE_DEFAULT_SKIP",
            )

        except Exception as e:
            # Conservative default = silence
            return InterventionPrediction(
                expected_delta_utility=-999.0,
                harm_probability=1.0,
                help_probability=0.0,
                confidence=0.0,
                reason=f"EXCEPTION_DEFAULT_SKIP:{e}",
            )


class CalibratedLinearPredictor(BaseInterventionPredictor):
    """Linear regression / logistic model with frozen weights for continuous delta-U prediction."""

    def __init__(
        self,
        weights: list[float] | None = None,
        bias: float = -5.0,
        harm_weights: list[float] | None = None,
        harm_bias: float = 1.5,
    ):
        self.weights = weights or [0.0] * len(FEATURE_NAMES)
        self.bias = bias
        self.harm_weights = harm_weights or [0.0] * len(FEATURE_NAMES)
        self.harm_bias = harm_bias

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        try:
            vec = features.to_numeric_vector()
            if len(vec) != len(self.weights):
                return InterventionPrediction(
                    expected_delta_utility=-999.0,
                    harm_probability=1.0,
                    help_probability=0.0,
                    confidence=0.0,
                    reason="DIMENSION_MISMATCH_DEFAULT_SKIP",
                )

            # Linear prediction for expected ΔU
            expected_du = self.bias + sum(w * x for w, x in zip(self.weights, vec))

            # Logistic prediction for P(HARM)
            harm_logit = self.harm_bias + sum(w * x for w, x in zip(self.harm_weights, vec))
            harm_prob = 1.0 / (1.0 + math.exp(-max(min(harm_logit, 20.0), -20.0)))

            help_prob = max(0.0, 1.0 - harm_prob) if expected_du > 0 else 0.0
            confidence = abs(harm_prob - 0.5) * 2.0

            return InterventionPrediction(
                expected_delta_utility=expected_du,
                harm_probability=harm_prob,
                help_probability=help_prob,
                confidence=confidence,
                reason="CALIBRATED_LINEAR_MODEL",
            )
        except Exception as e:
            return InterventionPrediction(
                expected_delta_utility=-999.0,
                harm_probability=1.0,
                help_probability=0.0,
                confidence=0.0,
                reason=f"EXCEPTION_DEFAULT_SKIP:{e}",
            )
