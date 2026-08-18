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
    """Calibrated rule-based predictor derived from I3.5.1 divergence analysis.

    Key empirical findings from development data:
    1. At step 0 when verification is SUFFICIENT and temporal is CURRENT:
       Unaided model chooses STOP (which succeeds). Governor intervenes with ANSWER (-120 ΔU penalty).
       -> RULE: STOP_ANSWER_HAZARD -> Predict HARM (-120.0, harm_prob=1.0)

    2. At step 0 when verification is MISSING and conflict is 1:
       Unaided model retrieves evidence first. Governor prematurely forces VERIFY (-9.5 ΔU).
       -> RULE: PREMATURE_VERIFY_HAZARD -> Predict HARM (-9.5, harm_prob=0.95)

    3. After 2+ steps when last_action was VERIFY and verification remains MISSING:
       Unaided model terminates with ANSWER (cuts loss). Governor forces SEARCH_MORE loop (-9.3 ΔU).
       -> RULE: IRREDUCIBLE_LOOP_HAZARD -> Predict HARM (-9.3, harm_prob=0.95)

    4. When repeated_no_gain is True:
       Governor intervention has 0% empirical benefit rate.
       -> RULE: NO_GAIN_HAZARD -> Predict HARM (-10.0, harm_prob=0.90)

    5. General conservative default:
       Governor has 0% empirical positive rate in development data.
       Any unhandled state defaults to NEUTRAL/HARM -> SKIP.
    """

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        try:
            # Rule 1: Step 0 with SUFFICIENT / CURRENT evidence (State-irrelevant STOP hazard)
            if (
                features.prior_action_count == 0
                and features.verification_state == "SUFFICIENT"
                and features.temporal_status == "CURRENT"
                and features.conflict_count == 0
            ):
                return InterventionPrediction(
                    expected_delta_utility=-120.0,
                    harm_probability=1.0,
                    help_probability=0.0,
                    confidence=1.0,
                    reason="RULE:STEP0_SUFFICIENT_STOP_HAZARD",
                )

            # Rule 2: Step 0 with MISSING verification and conflict (Premature VERIFY hazard)
            if (
                features.prior_action_count == 0
                and features.verification_state == "MISSING"
                and features.last_action is None
            ):
                return InterventionPrediction(
                    expected_delta_utility=-9.5,
                    harm_probability=0.95,
                    help_probability=0.0,
                    confidence=0.95,
                    reason="RULE:STEP0_MISSING_PREMATURE_VERIFY_HAZARD",
                )

            # Rule 3: Irreducible loop hazard (after VERIFY, verification still MISSING)
            if (
                features.prior_action_count >= 2
                and features.last_action == "VERIFY"
                and features.verification_state in ("MISSING", "FALSIFIED")
            ):
                return InterventionPrediction(
                    expected_delta_utility=-9.3,
                    harm_probability=0.95,
                    help_probability=0.0,
                    confidence=0.95,
                    reason="RULE:POST_VERIFY_IRREDUCIBLE_SEARCH_LOOP_HAZARD",
                )

            # Rule 4: Repeated no gain
            if features.repeated_no_gain:
                return InterventionPrediction(
                    expected_delta_utility=-10.0,
                    harm_probability=0.90,
                    help_probability=0.0,
                    confidence=0.90,
                    reason="RULE:REPEATED_NO_GAIN_HAZARD",
                )

            # Rule 5: FALSIFIED verification state (Model terminates, gov forces re-verify)
            if features.verification_state == "FALSIFIED":
                return InterventionPrediction(
                    expected_delta_utility=-8.7,
                    harm_probability=0.90,
                    help_probability=0.0,
                    confidence=0.90,
                    reason="RULE:FALSIFIED_STATE_OVER_INTERVENTION_HAZARD",
                )

            # Rule 6: Resource exhaustion (low remaining steps or budget)
            if (
                features.remaining_steps <= 2
                or features.retrieval_budget_remaining == 0
                and features.verification_budget_remaining == 0
            ):
                return InterventionPrediction(
                    expected_delta_utility=-5.0,
                    harm_probability=0.85,
                    help_probability=0.0,
                    confidence=0.85,
                    reason="RULE:LOW_RESOURCE_HAZARD",
                )

            # Conservative fallback: unclassified state
            return InterventionPrediction(
                expected_delta_utility=-5.0,
                harm_probability=0.80,
                help_probability=0.0,
                confidence=0.50,
                reason="RULE:CONSERVATIVE_DEFAULT_NO_BENEFIT_OBSERVED",
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
