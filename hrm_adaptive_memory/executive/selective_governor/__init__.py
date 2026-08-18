"""Selective Governor Intervention Gate package for DAPH V2B-I3.5.2."""
from .features import InterventionFeatures, extract_features, FEATURE_NAMES
from .intervention_gate import (
    SelectiveGovernorGate,
    InterventionDecision,
    DEFAULT_DELTA_U_THRESHOLD,
    DEFAULT_MAX_HARM_PROBABILITY,
    DEFAULT_MIN_CONFIDENCE,
)
from .model import (
    BaseInterventionPredictor,
    InterventionPrediction,
    RuleBasedInterventionPredictor,
    CalibratedLinearPredictor,
)
from .identity import compute_gate_identity
from .serializer import serialize_decision, decision_sha256
from .calibration import (
    evaluate_gate_on_counterfactual_states,
    evaluate_gate_on_dataset,
    GateEvaluationMetrics,
)

__all__ = [
    "InterventionFeatures",
    "extract_features",
    "FEATURE_NAMES",
    "SelectiveGovernorGate",
    "InterventionDecision",
    "BaseInterventionPredictor",
    "InterventionPrediction",
    "RuleBasedInterventionPredictor",
    "CalibratedLinearPredictor",
    "compute_gate_identity",
    "serialize_decision",
    "decision_sha256",
    "evaluate_gate_on_counterfactual_states",
    "evaluate_gate_on_dataset",
    "GateEvaluationMetrics",
    "DEFAULT_DELTA_U_THRESHOLD",
    "DEFAULT_MAX_HARM_PROBABILITY",
    "DEFAULT_MIN_CONFIDENCE",
]
