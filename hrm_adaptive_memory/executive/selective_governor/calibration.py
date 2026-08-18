"""Calibration and evaluation tools for the selective intervention gate."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation

from .features import InterventionFeatures
from .intervention_gate import SelectiveGovernorGate, InterventionDecision
from .model import RuleBasedInterventionPredictor, CalibratedLinearPredictor


@dataclass(frozen=True)
class GateEvaluationMetrics:
    """State-level evaluation metrics for an intervention gate over counterfactual states."""
    total_states: int
    interventions_approved: int
    skips_approved: int
    intervention_rate: float
    # Among intervened states
    intervened_harm_count: int
    intervened_help_count: int
    intervened_neutral_count: int
    harm_rate: float
    precision_help: float
    mean_delta_q_intervened: float
    total_realized_delta_q: float


def evaluate_gate_on_counterfactual_states(
    gate: SelectiveGovernorGate,
    state_records: list[dict[str, Any]],
) -> GateEvaluationMetrics:
    """Evaluate a selective gate over state-level counterfactual records using exact ΔQ(s)."""
    total_states = len(state_records)
    interventions = 0
    skips = 0
    intervened_deltas: list[float] = []
    intervened_harms = 0
    intervened_helps = 0
    intervened_neutrals = 0

    for rec in state_records:
        delta_q = rec["delta_q"]
        state_dict = rec["features"]
        features = InterventionFeatures(**state_dict)

        pred = gate.predictor.predict(features)
        should_intervene = (
            pred.expected_delta_utility > gate.delta_u_threshold
            and pred.harm_probability < gate.max_harm_probability
            and pred.confidence >= gate.min_confidence
        )

        if should_intervene:
            interventions += 1
            intervened_deltas.append(delta_q)
            if delta_q < -5.0:
                intervened_harms += 1
            elif delta_q > 5.0:
                intervened_helps += 1
            else:
                intervened_neutrals += 1
        else:
            skips += 1

    int_rate = interventions / total_states if total_states > 0 else 0.0
    harm_rate = intervened_harms / interventions if interventions > 0 else 0.0
    precision_help = intervened_helps / interventions if interventions > 0 else 0.0
    mean_gain = statistics.mean(intervened_deltas) if intervened_deltas else 0.0
    total_gain = sum(intervened_deltas)

    return GateEvaluationMetrics(
        total_states=total_states,
        interventions_approved=interventions,
        skips_approved=skips,
        intervention_rate=round(int_rate, 4),
        intervened_harm_count=intervened_harms,
        intervened_help_count=intervened_helps,
        intervened_neutral_count=intervened_neutrals,
        harm_rate=round(harm_rate, 4),
        precision_help=round(precision_help, 4),
        mean_delta_q_intervened=round(mean_gain, 4),
        total_realized_delta_q=round(total_gain, 4),
    )


# Backward compatibility alias
evaluate_gate_on_dataset = evaluate_gate_on_counterfactual_states
