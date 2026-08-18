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
    """Evaluation metrics for an intervention gate over an offline dataset."""
    total_states: int
    interventions_approved: int
    skips_approved: int
    intervention_rate: float
    # Among intervened states
    intervened_harm_count: int
    intervened_benefit_count: int
    intervened_neutral_count: int
    harm_rate: float
    benefit_rate: float
    net_intervention_value: float
    # Utility impact vs always-on
    total_utility_with_gate: float
    total_utility_always_on: float
    utility_saved: float


def evaluate_gate_on_dataset(
    gate: SelectiveGovernorGate,
    divergence_data: dict[str, Any],
) -> GateEvaluationMetrics:
    """Evaluate a gate's decisions against the offline divergence dataset."""
    records = divergence_data["task_records"]
    total_states = len(records)

    interventions = 0
    skips = 0
    intervened_deltas: list[float] = []
    intervened_harms = 0
    intervened_benefits = 0
    intervened_neutrals = 0

    total_u_base = 0.0
    total_u_always_gov = 0.0
    total_u_gated = 0.0

    for rec in records:
        u_base = rec["baseline_utility"]
        u_gov = rec["governor_utility"]
        delta_u = rec["utility_delta"]
        state_dict = rec["state_before_divergence"]

        total_u_base += u_base
        total_u_always_gov += u_gov

        if state_dict is not None:
            features = InterventionFeatures(**state_dict)
            pred = gate.predictor.predict(features)
            should_intervene = (
                pred.expected_delta_utility > gate.delta_u_threshold
                and pred.harm_probability < gate.max_harm_probability
                and pred.confidence >= gate.min_confidence
            )
        else:
            should_intervene = False

        if should_intervene:
            interventions += 1
            total_u_gated += u_gov
            intervened_deltas.append(delta_u)
            if delta_u < -5.0:
                intervened_harms += 1
            elif delta_u > 5.0:
                intervened_benefits += 1
            else:
                intervened_neutrals += 1
        else:
            skips += 1
            total_u_gated += u_base

    int_rate = interventions / total_states if total_states > 0 else 0.0
    harm_rate = intervened_harms / interventions if interventions > 0 else 0.0
    benefit_rate = intervened_benefits / interventions if interventions > 0 else 0.0
    net_val = statistics.mean(intervened_deltas) if intervened_deltas else 0.0
    utility_saved = total_u_gated - total_u_always_gov

    return GateEvaluationMetrics(
        total_states=total_states,
        interventions_approved=interventions,
        skips_approved=skips,
        intervention_rate=int_rate,
        intervened_harm_count=intervened_harms,
        intervened_benefit_count=intervened_benefits,
        intervened_neutral_count=intervened_neutrals,
        harm_rate=harm_rate,
        benefit_rate=benefit_rate,
        net_intervention_value=net_val,
        total_utility_with_gate=total_u_gated,
        total_utility_always_on=total_u_always_gov,
        utility_saved=utility_saved,
    )
