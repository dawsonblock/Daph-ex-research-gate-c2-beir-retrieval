"""Value of information proxy: observable approximation of VOI.

Not the true oracle VOI. An observable approximation:

VOI(a) = P(action changes decision-relevant state) × decision_sensitivity - C(a)

Categorical is fine initially. DeepSeek is good at reasoning over
ordinal features without fake calibration.
"""
from __future__ import annotations

from hrm_adaptive_memory.executive.governor.state import GovernorState
from hrm_adaptive_memory.executive.governor.action_semantics import get_action_semantics
from hrm_adaptive_memory.executive.governor.bottlenecks import (
    DecisionBottleneck, NONE, LOW, MEDIUM, HIGH)
from hrm_adaptive_memory.executive.governor.transition_model import PredictedActionOutcome
from hrm_adaptive_memory.executive.governor.redundancy import compute_redundancy


VOI_SCHEMA = "DAPH_V2B_I3_5_VOI_V1"
VOI_VERSION = 1


def estimate_voi(
    state: GovernorState,
    action: str,
    outcome: PredictedActionOutcome,
    bottlenecks: tuple[DecisionBottleneck, ...],
) -> str:
    """Estimate the value of information for an action.

    Returns: NONE, LOW, MEDIUM, or HIGH

    Factors:
    - Decision sensitivity: does the action affect decision-relevant state?
    - Probable state change: how likely is the action to change something?
    - Cost: resource cost of the action
    - Redundancy: has the action already been tried without gain?
    """
    semantics = get_action_semantics(action)
    active = bottlenecks[0] if bottlenecks else None

    # Decision sensitivity: does the action target the active bottleneck?
    sensitivity = NONE
    if active and active.kind != "READY_TO_ANSWER":
        if outcome.targets_active_bottleneck:
            sensitivity = HIGH
        elif outcome.adds_new_information:
            sensitivity = MEDIUM
        elif outcome.only_transforms_existing:
            sensitivity = LOW
    elif active and active.kind == "READY_TO_ANSWER":
        # Ready to answer — only ANSWER has high VOI
        if action == "ANSWER":
            sensitivity = HIGH
        else:
            sensitivity = NONE
    else:
        sensitivity = LOW

    # Probable state change
    probable_change = NONE
    if outcome.possible_changes:
        if outcome.adds_new_information:
            probable_change = HIGH
        elif outcome.can_resolve_current_blocker:
            probable_change = HIGH
        elif len(outcome.possible_changes) >= 2:
            probable_change = MEDIUM
        else:
            probable_change = LOW

    # Cost
    cost = LOW
    if len(semantics.cost_channels) >= 2:
        cost = MEDIUM
    if len(semantics.cost_channels) >= 3:
        cost = HIGH

    # Redundancy penalty
    redundancy = compute_redundancy(state, action)

    # Combine: VOI = sensitivity × probable_change - cost - redundancy
    # Simple categorical combination
    if redundancy == HIGH:
        return NONE
    if redundancy == MEDIUM:
        return LOW

    if sensitivity == HIGH and probable_change in (HIGH, MEDIUM):
        if cost in (NONE, LOW):
            return HIGH
        return MEDIUM

    if sensitivity == MEDIUM and probable_change in (HIGH, MEDIUM):
        if cost in (NONE, LOW):
            return MEDIUM
        return LOW

    if sensitivity == LOW or probable_change in (LOW, NONE):
        return LOW

    return NONE
