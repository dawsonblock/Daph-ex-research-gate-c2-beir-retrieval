"""Candidate features: topology-invariant features for action scoring.

These features describe action/state relationships rather than benchmark
labels. They survive topology changes better than raw state fields.

Features:
    does_action_target_current_blocker
    does_action_create_external_information
    does_action_only_transform_existing_information
    has_action_recently_failed
    does_action_preserve_future_options
    does_action_consume_last_remaining_resource
    does_action_terminate_under_uncertainty
    action_redundancy_level
    action_information_value
    action_expected_progress
"""
from __future__ import annotations

from dataclasses import dataclass
from hrm_adaptive_memory.executive.governor.state import GovernorState
from hrm_adaptive_memory.executive.governor.action_semantics import get_action_semantics
from hrm_adaptive_memory.executive.governor.bottlenecks import (
    DecisionBottleneck, NONE, LOW, MEDIUM, HIGH)
from hrm_adaptive_memory.executive.governor.transition_model import PredictedActionOutcome
from hrm_adaptive_memory.executive.governor.redundancy import compute_redundancy
from hrm_adaptive_memory.executive.governor.value_of_information import estimate_voi
from hrm_adaptive_memory.executive.governor.option_value import estimate_option_value


CANDIDATE_FEATURES_SCHEMA = "DAPH_V2B_I3_5_CANDIDATE_FEATURES_V1"
CANDIDATE_FEATURES_VERSION = 1

# Ordinal mapping for scoring
_ORDINAL_VALUE = {NONE: 0, LOW: 1, MEDIUM: 2, HIGH: 3}


@dataclass(frozen=True)
class CandidateActionAssessment:
    """Full assessment of a candidate action by the governor."""
    action: str
    blocker_alignment: str  # NONE, LOW, MEDIUM, HIGH
    information_value: str  # NONE, LOW, MEDIUM, HIGH
    expected_progress: str  # NONE, LOW, MEDIUM, HIGH
    resource_cost: str  # NONE, LOW, MEDIUM, HIGH
    repeat_penalty: str  # NONE, LOW, MEDIUM, HIGH
    option_preservation: str  # NONE, LOW, MEDIUM, HIGH
    policy_risk: str  # NONE, LOW, MEDIUM, HIGH
    # Topology-invariant boolean features
    targets_current_blocker: bool
    creates_external_information: bool
    only_transforms_existing: bool
    recently_failed: bool
    preserves_future_options: bool
    consumes_last_resource: bool
    terminates_under_uncertainty: bool

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "blocker_alignment": self.blocker_alignment,
            "information_value": self.information_value,
            "expected_progress": self.expected_progress,
            "resource_cost": self.resource_cost,
            "repeat_penalty": self.repeat_penalty,
            "option_preservation": self.option_preservation,
            "policy_risk": self.policy_risk,
            "targets_current_blocker": self.targets_current_blocker,
            "creates_external_information": self.creates_external_information,
            "only_transforms_existing": self.only_transforms_existing,
            "recently_failed": self.recently_failed,
            "preserves_future_options": self.preserves_future_options,
            "consumes_last_resource": self.consumes_last_resource,
            "terminates_under_uncertainty": self.terminates_under_uncertainty,
        }

    def score(self) -> float:
        """Compute a scalar score for ranking.

        S(a) = w_p * P(a) + w_i * I(a) - w_c * C(a) - w_r * R(a) - w_d * D(a)

        Where:
            P(a): expected progress
            I(a): information value
            C(a): resource cost
            R(a): execution/policy risk
            D(a): redundancy/diminishing-return penalty

        Weights are frozen for V1.
        """
        w_p, w_i, w_c, w_r, w_d, w_o = 1.0, 1.0, 0.5, 0.5, 1.0, 0.3
        progress = _ORDINAL_VALUE.get(self.expected_progress, 0)
        info = _ORDINAL_VALUE.get(self.information_value, 0)
        cost = _ORDINAL_VALUE.get(self.resource_cost, 0)
        risk = _ORDINAL_VALUE.get(self.policy_risk, 0)
        redundancy = _ORDINAL_VALUE.get(self.repeat_penalty, 0)
        options = _ORDINAL_VALUE.get(self.option_preservation, 0)

        return (w_p * progress
                + w_i * info
                + w_o * options
                - w_c * cost
                - w_r * risk
                - w_d * redundancy)


def assess_candidate(
    state: GovernorState,
    action: str,
    outcome: PredictedActionOutcome,
    bottlenecks: tuple[DecisionBottleneck, ...],
) -> CandidateActionAssessment:
    """Assess a single candidate action."""
    semantics = get_action_semantics(action)
    active = bottlenecks[0] if bottlenecks else None

    # Blocker alignment
    blocker_alignment = NONE
    if active and active.kind != "READY_TO_ANSWER":
        if outcome.targets_active_bottleneck and outcome.can_resolve_current_blocker:
            blocker_alignment = HIGH
        elif outcome.targets_active_bottleneck:
            blocker_alignment = MEDIUM
        elif outcome.adds_new_information:
            blocker_alignment = LOW
    elif active and active.kind == "READY_TO_ANSWER":
        blocker_alignment = HIGH if action == "ANSWER" else NONE

    # Information value
    information_value = estimate_voi(state, action, outcome, bottlenecks)

    # Expected progress
    expected_progress = NONE
    if outcome.terminal and active and active.kind == "READY_TO_ANSWER":
        expected_progress = HIGH
    elif outcome.can_resolve_current_blocker:
        expected_progress = HIGH
    elif outcome.targets_active_bottleneck:
        expected_progress = MEDIUM
    elif outcome.adds_new_information:
        expected_progress = MEDIUM
    elif outcome.only_transforms_existing:
        expected_progress = LOW

    # Resource cost
    resource_cost = LOW
    if len(semantics.cost_channels) >= 2:
        resource_cost = MEDIUM
    # Check if consuming last resource
    consumes_last = False
    for ch in semantics.cost_channels:
        if ch == "steps":
            continue
        if state.resource_state.get(ch, 0) <= 1:
            resource_cost = HIGH
            consumes_last = True
            break

    # Repeat penalty
    repeat_penalty = compute_redundancy(state, action)

    # Option preservation
    option_preservation = estimate_option_value(state, action)

    # Policy risk
    policy_risk = LOW
    if action not in state.legal_actions:
        policy_risk = HIGH
    if outcome.terminal and active and active.kind != "READY_TO_ANSWER":
        policy_risk = HIGH  # terminating under uncertainty

    # Boolean features
    targets_blocker = outcome.targets_active_bottleneck
    creates_external = outcome.adds_new_information
    only_transforms = outcome.only_transforms_existing
    recently_failed = repeat_penalty in (MEDIUM, HIGH)
    preserves_options = not semantics.is_terminal
    terminates_uncertain = (semantics.is_terminal
                            and active is not None
                            and active.kind != "READY_TO_ANSWER")

    return CandidateActionAssessment(
        action=action,
        blocker_alignment=blocker_alignment,
        information_value=information_value,
        expected_progress=expected_progress,
        resource_cost=resource_cost,
        repeat_penalty=repeat_penalty,
        option_preservation=option_preservation,
        policy_risk=policy_risk,
        targets_current_blocker=targets_blocker,
        creates_external_information=creates_external,
        only_transforms_existing=only_transforms,
        recently_failed=recently_failed,
        preserves_future_options=preserves_options,
        consumes_last_resource=consumes_last,
        terminates_under_uncertainty=terminates_uncertain,
    )
