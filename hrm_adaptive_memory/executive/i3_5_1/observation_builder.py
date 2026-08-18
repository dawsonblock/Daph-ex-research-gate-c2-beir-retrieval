"""Observation builder for I3.5.1.

Builds ControllerObservation from the task runtime, applying the
observation mask based on the condition's observation_mode.

The observation is identical for conditions that share the same
observation_mode, regardless of whether the governor is on or off.
This is the key invariant: the governor does not change what
information is observable.

V_O(BLIND, OFF) == V_O(BLIND, ON)
V_O(AWARE, OFF) == V_O(AWARE, ON)
"""
from __future__ import annotations

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, DecisionSummary,
)
from hrm_adaptive_memory.executive.executor import (
    TaskRuntime, build_cognitive_state,
)
from hrm_adaptive_memory.executive.metareasoning_controller import (
    ControllerObservation, ObservationMask,
    STATE_BLIND_MASK, STATE_AWARE_MASK,
    apply_observation_mask,
)
from hrm_adaptive_memory.executive.metareasoning_benchmark import I3BenchmarkTask

from .conditions import ExperimentalCondition, ObservationMode


def mask_for_mode(mode: ObservationMode) -> ObservationMask:
    """Return the frozen observation mask for the given mode."""
    if mode == ObservationMode.BLIND:
        return STATE_BLIND_MASK
    return STATE_AWARE_MASK


def build_observation(
    runtime: TaskRuntime,
    task: I3BenchmarkTask,
    condition: ExperimentalCondition,
    prior_decisions: tuple[DecisionSummary, ...],
    prior_outcomes: tuple[str, ...],
) -> ControllerObservation:
    """Build a ControllerObservation with the condition's mask applied.

    The observation depends only on observation_mode, not on
    governor_enabled. This is the key invariant.
    """
    mask = mask_for_mode(condition.observation_mode)
    snapshot = build_cognitive_state(
        runtime, prior_decisions=prior_decisions, prior_outcomes=prior_outcomes)
    masked_state = apply_observation_mask(snapshot, mask)
    resources = runtime.resources.as_dict()
    allowed = tuple(
        action for action in V2B_ACTIONS if runtime.resources.can_execute(action))
    return ControllerObservation(
        task_id=task.controller_instance_id or task.task_id,
        task_summary=task.task_summary,
        resource_state=resources,
        allowed_actions=allowed,
        executed_actions=tuple(
            DecisionAction(d.selected_action) if isinstance(d.selected_action, str)
            else d.selected_action for d in prior_decisions),
        rejected_actions=(),
        cognitive_state=masked_state,
    )
