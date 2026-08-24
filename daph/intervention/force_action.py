"""Force-action execution from checkpoints.

Forces a specific action from a checkpoint, then either:
  - Returns the result (for terminal actions: ANSWER, DEFER, STOP)
  - Continues with the pinned policy (for non-terminal actions: RETRIEVE, VERIFY, SEARCH_MORE)

This is the core mechanism for collecting causal action data:
    Q*(s,a) ≈ E[U | do(a), s]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceRuntime, EvidenceActionExecution, EvidenceTask,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceExhausted

from .checkpoint import StateCheckpoint, create_checkpoint, compute_state_features
from .restore import restore_runtime


@dataclass(frozen=True)
class ForcedActionResult:
    """Result of forcing an action from a checkpoint.

    Attributes:
        checkpoint_id: The checkpoint we intervened from
        action: The forced action
        intervention_type: "CAUSAL_DETERMINISTIC" or "FORCED_ACTION_ROLLOUT"
        immediate_utility: Utility gained from the forced action itself
        terminal_utility: Total utility at trajectory end
        success: Whether the task succeeded
        terminal: Whether the forced action was terminal
        terminal_action: The final action taken (may differ from forced action
                         if the policy continued after a non-terminal forced action)
        steps_to_terminal: Total steps from checkpoint to terminal
        outcome_code: Outcome of the forced action
        premature_defer: Whether DEFER was taken before evidence was sufficient
        premature_answer: Whether ANSWER was taken before evidence was sufficient
        resource_exhaustion: Whether resources were exhausted
        loop: Whether the trajectory looped without progress
        forced_action_execution: The raw execution result of the forced action
        downstream_actions: Actions taken by the policy after the forced action
    """
    checkpoint_id: str
    action: str
    intervention_type: str
    immediate_utility: float
    terminal_utility: float
    success: bool
    terminal: bool
    terminal_action: str | None
    steps_to_terminal: int
    outcome_code: str
    premature_defer: bool
    premature_answer: bool
    resource_exhaustion: bool
    loop: bool
    forced_action_execution: dict
    downstream_actions: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "action": self.action,
            "intervention_type": self.intervention_type,
            "immediate_utility": self.immediate_utility,
            "terminal_utility": self.terminal_utility,
            "success": self.success,
            "terminal": self.terminal,
            "terminal_action": self.terminal_action,
            "steps_to_terminal": self.steps_to_terminal,
            "outcome_code": self.outcome_code,
            "premature_defer": self.premature_defer,
            "premature_answer": self.premature_answer,
            "resource_exhaustion": self.resource_exhaustion,
            "loop": self.loop,
            "forced_action_execution": self.forced_action_execution,
            "downstream_actions": list(self.downstream_actions),
        }


def force_action(
    checkpoint: StateCheckpoint,
    task: EvidenceTask,
    action: DecisionAction,
    target_evidence_id: str | None = None,
) -> tuple[ForcedActionResult, EvidenceRuntime]:
    """Force a single action from a checkpoint.

    This executes the forced action deterministically using the EvidenceExecutor.
    It does NOT call the LLM — the forced action is applied directly.

    For terminal actions (ANSWER, DEFER, STOP), the result is immediate.
    For non-terminal actions (RETRIEVE, VERIFY, SEARCH_MORE), the result
    includes the post-action runtime for downstream policy rollout.

    Args:
        checkpoint: The checkpoint to intervene from
        task: The original EvidenceTask
        action: The action to force
        target_evidence_id: Optional evidence target for VERIFY

    Returns:
        Tuple of (ForcedActionResult, post-action EvidenceRuntime).
        The post-action runtime is the state after the forced action,
        ready for downstream policy rollout if non-terminal.
    """
    runtime = restore_runtime(checkpoint, task)
    executor = EvidenceExecutor()

    # Execute the forced action
    result = executor.execute(runtime, action, target_evidence_id=target_evidence_id)

    # Determine intervention type
    from .schedule import classify_intervention_type
    itype = classify_intervention_type(action.value)

    # Check for premature actions
    premature_defer = False
    premature_answer = False
    resource_exhaustion = (result.outcome_code == "RESOURCE_EXHAUSTED")

    if action is DecisionAction.DEFER and result.task_success is False:
        premature_defer = True
    if action is DecisionAction.ANSWER and result.task_success is False:
        premature_answer = True

    # Compute immediate utility (simple: +1 for success, -1 for failure, 0 for non-terminal)
    if result.terminal:
        immediate_utility = 1.0 if result.task_success else -1.0
    else:
        immediate_utility = 0.0

    # For terminal actions, terminal_utility = immediate_utility
    # For non-terminal, terminal_utility will be filled by downstream rollout
    terminal_utility = immediate_utility if result.terminal else 0.0

    forced_result = ForcedActionResult(
        checkpoint_id=checkpoint.checkpoint_id,
        action=action.value,
        intervention_type=itype,
        immediate_utility=immediate_utility,
        terminal_utility=terminal_utility,
        success=result.task_success if result.terminal else False,
        terminal=result.terminal,
        terminal_action=action.value if result.terminal else None,
        steps_to_terminal=1 if result.terminal else 0,
        outcome_code=result.outcome_code,
        premature_defer=premature_defer,
        premature_answer=premature_answer,
        resource_exhaustion=resource_exhaustion,
        loop=False,
        forced_action_execution={
            "action": result.action.value,
            "terminal": result.terminal,
            "task_success": result.task_success,
            "outcome_code": result.outcome_code,
            "evidence_exposed": list(result.evidence_exposed),
            "evidence_verified": list(result.evidence_verified),
        },
        downstream_actions=(),
    )

    return forced_result, result.runtime


def force_action_with_rollout(
    checkpoint: StateCheckpoint,
    task: EvidenceTask,
    action: DecisionAction,
    policy_fn: Callable[[EvidenceRuntime, tuple[str, ...], tuple[str, ...]], DecisionAction],
    target_evidence_id: str | None = None,
    max_steps: int = 10,
) -> ForcedActionResult:
    """Force an action from a checkpoint, then continue with the pinned policy.

    This is for FORCED_ACTION_ROLLOUT interventions where the forced action
    is non-terminal and the policy continues afterward.

    Args:
        checkpoint: The checkpoint to intervene from
        task: The original EvidenceTask
        action: The action to force
        policy_fn: Function that takes (runtime, prior_actions, prior_outcomes)
                   and returns the next action. This is the pinned LLM policy.
        target_evidence_id: Optional evidence target for VERIFY
        max_steps: Maximum steps after the forced action

    Returns:
        A ForcedActionResult with the complete trajectory outcome.
    """
    # First, force the action
    forced_result, post_runtime = force_action(
        checkpoint, task, action, target_evidence_id,
    )

    if forced_result.terminal:
        return forced_result

    # Continue with the pinned policy
    executor = EvidenceExecutor()
    runtime = post_runtime
    prior_actions = list(checkpoint.prior_actions) + [action.value]
    prior_outcomes = list(checkpoint.prior_outcomes) + [forced_result.outcome_code]
    downstream_actions: list[str] = []
    steps = 1

    loop_detected = False
    last_action = None
    same_action_count = 0

    while steps < max_steps:
        # Check for loop (same action repeated 3+ times)
        if last_action and len(downstream_actions) >= 3:
            recent = downstream_actions[-3:]
            if all(a == recent[0] for a in recent):
                loop_detected = True
                break

        # Get next action from policy
        try:
            next_action = policy_fn(runtime, tuple(prior_actions), tuple(prior_outcomes))
        except Exception:
            break

        if next_action is None:
            break

        # Execute the policy action
        try:
            result = executor.execute(runtime, next_action)
        except Exception:
            break

        runtime = result.runtime
        downstream_actions.append(next_action.value)
        prior_actions.append(next_action.value)
        prior_outcomes.append(result.outcome_code)
        steps += 1
        last_action = next_action.value

        if result.terminal:
            # Compute terminal utility
            terminal_utility = 1.0 if result.task_success else -1.0

            # Check for premature actions
            premature_defer = forced_result.premature_defer
            premature_answer = forced_result.premature_answer
            if next_action is DecisionAction.DEFER and result.task_success is False:
                premature_defer = True
            if next_action is DecisionAction.ANSWER and result.task_success is False:
                premature_answer = True

            return ForcedActionResult(
                checkpoint_id=checkpoint.checkpoint_id,
                action=action.value,
                intervention_type=forced_result.intervention_type,
                immediate_utility=forced_result.immediate_utility,
                terminal_utility=terminal_utility,
                success=result.task_success,
                terminal=True,
                terminal_action=next_action.value,
                steps_to_terminal=steps,
                outcome_code=result.outcome_code,
                premature_defer=premature_defer,
                premature_answer=premature_answer,
                resource_exhaustion=(result.outcome_code == "RESOURCE_EXHAUSTED"),
                loop=loop_detected,
                forced_action_execution=forced_result.forced_action_execution,
                downstream_actions=tuple(downstream_actions),
            )

    # Step limit reached without terminal
    return ForcedActionResult(
        checkpoint_id=checkpoint.checkpoint_id,
        action=action.value,
        intervention_type=forced_result.intervention_type,
        immediate_utility=forced_result.immediate_utility,
        terminal_utility=-0.5,  # penalty for not reaching terminal
        success=False,
        terminal=False,
        terminal_action=None,
        steps_to_terminal=steps,
        outcome_code="STEP_LIMIT",
        premature_defer=forced_result.premature_defer,
        premature_answer=forced_result.premature_answer,
        resource_exhaustion=False,
        loop=loop_detected,
        forced_action_execution=forced_result.forced_action_execution,
        downstream_actions=tuple(downstream_actions),
    )
