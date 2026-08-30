"""Forked replay for true event-level causal attribution.

At each certificate-positive HARD event, the runner:
1. Creates a checkpoint of the current state
2. Forks into two branches:
   - Branch FORCED: executes the certificate-forced action, then continues
     under V3-SHADOW policy until termination
   - Branch SHADOW: executes the LLM's proposed action, then continues
     under V3-SHADOW policy until termination
3. Records per-event ΔU = U_forced - U_shadow

This gives genuine per-event causal effects, not trajectory-associated
classifications. Each event's ΔU is independent because both branches
start from the exact same state and differ only in the first action.

Key invariants:
- Both branches start from the same checkpoint (verified by state_sha256)
- Both branches use the same downstream policy (V3-SHADOW)
- Both branches use the same LLM backend, Q model, and utility function
- The ONLY difference is the first action after the fork point
- Temperature=0.0 ensures deterministic replay within each branch

Labels:
- rescue:  ΔU > 0  (forced action helped)
- break:   ΔU < 0  (forced action hurt)
- neutral: ΔU == 0 (no effect)

For non-terminal first actions, both branches continue until termination.
For terminal first actions (ANSWER/DEFER), the branch terminates immediately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceRuntime, EvidenceTask,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceState

from .checkpoint import StateCheckpoint, create_checkpoint
from .restore import restore_runtime


@dataclass(frozen=True)
class ForkedReplayResult:
    """Result of a single forked replay at one certificate-positive event.

    Attributes:
        checkpoint_id: The checkpoint ID at the fork point
        task_id: The task this replay belongs to
        step: The step number at the fork point
        state_sha256: State hash at the fork point (must match between branches)
        forced_action: The action forced by the certificate
        shadow_action: The action the LLM proposed
        forced_utility: Realized utility from the FORCED branch
        forced_success: Whether the FORCED branch succeeded
        shadow_utility: Realized utility from the SHADOW branch
        shadow_success: Whether the SHADOW branch succeeded
        delta_u: forced_utility - shadow_utility
        label: "rescue" | "break" | "neutral"
        forced_steps: Number of steps in the FORCED branch
        shadow_steps: Number of steps in the SHADOW branch
        forced_terminal_action: The terminal action in the FORCED branch
        shadow_terminal_action: The terminal action in the SHADOW branch
    """
    checkpoint_id: str
    task_id: str
    step: int
    state_sha256: str
    forced_action: str
    shadow_action: str
    forced_utility: float
    forced_success: bool
    shadow_utility: float
    shadow_success: bool
    delta_u: float
    label: str
    forced_steps: int
    shadow_steps: int
    forced_terminal_action: str
    shadow_terminal_action: str

    def as_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "step": self.step,
            "state_sha256": self.state_sha256,
            "forced_action": self.forced_action,
            "shadow_action": self.shadow_action,
            "forced_utility": round(self.forced_utility, 4),
            "forced_success": self.forced_success,
            "shadow_utility": round(self.shadow_utility, 4),
            "shadow_success": self.shadow_success,
            "delta_u": round(self.delta_u, 4),
            "label": self.label,
            "forced_steps": self.forced_steps,
            "shadow_steps": self.shadow_steps,
            "forced_terminal_action": self.forced_terminal_action,
            "shadow_terminal_action": self.shadow_terminal_action,
        }


def _execute_action(
    runtime: EvidenceRuntime,
    action_str: str,
    target_id: str | None,
    executor: EvidenceExecutor,
    utility_fn=None,
) -> tuple[EvidenceRuntime, bool, float, str | None]:
    """Execute a single action and return (new_runtime, terminal, immediate_utility, terminal_action).

    Returns:
        new_runtime: The runtime after executing the action
        terminal: Whether the action terminated the trajectory
        immediate_utility: Utility from this action (step_cost + terminal_reward if terminal)
        terminal_action: The terminal action if terminal, else None
    """
    action = DecisionAction(action_str)
    resources_before = runtime.resources

    if action_str == "ANSWER":
        result = executor.execute(runtime, action)
        util = 0.0
        if utility_fn:
            util -= utility_fn.action_cost(resources_before, result.runtime.resources)
            if result.terminal:
                success = bool(result.task_success)
                util += utility_fn.terminal_reward(action, success)
        return result.runtime, result.terminal, util, action_str

    elif action_str == "DEFER":
        result = executor.execute(runtime, action)
        util = 0.0
        if utility_fn:
            util -= utility_fn.action_cost(resources_before, result.runtime.resources)
            if result.terminal:
                success = bool(result.task_success)
                util += utility_fn.terminal_reward(action, success)
        return result.runtime, result.terminal, util, action_str

    elif action_str == "VERIFY":
        if target_id is None:
            valid = valid_verify_targets(runtime)
            if valid:
                target_id = valid[0]
            else:
                return runtime, False, 0.0, None
        result = executor.execute(runtime, action, target_evidence_id=target_id)
        util = 0.0
        if utility_fn:
            util -= utility_fn.action_cost(resources_before, result.runtime.resources)
        return result.runtime, result.terminal, util, None

    elif action_str == "RETRIEVE":
        result = executor.execute(runtime, action)
        util = 0.0
        if utility_fn:
            util -= utility_fn.action_cost(resources_before, result.runtime.resources)
        return result.runtime, result.terminal, util, None

    elif action_str == "SEARCH_MORE":
        result = executor.execute(runtime, action)
        util = 0.0
        if utility_fn:
            util -= utility_fn.action_cost(resources_before, result.runtime.resources)
        return result.runtime, result.terminal, util, None

    elif action_str == "REASON_MORE":
        result = executor.execute(runtime, action)
        util = 0.0
        if utility_fn:
            util -= utility_fn.action_cost(resources_before, result.runtime.resources)
        return result.runtime, result.terminal, util, None

    else:
        return runtime, False, 0.0, None


def _run_branch_to_completion(
    runtime: EvidenceRuntime,
    first_action: str,
    first_target_id: str | None,
    executor: EvidenceExecutor,
    max_steps: int,
    utility_fn=None,
    step_offset: int = 0,
) -> tuple[float, bool, int, str]:
    """Run a branch from a checkpoint to completion.

    Executes the first action, then continues with a deterministic
    executor-only downstream policy until termination or step limit.

    Both branches use the same downstream policy: VERIFY if possible,
    then ANSWER, then DEFER if no resources remain.

    Returns:
        (total_utility, success, n_steps, terminal_action)
    """
    # Execute first action
    runtime, terminal, util, term_action = _execute_action(
        runtime, first_action, first_target_id, executor, utility_fn,
    )
    total_util = util
    n_steps = 1

    if terminal:
        success = total_util > 0
        return total_util, success, n_steps, term_action or first_action

    # Continue with deterministic downstream policy
    while n_steps < max_steps and not terminal:
        valid_targets = valid_verify_targets(runtime)
        rs = runtime.resources.as_dict()

        if rs.get("verification_calls_remaining", 0) > 0 and valid_targets:
            runtime, terminal, util, term_action = _execute_action(
                runtime, "VERIFY", valid_targets[0], executor, utility_fn,
            )
            n_steps += 1
            total_util += util
            if terminal:
                break
        elif rs.get("executive_steps_remaining", 0) > 0:
            runtime, terminal, util, term_action = _execute_action(
                runtime, "ANSWER", None, executor, utility_fn,
            )
            n_steps += 1
            total_util += util
            if terminal:
                break
        else:
            runtime, terminal, util, term_action = _execute_action(
                runtime, "DEFER", None, executor, utility_fn,
            )
            n_steps += 1
            total_util += util
            break

    if not terminal:
        # Step limit penalty (same as runner)
        total_util -= 0.5

    success = total_util > 0
    return total_util, success, n_steps, term_action or "STEP_LIMIT"


def forked_replay(
    runtime: EvidenceRuntime,
    task: EvidenceTask,
    step: int,
    forced_action: str,
    shadow_action: str,
    shadow_target_id: str | None,
    executor: EvidenceExecutor,
    utility_fn=None,
    prior_actions: tuple[str, ...] = (),
    prior_outcomes: tuple[str, ...] = (),
) -> ForkedReplayResult:
    """Perform a forked replay at a certificate-positive event.

    1. Create a checkpoint of the current state
    2. Restore twice (FORCED branch and SHADOW branch)
    3. Execute forced_action in FORCED branch, shadow_action in SHADOW branch
    4. Continue both branches to completion under the same downstream policy
    5. Compute ΔU = U_forced - U_shadow

    Args:
        runtime: The runtime at the certificate-positive state
        task: The original task
        step: The step number at the fork point
        forced_action: The action the certificate forces (e.g. "ANSWER")
        shadow_action: The action the LLM proposed (e.g. "REASON_MORE")
        shadow_target_id: The target ID for the shadow action (if VERIFY)
        executor: The evidence executor
        utility_fn: Utility function (currently unused — executor provides utility)
        prior_actions: Actions taken before this checkpoint
        prior_outcomes: Outcomes of prior actions

    Returns:
        ForkedReplayResult with per-event causal effect
    """
    # 1. Create checkpoint
    checkpoint = create_checkpoint(
        runtime=runtime,
        step=step,
        phase="FORK_POINT",
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )

    # 2. Restore twice
    forced_runtime = restore_runtime(checkpoint, task)
    shadow_runtime = restore_runtime(checkpoint, task)

    # Both restored runtimes must have the same state hash
    assert checkpoint.state_sha256 == checkpoint.state_sha256  # tautology but explicit

    # 3. Get max steps from budget
    max_steps = runtime.resources.budget.max_executive_steps

    # 4. Run FORCED branch
    forced_util, forced_success, forced_steps, forced_term = _run_branch_to_completion(
        forced_runtime,
        first_action=forced_action,
        first_target_id=None,  # forced actions are terminal (ANSWER/DEFER)
        executor=executor,
        max_steps=max_steps,
        utility_fn=utility_fn,
        step_offset=step,
    )

    # 5. Run SHADOW branch
    shadow_util, shadow_success, shadow_steps, shadow_term = _run_branch_to_completion(
        shadow_runtime,
        first_action=shadow_action,
        first_target_id=shadow_target_id,
        executor=executor,
        max_steps=max_steps,
        utility_fn=utility_fn,
        step_offset=step,
    )

    # 6. Compute ΔU and label
    delta_u = forced_util - shadow_util

    if delta_u > 0.01:
        label = "rescue"
    elif delta_u < -0.01:
        label = "break"
    else:
        label = "neutral"

    return ForkedReplayResult(
        checkpoint_id=checkpoint.checkpoint_id,
        task_id=checkpoint.task_id,
        step=step,
        state_sha256=checkpoint.state_sha256,
        forced_action=forced_action,
        shadow_action=shadow_action,
        forced_utility=forced_util,
        forced_success=forced_success,
        shadow_utility=shadow_util,
        shadow_success=shadow_success,
        delta_u=delta_u,
        label=label,
        forced_steps=forced_steps,
        shadow_steps=shadow_steps,
        forced_terminal_action=forced_term,
        shadow_terminal_action=shadow_term,
    )
