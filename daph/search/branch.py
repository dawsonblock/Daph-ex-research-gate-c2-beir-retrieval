"""Branch simulation for selective search.

Uses checkpoint/restore/force_action to simulate branches.
Does NOT create another executor — reuses EvidenceExecutor.
"""
from __future__ import annotations

import hashlib
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceTask, EvidenceRuntime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.checkpoint import StateCheckpoint, create_checkpoint
from daph.intervention.restore import restore_runtime
from daph.progress.progress_rule_v1 import compute_progress
from daph.search.types import BranchNode


def simulate_branch_step(
    checkpoint: StateCheckpoint,
    task: EvidenceTask,
    action: str,
    utility: MetareasoningUtility,
    executor: EvidenceExecutor,
    parent_id: str | None,
    depth: int,
) -> tuple[BranchNode, EvidenceRuntime | None]:
    """Simulate one step from a checkpoint.

    Executes the action deterministically (no LLM).
    Returns the branch node and the post-action runtime (or None if terminal).

    Args:
        checkpoint: The checkpoint to simulate from
        task: The EvidenceTask
        action: The action to simulate
        utility: Utility function for cost computation
        executor: EvidenceExecutor instance
        parent_id: Parent node ID (None for root children)
        depth: Depth of this node in the search tree

    Returns:
        Tuple of (BranchNode, post-action runtime or None if terminal)
    """
    action_enum = DecisionAction(action)

    # Determine verify target
    target_eid = None
    if action_enum is DecisionAction.VERIFY:
        runtime = restore_runtime(checkpoint, task)
        valid = valid_verify_targets(runtime)
        if valid:
            target_eid = valid[0]
        else:
            # Cannot verify — return a negative node
            node = BranchNode(
                action=action,
                depth=depth,
                parent_id=parent_id,
                node_id=_make_node_id(checkpoint, action, depth),
                checkpoint_id=checkpoint.checkpoint_id,
                state_sha=checkpoint.state_sha256,
                q_value=0.0,
                pav_score=-0.2,
                terminal=False,
            )
            return node, None

    # Execute the action
    runtime_before = restore_runtime(checkpoint, task)
    try:
        exec_result = executor.execute(runtime_before, action_enum, target_evidence_id=target_eid)
    except Exception:
        # Execution failed — return a negative node
        node = BranchNode(
            action=action,
            depth=depth,
            parent_id=parent_id,
            node_id=_make_node_id(checkpoint, action, depth),
            checkpoint_id=checkpoint.checkpoint_id,
            state_sha=checkpoint.state_sha256,
            q_value=0.0,
            pav_score=-0.2,
            terminal=False,
        )
        return node, None

    # Compute progress
    try:
        progress = compute_progress(runtime_before, exec_result, utility)
        pav_score = progress.progress
    except Exception:
        pav_score = 0.0

    # Compute action cost
    cost = utility.action_cost(runtime_before.resources, exec_result.runtime.resources)

    # Check terminal
    terminal = exec_result.terminal
    terminal_utility = None
    success = None
    if terminal:
        terminal_utility = utility.terminal_reward(
            exec_result.action, bool(exec_result.task_success),
        )
        success = bool(exec_result.task_success)

    # Create post-action checkpoint for further expansion
    post_checkpoint = create_checkpoint(
        exec_result.runtime,
        step=checkpoint.step + 1,
        phase=checkpoint.phase,
        prior_actions=checkpoint.prior_actions + (action,),
        prior_outcomes=checkpoint.prior_outcomes + (exec_result.outcome_code,),
    )

    node = BranchNode(
        action=action,
        depth=depth,
        parent_id=parent_id,
        node_id=_make_node_id(checkpoint, action, depth),
        checkpoint_id=checkpoint.checkpoint_id,
        state_sha=checkpoint.state_sha256,
        q_value=0.0,  # Filled by caller if available
        pav_score=pav_score,
        terminal=terminal,
        terminal_utility=terminal_utility,
        success=success,
        cumulative_cost=cost,
        cumulative_pav=pav_score,
    )

    # Return node and post-action runtime (for further expansion)
    # But we need the checkpoint, not the runtime, for further expansion
    # So return the post-checkpoint via the node's checkpoint_id
    # Actually, let's return the runtime for now
    if terminal:
        return node, None
    return node, exec_result.runtime


def _make_node_id(checkpoint: StateCheckpoint, action: str, depth: int) -> str:
    content = f"{checkpoint.checkpoint_id}:{action}:{depth}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
