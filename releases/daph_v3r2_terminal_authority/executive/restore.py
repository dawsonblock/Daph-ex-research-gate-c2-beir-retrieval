"""State restoration from checkpoints.

Restoring a checkpoint must reconstruct the same visible state.
Test: restore(checkpoint) -> hash(state) == checkpoint.state_sha256  # 100% required
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceItem, EvidenceRuntime, EvidenceTask, EvidenceHypothesis,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import valid_verify_targets
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

from .checkpoint import (
    StateCheckpoint,
    _dict_to_evidence,
    _compute_state_sha256,
    compute_state_features,
    compute_legal_actions,
)


def restore_runtime(
    checkpoint: StateCheckpoint,
    task: EvidenceTask,
) -> EvidenceRuntime:
    """Restore an EvidenceRuntime from a checkpoint.

    This reconstructs the complete evidence state and resource state
    from the checkpoint's frozen data.

    Args:
        checkpoint: The frozen checkpoint to restore
        task: The original EvidenceTask (needed for hypothesis definitions
              and hidden evidence structure)

    Returns:
        An EvidenceRuntime matching the checkpoint's state.

    Raises:
        ValueError: If the restored state hash does not match the checkpoint.
    """
    # Reconstruct evidence items from checkpoint
    # The checkpoint stores visible evidence; hidden evidence comes from the task
    visible_evidence = tuple(_dict_to_evidence(d) for d in checkpoint.evidence)
    visible_ids = {ev.evidence_id for ev in visible_evidence}

    # Hidden evidence from the task that hasn't been retrieved
    hidden_evidence = tuple(
        ev for ev in task.evidence_items
        if ev.evidence_id not in visible_ids
    )

    # Combine: visible (from checkpoint) + hidden (from task, unretrieved)
    all_evidence = visible_evidence + hidden_evidence

    # Reconstruct resource state
    budget = _reconstruct_budget(checkpoint.resources)
    resources = _reconstruct_resource_state(checkpoint.resources, budget)

    # Build runtime
    retrieved_ids = tuple(ev.evidence_id for ev in visible_evidence)
    verified_ids = tuple(
        ev.evidence_id for ev in visible_evidence
        if ev.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED)
    )
    searched = checkpoint.state_features.get("searched", False)
    reasoning_complete = checkpoint.state_features.get("reasoning_complete", False)

    runtime = EvidenceRuntime(
        task=task,
        resources=resources,
        evidence=all_evidence,
        retrieved_evidence_ids=retrieved_ids,
        verified_evidence_ids=verified_ids,
        searched=searched,
        reasoning_complete=reasoning_complete,
    )

    # Verify state hash matches
    evidence_dicts = tuple(
        {
            "evidence_id": ev.evidence_id,
            "proposition": ev.proposition,
            "source_class": ev.source_class,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.value,
            "temporal_status": ev.temporal_status.value,
            "retrieved": ev.retrieved,
            "verify_result": ev.verify_result,
        }
        for ev in runtime.visible_evidence
    )
    restored_sha = _compute_state_sha256(
        checkpoint.task_id, checkpoint.step,
        evidence_dicts, resources.as_dict(),
        checkpoint.prior_actions,
    )

    if restored_sha != checkpoint.state_sha256:
        raise ValueError(
            f"State hash mismatch on restore: {restored_sha[:16]}... != {checkpoint.state_sha256[:16]}... "
            f"(task={checkpoint.task_id}, step={checkpoint.step})"
        )

    return runtime


def _reconstruct_budget(resources_dict: dict) -> ResourceBudget:
    """Reconstruct a ResourceBudget from the resources dict."""
    # Infer budget max values from used + remaining
    max_steps = resources_dict.get("executive_steps_used", 0) + resources_dict.get("executive_steps_remaining", 0)
    max_retrieval = resources_dict.get("retrieval_calls_used", 0) + resources_dict.get("retrieval_calls_remaining", 0)
    max_search = resources_dict.get("search_calls_used", 0) + resources_dict.get("search_calls_remaining", 0)
    max_verify = resources_dict.get("verification_calls_used", 0) + resources_dict.get("verification_calls_remaining", 0)
    max_reasoning = resources_dict.get("reasoning_tokens_used", 0) + resources_dict.get("reasoning_tokens_remaining", 0)
    max_elapsed = resources_dict.get("elapsed_ms", 0) + resources_dict.get("elapsed_ms_remaining", 0)

    return ResourceBudget(
        max_executive_steps=max(max_steps, 1),
        max_retrieval_calls=max(max_retrieval, 0),
        max_search_calls=max(max_search, 0),
        max_verification_calls=max(max_verify, 0),
        max_reasoning_tokens=max(max_reasoning, 1),
        max_elapsed_ms=max(max_elapsed, 1),
    )


def _reconstruct_resource_state(resources_dict: dict, budget: ResourceBudget) -> ResourceState:
    """Reconstruct a ResourceState from the resources dict and budget."""
    return ResourceState(
        budget=budget,
        executive_steps_used=resources_dict.get("executive_steps_used", 0),
        reasoning_tokens_used=resources_dict.get("reasoning_tokens_used", 0),
        retrieval_calls_used=resources_dict.get("retrieval_calls_used", 0),
        verification_calls_used=resources_dict.get("verification_calls_used", 0),
        search_calls_used=resources_dict.get("search_calls_used", 0),
        elapsed_ms=resources_dict.get("elapsed_ms", 0),
        monetary_cost_microusd=resources_dict.get("monetary_cost_microusd", 0),
        policy_rejections_used=resources_dict.get("policy_rejections_used", 0),
    )


def verify_checkpoint_integrity(checkpoint: StateCheckpoint, task: EvidenceTask) -> bool:
    """Verify that a checkpoint can be restored correctly.

    Returns True if restore produces a matching state hash.
    """
    try:
        runtime = restore_runtime(checkpoint, task)
        return True
    except (ValueError, KeyError, TypeError):
        return False
