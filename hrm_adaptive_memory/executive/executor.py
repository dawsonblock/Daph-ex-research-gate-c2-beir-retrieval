"""Deterministic synthetic action executor for V2B-I2 benchmark tasks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.datalog import DatalogFact
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, ConflictSummary, DecisionSummary, MemorySummary,
    TemporalStatus, VerificationState, VerificationSummary)

from .benchmark import BenchmarkTask
from .resources import ResourceState


@dataclass(frozen=True)
class TaskRuntime:
    task: BenchmarkTask
    resources: ResourceState
    verification_state: VerificationState
    temporal_status: TemporalStatus
    unresolved_conflict: bool
    reasoning_complete: bool = False
    retrieved: bool = False
    searched: bool = False


@dataclass(frozen=True)
class ActionExecution:
    action: DecisionAction
    runtime: TaskRuntime
    terminal: bool
    task_success: bool | None
    outcome_code: str


def initial_runtime(task: BenchmarkTask, resources: ResourceState) -> TaskRuntime:
    return TaskRuntime(task, resources, task.initial_verification_state,
                       task.initial_temporal_status, task.unresolved_conflict)


def build_cognitive_state(runtime: TaskRuntime, *, prior_decisions: tuple[DecisionSummary, ...],
                          prior_outcomes: tuple[str, ...]) -> CognitiveStateSnapshot:
    task = runtime.task
    facts = []
    if task.high_stakes:
        facts.append(DatalogFact("high_stakes", (task.task_id,)))
    if runtime.verification_state is VerificationState.UNVERIFIED:
        facts.append(DatalogFact("unverified", (task.task_id,)))
    if runtime.verification_state is VerificationState.FALSIFIED:
        facts.append(DatalogFact("falsified", (task.task_id,)))
    if runtime.temporal_status is TemporalStatus.STALE or runtime.verification_state is VerificationState.STALE:
        facts.append(DatalogFact("stale", (task.task_id,)))
    if runtime.unresolved_conflict:
        facts.append(DatalogFact("unresolved_conflict", (task.task_id,)))
    if task.reasoning_required and not runtime.reasoning_complete:
        facts.append(DatalogFact("reasoning_required", (task.task_id,)))
    if (runtime.verification_state is VerificationState.SUFFICIENT
            and runtime.temporal_status is TemporalStatus.CURRENT
            and not runtime.unresolved_conflict and (not task.reasoning_required or runtime.reasoning_complete)):
        facts.append(DatalogFact("evidence_sufficient", (task.task_id,)))
    resources = runtime.resources.as_dict()
    if resources["retrieval_calls_remaining"] == 0:
        facts.append(DatalogFact("retrieval_exhausted", (task.task_id,)))
    if resources["verification_calls_remaining"] == 0:
        facts.append(DatalogFact("verification_exhausted", (task.task_id,)))
    conflicts = (() if not runtime.unresolved_conflict else (
        ConflictSummary(f"conflict-{task.task_id}", "benchmark_relation", 2, "UNRESOLVED"),))
    return CognitiveStateSnapshot(
        task_id=task.task_id, task_summary=task.task_summary,
        relevant_memories=(MemorySummary(
            f"memory-{task.task_id}", 1.0, runtime.verification_state, 1,
            1, "UNRESOLVED" if runtime.unresolved_conflict else "NONE", runtime.temporal_status),),
        verification_states=(VerificationSummary(
            f"verification-{task.task_id}", runtime.verification_state, 1, None),),
        provenance_summaries=(f"lineage_count=1",), temporal_status=runtime.temporal_status,
        unresolved_conflicts=conflicts, prior_decisions=prior_decisions[-16:],
        prior_outcomes=prior_outcomes[-16:], resource_state=resources,
        policy_facts=tuple(sorted(facts)),
    )


class DeterministicActionExecutor:
    """Applies frozen task effects; it never calls retrieval, HTTP, or an LLM."""

    @staticmethod
    def _apply_effect(runtime: TaskRuntime, effect: Mapping[str, str]) -> TaskRuntime:
        updates = {}
        if "verification_state" in effect:
            updates["verification_state"] = VerificationState(effect["verification_state"])
        if "temporal_status" in effect:
            updates["temporal_status"] = TemporalStatus(effect["temporal_status"])
        if "unresolved_conflict" in effect:
            updates["unresolved_conflict"] = effect["unresolved_conflict"] == "true"
        if effect.get("reasoning_complete") == "true":
            updates["reasoning_complete"] = True
        return replace(runtime, **updates)

    def execute(self, runtime: TaskRuntime, action: DecisionAction) -> ActionExecution:
        next_runtime = replace(runtime, resources=runtime.resources.consume(action))
        if action is DecisionAction.RETRIEVE:
            next_runtime = replace(next_runtime, retrieved=True)
        elif action is DecisionAction.SEARCH_MORE:
            next_runtime = replace(next_runtime, searched=True)
        next_runtime = self._apply_effect(next_runtime, next_runtime.task.action_effects.get(action, {}))
        if action not in {DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            return ActionExecution(action, next_runtime, False, None, f"{action.value}_COMPLETED")
        evidence_sufficient = (
            next_runtime.verification_state is VerificationState.SUFFICIENT
            and next_runtime.temporal_status is TemporalStatus.CURRENT
            and not next_runtime.unresolved_conflict
            and (not next_runtime.task.reasoning_required or next_runtime.reasoning_complete))
        if action is DecisionAction.ANSWER:
            success = next_runtime.task.expected_terminal is DecisionAction.ANSWER and evidence_sufficient
        else:
            success = next_runtime.task.expected_terminal is action
        return ActionExecution(action, next_runtime, True, success,
                               "TASK_SUCCESS" if success else "TASK_FAILURE")
