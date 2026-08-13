"""Deterministic V2B-I3 environment and projection functions.

`I3Runtime` is latent environment state.  Controllers receive only the
`CognitiveStateSnapshot` returned by :func:`build_observable_snapshot` (or a
masked observation), never this runtime object or a task's terminal label.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.datalog import DatalogFact
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, ConflictSummary, DecisionSummary, MemorySummary,
    TemporalStatus, VerificationState, VerificationSummary)

from .metareasoning_benchmark import I3BenchmarkTask
from .resources import ResourceState


@dataclass(frozen=True)
class I3Runtime:
    task: I3BenchmarkTask
    resources: ResourceState
    verification_state: VerificationState
    temporal_status: TemporalStatus
    unresolved_conflict: bool
    composition_complete: bool
    retrieved: bool = False
    searched: bool = False


@dataclass(frozen=True)
class I3ActionExecution:
    action: DecisionAction
    runtime: I3Runtime
    terminal: bool
    task_success: bool | None
    outcome_code: str


def initial_i3_runtime(task: I3BenchmarkTask, resources: ResourceState) -> I3Runtime:
    latent = task.latent
    return I3Runtime(task, resources, latent.verification_state, latent.temporal_status,
                     latent.unresolved_conflict, latent.composition_complete)


def runtime_state_hash(runtime: I3Runtime) -> str:
    """Commit state that affects transitions, excluding hidden scoring labels."""
    material = {
        "task_id": runtime.task.task_id,
        "verification_state": runtime.verification_state.value,
        "temporal_status": runtime.temporal_status.value,
        "unresolved_conflict": runtime.unresolved_conflict,
        "composition_complete": runtime.composition_complete,
        "retrieved": runtime.retrieved,
        "searched": runtime.searched,
        "resources": runtime.resources.as_dict(),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def policy_facts(runtime: I3Runtime) -> tuple[DatalogFact, ...]:
    """Shared safety substrate; these facts are deliberately not controller input."""
    task_id = runtime.task.task_id
    facts: list[DatalogFact] = []
    if runtime.task.high_stakes:
        facts.append(DatalogFact("high_stakes", (task_id,)))
    if runtime.verification_state is VerificationState.UNVERIFIED:
        facts.append(DatalogFact("unverified", (task_id,)))
    if runtime.verification_state is VerificationState.FALSIFIED:
        facts.append(DatalogFact("falsified", (task_id,)))
    if (runtime.temporal_status is TemporalStatus.STALE
            or runtime.verification_state is VerificationState.STALE):
        facts.append(DatalogFact("stale", (task_id,)))
    if runtime.unresolved_conflict:
        facts.append(DatalogFact("unresolved_conflict", (task_id,)))
    # This is a policy-only determination, never a public controller fact.
    if answerable(runtime):
        facts.append(DatalogFact("evidence_sufficient", (task_id,)))
    resources = runtime.resources.as_dict()
    for predicate, field in (
        ("retrieval_exhausted", "retrieval_calls_remaining"),
        ("verification_exhausted", "verification_calls_remaining"),
        ("search_exhausted", "search_calls_remaining"),
    ):
        if resources[field] == 0:
            facts.append(DatalogFact(predicate, (task_id,)))
    return tuple(sorted(facts))


def answerable(runtime: I3Runtime) -> bool:
    return (
        runtime.verification_state is VerificationState.SUFFICIENT
        and runtime.temporal_status is TemporalStatus.CURRENT
        and not runtime.unresolved_conflict
        and runtime.composition_complete
    )


def build_observable_snapshot(runtime: I3Runtime, *, prior_decisions: tuple[DecisionSummary, ...],
                              prior_outcomes: tuple[str, ...]) -> CognitiveStateSnapshot:
    """Project bounded observable cognitive state without terminal/oracle labels."""
    task = runtime.task
    conflicts = (() if not runtime.unresolved_conflict else (
        ConflictSummary(f"conflict-{task.task_id}", "benchmark_relation", 2, "UNRESOLVED"),))
    signals = ("COMPOSITION_COMPLETE" if runtime.composition_complete
               else "COMPOSITION_INCOMPLETE",)
    return CognitiveStateSnapshot(
        task_id=task.task_id, task_summary=task.task_summary,
        relevant_memories=(MemorySummary(
            f"memory-{task.task_id}", 1.0, runtime.verification_state,
            task.observable_provenance_count, task.observable_provenance_count,
            "UNRESOLVED" if runtime.unresolved_conflict else "NONE", runtime.temporal_status),),
        verification_states=(VerificationSummary(
            f"verification-{task.task_id}", runtime.verification_state,
            task.observable_provenance_count, None),),
        provenance_summaries=(f"lineage_count={task.observable_provenance_count}",),
        temporal_status=runtime.temporal_status, unresolved_conflicts=conflicts,
        prior_decisions=prior_decisions[-16:], prior_outcomes=prior_outcomes[-16:],
        resource_state=runtime.resources.as_dict(), policy_facts=(),
        observation_signals=signals,
    )


def state_delta(before: I3Runtime, after: I3Runtime) -> dict[str, object]:
    """Per-action semantic delta used for utility attribution, not task success proxy."""
    fields = {
        "verification_state": (before.verification_state.value, after.verification_state.value),
        "temporal_status": (before.temporal_status.value, after.temporal_status.value),
        "unresolved_conflict": (before.unresolved_conflict, after.unresolved_conflict),
        "composition_complete": (before.composition_complete, after.composition_complete),
    }
    changed = {name: {"before": old, "after": new}
               for name, (old, new) in fields.items() if old != new}
    improved = (
        (before.verification_state is not VerificationState.SUFFICIENT
         and after.verification_state is VerificationState.SUFFICIENT)
        or (before.temporal_status is not TemporalStatus.CURRENT
            and after.temporal_status is TemporalStatus.CURRENT)
        or (before.unresolved_conflict and not after.unresolved_conflict)
        or (not before.composition_complete and after.composition_complete)
    )
    return {
        "changed": changed,
        "evidence_delta": {"before": before.verification_state.value,
                           "after": after.verification_state.value},
        "verification_delta": {"before": before.verification_state.value,
                               "after": after.verification_state.value},
        "temporal_delta": {"before": before.temporal_status.value,
                           "after": after.temporal_status.value},
        "conflict_delta": {"before": before.unresolved_conflict,
                           "after": after.unresolved_conflict},
        "reasoning_delta": {"before": before.composition_complete,
                            "after": after.composition_complete},
        "answerability_delta": {"before": answerable(before), "after": answerable(after)},
        "decision_relevant_improvement": improved,
    }


class DeterministicMetareasoningExecutor:
    """Applies benchmark dynamics. It never calls an LLM, retrieval service, or HTTP."""

    @staticmethod
    def _apply_effect(runtime: I3Runtime, effect: Mapping[str, str]) -> I3Runtime:
        updates: dict[str, object] = {}
        if "verification_state" in effect:
            updates["verification_state"] = VerificationState(effect["verification_state"])
        if "temporal_status" in effect:
            updates["temporal_status"] = TemporalStatus(effect["temporal_status"])
        if "unresolved_conflict" in effect:
            updates["unresolved_conflict"] = effect["unresolved_conflict"] == "true"
        if "composition_complete" in effect:
            updates["composition_complete"] = effect["composition_complete"] == "true"
        return replace(runtime, **updates)

    def execute(self, runtime: I3Runtime, action: DecisionAction) -> I3ActionExecution:
        next_runtime = replace(runtime, resources=runtime.resources.consume(action))
        if action is DecisionAction.RETRIEVE:
            next_runtime = replace(next_runtime, retrieved=True)
        elif action is DecisionAction.SEARCH_MORE:
            next_runtime = replace(next_runtime, searched=True)
        next_runtime = self._apply_effect(next_runtime, next_runtime.task.action_effects.get(action, {}))
        if action not in {DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            # A transition that consumes the final executable budget must be
            # terminal. This prevents nonterminal dead-end states with no
            # legal action for either the policy-constrained oracle or runner.
            if not any(next_runtime.resources.can_execute(candidate) for candidate in V2B_ACTIONS):
                return I3ActionExecution(action, next_runtime, True, False, "RESOURCE_EXHAUSTED")
            return I3ActionExecution(action, next_runtime, False, None, f"{action.value}_COMPLETED")
        if action is DecisionAction.ANSWER:
            success = next_runtime.task.latent.expected_terminal is DecisionAction.ANSWER and answerable(next_runtime)
        else:
            success = next_runtime.task.latent.expected_terminal is action
        return I3ActionExecution(action, next_runtime, True, success,
                                 "TASK_SUCCESS" if success else "TASK_FAILURE")
