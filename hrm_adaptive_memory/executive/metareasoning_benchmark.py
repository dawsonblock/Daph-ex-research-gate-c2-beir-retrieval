"""Frozen V2B-I3 metareasoning benchmark with explicit latent/observable layers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hrm_adaptive_memory.cognitive_control.actions import validate_v2b_action
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import TemporalStatus, VerificationState

from .resources import ResourceBudget


BENCHMARK_SCHEMA = "DAPH_V2B_I3_METAREASONING_BENCHMARK_V1"
FROZEN_DEVELOPMENT_STATUS = "FROZEN_FOR_DEVELOPMENT"


@dataclass(frozen=True)
class LatentTaskState:
    """Environment state used for dynamics/scoring and never passed to a controller."""

    verification_state: VerificationState
    temporal_status: TemporalStatus
    unresolved_conflict: bool
    composition_complete: bool
    expected_terminal: DecisionAction


@dataclass(frozen=True)
class I3BenchmarkTask:
    task_id: str
    category: str
    task_summary: str
    high_stakes: bool
    budget_profile: str
    latent: LatentTaskState
    observable_provenance_count: int
    action_effects: Mapping[DecisionAction, Mapping[str, str]]

    def __post_init__(self) -> None:
        if (not self.task_id or self.task_id != self.task_id.lower()
                or not self.category or not self.task_summary):
            raise ValueError("I3 tasks require lowercase ids, a category, and a summary")
        if self.latent.expected_terminal not in {
                DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            raise ValueError("I3 tasks require a terminal action")
        if self.observable_provenance_count < 0:
            raise ValueError("observable provenance count must be nonnegative")
        for action, effect in self.action_effects.items():
            validate_v2b_action(action)
            if not isinstance(effect, Mapping):
                raise ValueError("I3 action effects must be mappings")


@dataclass(frozen=True)
class MetareasoningBenchmark:
    benchmark_id: str
    tasks: tuple[I3BenchmarkTask, ...]
    budget_profiles: Mapping[str, ResourceBudget]
    utility_weights: Mapping[str, float]
    metadata: Mapping[str, Any]

    def budget_for(self, task: I3BenchmarkTask) -> ResourceBudget:
        try:
            return self.budget_profiles[task.budget_profile]
        except KeyError as error:
            raise ValueError(f"unknown I3 budget profile: {task.budget_profile}") from error


def _load_budget_profiles(raw: object) -> Mapping[str, ResourceBudget]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("I3 benchmark needs nonempty named budget profiles")
    profiles: dict[str, ResourceBudget] = {}
    for name, values in raw.items():
        if (not isinstance(name, str) or not name.isupper() or not isinstance(values, Mapping)):
            raise ValueError("I3 budget profiles require uppercase names and mapping values")
        profiles[name] = ResourceBudget(**dict(values))
    return profiles


def _load_utility_weights(raw: object) -> Mapping[str, float]:
    required = {"success_reward", "failure_penalty", "executive_step", "retrieval",
                "verification", "search", "reasoning_128_tokens", "logical_ms"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("I3 benchmark has an invalid frozen utility-weight set")
    weights = {str(name): float(value) for name, value in raw.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError("I3 utility weights must be nonnegative")
    if weights["success_reward"] == 0 or weights["failure_penalty"] == 0:
        raise ValueError("I3 terminal utility weights must be positive")
    return weights


def load_metareasoning_benchmark(path: str | Path) -> MetareasoningBenchmark:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError("unsupported V2B-I3 metareasoning benchmark schema")
    if payload.get("status") != FROZEN_DEVELOPMENT_STATUS:
        raise ValueError("V2B-I3 benchmark must be frozen for development")
    profiles = _load_budget_profiles(payload.get("budget_profiles"))
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("V2B-I3 frozen benchmark must contain tasks")
    tasks: list[I3BenchmarkTask] = []
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise ValueError("I3 tasks must be mappings")
        latent = raw.get("latent")
        if not isinstance(latent, Mapping):
            raise ValueError("I3 task needs a latent environment state")
        effects = {
            validate_v2b_action(action): dict(effect)
            for action, effect in dict(raw.get("action_effects", {})).items()
        }
        task = I3BenchmarkTask(
            task_id=str(raw["task_id"]), category=str(raw["category"]),
            task_summary=str(raw["task_summary"]), high_stakes=bool(raw["high_stakes"]),
            budget_profile=str(raw["budget_profile"]),
            latent=LatentTaskState(
                verification_state=VerificationState(latent["verification_state"]),
                temporal_status=TemporalStatus(latent["temporal_status"]),
                unresolved_conflict=bool(latent["unresolved_conflict"]),
                composition_complete=bool(latent["composition_complete"]),
                expected_terminal=validate_v2b_action(latent["expected_terminal"]),
            ),
            observable_provenance_count=int(raw.get("observable_provenance_count", 0)),
            action_effects=effects,
        )
        if task.budget_profile not in profiles:
            raise ValueError(f"task {task.task_id} references an unknown budget profile")
        tasks.append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("V2B-I3 task ids must be unique")
    return MetareasoningBenchmark(
        benchmark_id=str(payload.get("benchmark_id", "")), tasks=tuple(tasks),
        budget_profiles=profiles, utility_weights=_load_utility_weights(payload.get("utility_weights")),
        metadata={key: value for key, value in payload.items()
                  if key not in {"tasks", "budget_profiles", "utility_weights"}},
    )
