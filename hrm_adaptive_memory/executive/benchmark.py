"""Frozen synthetic task corpus for V2B-I2 controller infrastructure."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hrm_adaptive_memory.cognitive_control.actions import validate_v2b_action
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import TemporalStatus, VerificationState


BENCHMARK_SCHEMA = "DAPH_V2B_I2_BENCHMARK_V1"
FROZEN_DEVELOPMENT_STATUS = "FROZEN_FOR_DEVELOPMENT"


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    category: str
    task_summary: str
    high_stakes: bool
    initial_verification_state: VerificationState
    initial_temporal_status: TemporalStatus
    unresolved_conflict: bool
    reasoning_required: bool
    expected_terminal: DecisionAction
    action_effects: Mapping[DecisionAction, Mapping[str, str]]

    def __post_init__(self) -> None:
        if (not self.task_id or self.task_id != self.task_id.lower()
                or not self.category or not self.task_summary):
            raise ValueError("benchmark tasks require lowercase ids, a category, and a summary")
        if self.expected_terminal not in {DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            raise ValueError("benchmark tasks must define one terminal V2B action")
        for action, effect in self.action_effects.items():
            validate_v2b_action(action)
            if not isinstance(effect, Mapping):
                raise ValueError("benchmark action effects must be mappings")


@dataclass(frozen=True)
class FrozenBenchmark:
    benchmark_id: str
    tasks: tuple[BenchmarkTask, ...]
    metadata: Mapping[str, Any]


def load_frozen_benchmark(path: str | Path) -> FrozenBenchmark:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError("unsupported V2B-I2 benchmark schema")
    if payload.get("status") != FROZEN_DEVELOPMENT_STATUS:
        raise ValueError("V2B-I2 benchmark must be frozen for development")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("V2B-I2 frozen benchmark must contain tasks")
    tasks = []
    for raw in raw_tasks:
        effects = {
            validate_v2b_action(action): dict(value)
            for action, value in dict(raw.get("action_effects", {})).items()
        }
        tasks.append(BenchmarkTask(
            task_id=raw["task_id"], category=raw["category"], task_summary=raw["task_summary"],
            high_stakes=bool(raw["high_stakes"]),
            initial_verification_state=VerificationState(raw["initial_verification_state"]),
            initial_temporal_status=TemporalStatus(raw["initial_temporal_status"]),
            unresolved_conflict=bool(raw["unresolved_conflict"]),
            reasoning_required=bool(raw["reasoning_required"]),
            expected_terminal=validate_v2b_action(raw["expected_terminal"]),
            action_effects=effects,
        ))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("V2B-I2 benchmark task ids must be unique")
    return FrozenBenchmark(str(payload.get("benchmark_id", "")), tuple(tasks),
                           {key: value for key, value in payload.items() if key != "tasks"})
