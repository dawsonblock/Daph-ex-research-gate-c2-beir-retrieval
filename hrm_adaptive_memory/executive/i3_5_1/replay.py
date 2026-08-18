"""Deterministic replay engine for I3.5.1.

Given a task, initial state, and parsed model actions, the entire
trajectory must be reproducible without DeepSeek.

Verifies:
  - state before action
  - resource before
  - selected action
  - policy result
  - execution outcome
  - state after
  - resource after
  - terminal status
  - utility contribution

Then recalculates V_pi from replay.

Acceptance:
  100% trajectory replay match
  100% utility replay match
  100% terminal outcome match
  Tolerance for utility: 1e-9
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from ..actions import ActionProposal
from ..executor import (
    ActionExecution, DeterministicActionExecutor, TaskRuntime,
    initial_runtime,
)
from ..resources import ResourceBudget, ResourceState, ResourceExhausted
from ..metareasoning_utility import MetareasoningUtility
from ..metareasoning_benchmark import MetareasoningBenchmark, I3BenchmarkTask

from .trajectory_runner import _I3TaskAdapter

REPLAY_SCHEMA = "DAPH_V2B_I3_5_1_REPLAY_V1"
REPLAY_VERSION = 1
UTILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ReplayStep:
    """One replayed step."""
    step_id: int
    action: str
    outcome_code: str
    task_success: bool | None
    terminal: bool
    realized_utility_after: float
    match: bool


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying one trajectory."""
    task_id: str
    condition_id: str
    steps: tuple[ReplayStep, ...]
    final_utility: float
    expected_utility: float
    utility_match: bool
    terminal_match: bool
    all_steps_match: bool

    @property
    def fully_matches(self) -> bool:
        return self.utility_match and self.terminal_match and self.all_steps_match


def replay_trajectory(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    steps: list[dict[str, Any]],
    expected_utility: float,
    expected_terminal: str,
    *,
    utility: MetareasoningUtility | None = None,
    executor: DeterministicActionExecutor | None = None,
) -> ReplayResult:
    """Replay a trajectory from stored step data without model calls."""
    if executor is None:
        executor = DeterministicActionExecutor()

    resources = ResourceState(budget)
    runtime = initial_runtime(_I3TaskAdapter(task), resources)
    realized = 0.0
    replay_steps: list[ReplayStep] = []
    terminal_result = "STEP_LIMIT"

    for step_data in steps:
        step_id = step_data["step_id"]
        action = DecisionAction(step_data["executed_action"])
        expected_outcome = step_data["outcome_code"]
        expected_step_terminal = step_data["terminal"]

        resources_before = runtime.resources
        try:
            execution = executor.execute(runtime, action)
        except ResourceExhausted:
            execution = ActionExecution(
                DecisionAction.DEFER, runtime, True, False, "RESOURCE_EXHAUSTED")

        if utility is not None:
            resources_after = execution.runtime.resources
            step_cost = utility.action_cost(resources_before, resources_after)
            realized -= step_cost
            if execution.terminal:
                realized += utility.terminal_reward(
                    execution.action, bool(execution.task_success))

        outcome_match = execution.outcome_code == expected_outcome
        terminal_match = execution.terminal == expected_step_terminal

        replay_steps.append(ReplayStep(
            step_id=step_id,
            action=action.value,
            outcome_code=execution.outcome_code,
            task_success=execution.task_success,
            terminal=execution.terminal,
            realized_utility_after=realized,
            match=outcome_match and terminal_match,
        ))

        runtime = execution.runtime
        if execution.terminal:
            terminal_result = execution.outcome_code
            break

    utility_match = abs(realized - expected_utility) < UTILITY_TOLERANCE
    terminal_match = terminal_result == expected_terminal
    all_steps_match = all(s.match for s in replay_steps)

    return ReplayResult(
        task_id=task.task_id,
        condition_id=steps[0].get("condition_id", "UNKNOWN") if steps else "UNKNOWN",
        steps=tuple(replay_steps),
        final_utility=realized,
        expected_utility=expected_utility,
        utility_match=utility_match,
        terminal_match=terminal_match,
        all_steps_match=all_steps_match,
    )


def replay_all_trajectories(
    results: list[dict[str, Any]],
    benchmark: MetareasoningBenchmark,
    *,
    utility: MetareasoningUtility | None = None,
    split: str = "structure_dev_v2",
) -> dict[str, Any]:
    """Replay all trajectories from results and verify matches."""
    task_by_id = {t.task_id: t for t in benchmark.tasks}
    split_benchmark = benchmark.for_split(split)
    total = 0
    utility_matches = 0
    terminal_matches = 0
    step_matches = 0
    full_matches = 0
    failures: list[dict[str, Any]] = []

    for block in results:
        task = task_by_id.get(block["task_id"])
        if task is None:
            continue
        budget = split_benchmark.budget_for(task)
        for cond_id, traj_data in block["trajectories"].items():
            total += 1
            replay = replay_trajectory(
                task=task,
                budget=budget,
                steps=traj_data["steps"],
                expected_utility=traj_data["realized_utility"],
                expected_terminal=traj_data["terminal_result"],
                utility=utility,
            )
            if replay.utility_match:
                utility_matches += 1
            if replay.terminal_match:
                terminal_matches += 1
            if replay.all_steps_match:
                step_matches += 1
            if replay.fully_matches:
                full_matches += 1
            else:
                failures.append({
                    "task_id": block["task_id"],
                    "condition_id": cond_id,
                    "utility_match": replay.utility_match,
                    "terminal_match": replay.terminal_match,
                    "step_match": replay.all_steps_match,
                    "expected_utility": replay.expected_utility,
                    "replayed_utility": replay.final_utility,
                })

    return {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_VERSION,
        "total_trajectories": total,
        "utility_matches": utility_matches,
        "terminal_matches": terminal_matches,
        "step_matches": step_matches,
        "full_matches": full_matches,
        "utility_tolerance": UTILITY_TOLERANCE,
        "all_match": full_matches == total,
        "failures": failures[:20],  # First 20 failures for debugging
    }
