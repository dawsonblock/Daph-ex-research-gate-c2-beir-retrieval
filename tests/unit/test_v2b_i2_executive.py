"""V2B-I2 deterministic executive experiment infrastructure contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.benchmark import load_frozen_benchmark
from hrm_adaptive_memory.executive.controller import (
    ControlObservation, DeterministicCognitiveStateController, FixedBaselineController)
from hrm_adaptive_memory.executive.loop import V2BExperimentLoop
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import (
    ActionCost, ResourceBudget, ResourceExhausted, ResourceState)


ROOT = Path(__file__).parents[2]
BENCHMARK_PATH = ROOT / "experiments/v2b/tasks/v2b_i2_frozen_benchmark_v1.json"
POLICY_PATH = ROOT / "configs/v2b_i2_policy_v1.json"


def _loop():
    return V2BExperimentLoop(policy=load_frozen_policy(POLICY_PATH))


def test_i2_benchmark_is_frozen_nonempty_and_covers_resource_allocation_task_classes():
    benchmark = load_frozen_benchmark(BENCHMARK_PATH)
    assert {task.category for task in benchmark.tasks} == {
        "immediate_answer", "retrieval_required", "verification_required", "conflict",
        "stale_temporal", "false_memory", "search_more_required", "reason_more_required",
        "insufficient_evidence"}


def test_i2_control_and_cognitive_conditions_use_one_executor_policy_and_budget(tmp_path):
    benchmark = load_frozen_benchmark(BENCHMARK_PATH)
    loop = _loop()
    control = loop.run_condition(benchmark, condition="CONTROL", controller=FixedBaselineController(),
                                 store_root=tmp_path)
    v2b = loop.run_condition(benchmark, condition="V2B",
                             controller=DeterministicCognitiveStateController(), store_root=tmp_path)
    assert control.controller_id != v2b.controller_id
    assert v2b.metrics["tasks"] == len(benchmark.tasks)
    assert v2b.metrics["task_successes"] == len(benchmark.tasks)
    assert v2b.metrics["retrieval_calls"] < control.metrics["retrieval_calls"]
    assert v2b.metrics["reasoning_tokens"] < control.metrics["reasoning_tokens"]

    verification = next(task for task in v2b.tasks if task.task_id == "v2b_i2_verification_required")
    assert tuple(trace.executed_action for trace in verification.traces) == (
        DecisionAction.VERIFY, DecisionAction.ANSWER)
    conflict = next(task for task in v2b.tasks if task.task_id == "v2b_i2_conflict")
    assert tuple(trace.executed_action for trace in conflict.traces) == (DecisionAction.DEFER,)


def test_i2_control_controller_does_not_receive_cognitive_state_or_choose_tools_directly():
    proposal = FixedBaselineController().choose(ControlObservation(
        "v2b_i2_task", "task summary", executive_steps_used=0, executive_steps_remaining=12))
    assert proposal.action is DecisionAction.RETRIEVE
    assert proposal.reason_code == "FIXED_BASELINE"


def test_i2_resource_state_is_hard_limited_for_each_action():
    resources = ResourceState(ResourceBudget(max_retrieval_calls=0))
    assert not resources.can_execute(DecisionAction.RETRIEVE)
    with pytest.raises(ResourceExhausted, match="resource budget"):
        resources.consume(DecisionAction.RETRIEVE)


def test_i2_resource_costs_cannot_reduce_accounted_usage():
    with pytest.raises(ValueError, match="nonnegative"):
        ActionCost(reasoning_tokens=-1)


def test_i2_benchmark_refuses_non_frozen_or_empty_inputs(tmp_path):
    payload = json.loads(BENCHMARK_PATH.read_text())
    payload["status"] = "NOT_FROZEN"
    path = tmp_path / "bad_benchmark.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen for development"):
        load_frozen_benchmark(path)
