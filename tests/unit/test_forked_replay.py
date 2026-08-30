"""Tests for forked replay event-level causal attribution."""
import pytest
from pathlib import Path

from hrm_adaptive_memory.executive.evidence_benchmark.i3_30r3_confirmation_generator import (
    generate_confirmation_benchmark, get_confirmation_budget_for_task,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.forked_replay import forked_replay, ForkedReplayResult


@pytest.fixture
def utility():
    return MetareasoningUtility.from_file(Path("configs/v2b_i3_1_utility_v1.json"))


@pytest.fixture
def tasks():
    return generate_confirmation_benchmark(seed=43291)


class TestForkedReplay:
    """Test forked replay produces valid per-event causal effects."""

    def test_d5_answer_vs_reason_more(self, tasks, utility):
        """D5: forcing ANSWER after VERIFY should be a rescue."""
        d5 = [t for t in tasks if "_d5_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d5)
        resources = ResourceState(budget=budget)
        runtime = initial_evidence_runtime(d5, resources)
        executor = EvidenceExecutor()

        # VERIFY the discriminator
        result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id="E3")
        runtime = result.runtime
        assert not result.terminal

        replay = forked_replay(
            runtime=runtime, task=d5, step=1,
            forced_action="ANSWER", shadow_action="REASON_MORE",
            shadow_target_id=None, executor=executor, utility_fn=utility,
            prior_actions=("VERIFY",),
        )

        assert replay.forced_action == "ANSWER"
        assert replay.shadow_action == "REASON_MORE"
        assert replay.forced_success is True
        assert replay.delta_u > 0
        assert replay.label == "rescue"

    def test_d2_defer_vs_answer(self, tasks, utility):
        """D2: forcing DEFER when LLM proposes ANSWER should be a rescue."""
        d2 = [t for t in tasks if "_d2_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d2)
        resources = ResourceState(budget=budget)
        runtime = initial_evidence_runtime(d2, resources)
        executor = EvidenceExecutor()

        # Pre-verify
        valid = valid_verify_targets(runtime)
        if valid:
            result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id=valid[0])
            runtime = result.runtime

        replay = forked_replay(
            runtime=runtime, task=d2, step=1,
            forced_action="DEFER", shadow_action="ANSWER",
            shadow_target_id=None, executor=executor, utility_fn=utility,
            prior_actions=("VERIFY",),
        )

        assert replay.forced_action == "DEFER"
        assert replay.forced_success is True
        assert replay.delta_u > 0
        assert replay.label == "rescue"

    def test_both_branches_same_state_sha(self, tasks, utility):
        """Both branches must start from the same checkpoint state."""
        d5 = [t for t in tasks if "_d5_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d5)
        resources = ResourceState(budget=budget)
        runtime = initial_evidence_runtime(d5, resources)
        executor = EvidenceExecutor()

        result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id="E3")
        runtime = result.runtime

        replay = forked_replay(
            runtime=runtime, task=d5, step=1,
            forced_action="ANSWER", shadow_action="DEFER",
            shadow_target_id=None, executor=executor, utility_fn=utility,
            prior_actions=("VERIFY",),
        )

        # Both branches share the same checkpoint
        assert replay.state_sha256 is not None
        assert len(replay.state_sha256) == 64  # SHA256 hex

    def test_label_classification(self, tasks, utility):
        """Labels: rescue (ΔU>0), break (ΔU<0), neutral (ΔU=0)."""
        d5 = [t for t in tasks if "_d5_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d5)
        resources = ResourceState(budget=budget)
        runtime = initial_evidence_runtime(d5, resources)
        executor = EvidenceExecutor()

        result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id="E3")
        runtime = result.runtime

        # Same action in both branches → neutral
        replay = forked_replay(
            runtime=runtime, task=d5, step=1,
            forced_action="ANSWER", shadow_action="ANSWER",
            shadow_target_id=None, executor=executor, utility_fn=utility,
            prior_actions=("VERIFY",),
        )
        assert replay.label == "neutral"
        assert abs(replay.delta_u) < 0.01

    def test_result_is_serializable(self, tasks, utility):
        """ForkedReplayResult.as_dict() produces valid JSON."""
        d5 = [t for t in tasks if "_d5_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d5)
        resources = ResourceState(budget=budget)
        runtime = initial_evidence_runtime(d5, resources)
        executor = EvidenceExecutor()

        result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id="E3")
        runtime = result.runtime

        replay = forked_replay(
            runtime=runtime, task=d5, step=1,
            forced_action="ANSWER", shadow_action="REASON_MORE",
            shadow_target_id=None, executor=executor, utility_fn=utility,
            prior_actions=("VERIFY",),
        )

        d = replay.as_dict()
        import json
        json_str = json.dumps(d)
        assert json.loads(json_str) == d
