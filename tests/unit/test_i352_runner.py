"""Unit tests for I3.5.2 Selective Governor Runner."""
import pytest
from unittest.mock import MagicMock
from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.i3_5_2.modes import GovernorMode
from hrm_adaptive_memory.executive.i3_5_2.trajectory_runner import (
    I352FactorialRunner,
)
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus,
    VerificationState,
)
from hrm_adaptive_memory.executive.metareasoning_benchmark import I3BenchmarkTask, LatentTaskState
from hrm_adaptive_memory.executive.resources import ResourceBudget
from hrm_adaptive_memory.executive.selective_governor import (
    SelectiveGovernorGate,
    BaseInterventionPredictor,
    InterventionPrediction,
)
from hrm_adaptive_memory.executive.model_backend import ModelCallResult


def _make_mock_task() -> I3BenchmarkTask:
    latent = LatentTaskState(
        verification_state=VerificationState.SUFFICIENT,
        temporal_status=TemporalStatus.CURRENT,
        unresolved_conflict=False,
        composition_complete=True,
        expected_terminal=DecisionAction.ANSWER,
    )
    return I3BenchmarkTask(
        task_id="test_001",
        category="state_irrelevant_answer",
        task_summary="Test summary",
        high_stakes=False,
        semantic_structure_coarse="c_01",
        semantic_structure_exact="e_01",
        budget_profile="standard",
        observable_provenance_count=1,
        split="structure_dev_v2",
        controller_instance_id="inst_01",
        action_effects={},
        latent=latent,
    )


def _make_mock_budget() -> ResourceBudget:
    return ResourceBudget(
        max_executive_steps=8,
        max_retrieval_calls=5,
        max_verification_calls=5,
        max_search_calls=5,
        max_reasoning_tokens=512,
        max_elapsed_ms=10000,
    )


def _make_mock_call_result() -> ModelCallResult:
    return ModelCallResult(
        raw_output='{"action": "ANSWER", "reason_code": "READY", "target_id": null}',
        model_name="deepseek-chat",
        system_fingerprint="fp_01",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        finish_reason="stop",
    )


class TestI352Modes:
    def test_modes_exist(self):
        assert GovernorMode.OFF.value == "OFF"
        assert GovernorMode.ALWAYS_ON.value == "ALWAYS_ON"
        assert GovernorMode.SELECTIVE.value == "SELECTIVE"
        assert GovernorMode.SHADOW_SELECTIVE.value == "SHADOW_SELECTIVE"

    def test_runner_off_mode_never_calls_governor(self):
        backend = MagicMock()
        backend.generate.return_value = _make_mock_call_result()
        governor = MagicMock()
        runner = I352FactorialRunner(
            backend=backend,
            governor=governor,
        )
        task = _make_mock_task()
        budget = _make_mock_budget()
        cond = get_condition(ConditionID.AWARE_GOVERNOR)

        traj = runner._run_trajectory(task, budget, cond, governor_mode=GovernorMode.OFF)
        assert traj.governor_mode == "OFF"
        governor.assess.assert_not_called()
        assert traj.steps[0].packet_schema == "DAPH_V2B_I3_5_1_BASE_PACKET_V1"

    def test_runner_always_on_mode_calls_governor(self):
        backend = MagicMock()
        backend.generate.return_value = _make_mock_call_result()
        runner = I352FactorialRunner(backend=backend)
        task = _make_mock_task()
        budget = _make_mock_budget()
        cond = get_condition(ConditionID.AWARE_GOVERNOR)

        traj = runner._run_trajectory(task, budget, cond, governor_mode=GovernorMode.ALWAYS_ON)
        assert traj.governor_mode == "ALWAYS_ON"
        assert traj.interventions_approved >= 1
        assert traj.steps[0].packet_schema == "DAPH_V2B_I3_5_1_GOVERNOR_PACKET_V1"

    def test_runner_selective_mode_skips_when_harmful(self):
        backend = MagicMock()
        backend.generate.return_value = _make_mock_call_result()
        # Default rule-based gate predicts HARM for step 0 SUFFICIENT -> skips
        runner = I352FactorialRunner(backend=backend)
        task = _make_mock_task()
        budget = _make_mock_budget()
        cond = get_condition(ConditionID.AWARE_GOVERNOR)

        traj = runner._run_trajectory(task, budget, cond, governor_mode=GovernorMode.SELECTIVE)
        assert traj.governor_mode == "SELECTIVE"
        assert traj.interventions_approved == 0
        assert traj.steps[0].gate_intervened is False
        assert traj.steps[0].packet_schema == "DAPH_V2B_I3_5_1_BASE_PACKET_V1"

    def test_runner_selective_mode_intervenes_when_beneficial(self):
        backend = MagicMock()
        backend.generate.return_value = _make_mock_call_result()
        class MockApprovePredictor(BaseInterventionPredictor):
            def predict(self, features):
                return InterventionPrediction(
                    expected_delta_utility=25.0,
                    harm_probability=0.01,
                    help_probability=0.99,
                    confidence=0.95,
                    reason="MOCK_APPROVE",
                )

        gate = SelectiveGovernorGate(predictor=MockApprovePredictor())
        runner = I352FactorialRunner(backend=backend, gate=gate)
        task = _make_mock_task()
        budget = _make_mock_budget()
        cond = get_condition(ConditionID.AWARE_GOVERNOR)

        traj = runner._run_trajectory(task, budget, cond, governor_mode=GovernorMode.SELECTIVE)
        assert traj.governor_mode == "SELECTIVE"
        assert traj.interventions_approved >= 1
        assert traj.steps[0].gate_intervened is True
        assert traj.steps[0].packet_schema == "DAPH_V2B_I3_5_1_GOVERNOR_PACKET_V1"
