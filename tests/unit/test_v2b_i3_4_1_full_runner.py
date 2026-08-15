"""Tests for the I3.4.1 full experiment runner."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.model_backend import ModelCallResult, StubBackend
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    STATE_BLIND_MASK, STATE_AWARE_MASK)
from hrm_adaptive_memory.executive.i3_4_full_runner import (
    FullExperimentRunner, RUNNER_SCHEMA, RUNNER_VERSION,
    TrajectoryStep, ConditionTrajectory, PairedTrajectoryResult,
    save_receipts, save_results, score_results, run_statistical_analysis,
    _I3TaskAdapter)

BENCHMARK_PATH = "experiments/v2b_i3_3/manifests/v2b_i3_3_benchmark_manifest_v1.json"


class _TestBackend(StubBackend):
    """StubBackend with the extra attributes the runner expects."""
    def __init__(self):
        super().__init__()
        self.task_id = ""
        self.condition = ""
        self.pair_id = ""
        self.call_receipts: list = []

    def generate(self, *, system_prompt, user_prompt, temperature, max_tokens):
        result = super().generate(
            system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=temperature, max_tokens=max_tokens)
        return ModelCallResult(
            raw_output=result.raw_output,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=0,
            latency_ms=10,
            model_name="stub-deterministic-v1",
            system_fingerprint="stub-fp-001",
            finish_reason="stop")


@pytest.fixture(scope="module")
def benchmark():
    return load_metareasoning_benchmark(BENCHMARK_PATH)


@pytest.fixture
def dev_tasks(benchmark):
    return [t for t in benchmark.tasks if t.split == "development"]


@pytest.fixture
def runner():
    return FullExperimentRunner(backend=_TestBackend())


class TestRunnerSchema:
    def test_runner_schema_is_frozen(self):
        assert RUNNER_SCHEMA == "DAPH_V2B_I3_4_FULL_RUNNER_V1"
        assert RUNNER_VERSION == 1

    def test_runner_max_steps_default(self):
        r = FullExperimentRunner(backend=_TestBackend())
        assert r.max_steps == 24


class TestTaskAdapter:
    def test_adapter_exposes_i2_compatible_fields(self, dev_tasks):
        task = dev_tasks[0]
        adapter = _I3TaskAdapter(task)
        assert adapter.task_id == task.task_id
        assert adapter.task_summary == task.task_summary
        assert adapter.high_stakes == task.high_stakes
        assert adapter.initial_verification_state == task.latent.verification_state
        assert adapter.initial_temporal_status == task.latent.temporal_status
        assert adapter.unresolved_conflict == task.latent.unresolved_conflict
        assert adapter.expected_terminal == task.latent.expected_terminal
        assert adapter.action_effects == task.action_effects


class TestObservationConstruction:
    def test_blind_mask_produces_none_cognitive_state(self, runner, dev_tasks, benchmark):
        task = dev_tasks[0]
        from hrm_adaptive_memory.executive.resources import ResourceState
        from hrm_adaptive_memory.executive.executor import initial_runtime
        budget = benchmark.budget_for(task)
        runtime = initial_runtime(_I3TaskAdapter(task), ResourceState(budget))
        obs = runner._make_controller_observation(
            runtime, task, STATE_BLIND_MASK, (), ())
        assert obs.cognitive_state is None

    def test_aware_mask_produces_cognitive_state(self, runner, dev_tasks, benchmark):
        task = dev_tasks[0]
        from hrm_adaptive_memory.executive.resources import ResourceState
        from hrm_adaptive_memory.executive.executor import initial_runtime
        budget = benchmark.budget_for(task)
        runtime = initial_runtime(_I3TaskAdapter(task), ResourceState(budget))
        obs = runner._make_controller_observation(
            runtime, task, STATE_AWARE_MASK, (), ())
        assert obs.cognitive_state is not None


class TestTrajectoryExecution:
    def test_pair_produces_two_trajectories(self, runner, dev_tasks, benchmark):
        result = runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        assert result.blind.condition == "BLIND"
        assert result.aware.condition == "AWARE"
        assert len(result.blind.steps) >= 1
        assert len(result.aware.steps) >= 1

    def test_pair_has_fingerprint_check(self, runner, dev_tasks, benchmark):
        result = runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        assert result.fingerprint_match is True
        assert result.pair_valid is True

    def test_pair_id_format(self, runner, dev_tasks, benchmark):
        result = runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        assert result.pair_id == f"v2b_i3_4_experiment_v1:{dev_tasks[0].task_id}"

    def test_trajectory_steps_have_required_fields(self, runner, dev_tasks, benchmark):
        result = runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        for step in result.blind.steps:
            assert isinstance(step.step_id, int)
            assert isinstance(step.proposed_action, str)
            assert isinstance(step.reason_code, str)
            assert isinstance(step.executed_action, str)
            assert isinstance(step.terminal, bool)

    def test_terminal_trajectory_has_terminal_step(self, runner, dev_tasks, benchmark):
        result = runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        assert result.blind.terminal_result != ""
        assert result.aware.terminal_result != ""
        # Last step should be terminal
        if len(result.blind.steps) > 0:
            assert result.blind.steps[-1].terminal is True

    def test_model_calls_match_steps(self, runner, dev_tasks, benchmark):
        result = runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        assert result.blind.model_calls == len(result.blind.steps)
        assert result.aware.model_calls == len(result.aware.steps)


class TestRunnerSummary:
    def test_summary_has_schema(self, runner, dev_tasks, benchmark):
        runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        summary = runner.runner_summary()
        assert summary["schema"] == RUNNER_SCHEMA
        assert summary["schema_version"] == RUNNER_VERSION

    def test_summary_counts_pairs(self, runner, dev_tasks, benchmark):
        for task in dev_tasks[:3]:
            runner.run_pair(task, benchmark.budget_for(task))
        summary = runner.runner_summary()
        assert summary["pairs_completed"] == 3

    def test_summary_counts_model_calls(self, runner, dev_tasks, benchmark):
        for task in dev_tasks[:3]:
            runner.run_pair(task, benchmark.budget_for(task))
        summary = runner.runner_summary()
        assert summary["total_model_calls"] > 0


class TestSplitExecution:
    def test_run_split_with_max_tasks(self, runner, benchmark):
        results = runner.run_split(benchmark, "development", max_tasks=3)
        assert len(results) == 3
        assert all(r.pair_valid for r in results)


class TestPersistence:
    def test_save_results_roundtrip(self, runner, dev_tasks, benchmark, tmp_path):
        for task in dev_tasks[:3]:
            runner.run_pair(task, benchmark.budget_for(task))
        path = tmp_path / "results.json"
        sha = save_results(runner.results, path)
        assert len(sha) == 64
        data = json.loads(path.read_text())
        assert data["schema"] == RUNNER_SCHEMA
        assert len(data["results"]) == 3

    def test_save_results_has_step_details(self, runner, dev_tasks, benchmark, tmp_path):
        runner.run_pair(dev_tasks[0], benchmark.budget_for(dev_tasks[0]))
        path = tmp_path / "results.json"
        save_results(runner.results, path)
        data = json.loads(path.read_text())
        result = data["results"][0]
        assert "blind" in result
        assert "aware" in result
        assert "steps" in result["blind"]
        assert "steps" in result["aware"]


class TestScoringIntegration:
    def test_score_results_returns_contributions_and_deltas(self, runner, benchmark, dev_tasks):
        for task in dev_tasks[:5]:
            runner.run_pair(task, benchmark.budget_for(task))

        oracle_views_path = "experiments/v2b_i3_4/oracle_views/v2b_i3_4_observable_oracle_views_v1.json"
        latent_oracle_path = "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_latent_oracles_v1.jsonl.gz"

        if not Path(oracle_views_path).exists() or not Path(latent_oracle_path).exists():
            pytest.skip("Oracle view or latent oracle files not available")

        contributions, deltas = score_results(
            runner.results, benchmark, oracle_views_path, latent_oracle_path,
            benchmark.utility_weights)
        assert len(contributions) == 10  # 5 tasks × 2 conditions
        assert len(deltas) == 5

    def test_paired_deltas_have_delta_dg(self, runner, benchmark, dev_tasks):
        for task in dev_tasks[:3]:
            runner.run_pair(task, benchmark.budget_for(task))

        oracle_views_path = "experiments/v2b_i3_4/oracle_views/v2b_i3_4_observable_oracle_views_v1.json"
        latent_oracle_path = "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_latent_oracles_v1.jsonl.gz"

        if not Path(oracle_views_path).exists() or not Path(latent_oracle_path).exists():
            pytest.skip("Oracle view or latent oracle files not available")

        contributions, deltas = score_results(
            runner.results, benchmark, oracle_views_path, latent_oracle_path,
            benchmark.utility_weights)
        for d in deltas:
            assert hasattr(d, "delta_dg")
            assert hasattr(d, "delta_ig")
            assert hasattr(d, "delta_tr")


class TestStatisticalAnalysis:
    def test_run_statistical_analysis_returns_bootstrap(self):
        from hrm_adaptive_memory.executive.i3_4_scientific_scoring import I34PairedDelta
        deltas = [
            I34PairedDelta(task_id=f"task_{i}", delta_ig=0.1, delta_dg=0.2,
                           delta_tr=0.3, delta_cost=0.0)
            for i in range(10)
        ]
        result = run_statistical_analysis(deltas)
        assert result["n_paired_tasks"] == 10
        assert "task_level_bootstrap" in result
        assert result["task_level_bootstrap"]["iterations"] > 0

    def test_run_statistical_analysis_empty_deltas(self):
        result = run_statistical_analysis([])
        assert "error" in result
