"""V2B-I3 experimental-validity contracts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.actions import ActionProposal
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    ControllerObservation, MatchedMetareasoningController, load_observation_masks)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, build_observable_snapshot, initial_i3_runtime)
from hrm_adaptive_memory.executive.metareasoning_loop import (
    STATE_AWARE, STATE_BLIND, V2BMetareasoningExperiment)
from hrm_adaptive_memory.executive.metareasoning_oracle import ExactOptimalPolicyOracle
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


ROOT = Path(__file__).parents[2]
BENCHMARK_PATH = ROOT / "experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json"
PRIVATE_BENCHMARK_PATH = ROOT / "experiments/v2b/tasks/v2b_i3_metareasoning_benchmark_v1.json"
PACKETS_PATH = ROOT / "experiments/v2b/benchmark/controller_packets/v2b_i3_controller_packets_v1.json"
MASKS_PATH = ROOT / "configs/v2b_i3_observation_masks_v1.json"
POLICY_PATH = ROOT / "configs/v2b_i3_policy_v1.json"


def _benchmark():
    return load_metareasoning_benchmark(BENCHMARK_PATH)


def _experiment():
    return V2BMetareasoningExperiment(benchmark=_benchmark(), policy=load_frozen_policy(POLICY_PATH))


def test_i3_declares_latent_observable_separation_and_budget_conditioning():
    raw = json.loads(PRIVATE_BENCHMARK_PATH.read_text())
    manifest = json.loads(BENCHMARK_PATH.read_text())
    packets = json.loads(PACKETS_PATH.read_text())
    assert manifest["private_environment_path"]
    assert manifest["controller_packets_path"]
    assert len(packets["packets"]) == len(raw["tasks"])
    assert "latent_state" in raw["protocol"]
    assert "reasoning_required" in raw["protocol"]["forbidden_controller_inputs"]
    assert {task["budget_profile"] for task in raw["tasks"]} == {"TIGHT", "STANDARD", "GENEROUS"}
    ambiguous = [task for task in raw["tasks"] if task["task_id"].startswith("i3_ambiguous_")]
    assert len(ambiguous) == 2 and ambiguous[0]["task_summary"] == ambiguous[1]["task_summary"]
    assert all("expected_terminal" not in task and "reasoning_required" not in task
               for task in raw["tasks"])


def test_i3_has_a_partially_observable_transition_pair():
    benchmark = _benchmark()
    retrieve = next(task for task in benchmark.tasks if task.task_id == "i3_hidden_transition_retrieve")
    verify = next(task for task in benchmark.tasks if task.task_id == "i3_hidden_transition_verify")
    assert retrieve.task_summary == verify.task_summary
    assert retrieve.observable_provenance_count == verify.observable_provenance_count
    assert retrieve.latent.verification_state is verify.latent.verification_state
    assert set(retrieve.action_effects) == {DecisionAction.RETRIEVE}
    assert set(verify.action_effects) == {DecisionAction.VERIFY}
    retrieve_state = initial_i3_runtime(retrieve, ResourceState(benchmark.budget_for(retrieve)))
    verify_state = initial_i3_runtime(verify, ResourceState(benchmark.budget_for(verify)))
    retrieve_observation = build_observable_snapshot(retrieve_state, prior_decisions=(), prior_outcomes=())
    verify_observation = build_observable_snapshot(verify_state, prior_decisions=(), prior_outcomes=())
    assert retrieve_observation.task_summary == verify_observation.task_summary
    assert retrieve_observation.relevant_memories[0].verification_state == verify_observation.relevant_memories[0].verification_state
    assert retrieve_observation.resource_state == verify_observation.resource_state


def test_i3_budget_profile_changes_the_optimal_action_for_the_same_task_specification():
    benchmark = _benchmark(); policy = load_frozen_policy(POLICY_PATH)
    tight = next(task for task in benchmark.tasks if task.task_id == "i3_tight_budget")
    generous = next(task for task in benchmark.tasks if task.task_id == "i3_generous_search")
    assert tight.task_summary == generous.task_summary
    assert tight.action_effects == generous.action_effects
    tight_oracle = ExactOptimalPolicyOracle(
        task=tight, policy=policy, utility_weights=benchmark.utility_weights)
    generous_oracle = ExactOptimalPolicyOracle(
        task=generous, policy=policy, utility_weights=benchmark.utility_weights)
    assert tight_oracle.solve(initial_i3_runtime(tight, ResourceState(benchmark.budget_for(tight)))).action is DecisionAction.DEFER
    assert generous_oracle.solve(initial_i3_runtime(
        generous, ResourceState(benchmark.budget_for(generous)))).action is DecisionAction.SEARCH_MORE


def test_i3_controller_packets_fail_closed_on_latent_or_oracle_leakage(tmp_path):
    manifest = json.loads(BENCHMARK_PATH.read_text())
    packets = json.loads(PACKETS_PATH.read_text())
    packets["packets"][0]["optimal_action"] = "ANSWER"
    packet_path = tmp_path / "packets.json"; packet_path.write_text(json.dumps(packets))
    private_path = tmp_path / "private.json"; private_path.write_bytes(PRIVATE_BENCHMARK_PATH.read_bytes())
    manifest["private_environment_path"] = "private.json"
    manifest["controller_packets_path"] = "packets.json"
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="forbidden"):
        load_metareasoning_benchmark(manifest_path)
    packets = json.loads(PACKETS_PATH.read_text())
    packets["packets"][0]["task_summary"] = "oracle_value is 100"
    packet_path.write_text(json.dumps(packets))
    with pytest.raises(ValueError, match="forbidden"):
        load_metareasoning_benchmark(manifest_path)


def test_i3_private_benchmark_has_disjoint_development_validation_and_held_out_splits():
    benchmark = _benchmark()
    splits = {split: {task.task_id for task in benchmark.for_split(split).tasks}
              for split in ("development", "validation", "held_out")}
    assert all(splits.values())
    assert not (splits["development"] & splits["validation"])
    assert not (splits["development"] & splits["held_out"])
    assert not (splits["validation"] & splits["held_out"])
    assert set().union(*splits.values()) == {task.task_id for task in benchmark.tasks}


def test_i3_full_and_ablation_masks_are_explicit_and_hashable():
    masks = load_observation_masks(MASKS_PATH)
    assert masks["STATE_BLIND_CONTROLLER"].sha256() != masks["STATE_AWARE_CONTROLLER"].sha256()
    assert {"NO_VERIFICATION", "NO_PROVENANCE", "NO_TEMPORAL", "NO_CONFLICT", "NO_HISTORY"} <= set(masks)


def test_i3_matched_controller_changes_only_cognitive_state_visibility(tmp_path):
    experiment = _experiment()
    blind_controller = MatchedMetareasoningController()
    aware_controller = MatchedMetareasoningController()
    assert type(blind_controller) is type(aware_controller)
    assert blind_controller.__dict__ == aware_controller.__dict__ == {}
    blind = experiment.run_condition(condition=STATE_BLIND, controller=blind_controller, store_root=tmp_path)
    aware = experiment.run_condition(condition=STATE_AWARE, controller=aware_controller, store_root=tmp_path)
    assert blind.controller_algorithm_id == aware.controller_algorithm_id
    assert aware.metrics["normalized_executive_regret"] < blind.metrics["normalized_executive_regret"]
    assert aware.metrics["task_successes"] > blind.metrics["task_successes"]
    assert aware.metrics["failure_to_stop_rate"] == 0.0
    assert aware.metrics["premature_stop_rate"] == 0.0


def test_i3_state_blind_observation_is_masked_but_policy_is_shared(tmp_path):
    experiment = _experiment()
    blind = experiment.run_condition(
        condition=STATE_BLIND, controller=MatchedMetareasoningController(),
        store_root=tmp_path)
    high_stakes = next(task for task in blind.tasks if task.task_id == "i3_verification_required")
    first = high_stakes.traces[0]
    assert first.proposed_action is DecisionAction.RETRIEVE
    assert first.policy_resolved_action is DecisionAction.VERIFY
    assert first.executed_action is DecisionAction.VERIFY


def test_i3_action_deltas_measure_usefulness_without_task_success_proxy(tmp_path):
    aware = _experiment().run_condition(
        condition=STATE_AWARE, controller=MatchedMetareasoningController(),
        store_root=tmp_path)
    search = next(task for task in aware.tasks if task.task_id == "i3_search_after_retrieval")
    retrieval = search.traces[0]
    expansion = next(trace for trace in search.traces
                     if trace.executed_action is DecisionAction.SEARCH_MORE)
    assert retrieval.state_delta is not None
    assert retrieval.state_delta["decision_relevant_improvement"] is False
    assert expansion.state_delta is not None
    assert expansion.state_delta["decision_relevant_improvement"] is True
    assert retrieval.pre_state_hash != retrieval.post_state_hash
    assert set(expansion.state_delta) >= {
        "evidence_delta", "verification_delta", "temporal_delta", "conflict_delta",
        "reasoning_delta", "answerability_delta"}
    assert expansion.observation_hash and expansion.resources_before and expansion.resources_after


@dataclass
class _DenyThenReplanController:
    controller_id: str = "test_deny_then_replan"
    algorithm_id: str = "test_deny_then_replan"

    def choose(self, observation: ControllerObservation) -> ActionProposal:
        if not observation.rejected_actions:
            return ActionProposal(DecisionAction.SEARCH_MORE, "TEST_DENIED_SEARCH")
        return ActionProposal(DecisionAction.ANSWER, "TEST_REPLAN_AFTER_DENY")


def test_i3_policy_deny_records_rejection_then_allows_replanning(tmp_path):
    benchmark = _benchmark()
    immediate = next(task for task in benchmark.tasks if task.task_id == "i3_immediate_answer")
    single = type(benchmark)(benchmark.benchmark_id, (immediate,), benchmark.budget_profiles,
                             benchmark.utility_weights, benchmark.metadata, benchmark.artifact_hashes)
    run = V2BMetareasoningExperiment(benchmark=single, policy=load_frozen_policy(POLICY_PATH)).run_condition(
        condition=STATE_AWARE, controller=_DenyThenReplanController(), store_root=tmp_path)
    traces = run.tasks[0].traces
    assert traces[0].execution_status == "POLICY_REJECTED"
    assert traces[0].executed_action is None
    assert traces[1].executed_action is DecisionAction.ANSWER
    assert run.tasks[0].task_success


@dataclass
class _AlwaysDeniedController:
    controller_id: str = "test_always_denied"
    algorithm_id: str = "test_always_denied"

    def choose(self, observation: ControllerObservation) -> ActionProposal:
        return ActionProposal(DecisionAction.SEARCH_MORE, "TEST_ALWAYS_DENIED")


def test_i3_policy_rejection_limit_terminates_explicitly(tmp_path):
    benchmark = _benchmark()
    immediate = next(task for task in benchmark.tasks if task.task_id == "i3_immediate_answer")
    single = type(benchmark)(benchmark.benchmark_id, (immediate,), benchmark.budget_profiles,
                             benchmark.utility_weights, benchmark.metadata, benchmark.artifact_hashes)
    run = V2BMetareasoningExperiment(benchmark=single, policy=load_frozen_policy(POLICY_PATH)).run_condition(
        condition=STATE_AWARE, controller=_AlwaysDeniedController(), store_root=tmp_path)
    task = run.tasks[0]
    assert task.terminal_result == "POLICY_REJECTION_LIMIT"
    assert len(task.rejected_actions) == 3
    assert all(trace.executed_action is None for trace in task.traces)


def test_i3_exact_oracle_uses_latent_environment_only_for_evaluation():
    benchmark = _benchmark()
    policy = load_frozen_policy(POLICY_PATH)
    composition = next(task for task in benchmark.tasks if task.task_id == "i3_composition")
    runtime = initial_i3_runtime(composition, ResourceState(benchmark.budget_for(composition)))
    oracle = ExactOptimalPolicyOracle(task=composition, policy=policy,
                                      utility_weights=benchmark.utility_weights)
    decision = oracle.solve(runtime)
    assert decision.action is DecisionAction.REASON_MORE
    assert decision.utility > 0
    assert decision.state_hash
    assert decision.q_values[DecisionAction.REASON_MORE.value] == decision.utility
    assert decision.minimum_remaining_cost > 0


def test_i3_stop_is_not_an_answer_or_deferral_alias(tmp_path):
    run = _experiment().run_condition(
        condition=STATE_AWARE, controller=MatchedMetareasoningController(), store_root=tmp_path)
    task = next(task for task in run.tasks if task.task_id == "i3_stop_without_assertion")
    assert task.traces[0].executed_action is DecisionAction.STOP
    assert task.terminal_result == "TASK_SUCCESS"
    assert task.task_success


def test_i3_oracle_is_legal_deterministic_and_finite_over_reachable_states():
    benchmark = _benchmark(); policy = load_frozen_policy(POLICY_PATH)
    executor = DeterministicMetareasoningExecutor()
    for task in benchmark.tasks:
        initial = initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
        oracle = ExactOptimalPolicyOracle(task=task, policy=policy,
                                          utility_weights=benchmark.utility_weights)
        decision = oracle.solve(initial)
        assert decision.action in oracle.legal_actions(initial)
        for state in oracle.reachable_states(initial):
            legal = oracle.legal_actions(state)
            assert legal
            for action in legal:
                first = executor.execute(state, action)
                second = executor.execute(state, action)
                assert first == second
                assert first.runtime.resources.executive_steps_used >= state.resources.executive_steps_used
                assert first.runtime.resources.executive_steps_used > state.resources.executive_steps_used
        snapshot = build_observable_snapshot(initial, prior_decisions=(), prior_outcomes=())
        serialized = json.dumps({"signals": snapshot.observation_signals,
                                 "policy_facts": [fact.predicate for fact in snapshot.policy_facts]})
        assert "reasoning_required" not in serialized and "evidence_sufficient" not in serialized


def test_i3_loader_rejects_nondevelopment_or_missing_latent_state(tmp_path):
    raw = json.loads(PRIVATE_BENCHMARK_PATH.read_text())
    raw["status"] = "NOT_FROZEN"
    invalid = tmp_path / "invalid.json"; invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="frozen for development"):
        load_metareasoning_benchmark(invalid)
    raw = json.loads(PRIVATE_BENCHMARK_PATH.read_text())
    del raw["tasks"][0]["latent"]
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="latent"):
        load_metareasoning_benchmark(invalid)
