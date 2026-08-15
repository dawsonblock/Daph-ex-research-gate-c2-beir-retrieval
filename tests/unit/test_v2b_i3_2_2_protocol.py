"""V2B-I3.2.2 frozen aggregate, utility, and artifact-closure contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.i3_2_2_qualification import validate_protocol
from hrm_adaptive_memory.executive.metareasoning_artifacts import (
    artifact_graph_sha256, resolve_benchmark_artifact_graph)
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    MatchedMetareasoningController, load_observation_masks)
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_i3_2 import (
    aggregate_metrics, class_decomposition, run_condition)
from hrm_adaptive_memory.executive.metareasoning_sequential_oracle import (
    build_sequential_observable_oracle)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "experiments/v2b_i3_2/benchmark/v2b_i3_2_benchmark_manifest_v1.json"
PROTOCOL = ROOT / "configs/v2b_i3_2_2_protocol_v1.json"


def _small_evaluation():
    benchmark = load_metareasoning_benchmark(MANIFEST)
    selected_ids = {"i3_2_verify_supported", "i3_2_verify_irreducible", "i3_immediate_answer"}
    tasks = tuple(task for task in benchmark.tasks if task.task_id in selected_ids)
    small = type(benchmark)(benchmark.benchmark_id, tasks, benchmark.budget_profiles,
                            benchmark.utility_weights, benchmark.metadata, benchmark.artifact_hashes)
    policy = load_frozen_policy(ROOT / "configs/v2b_i3_policy_v1.json")
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")
    mask = load_observation_masks(ROOT / "configs/v2b_i3_observation_masks_v1.json")[
        "STATE_BLIND_CONTROLLER"]
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(small.budget_for(task)))
                for task in tasks}
    cache = OracleTableCache()
    latent = {task_id: cache.get_or_build(initial_runtime=runtime, policy=policy, utility=utility,
                                           include_policy_feedback=True)
              for task_id, runtime in runtimes.items()}
    observable = build_sequential_observable_oracle(
        runtime_tables=((runtimes[key], latent[key]) for key in sorted(runtimes)),
        mask=mask, policy=policy, utility=utility, benchmark_hash="i3-2-2-test")
    runs = run_condition(benchmark=small, condition="STATE_BLIND_CONTROLLER",
                         controller=MatchedMetareasoningController(), mask=mask, policy=policy,
                         utility=utility, latent_tables=latent, oracle_set=observable)
    decomposition = class_decomposition(runs=runs, oracle_set=observable, latent_tables=latent)
    return small, mask, observable, runs, decomposition


def test_i3_2_2_reports_both_explicit_aggregate_priors_and_preserves_decomposition():
    benchmark, mask, observable, runs, decomposition = _small_evaluation()
    metrics = aggregate_metrics(runs=runs, decomposition=decomposition, oracle_set=observable,
                                benchmark=benchmark, mask=mask)
    for prefix in ("task_uniform", "class_uniform"):
        assert metrics[f"{prefix}_total_regret"] == pytest.approx(
            metrics[f"{prefix}_information_gap"] + metrics[f"{prefix}_decision_gap"])
    assert "mean_information_gap" not in metrics
    assert "policy_probe_count" not in metrics
    assert "policy_intervention_count" in metrics


def test_i3_2_2_trace_separates_cost_reward_and_net_utility():
    _, _, _, runs, _ = _small_evaluation()
    for trace in (trace for run in runs for trace in run.traces):
        assert trace.action_cost >= 0
        assert trace.net_step_utility == pytest.approx(
            trace.immediate_reward - trace.action_cost + float(trace.terminal_reward or 0.0))


def test_i3_2_2_artifact_closure_is_canonical_and_hashable():
    manifest = json.loads(MANIFEST.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    graph = resolve_benchmark_artifact_graph(
        manifest_path=MANIFEST.relative_to(ROOT).as_posix(), manifest=manifest,
        protocol_path=PROTOCOL.relative_to(ROOT).as_posix(), protocol=protocol)
    assert set(graph) >= {"benchmark_manifest", "private_environment", "controller_packets",
                          "task_extension", "controller_packets_extension", "protocol",
                          "observation_masks", "policy", "utility", "resource_profiles"}
    assert artifact_graph_sha256(ROOT, graph) == artifact_graph_sha256(ROOT, dict(reversed(graph.items())))


def test_i3_2_2_protocol_is_frozen_and_other_statuses_fail_closed():
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    protocol["status"] = "FROZEN_FOR_DEVELOPMENT"
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_protocol(protocol)
