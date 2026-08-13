"""V2B-I3.1 methodology tests: table parity, information bounds, and replay."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    STATE_AWARE_MASK, STATE_BLIND_MASK, MatchedMetareasoningController)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    build_observable_snapshot, initial_i3_runtime)
from hrm_adaptive_memory.executive.metareasoning_i3_1 import replay_trajectory, trajectory_payload
from hrm_adaptive_memory.executive.metareasoning_loop import STATE_AWARE, V2BMetareasoningExperiment
from hrm_adaptive_memory.executive.metareasoning_observable_oracle import build_observable_oracle
from hrm_adaptive_memory.executive.metareasoning_state import canonicalize_runtime_state
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    OracleTableCache, build_oracle_policy_table_for_runtime)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "experiments/v2b_i3_1/benchmark/v2b_i3_1_benchmark_manifest_v1.json"
POLICY = ROOT / "configs/v2b_i3_policy_v1.json"
UTILITY = ROOT / "configs/v2b_i3_1_utility_v1.json"


def _inputs():
    benchmark = load_metareasoning_benchmark(MANIFEST)
    policy = load_frozen_policy(POLICY)
    utility = MetareasoningUtility.from_file(UTILITY)
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
                for task in benchmark.tasks}
    return benchmark, policy, utility, runtimes


def test_i3_1_canonical_state_excludes_derivable_instrumentation():
    _, _, _, runtimes = _inputs()
    runtime = runtimes["i3_retrieval_required"]
    altered = replace(runtime, resources=replace(runtime.resources, elapsed_ms=777))
    assert canonicalize_runtime_state(runtime) == canonicalize_runtime_state(altered)


def test_i3_1_table_is_finite_and_every_nonterminal_edge_reduces_steps():
    _, policy, utility, runtimes = _inputs()
    table = build_oracle_policy_table_for_runtime(
        initial_runtime=runtimes["i3_retrieval_required"], policy=policy, utility=utility)
    assert table.initial_value > 0
    rebuilt = build_oracle_policy_table_for_runtime(
        initial_runtime=runtimes["i3_retrieval_required"], policy=policy, utility=utility)
    assert table.table_sha256 == rebuilt.table_sha256
    for (origin, _), transition in table.transitions.items():
        if not transition.terminal:
            assert transition.next_state_id is not None
            assert table.states[transition.next_state_id].steps_remaining < table.states[origin].steps_remaining


def test_i3_1_action_regret_is_a_table_lookup_with_runtime_parity():
    _, policy, utility, runtimes = _inputs()
    table = build_oracle_policy_table_for_runtime(
        initial_runtime=runtimes["i3_retrieval_required"], policy=policy, utility=utility)
    state_id = table.initial_state_id
    action = next(iter(table.optimal_actions[state_id]))
    assert table.action_regret(state_id, action) == 0.0
    for (origin, candidate), q_value in table.q_values.items():
        if origin == state_id:
            assert table.action_regret(origin, candidate) == pytest.approx(
                max(0.0, table.state_values[origin] - q_value))


def test_i3_1_table_cache_reuses_latent_table_across_conditions():
    _, policy, utility, runtimes = _inputs()
    cache = OracleTableCache()
    first = cache.get_or_build(initial_runtime=runtimes["i3_retrieval_required"], policy=policy, utility=utility)
    second = cache.get_or_build(initial_runtime=runtimes["i3_retrieval_required"], policy=policy, utility=utility)
    assert first is second
    assert cache.hits == 1 and cache.misses == 1


def test_i3_1_observable_oracle_detects_hidden_transition_ambiguity():
    _, policy, utility, runtimes = _inputs()
    cache = OracleTableCache()
    selected = ("i3_hidden_transition_retrieve", "i3_hidden_transition_verify")
    pairs = [(runtimes[task_id], cache.get_or_build(
        initial_runtime=runtimes[task_id], policy=policy, utility=utility)) for task_id in selected]
    blind = build_observable_oracle(runtime_tables=pairs, mask=STATE_BLIND_MASK)
    aware = build_observable_oracle(runtime_tables=pairs, mask=STATE_AWARE_MASK)
    assert len(blind.classes) == 1
    assert len(aware.classes) == 1
    assert blind.ambiguity_count == 1
    assert aware.ambiguity_count == 1


def test_i3_1_controller_packet_has_no_private_task_id_or_forbidden_labels():
    benchmark, _, _, runtimes = _inputs()
    task = next(item for item in benchmark.tasks if item.task_id == "i3_hidden_transition_retrieve")
    snapshot = build_observable_snapshot(runtimes[task.task_id], prior_decisions=(), prior_outcomes=())
    serialized = json.dumps({"task_id": snapshot.task_id, "task_summary": snapshot.task_summary})
    assert task.task_id not in serialized
    for forbidden in ("optimal_action", "oracle_value", "reasoning_required", "evidence_sufficient"):
        assert forbidden not in serialized.lower()


def test_i3_1_state_space_guard_fails_closed():
    _, policy, utility, runtimes = _inputs()
    with pytest.raises(RuntimeError, match="ORACLE_STATE_SPACE_LIMIT"):
        build_oracle_policy_table_for_runtime(
            initial_runtime=runtimes["i3_retrieval_required"], policy=policy, utility=utility,
            max_states=1)


def test_i3_1_runtime_table_and_receipt_replay_are_exact(tmp_path):
    benchmark, policy, utility, runtimes = _inputs()
    cache = OracleTableCache()
    protocol = V2BMetareasoningExperiment(
        benchmark=benchmark, policy=policy, utility=utility, oracle_table_cache=cache)
    run = protocol.run_condition(condition=STATE_AWARE, controller=MatchedMetareasoningController(),
                                 store_root=tmp_path, mask=STATE_AWARE_MASK)
    task_run = next(item for item in run.tasks if item.task_id == "i3_retrieval_required")
    table = cache.get_or_build(initial_runtime=runtimes[task_run.task_id], policy=policy, utility=utility)
    observable = build_observable_oracle(runtime_tables=[(runtimes[task_run.task_id], table)],
                                         mask=STATE_AWARE_MASK)
    receipt = trajectory_payload(
        run=task_run, table=table, observable=observable, condition=STATE_AWARE,
        observation_mask_sha256=STATE_AWARE_MASK.sha256(), controller_revision=run.controller_algorithm_id,
        policy_sha256=policy.sha256, utility_sha256=utility.sha256, budget_sha256="test-budget")
    replay = replay_trajectory(benchmark=benchmark, task_id=task_run.task_id,
                               traces=receipt["steps"], utility=utility)
    assert replay["state_hash"] == task_run.traces[-1].post_state_hash
    assert replay["trajectory_utility"] == pytest.approx(task_run.realized_utility)
