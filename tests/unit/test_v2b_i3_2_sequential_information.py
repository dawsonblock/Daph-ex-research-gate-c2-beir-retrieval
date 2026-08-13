"""V2B-I3.2 sequential-information oracle and policy-feedback contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.i3_2_qualification import validate_i3_2_configuration
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    MatchedMetareasoningController, load_observation_masks)
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_i3_2 import (
    class_decomposition, replay_trajectory, run_condition, trajectory_payload)
from hrm_adaptive_memory.executive.metareasoning_observable_oracle import (
    OpeningObservableOracle, build_opening_observable_oracle)
from hrm_adaptive_memory.executive.metareasoning_sequential_oracle import (
    _apply_proposal, build_sequential_observable_oracle, canonical_packet,
    controller_observation)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "experiments/v2b_i3_2/benchmark/v2b_i3_2_benchmark_manifest_v1.json"
POLICY = ROOT / "configs/v2b_i3_policy_v1.json"
UTILITY = ROOT / "configs/v2b_i3_1_utility_v1.json"
MASKS = ROOT / "configs/v2b_i3_observation_masks_v1.json"
CONFIGURATION = ROOT / "experiments/v2b_i3_2/configs/v2b_i3_2_development.json"


def _inputs(task_ids: tuple[str, ...]):
    benchmark = load_metareasoning_benchmark(MANIFEST)
    policy = load_frozen_policy(POLICY)
    utility = MetareasoningUtility.from_file(UTILITY)
    masks = load_observation_masks(MASKS)
    selected = tuple(task for task in benchmark.tasks if task.task_id in task_ids)
    small = type(benchmark)(benchmark.benchmark_id, selected, benchmark.budget_profiles,
                            benchmark.utility_weights, benchmark.metadata, benchmark.artifact_hashes)
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(small.budget_for(task)))
                for task in selected}
    cache = OracleTableCache()
    tables = {task_id: cache.get_or_build(initial_runtime=runtime, policy=policy, utility=utility,
                                          include_policy_feedback=True)
              for task_id, runtime in runtimes.items()}
    return small, policy, utility, masks, runtimes, tables


def _sequential(task_ids: tuple[str, ...], condition: str):
    benchmark, policy, utility, masks, runtimes, tables = _inputs(task_ids)
    oracle = build_sequential_observable_oracle(
        runtime_tables=((runtimes[key], tables[key]) for key in sorted(runtimes)),
        mask=masks[condition], policy=policy, utility=utility, benchmark_hash="test-i3-2",
        max_information_states=4_000, max_information_transitions=24_000)
    return benchmark, policy, utility, masks[condition], runtimes, tables, oracle


def _root_with_members(oracle, member_count: int):
    return next(table for table in oracle.tables.values()
                if len(table.information_states[table.initial_information_state_id].members) == member_count)


def test_i3_2_preserves_the_opening_observable_diagnostic_under_its_real_name():
    _, _, _, masks, runtimes, tables = _inputs(("i3_2_verify_supported", "i3_2_verify_irreducible"))
    opening = build_opening_observable_oracle(
        runtime_tables=[(runtimes[key], tables[key]) for key in sorted(runtimes)],
        mask=masks["STATE_BLIND_CONTROLLER"])
    assert isinstance(opening, OpeningObservableOracle)
    assert opening.serializable()["schema"] == "DAPH_V2B_OPENING_OBSERVABLE_ORACLE_TABLE_V1"
    assert opening.ambiguity_count >= 1


def test_i3_2_initial_packets_are_opaque_and_aliased_only_when_the_mask_allows_it():
    benchmark, _, _, masks, runtimes, _ = _inputs(("i3_2_aware_sufficient", "i3_2_aware_missing"))
    blind = [canonical_packet(controller_observation(runtime=runtimes[task.task_id], history=(),
                                                     mask=masks["STATE_BLIND_CONTROLLER"]))
             for task in benchmark.tasks]
    aware = [canonical_packet(controller_observation(runtime=runtimes[task.task_id], history=(),
                                                     mask=masks["STATE_AWARE_CONTROLLER"]))
             for task in benchmark.tasks]
    assert blind[0] == blind[1]
    assert aware[0] != aware[1]
    serialized = json.dumps(blind[0], sort_keys=True).lower()
    for forbidden in ("i3_2_aware", "latent", "optimal_action", "oracle_value",
                      "reasoning_required", "evidence_sufficient", "expected_terminal"):
        assert forbidden not in serialized


def test_i3_2_sequential_verify_action_splits_posterior_and_closes_uncertainty():
    _, _, _, _, _, _, oracle = _sequential(
        ("i3_2_verify_supported", "i3_2_verify_irreducible"), "STATE_AWARE_CONTROLLER")
    table = _root_with_members(oracle, 2)
    initial = table.initial_information_state_id
    transition = table.transitions[(initial, DecisionAction.VERIFY)]
    assert len(transition.outcomes) == 2
    assert transition.expected_information_gain_bits > 0
    assert all(outcome.next_information_state_id is not None for outcome in transition.outcomes)
    assert all(len(table.information_states[outcome.next_information_state_id].members) == 1
               for outcome in transition.outcomes)
    assert table.expected_latent_values[initial] >= table.belief_values[initial]
    assert table.information_gap(initial) > 0


def test_i3_2_policy_rejection_is_visible_costed_and_can_split_a_belief():
    _, _, _, _, _, _, oracle = _sequential(
        ("i3_2_policy_probe_sufficient", "i3_2_policy_probe_missing"), "STATE_BLIND_CONTROLLER")
    table = _root_with_members(oracle, 2)
    initial = table.initial_information_state_id
    transition = table.transitions[(initial, DecisionAction.SEARCH_MORE)]
    statuses = {table.member_transitions[(initial, DecisionAction.SEARCH_MORE, key)].history_event.execution_status
                for outcome in transition.outcomes for key in outcome.member_keys}
    assert statuses == {"POLICY_REJECTED", "EXECUTED"}
    assert transition.expected_information_gain_bits > 0
    rejected = next(member for member in table.information_states[initial].members
                    if table.member_transitions[(initial, DecisionAction.SEARCH_MORE, member.key)]
                    .history_event.execution_status == "POLICY_REJECTED")
    rejection = table.member_transitions[(initial, DecisionAction.SEARCH_MORE, rejected.key)]
    assert rejection.immediate_utility < 0
    assert rejection.feedback.reason_class == "POLICY_DENIED"


def test_i3_2_irreducible_alias_has_residual_information_gap_even_when_state_is_aware():
    _, _, _, _, _, _, oracle = _sequential(
        ("i3_2_permanent_alias_answer", "i3_2_permanent_alias_defer"), "STATE_AWARE_CONTROLLER")
    table = _root_with_members(oracle, 2)
    root = table.initial_information_state_id
    assert table.information_gap(root) > 0
    assert table.expected_latent_values[root] > table.belief_values[root]


def test_i3_2_runtime_latent_and_observable_transitions_share_resource_semantics():
    _, policy, utility, _, runtimes, tables, oracle = _sequential(
        ("i3_2_verify_supported",), "STATE_AWARE_CONTROLLER")
    runtime = runtimes["i3_2_verify_supported"]
    latent = tables[runtime.task.task_id]
    table = next(iter(oracle.tables.values()))
    root = table.initial_information_state_id
    member = table.information_states[root].members[0]
    actions = (DecisionAction.ANSWER, DecisionAction.RETRIEVE, DecisionAction.VERIFY,
               DecisionAction.SEARCH_MORE, DecisionAction.REASON_MORE, DecisionAction.DEFER,
               DecisionAction.STOP)
    for action in actions:
        outcome = _apply_proposal(runtime=runtime, proposed=action, policy=policy, utility=utility)
        transition = table.member_transitions[(root, action, member.key)]
        assert outcome.history_event == transition.history_event
        assert outcome.immediate_utility == pytest.approx(transition.immediate_utility)
        assert latent.proposal_transitions[(latent.initial_state_id, action)].action_cost == pytest.approx(
            outcome.immediate_utility)


def test_i3_2_decomposition_and_trajectory_replay_are_exact():
    ids = ("i3_2_policy_probe_sufficient", "i3_2_policy_probe_missing")
    benchmark, policy, utility, mask, _, tables, oracle = _sequential(ids, "STATE_BLIND_CONTROLLER")
    runs = run_condition(benchmark=benchmark, condition="STATE_BLIND_CONTROLLER",
                         controller=MatchedMetareasoningController(), mask=mask, policy=policy,
                         utility=utility, latent_tables=tables, oracle_set=oracle)
    decomposition = class_decomposition(runs=runs, oracle_set=oracle, latent_tables=tables)
    assert all(float(row["information_gap"]) >= 0 for row in decomposition.values())
    assert all(float(row["decision_gap"]) >= -1e-9 for row in decomposition.values())
    root_table = _root_with_members(oracle, 2)
    for run in runs:
        receipt = trajectory_payload(run=run, table=root_table, condition=run.condition,
                                     policy_sha256=policy.sha256, utility_sha256=utility.sha256,
                                     controller_revision=MatchedMetareasoningController.algorithm_id)
        replay = replay_trajectory(benchmark=benchmark, task_id=run.task_id,
                                   traces=receipt["steps"], policy=policy, utility=utility)
        assert replay["trajectory_utility"] == pytest.approx(run.realized_utility)


def test_i3_2_fails_closed_on_belief_space_limit_and_development_identity():
    _, policy, utility, masks, runtimes, tables = _inputs(("i3_2_verify_supported",))
    with pytest.raises(RuntimeError, match="INFORMATION_STATE_SPACE_LIMIT"):
        build_sequential_observable_oracle(
            runtime_tables=[(runtimes["i3_2_verify_supported"], tables["i3_2_verify_supported"])],
            mask=masks["STATE_AWARE_CONTROLLER"], policy=policy, utility=utility,
            benchmark_hash="test-i3-2", max_information_states=1, max_information_transitions=10)
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_i3_2_configuration(json.loads(CONFIGURATION.read_text()))
