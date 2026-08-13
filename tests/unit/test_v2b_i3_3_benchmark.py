"""V2B-I3.3 frozen corpus, balance, leakage, and oracle-solvability gates."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from hrm_adaptive_memory.cognitive_control.i3_3_qualification import validate_configuration
from hrm_adaptive_memory.executive.metareasoning_artifacts import (
    artifact_graph_sha256, resolve_benchmark_artifact_graph)
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_executor import (
    initial_i3_runtime)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "experiments/v2b_i3_3/manifests/v2b_i3_3_benchmark_manifest_v1.json"
PRIVATE = ROOT / "experiments/v2b_i3_3/private/v2b_i3_3_tasks_v1.json"
PACKETS = ROOT / "experiments/v2b_i3_3/controller_packets/v2b_i3_3_controller_packets_v1.json"
SPLITS = ROOT / "experiments/v2b_i3_3/splits/v2b_i3_3_splits_v1.json"
CONFIG = ROOT / "experiments/v2b_i3_3/configs/v2b_i3_3_benchmark_freeze_v1.json"
CACHE = ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_oracle_cache_manifest_v1.json"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_i3_3_generator_matches_all_frozen_concrete_task_and_packet_bytes():
    path = ROOT / "experiments/v2b_i3_3/generators/generate_v2b_i3_3.py"
    spec = importlib.util.spec_from_file_location("v2b_i3_3_generator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks, packets = module.generate()
    assert tasks == json.loads(PRIVATE.read_text())["tasks"]
    assert packets == json.loads(PACKETS.read_text())["packets"]


def test_i3_3_split_hashes_are_exact_disjoint_and_have_frozen_counts():
    private = json.loads(PRIVATE.read_text())
    splits = json.loads(SPLITS.read_text())["splits"]
    by_id = {task["task_id"]: task for task in private["tasks"]}
    seen: set[str] = set()
    assert {name: len(items) for name, items in splits.items()} == {
        "development": 300, "validation": 150, "held_out": 300}
    for split, items in splits.items():
        for item in items:
            assert item["task_id"] not in seen
            seen.add(item["task_id"])
            assert by_id[item["task_id"]]["split"] == split
            assert hashlib.sha256(canonical(by_id[item["task_id"]])).hexdigest() == item["task_sha256"]
    assert seen == set(by_id)


def test_i3_3_is_balanced_and_contains_required_channel_and_budget_pairs():
    tasks = json.loads(PRIVATE.read_text())["tasks"]
    actions = Counter(task["designed_optimal_action"] for task in tasks)
    channels = Counter(task["cognitive_channel"] for task in tasks)
    assert set(actions) == {"ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                            "REASON_MORE", "DEFER", "STOP"}
    assert min(actions.values()) >= 90
    for channel in ("verification", "temporal", "provenance", "conflict", "history",
                    "composition", "irreducible", "state_irrelevant",
                    "verification_x_budget"):
        assert channels[channel] > 0
    pairs: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        if task["cognitive_channel"] == "verification_x_budget":
            pairs.setdefault(task["generator_pair_id"], []).append(task)
    assert len(pairs) >= 30
    for members in pairs.values():
        assert {item["budget_profile"] for item in members} == {"TIGHT", "GENEROUS"}
        assert len({canonical(item["latent"]) for item in members}) == 1
        assert len({canonical(item["action_effects"]) for item in members}) == 1


def test_i3_3_public_packets_are_opaque_and_have_no_latent_or_oracle_leakage():
    packets = json.loads(PACKETS.read_text())["packets"]
    forbidden = ("optimal", "oracle", "latent", "correct_action", "reasoning_required",
                 "evidence_sufficient", "ground_truth", "expected_terminal")
    private_ids = {task["task_id"] for task in json.loads(PRIVATE.read_text())["tasks"]}
    for packet in packets:
        public = {"instance_id": packet["instance_id"], "task_summary": packet["task_summary"]}
        serialized = json.dumps(public, sort_keys=True).lower()
        assert not any(token in serialized for token in forbidden)
        assert not (private_ids & set(public.values()))


def test_i3_3_manifest_closure_and_loader_use_the_same_artifact_graph():
    manifest = json.loads(MANIFEST.read_text())
    protocol_path = ROOT / "configs/v2b_i3_2_2_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    graph = resolve_benchmark_artifact_graph(
        manifest_path=MANIFEST.relative_to(ROOT).as_posix(), manifest=manifest,
        protocol_path=protocol_path.relative_to(ROOT).as_posix(), protocol=protocol,
        json_loader=lambda relative: json.loads((ROOT / relative).read_text()))
    assert set(graph) >= {"benchmark_manifest", "private_environment", "controller_packets",
                          "task_families", "split_definitions", "surface_templates",
                          "balance_report", "protocol", "policy", "utility",
                          "observation_masks", "resource_profiles", "oracle_cache_manifest",
                          "oracle_latent_tables", "oracle_difficulty_report",
                          "oracle_sequential_state_aware_controller"}
    assert len(artifact_graph_sha256(ROOT, graph)) == 64
    benchmark = load_metareasoning_benchmark(MANIFEST)
    loader_graph = resolve_benchmark_artifact_graph(
        manifest_path=MANIFEST.relative_to(ROOT).as_posix(), manifest=manifest,
        json_loader=lambda relative: json.loads((ROOT / relative).read_text()))
    assert benchmark.artifact_hashes == {
        role: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for role, relative in loader_graph.items()}


def test_i3_3_every_task_has_a_bounded_exact_latent_oracle():
    benchmark = load_metareasoning_benchmark(MANIFEST)
    policy = load_frozen_policy(ROOT / "configs/v2b_i3_policy_v1.json")
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")
    cache = OracleTableCache()
    optimal_actions: Counter[str] = Counter()
    for task in benchmark.tasks:
        runtime = initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
        table = cache.get_or_build(initial_runtime=runtime, policy=policy, utility=utility,
                                   include_policy_feedback=True)
        assert table.initial_state_id in table.states
        assert table.optimal_actions[table.initial_state_id]
        assert len(table.states) <= 20_000
        assert len(table.transitions) <= 120_000
        optimal_actions.update(action.value for action in table.optimal_actions[table.initial_state_id])
    assert set(optimal_actions) == {"ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                                    "REASON_MORE", "DEFER", "STOP"}


def test_i3_3_freeze_configuration_is_fail_closed_and_not_a_scientific_claim():
    configuration = json.loads(CONFIG.read_text())
    validate_configuration(configuration)
    assert "NOT_A_SCIENTIFIC_RESULT" in configuration["status"]
    invalid = dict(configuration); invalid["primary_prior"] = "CLASS_UNIFORM"
    try:
        validate_configuration(invalid)
    except RuntimeError:
        pass
    else:
        raise AssertionError("I3.3 configuration accepted a non-primary task prior")


def test_i3_3_precomputed_oracle_cache_is_closed_semantic_and_information_bounded():
    cache = json.loads(CACHE.read_text())
    assert cache["benchmark_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    conditions = cache["sequential_observable_oracles"]
    assert set(conditions) == {
        "STATE_BLIND_CONTROLLER", "STATE_AWARE_CONTROLLER", "NO_VERIFICATION",
        "NO_TEMPORAL", "NO_PROVENANCE", "NO_CONFLICT", "NO_HISTORY"}
    assert (conditions["STATE_AWARE_CONTROLLER"]["task_uniform_information_gap"]
            < conditions["STATE_BLIND_CONTROLLER"]["task_uniform_information_gap"])
    entries = [cache["latent_oracles"], cache["difficulty_report"], *conditions.values()]
    for entry in entries:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    with gzip.open(ROOT / cache["latent_oracles"]["path"], "rt") as stream:
        first = json.loads(next(stream))
    assert "build_metrics" not in first["table"]
    with gzip.open(ROOT / conditions["STATE_AWARE_CONTROLLER"]["path"], "rt") as stream:
        first = json.loads(next(stream))
    assert "build_metrics" not in first["table"]
    assert set(cache["latent_optimal_action_counts_with_ties"]) == {
        "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"}
