"""V2B-I3.3 frozen corpus, balance, leakage, and oracle-solvability gates."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

from hrm_adaptive_memory.common.canonical_json import canonical_bytes, loads_strict
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


def test_i3_3_generator_matches_all_frozen_concrete_task_and_packet_bytes():
    path = ROOT / "experiments/v2b_i3_3/generators/generate_v2b_i3_3.py"
    spec = importlib.util.spec_from_file_location("v2b_i3_3_generator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks, packets = module.generate()
    assert tasks == json.loads(PRIVATE.read_text())["tasks"]
    assert packets == json.loads(PACKETS.read_text())["packets"]
    module.SEED += 1
    changed_tasks, changed_packets = module.generate()
    assert (changed_tasks, changed_packets) != (tasks, packets)


def test_i3_3_split_hashes_are_exact_disjoint_and_have_frozen_counts():
    private = json.loads(PRIVATE.read_text())
    splits = json.loads(SPLITS.read_text())["splits"]
    by_id = {task["task_id"]: task for task in private["tasks"]}
    seen: set[str] = set()
    assert {name: len(items) for name, items in splits.items()} == {
        "development": 300, "validation": 150,
        "held_out_instance": 100, "held_out_surface": 50,
        "held_out_structure": 150}
    for split, items in splits.items():
        for item in items:
            assert item["task_id"] not in seen
            seen.add(item["task_id"])
            assert by_id[item["task_id"]]["split"] == split
            assert hashlib.sha256(canonical_bytes(by_id[item["task_id"]])).hexdigest() == item["task_sha256"]
    assert seen == set(by_id)


def test_i3_3_is_balanced_and_contains_required_channel_and_budget_pairs():
    tasks = json.loads(PRIVATE.read_text())["tasks"]
    actions = Counter(task["designed_optimal_action"] for task in tasks)
    channels = Counter(task["cognitive_channel"] for task in tasks)
    assert set(actions) == {"ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                            "REASON_MORE", "DEFER", "STOP"}
    assert min(actions.values()) >= 80
    assert max(actions.values()) / min(actions.values()) < 2.0
    for channel in ("verification", "temporal", "provenance", "conflict", "history",
                    "composition", "irreducible", "state_irrelevant",
                    "verification_x_budget"):
        assert channels[channel] > 0
    pairs: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        if task["cognitive_channel"] == "verification_x_budget":
            pairs.setdefault(task["generator_pair_id"], []).append(task)
    assert len(pairs) >= 94
    for members in pairs.values():
        assert {item["budget_profile"] for item in members} == {"TIGHT", "GENEROUS"}
        assert len({canonical_bytes(item["latent"]) for item in members}) == 1
        assert len({canonical_bytes(item["action_effects"]) for item in members}) == 1
    assert sum(len(members) for members in pairs.values()) / len(tasks) >= 0.25


def test_i3_3_semantic_structure_splits_have_frozen_novelty_and_surface_isolation():
    tasks = json.loads(PRIVATE.read_text())["tasks"]
    by_split = {split: [task for task in tasks if task["split"] == split]
                for split in json.loads(SPLITS.read_text())["splits"]}
    development = {task["semantic_structure_exact"] for task in by_split["development"]}
    assert len(development) >= 100
    assert len({task["semantic_structure_exact"] for task in by_split["validation"]}
               - development) >= 30
    assert not ({task["semantic_structure_exact"] for task in by_split["held_out_structure"]}
                & development)
    assert len({task["semantic_structure_exact"] for task in by_split["held_out_structure"]}) >= 50
    surfaces = json.loads((ROOT / "experiments/v2b_i3_3/surface_templates/"
                           "v2b_i3_3_surface_templates_v1.json").read_text())
    held_out_templates = set(surfaces["held_out_surface_templates"])
    development_templates = set(surfaces["development_templates"])
    assert held_out_templates.isdisjoint(development_templates)
    assert all(any(task["task_summary"].startswith(template)
                   for template in held_out_templates)
               for task in by_split["held_out_surface"])


def test_i3_3_2_behavior_topologies_are_isolated_from_all_pretest_splits():
    report = json.loads((ROOT / "experiments/v2b_i3_3/reports/"
                         "v2b_i3_3_2_topology_diversity_report_v1.json").read_text())
    overlap = report["topology_overlap_matrix"]
    assert overlap["held_out_structure"]["development"] == 0
    assert overlap["held_out_structure"]["validation"] == 0
    assert report["held_out_structure_unseen_from_development_and_validation"] >= 50
    assert report["transition_topologies"]["development"] >= 50
    allocation = json.loads((ROOT / "experiments/v2b_i3_3/splits/"
                              "topology_allocation_v1.json").read_text())
    for item in allocation["topologies"].values():
        roles = set(item["roles"])
        assert not ("held_out_structure" in roles
                    and roles & {"development", "validation"})


def test_i3_3_all_frozen_json_is_strict_rfc_8259():
    for path in sorted((ROOT / "experiments/v2b_i3_3").rglob("*.json")):
        loads_strict(path.read_bytes())


def test_i3_3_public_packets_are_opaque_and_have_no_latent_or_oracle_leakage():
    packets = json.loads(PACKETS.read_text())["packets"]
    forbidden = ("optimal", "oracle", "latent", "correct_action", "reasoning_required",
                 "evidence_sufficient", "ground_truth", "expected_terminal", "topology",
                 "semantic_structure", "difficulty", "q_margin", "information_gap",
                 "held_out_structure")
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
                          "balance_report", "structural_diversity_report",
                          "protocol", "policy", "utility",
                          "observation_masks", "resource_profiles", "oracle_cache_manifest",
                          "oracle_latent_tables", "oracle_difficulty_report",
                          "oracle_balance_report",
                          "topology_allocation", "topology_diversity_report",
                          "oracle_sequential_state_aware_controller"}
    assert len(artifact_graph_sha256(ROOT, graph)) == 64
    benchmark = load_metareasoning_benchmark(MANIFEST)
    loader_graph = resolve_benchmark_artifact_graph(
        manifest_path=MANIFEST.relative_to(ROOT).as_posix(), manifest=manifest,
        json_loader=lambda relative: json.loads((ROOT / relative).read_text()))
    assert benchmark.artifact_hashes == {
        role: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for role, relative in loader_graph.items()}


def test_i3_3_representative_tasks_have_bounded_exact_latent_oracles():
    benchmark = load_metareasoning_benchmark(MANIFEST)
    policy = load_frozen_policy(ROOT / "configs/v2b_i3_policy_v1.json")
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")
    cache = OracleTableCache()
    optimal_actions: Counter[str] = Counter()
    samples = [next(task for task in benchmark.tasks
                    if task.split == split and task.category.endswith(action.lower()))
               for split, action in zip(
                   ("development", "development", "validation", "validation",
                    "held_out_instance", "held_out_surface", "held_out_structure"),
                   ("ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                    "REASON_MORE", "STOP", "DEFER"))]
    for task in samples:
        runtime = initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
        table = cache.get_or_build(initial_runtime=runtime, policy=policy, utility=utility,
                                   include_policy_feedback=True)
        assert table.initial_state_id in table.states
        assert table.optimal_actions[table.initial_state_id]
        assert len(table.states) <= 20_000
        assert len(table.transitions) <= 120_000
        optimal_actions.update(action.value for action in table.optimal_actions[table.initial_state_id])
    assert optimal_actions


def test_i3_3_every_designed_optimal_action_is_oracle_optimal():
    cache = json.loads(CACHE.read_text())
    report = json.loads((ROOT / cache["oracle_balance_report"]["path"]).read_text())
    assert report["designed_oracle_agreement_count"] == report["task_count"] == 750
    assert report["designed_oracle_disagreements"] == []
    assert sum(report["singleton_optimal_action_counts"].values()) + sum(
        report["tied_optimal_action_sets"].values()) == 750
    assert set(report["singleton_optimal_action_counts"]) == {
        "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP"}
    assert report["multi_optimal_task_count"] / report["task_count"] <= 0.15
    assert set(report["q_margin_bands"]) == {"EASY", "MEDIUM", "HARD", "TIE"}
    for count in report["q_margin_bands"].values():
        assert 0.05 <= count / report["task_count"] <= 0.55
    assert set(report["by_split"]) == {
        "development", "validation", "held_out_instance",
        "held_out_surface", "held_out_structure"}


def test_i3_3_freeze_configuration_is_fail_closed_and_not_a_scientific_claim():
    configuration = json.loads(CONFIG.read_text())
    validate_configuration(configuration)
    assert configuration["status"] == "FROZEN_SCIENTIFIC_BENCHMARK_NO_EXECUTIVE_RESULT"
    invalid = dict(configuration); invalid["primary_prior"] = "CLASS_UNIFORM"
    try:
        validate_configuration(invalid)
    except RuntimeError:
        pass
    else:
        raise AssertionError("I3.3 configuration accepted a non-primary task prior")


def test_i3_3_precomputed_oracle_cache_is_closed_semantic_and_information_bounded():
    cache = json.loads(CACHE.read_text())
    assert cache["benchmark_manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST.read_text()); manifest.pop("oracle_cache_manifest_path")
    protocol_path = ROOT / "configs/v2b_i3_2_2_protocol_v1.json"
    graph = resolve_benchmark_artifact_graph(
        manifest_path=MANIFEST.relative_to(ROOT).as_posix(), manifest=manifest,
        protocol_path=protocol_path.relative_to(ROOT).as_posix(),
        protocol=json.loads(protocol_path.read_text()))
    assert cache["benchmark_closure_sha256"] == artifact_graph_sha256(ROOT, graph)
    conditions = cache["sequential_observable_oracles"]
    assert set(conditions) == {
        "STATE_BLIND_CONTROLLER", "STATE_AWARE_CONTROLLER", "NO_VERIFICATION",
        "NO_TEMPORAL", "NO_PROVENANCE", "NO_CONFLICT", "NO_HISTORY"}
    assert (conditions["STATE_AWARE_CONTROLLER"]["task_uniform_information_gap"]
            < conditions["STATE_BLIND_CONTROLLER"]["task_uniform_information_gap"])
    aware_gap = conditions["STATE_AWARE_CONTROLLER"]["task_uniform_information_gap"]
    assert aware_gap > 0  # Frozen irreducible partial-observability controls.
    for ablation in ("NO_VERIFICATION", "NO_TEMPORAL", "NO_PROVENANCE",
                     "NO_CONFLICT", "NO_HISTORY"):
        assert conditions[ablation]["task_uniform_information_gap"] > aware_gap
    entries = [cache["latent_oracles"], cache["difficulty_report"],
               cache["oracle_balance_report"], *conditions.values()]
    entries.extend([cache["topology_allocation"], cache["topology_diversity_report"]])
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
