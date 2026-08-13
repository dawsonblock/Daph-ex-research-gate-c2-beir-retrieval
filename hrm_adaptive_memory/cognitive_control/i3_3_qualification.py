"""Fail-closed identity for the frozen V2B-I3.3 benchmark corpus.

This identity qualifies benchmark provenance and replay inputs only. It cannot
issue a model-controller result or a scientific V2B verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from hrm_adaptive_memory.executive.metareasoning_artifacts import (
    resolve_benchmark_artifact_graph)

from .qualification import _combined_hash, _git, _git_bytes, _tree_hash, dependency_environment


IDENTITY_VERSION = "DAPH_V2B_I3_3_1_BENCHMARK_INTEGRITY_IDENTITY_V1"
CONFIGURATION_PATH = "experiments/v2b_i3_3/configs/v2b_i3_3_benchmark_freeze_v1.json"
COMPONENTS = {
    "i3_2_2_protocol_baseline": "configs/v2b_i3_3_baseline.json",
    "i3_3_frozen_benchmark_baseline": "configs/v2b_i3_3_1_baseline.json",
    "protocol_identity_runtime": "hrm_adaptive_memory/cognitive_control/i3_2_2_qualification.py",
    "benchmark_identity_runtime": "hrm_adaptive_memory/cognitive_control/i3_3_qualification.py",
    "artifact_resolver": "hrm_adaptive_memory/executive/metareasoning_artifacts.py",
    "strict_canonical_json": "hrm_adaptive_memory/common/canonical_json.py",
    "benchmark_loader": "hrm_adaptive_memory/executive/metareasoning_benchmark.py",
    "latent_state": "hrm_adaptive_memory/executive/metareasoning_state.py",
    "latent_executor": "hrm_adaptive_memory/executive/metareasoning_executor.py",
    "latent_oracle": "hrm_adaptive_memory/executive/metareasoning_transition_table.py",
    "sequential_observable_oracle": "hrm_adaptive_memory/executive/metareasoning_sequential_oracle.py",
    "utility": "hrm_adaptive_memory/executive/metareasoning_utility.py",
    "metrics_and_replay": "hrm_adaptive_memory/executive/metareasoning_i3_2.py",
    "generator": "experiments/v2b_i3_3/generators/generate_v2b_i3_3.py",
    "oracle_cache_builder": "scripts/precompute_v2b_i3_3_oracles.py",
    "configuration": CONFIGURATION_PATH,
}
TEST_CORPUS = (
    "tests/unit/test_v2b_i3_metareasoning.py",
    "tests/unit/test_v2b_i3_1_oracle_efficiency.py",
    "tests/unit/test_v2b_i3_2_sequential_information.py",
    "tests/unit/test_v2b_i3_2_2_protocol.py",
    "tests/unit/test_v2b_i3_3_benchmark.py",
    "tests/qualification/test_v2b_i3_3_full_oracle_regeneration.py",
    "tests/adversarial/test_v2b_infrastructure_adversarial.py",
)


def validate_configuration(configuration: Mapping[str, object]) -> None:
    if configuration.get("schema") != "DAPH_V2B_I3_3_BENCHMARK_FREEZE_CONFIGURATION_V1":
        raise RuntimeError("I3.3 benchmark configuration schema is unsupported")
    if configuration.get("status") != "FROZEN_BENCHMARK_NOT_A_SCIENTIFIC_RESULT":
        raise RuntimeError("I3.3 benchmark is not frozen")
    if configuration.get("primary_prior") != "TASK_UNIFORM":
        raise RuntimeError("I3.3 primary aggregate prior must be task-uniform")
    if configuration.get("split_counts") != {
            "development": 300, "validation": 150,
            "held_out_instance": 100, "held_out_surface": 50,
            "held_out_structure": 150}:
        raise RuntimeError("I3.3 split counts are not frozen")
    novelty = configuration.get("structural_novelty_thresholds")
    if (not isinstance(novelty, Mapping)
            or set(novelty) != {
                "validation_exact_unseen_minimum",
                "held_out_structure_exact_unseen_minimum",
                "held_out_structure_task_fraction_minimum"}
            or isinstance(novelty["validation_exact_unseen_minimum"], bool)
            or not isinstance(novelty["validation_exact_unseen_minimum"], int)
            or novelty["validation_exact_unseen_minimum"] < 30
            or isinstance(novelty["held_out_structure_exact_unseen_minimum"], bool)
            or not isinstance(novelty["held_out_structure_exact_unseen_minimum"], int)
            or novelty["held_out_structure_exact_unseen_minimum"] < 50
            or isinstance(novelty["held_out_structure_task_fraction_minimum"], bool)
            or not isinstance(novelty["held_out_structure_task_fraction_minimum"], (int, float))
            or novelty["held_out_structure_task_fraction_minimum"] < 0.2):
        raise RuntimeError("I3.3.1 structural novelty thresholds are not frozen")
    if configuration.get("oracle_balance_thresholds") != {
            "designed_oracle_agreement": 1.0,
            "minimum_singleton_optimal_actions": 7,
            "maximum_tied_task_fraction": 0.15,
            "minimum_q_margin_bands": 2}:
        raise RuntimeError("I3.3.1 oracle-balance thresholds are not frozen")
    limits = configuration.get("oracle_limits")
    expected = {
        "max_latent_states_per_task", "max_latent_transitions_per_task",
        "max_information_states_per_condition",
        "max_information_transitions_per_condition", "max_members_per_belief",
    }
    if (not isinstance(limits, Mapping) or set(limits) != expected
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in limits.values())):
        raise RuntimeError("I3.3 oracle limits are incomplete or invalid")


def i3_3_benchmark_identity(root: str | Path, commit: str = "HEAD") -> dict[str, object]:
    root = Path(root).resolve()
    source_commit = _git(root, "rev-parse", commit)
    configuration = json.loads(_git_bytes(root, source_commit, CONFIGURATION_PATH))
    validate_configuration(configuration)
    manifest_path = str(configuration["benchmark_manifest_path"])
    manifest = json.loads(_git_bytes(root, source_commit, manifest_path))
    if manifest.get("status") != "FROZEN_FOR_BENCHMARK_QUALIFICATION":
        raise RuntimeError("I3.3 benchmark manifest is not frozen for benchmark qualification")
    protocol_path = str(configuration["protocol_path"])
    protocol = json.loads(_git_bytes(root, source_commit, protocol_path))
    graph = resolve_benchmark_artifact_graph(
        manifest_path=manifest_path, manifest=manifest,
        protocol_path=protocol_path, protocol=protocol,
        json_loader=lambda path: json.loads(_git_bytes(root, source_commit, path)))
    paths = {**COMPONENTS, **{f"benchmark_{role}": path for role, path in graph.items()}}
    for path in paths.values():
        _tree_hash(root, source_commit, path)
    return {
        "identity_version": IDENTITY_VERSION,
        "claim_boundary": "Frozen I3.3.1 benchmark integrity identity; no executive scientific result.",
        "source_commit": source_commit,
        "source_tree_hash": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "component_hashes": {
            name: {"path": path, "sha256": _tree_hash(root, source_commit, path)}
            for name, path in sorted(paths.items())
        },
        "test_corpus_sha256": _combined_hash(root, source_commit, TEST_CORPUS),
        "configuration": configuration,
        "environment": dependency_environment(),
    }
