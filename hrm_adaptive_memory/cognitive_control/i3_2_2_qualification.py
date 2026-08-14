"""Fail-closed V2B-I3.2.2 methodology identity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from hrm_adaptive_memory.executive.metareasoning_artifacts import (
    resolve_benchmark_artifact_graph)

from .qualification import _combined_hash, _git, _git_bytes, _tree_hash, dependency_environment


IDENTITY_VERSION = "DAPH_V2B_I3_2_2_PROTOCOL_IDENTITY_V1"
PROTOCOL_PATH = "configs/v2b_i3_2_2_protocol_v1.json"
COMPONENTS = {
    "integrity_baseline": "configs/v2b_i3_2_2_baseline.json",
    "artifact_resolver": "hrm_adaptive_memory/executive/metareasoning_artifacts.py",
    "benchmark_runtime": "hrm_adaptive_memory/executive/metareasoning_benchmark.py",
    "latent_oracle": "hrm_adaptive_memory/executive/metareasoning_transition_table.py",
    "sequential_observable_oracle": "hrm_adaptive_memory/executive/metareasoning_sequential_oracle.py",
    "metrics_and_replay": "hrm_adaptive_memory/executive/metareasoning_i3_2.py",
    "utility_runtime": "hrm_adaptive_memory/executive/metareasoning_utility.py",
    "resource_runtime": "hrm_adaptive_memory/executive/resources.py",
    "controller": "hrm_adaptive_memory/executive/metareasoning_controller.py",
    "protocol": PROTOCOL_PATH,
}
TEST_CORPUS = (
    "tests/unit/test_v2b_i3_metareasoning.py",
    "tests/unit/test_v2b_i3_1_oracle_efficiency.py",
    "tests/unit/test_v2b_i3_2_sequential_information.py",
    "tests/unit/test_v2b_i3_2_2_protocol.py",
    "tests/adversarial/test_v2b_infrastructure_adversarial.py",
)


def validate_protocol(protocol: Mapping[str, object]) -> None:
    if protocol.get("schema") != "DAPH_V2B_I3_2_2_PROTOCOL_V1":
        raise RuntimeError("I3.2.2 protocol schema is unsupported")
    if protocol.get("status") != "FROZEN_FOR_METHODOLOGY_QUALIFICATION":
        raise RuntimeError("I3.2.2 protocol is not frozen for methodology qualification")
    priors = protocol.get("aggregate_priors")
    if not isinstance(priors, Mapping) or priors.get("scientific_primary") != "TASK_UNIFORM":
        raise RuntimeError("I3.2.2 protocol lacks the primary task-uniform prior")


def protocol_identity(root: str | Path, commit: str = "HEAD") -> dict[str, object]:
    root = Path(root).resolve()
    source_commit = _git(root, "rev-parse", commit)
    protocol = json.loads(_git_bytes(root, source_commit, PROTOCOL_PATH))
    validate_protocol(protocol)
    manifest_path = str(protocol["benchmark_manifest_path"])
    manifest = json.loads(_git_bytes(root, source_commit, manifest_path))
    graph = resolve_benchmark_artifact_graph(
        manifest_path=manifest_path, manifest=manifest,
        protocol_path=PROTOCOL_PATH, protocol=protocol)
    paths = {**COMPONENTS, **{f"benchmark_{role}": path for role, path in graph.items()}}
    return {
        "identity_version": IDENTITY_VERSION,
        "source_commit": source_commit,
        "source_tree_hash": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "component_hashes": {name: {"path": path, "sha256": _tree_hash(root, source_commit, path)}
                             for name, path in paths.items()},
        "test_corpus_sha256": _combined_hash(root, source_commit, TEST_CORPUS),
        "environment": dependency_environment(),
    }
