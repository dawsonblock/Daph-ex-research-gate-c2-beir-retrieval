"""Fail-closed identity scaffold for a future qualified V2B-I3 experiment.

I3 is currently development-only. This module defines what must bind together
before a scientific receipt can exist; it refuses the current development
configuration by design.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .qualification import _combined_hash, _git, _tree_hash, dependency_environment


IDENTITY_VERSION = "DAPH_V2B_I3_METAREASONING_IDENTITY_V1"
I3_COMPONENTS = {
    "protocol": "configs/cognitive_control_v2b_design.json",
    "observation_generation": "hrm_adaptive_memory/executive/metareasoning_executor.py",
    "observation_masks": "configs/v2b_i3_observation_masks_v1.json",
    "controller": "hrm_adaptive_memory/executive/metareasoning_controller.py",
    "action_schema": "hrm_adaptive_memory/cognitive_control/actions.py",
    "resource_accounting": "hrm_adaptive_memory/executive/resources.py",
    "policy": "configs/v2b_i3_policy_v1.json",
    "policy_runtime": "hrm_adaptive_memory/executive/policy.py",
    "private_environment": "experiments/v2b/tasks/v2b_i3_metareasoning_benchmark_v1.json",
    "controller_packets": "experiments/v2b/benchmark/controller_packets/v2b_i3_controller_packets_v1.json",
    "benchmark_manifest": "experiments/v2b/benchmark/v2b_i3_benchmark_manifest_v1.json",
    "benchmark_runtime": "hrm_adaptive_memory/executive/metareasoning_benchmark.py",
    "oracle": "hrm_adaptive_memory/executive/metareasoning_oracle.py",
    "metric_runtime": "hrm_adaptive_memory/executive/metareasoning_loop.py",
    "development_configuration": "experiments/v2b/configs/v2b_i3_development.json",
}
I3_TEST_CORPUS = (
    "tests/unit/test_v2b_i3_metareasoning.py",
    "tests/adversarial/test_v2b_infrastructure_adversarial.py",
)


def validate_i3_configuration(configuration: Mapping[str, Any]) -> None:
    if configuration.get("schema") != "DAPH_V2B_EXPERIMENT_CONFIGURATION_V1":
        raise RuntimeError("I3 configuration has an unsupported schema")
    if configuration.get("status") != "FROZEN_FOR_QUALIFICATION":
        raise RuntimeError("I3 configuration is not frozen for qualification")
    controller = configuration.get("controller", {})
    benchmark = configuration.get("benchmark", {})
    model = configuration.get("model", {})
    required = (
        controller.get("algorithm_id"), controller.get("revision"), benchmark.get("path"),
        benchmark.get("private_environment_path"), benchmark.get("controller_packets_path"),
        model.get("id"), model.get("revision"), model.get("tokenizer_revision"),
        model.get("generation_configuration"),
    )
    if any(value in (None, "", "NONE", {}) for value in required):
        raise RuntimeError("I3 qualification configuration lacks a pinned controller/model/benchmark identity")


def metareasoning_identity(root: str | Path, configuration: Mapping[str, Any],
                           commit: str = "HEAD") -> dict[str, object]:
    """Construct the future I3 identity only after fail-closed qualification checks."""
    validate_i3_configuration(configuration)
    root = Path(root).resolve()
    source_commit = _git(root, "rev-parse", commit)
    return {
        "identity_version": IDENTITY_VERSION,
        "source_commit": source_commit,
        "source_tree_hash": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "component_hashes": {
            name: {"path": path, "sha256": _tree_hash(root, source_commit, path)}
            for name, path in I3_COMPONENTS.items()
        },
        "test_corpus_sha256": _combined_hash(root, source_commit, I3_TEST_CORPUS),
        "configuration": dict(configuration),
        "environment": dependency_environment(),
    }
