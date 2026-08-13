"""Fail-closed identity scaffold for V2B-I3.2 methodology qualification.

I3.2 remains a development-only sequential-information-oracle milestone.  The
identity deliberately refuses its development configuration so no receipt can
be mistaken for a V2B scientific qualification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .qualification import _combined_hash, _git, _tree_hash, dependency_environment


IDENTITY_VERSION = "DAPH_V2B_I3_2_SEQUENTIAL_INFORMATION_ORACLE_IDENTITY_V1"
I3_2_COMPONENTS = {
    "i3_1_baseline": "configs/v2b_i3_2_baseline.json",
    "private_environment": "experiments/v2b/tasks/v2b_i3_metareasoning_benchmark_v1.json",
    "private_extension": "experiments/v2b_i3_2/tasks/v2b_i3_2_information_state_extension_v1.json",
    "controller_packets": "experiments/v2b/benchmark/controller_packets/v2b_i3_controller_packets_v1.json",
    "controller_packet_extension": "experiments/v2b_i3_2/benchmark/controller_packets/v2b_i3_2_controller_packets_extension_v1.json",
    "benchmark_manifest": "experiments/v2b_i3_2/benchmark/v2b_i3_2_benchmark_manifest_v1.json",
    "benchmark_runtime": "hrm_adaptive_memory/executive/metareasoning_benchmark.py",
    "runtime_state": "hrm_adaptive_memory/executive/metareasoning_state.py",
    "latent_transition_table": "hrm_adaptive_memory/executive/metareasoning_transition_table.py",
    "latent_oracle": "hrm_adaptive_memory/executive/metareasoning_oracle.py",
    "opening_observable_oracle": "hrm_adaptive_memory/executive/metareasoning_observable_oracle.py",
    "sequential_observable_oracle": "hrm_adaptive_memory/executive/metareasoning_sequential_oracle.py",
    "i3_2_runtime": "hrm_adaptive_memory/executive/metareasoning_i3_2.py",
    "utility": "hrm_adaptive_memory/executive/metareasoning_utility.py",
    "controller": "hrm_adaptive_memory/executive/metareasoning_controller.py",
    "policy": "configs/v2b_i3_policy_v1.json",
    "observation_masks": "configs/v2b_i3_observation_masks_v1.json",
    "resource_accounting": "hrm_adaptive_memory/executive/resources.py",
    "utility_config": "configs/v2b_i3_1_utility_v1.json",
    "development_config": "experiments/v2b_i3_2/configs/v2b_i3_2_development.json",
    "development_runner": "scripts/run_v2b_i3_2_development.py",
}
I3_2_TEST_CORPUS = (
    "tests/unit/test_v2b_i3_metareasoning.py",
    "tests/unit/test_v2b_i3_1_oracle_efficiency.py",
    "tests/unit/test_v2b_i3_2_sequential_information.py",
    "tests/adversarial/test_v2b_infrastructure_adversarial.py",
)


def validate_i3_2_configuration(configuration: Mapping[str, Any]) -> None:
    if configuration.get("schema") != "DAPH_V2B_I3_2_EXPERIMENT_CONFIGURATION_V1":
        raise RuntimeError("I3.2 configuration has an unsupported schema")
    if configuration.get("status") != "FROZEN_FOR_QUALIFICATION":
        raise RuntimeError("I3.2 configuration is not frozen for qualification")
    for section, key in (("baseline", "path"), ("benchmark", "path"), ("policy", "path"),
                         ("observation_masks", "path"), ("utility", "path")):
        if not configuration.get(section, {}).get(key):
            raise RuntimeError(f"I3.2 qualification configuration lacks {section}.{key}")
    if configuration.get("prior_definition") != "UNIFORM_BY_INITIAL_OBSERVATION_CLASS_V1":
        raise RuntimeError("I3.2 qualification has no supported frozen prior definition")
    visibility = configuration.get("policy_feedback_visibility")
    if not isinstance(visibility, Mapping) or not visibility.get("rejection_consumes_control_step"):
        raise RuntimeError("I3.2 qualification lacks frozen policy-feedback semantics")


def i3_2_identity(root: str | Path, configuration: Mapping[str, Any],
                  commit: str = "HEAD") -> dict[str, object]:
    validate_i3_2_configuration(configuration)
    root = Path(root).resolve()
    source_commit = _git(root, "rev-parse", commit)
    return {
        "identity_version": IDENTITY_VERSION,
        "source_commit": source_commit,
        "source_tree_hash": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "component_hashes": {name: {"path": path, "sha256": _tree_hash(root, source_commit, path)}
                             for name, path in I3_2_COMPONENTS.items()},
        "test_corpus_sha256": _combined_hash(root, source_commit, I3_2_TEST_CORPUS),
        "configuration": dict(configuration),
        "environment": dependency_environment(),
    }
