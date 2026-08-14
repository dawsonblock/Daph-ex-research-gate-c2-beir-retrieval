"""Exhaustive I3.3.2 oracle regeneration gate (intentionally not a unit test)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hrm_adaptive_memory.common.canonical_json import canonical_bytes
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import load_observation_masks
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_sequential_oracle import (
    build_sequential_observable_oracle)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "experiments/v2b_i3_3/manifests/v2b_i3_3_benchmark_manifest_v1.json"
CONFIG = ROOT / "experiments/v2b_i3_3/configs/v2b_i3_3_benchmark_freeze_v1.json"
CACHE = ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_oracle_cache_manifest_v1.json"


def _combined(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def test_i3_3_qualification_regenerates_all_latent_and_sequential_oracle_set_hashes():
    configuration = json.loads(CONFIG.read_text())
    expected = json.loads(CACHE.read_text())
    benchmark = load_metareasoning_benchmark(MANIFEST)
    policy = load_frozen_policy(ROOT / configuration["policy_path"])
    utility = MetareasoningUtility.from_file(ROOT / configuration["utility_path"])
    masks = load_observation_masks(ROOT / configuration["observation_masks_path"])
    cache = OracleTableCache()
    runtimes = {
        task.task_id: initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
        for task in benchmark.tasks
    }
    latent = {
        task_id: cache.get_or_build(
            initial_runtime=runtime, policy=policy, utility=utility,
            include_policy_feedback=True)
        for task_id, runtime in sorted(runtimes.items())
    }
    assert _combined({key: table.table_sha256 for key, table in sorted(latent.items())}) == (
        expected["latent_oracles"]["table_set_sha256"])
    limits = configuration["oracle_limits"]
    for condition in configuration["required_conditions"]:
        oracle_set = build_sequential_observable_oracle(
            runtime_tables=((runtimes[key], latent[key]) for key in sorted(latent)),
            mask=masks[condition], policy=policy, utility=utility,
            benchmark_hash=expected["benchmark_closure_sha256"],
            max_information_states=limits["max_information_states_per_condition"],
            max_information_transitions=limits["max_information_transitions_per_condition"],
            max_members_per_belief=limits["max_members_per_belief"])
        assert oracle_set.table_sha256 == (
            expected["sequential_observable_oracles"][condition]["set_sha256"])
