#!/usr/bin/env python3
"""Precompute deterministic latent and sequential I3.5 V2 oracle ground truth.

Adapts the I3.3 oracle precomputation for V2 structural tasks.
Produces:
- Latent oracle tables (per-task V_L*)
- Sequential observable oracle tables (per-condition V_O*)
- Difficulty report (topology depth bands, Q-margins)
- Topology allocation
- Oracle cache manifest
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import load_observation_masks
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_sequential_oracle import (
    build_sequential_observable_oracle)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_topology import transition_topology
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.common.canonical_json import (
    canonical_bytes, write_json)


BENCHMARK_MANIFEST = ROOT / "experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json"
OUTPUT = ROOT / "experiments/v2b_i3_5/oracle_tables"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_hash(values: object) -> str:
    return hashlib.sha256(canonical_bytes(values)).hexdigest()


def semantic_table(table) -> dict[str, object]:
    payload = table.serializable()
    payload.pop("build_metrics", None)
    return payload


def write_gzip_jsonl(path: Path, rows: list[object]) -> None:
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))


def q_margin(table, state_id: str) -> float:
    values = sorted({round(value, 12) for (origin, _), value in table.q_values.items()
                     if origin == state_id}, reverse=True)
    return 0.0 if len(values) < 2 else values[0] - values[1]


def margin_band(value: float, tied: bool, reward_span: float) -> str:
    if tied:
        return "TIE"
    normalized = value / reward_span
    if normalized < 0.005:
        return "HARD"
    if normalized < 0.10:
        return "MEDIUM"
    return "EASY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-litlogger", action="store_true")
    args = parser.parse_args()

    benchmark = load_metareasoning_benchmark(BENCHMARK_MANIFEST, verify_oracle_cache=False)
    benchmark_manifest = json.loads(BENCHMARK_MANIFEST.read_text())
    private_path = (BENCHMARK_MANIFEST.parent / benchmark_manifest["private_environment_path"]).resolve()
    private_payload = json.loads(private_path.read_text())
    private_by_id = {str(task["task_id"]): task for task in private_payload["tasks"]}

    # Load frozen policy and utility (same as I3.3)
    policy_path = ROOT / "configs/v2b_i3_policy_v1.json"
    utility_path = ROOT / "configs/v2b_i3_1_utility_v1.json"
    policy = load_frozen_policy(policy_path)
    utility = MetareasoningUtility.from_file(utility_path)

    # Load observation masks (same as I3.3)
    masks_path = ROOT / "configs/v2b_i3_observation_masks_v1.json"
    masks = load_observation_masks(masks_path)

    limits = {
        "max_latent_states_per_task": 20000,
        "max_latent_transitions_per_task": 120000,
        "max_information_states_per_condition": 500000,
        "max_information_transitions_per_condition": 3000000,
        "max_members_per_belief": 32,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Build latent oracle tables
    print("Building latent oracle tables...", flush=True)
    started = perf_counter()
    cache = OracleTableCache()
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
                for task in benchmark.tasks}
    latent = {}
    latent_rows = []
    difficulty_rows = []
    topology_by_task = {}
    topology_by_split: dict[str, set[str]] = defaultdict(set)
    depth_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    reward_span = utility.correct_answer - utility.incorrect_answer

    for i, task_id in enumerate(sorted(runtimes)):
        table = cache.get_or_build(initial_runtime=runtimes[task_id], policy=policy,
                                   utility=utility, include_policy_feedback=True)
        latent[task_id] = table
        root = table.initial_state_id
        actions = [item.value for item in table.optimal_actions[root]]
        topology = transition_topology(table)
        topology_by_task[task_id] = topology
        split = str(private_by_id[task_id]["split"])
        topology_by_split[split].add(topology.sha256)
        depth_by_split[split][topology.depth_band] += 1
        minimum_cost = table.minimum_remaining_cost[root]
        successful_path_exists = minimum_cost != float("inf")
        latent_rows.append({"task_id": task_id, "table": semantic_table(table)})
        difficulty_rows.append({
            "task_id": task_id, "latent_value": table.state_values[root],
            "latent_optimal_actions": actions, "optimal_q_margin": q_margin(table, root),
            "normalized_optimal_q_margin": q_margin(table, root) / reward_span,
            "q_margin_band": margin_band(q_margin(table, root), len(actions) > 1, reward_span),
            "successful_path_exists": successful_path_exists,
            "minimum_remaining_cost": minimum_cost if successful_path_exists else None,
            "semantic_structure_coarse": private_by_id[task_id]["semantic_structure_coarse"],
            "semantic_structure_exact": private_by_id[task_id]["semantic_structure_exact"],
            "transition_topology_sha256": topology.sha256,
            "minimum_optimal_trajectory_depth": topology.minimum_optimal_trajectory_depth,
            "maximum_relevant_trajectory_depth": topology.maximum_relevant_trajectory_depth,
            "topology_depth_band": topology.depth_band,
            "decision_branch_points": topology.decision_branch_points,
            "split": split,
            "reachable_states": len(table.states),
            "reachable_transitions": len(table.transitions),
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(runtimes)}] latent tables built", flush=True)

    latent_seconds = perf_counter() - started
    print(f"Latent oracle built in {latent_seconds:.1f}s", flush=True)

    # Save latent oracle
    latent_path = OUTPUT / "v2b_i3_5_latent_oracles_v1.jsonl.gz"
    write_gzip_jsonl(latent_path, latent_rows)
    print(f"Latent oracle saved: {latent_path}", flush=True)

    # Save difficulty report
    difficulty_path = OUTPUT / "v2b_i3_5_difficulty_report_v1.json"
    write_json(difficulty_path, {
        "schema": "DAPH_V2B_I3_3_2_DIFFICULTY_REPORT_V1",
        "status": "FROZEN_BENCHMARK_NOT_A_SCIENTIFIC_RESULT",
        "tasks": difficulty_rows,
    })
    print(f"Difficulty report saved: {difficulty_path}", flush=True)

    # Build topology allocation
    topology_roles: dict[str, set[str]] = defaultdict(set)
    topology_tasks: dict[str, list[str]] = defaultdict(list)
    for task_id, topology in sorted(topology_by_task.items()):
        split = str(private_by_id[task_id]["split"])
        topology_roles[topology.sha256].add(split)
        topology_tasks[topology.sha256].append(task_id)

    topology_allocation = {
        "schema": "DAPH_V2B_I3_3_2_TOPOLOGY_ALLOCATION_V1",
        "status": "FROZEN_SCIENTIFIC_BENCHMARK_NO_EXECUTIVE_RESULT",
        "topologies": {
            key: {"roles": sorted(topology_roles[key]), "task_ids": sorted(topology_tasks[key])}
            for key in sorted(topology_roles)
        },
        "invariants": {
            "structure_held_out_v2_disjoint_from_structure_dev_v2": True,
            "structure_held_out_v2_disjoint_from_structure_validation_v2": True,
        },
    }
    topology_allocation_path = OUTPUT.parent / "splits/topology_allocation_v2.json"
    write_json(topology_allocation_path, topology_allocation)
    print(f"Topology allocation saved: {topology_allocation_path}", flush=True)

    # Build topology diversity report
    splits = tuple(sorted(topology_by_split))
    overlap = {
        left: {right: len(topology_by_split[left] & topology_by_split[right])
               for right in splits}
        for left in splits
    }
    topology_report = {
        "schema": "DAPH_V2B_I3_3_2_TOPOLOGY_DIVERSITY_REPORT_V1",
        "status": "FROZEN_SCIENTIFIC_BENCHMARK_NO_EXECUTIVE_RESULT",
        "transition_topologies": {
            split: len(topology_by_split[split]) for split in splits},
        "topology_overlap_matrix": overlap,
        "topology_depth_bands": {
            split: dict(sorted(depth_by_split[split].items())) for split in splits},
        "identity_semantics": (
            "Behavior-derived proposal/policy/transition connectivity; excludes task ids, "
            "surface text, generator channel labels, state labels, and budget-profile names."),
    }
    topology_report_path = OUTPUT.parent / "reports/v2b_i3_5_topology_diversity_report_v1.json"
    write_json(topology_report_path, topology_report)
    print(f"Topology diversity report saved: {topology_report_path}", flush=True)

    # Build sequential observable oracles for each condition
    condition_records = {}
    for condition in ("STATE_BLIND_CONTROLLER", "STATE_AWARE_CONTROLLER"):
        print(f"\nBuilding sequential oracle for {condition}...", flush=True)
        condition_started = perf_counter()
        oracle_set = build_sequential_observable_oracle(
            runtime_tables=((runtimes[key], latent[key]) for key in sorted(latent)),
            mask=masks[condition], policy=policy, utility=utility,
            benchmark_hash=sha256(BENCHMARK_MANIFEST),
            max_information_states=limits["max_information_states_per_condition"],
            max_information_transitions=limits["max_information_transitions_per_condition"],
            max_members_per_belief=limits["max_members_per_belief"])
        rows = [{"initial_information_state_id": key, "table": semantic_table(table)}
                for key, table in sorted(oracle_set.tables.items())]
        target = OUTPUT / f"v2b_i3_5_sequential_{condition.lower()}_v1.jsonl.gz"
        write_gzip_jsonl(target, rows)
        states = sum(len(table.information_states) for table in oracle_set.tables.values())
        transitions = sum(len(table.transitions) for table in oracle_set.tables.values())
        condition_records[condition] = {
            "path": target.relative_to(ROOT).as_posix(), "sha256": sha256(target),
            "set_sha256": oracle_set.table_sha256,
            "table_count": len(oracle_set.tables), "information_states": states,
            "information_transitions": transitions,
        }
        elapsed = perf_counter() - condition_started
        print(f"  {condition}: {len(oracle_set.tables)} tables, "
              f"{states} states, {transitions} transitions, {elapsed:.1f}s", flush=True)

    # Save oracle cache manifest
    manifest = {
        "schema": "DAPH_V2B_I3_3_ORACLE_CACHE_MANIFEST_V1",
        "status": "FROZEN_BENCHMARK_NOT_A_SCIENTIFIC_RESULT",
        "benchmark_manifest_sha256": sha256(BENCHMARK_MANIFEST),
        "latent_oracles": {
            "path": latent_path.relative_to(ROOT).as_posix(), "sha256": sha256(latent_path),
            "table_set_sha256": combined_hash({key: table.table_sha256
                                                 for key, table in sorted(latent.items())}),
            "table_count": len(latent),
            "reachable_states": sum(len(table.states) for table in latent.values()),
            "reachable_transitions": sum(len(table.transitions) for table in latent.values()),
        },
        "sequential_observable_oracles": condition_records,
        "difficulty_report": {"path": difficulty_path.relative_to(ROOT).as_posix(),
                              "sha256": sha256(difficulty_path)},
        "topology_allocation": {
            "path": topology_allocation_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(topology_allocation_path),
        },
        "topology_diversity_report": {
            "path": topology_report_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(topology_report_path),
        },
        "limits": limits,
        "claim_boundary": "Precomputed deterministic oracle ground truth only; no executive result.",
    }
    manifest_path = OUTPUT / "v2b_i3_5_oracle_cache_manifest_v1.json"
    write_json(manifest_path, manifest)
    print(f"\nOracle cache manifest saved: {manifest_path}", flush=True)

    # Print summary
    print(f"\n=== V2 ORACLE PRECOMPUTATION COMPLETE ===")
    print(f"Tasks: {len(latent)}")
    print(f"Latent tables: {len(latent)}")
    print(f"Latent build time: {latent_seconds:.1f}s")
    for condition, record in condition_records.items():
        print(f"{condition}: {record['table_count']} tables, "
              f"{record['information_states']} states")
    print(f"Topologies by split:")
    for split in splits:
        print(f"  {split}: {len(topology_by_split[split])} topologies, "
              f"depths={dict(sorted(depth_by_split[split].items()))}")


if __name__ == "__main__":
    main()
