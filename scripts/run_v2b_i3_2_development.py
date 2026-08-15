#!/usr/bin/env python3
"""Run the V2B-I3.2 sequential-information oracle development protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    MatchedMetareasoningController, load_observation_masks)
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_i3_2 import (
    DEVELOPMENT_RECEIPT_SCHEMA, aggregate_metrics, class_decomposition, replay_trajectory,
    run_condition, trajectory_payload)
from hrm_adaptive_memory.executive.metareasoning_sequential_oracle import (
    build_sequential_observable_oracle, policy_feedback_visibility_hash)
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility, frozen_action_cost_hash
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.cognitive_control.i3_2_qualification import validate_oracle_limits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _log(receipt: dict[str, object], path: Path) -> None:
    import litlogger
    experiment = litlogger.init(
        name=f"v2b-i3.2-sequential-information-{str(receipt['source_commit'])[:7]}",
        teamspace="deep-gpu-acceleration-project",
        metadata={"protocol": "DAPH_V2B_I3_2", "status": receipt["status"],
                  "source_commit": receipt["source_commit"], "source_tree_hash": receipt["source_tree_hash"],
                  "receipt_sha256": _sha256(path), "location": "US", "altitude": "1334"},
        print_url=True)
    for condition, data in receipt["conditions"].items():  # type: ignore[index]
        for metric, value in data["metrics"].items():  # type: ignore[index]
            experiment[f"{str(condition).lower()}_{metric}"].append(value)
    for metric, value in receipt["timing"].items():  # type: ignore[index]
        experiment[f"timing_{metric}"].append(value)
    experiment["run_valid"] = "true"; experiment["qualification"] = "NOT_QUALIFIED_DEVELOPMENT_ONLY"
    experiment.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="v2b-i3-2-development")
    args = parser.parse_args()
    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.run_id):
        raise ValueError("run-id must contain only letters, digits, underscores, or hyphens")
    if _git("status", "--porcelain"):
        raise RuntimeError("I3.2 receipts require a clean committed checkout")
    config_path = ROOT / "experiments/v2b_i3_2/configs/v2b_i3_2_development.json"
    config = json.loads(config_path.read_text())
    if config.get("status") != "DEVELOPMENT_EXPERIMENT_NOT_QUALIFIED":
        raise RuntimeError("I3.2 runner accepts only development-only configuration")
    oracle_limits = validate_oracle_limits(config)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("I3.2 output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    benchmark_path = ROOT / config["benchmark"]["path"]
    policy_path = ROOT / config["policy"]["path"]
    masks_path = ROOT / config["observation_masks"]["path"]
    utility_path = ROOT / config["utility"]["path"]
    benchmark = load_metareasoning_benchmark(benchmark_path)
    policy = load_frozen_policy(policy_path); masks = load_observation_masks(masks_path)
    utility = MetareasoningUtility.from_file(utility_path)
    cache = OracleTableCache(); latent_started = perf_counter()
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
                for task in benchmark.tasks}
    latent_tables = {task_id: cache.get_or_build(initial_runtime=runtime, policy=policy, utility=utility,
                                                  include_policy_feedback=True)
                     for task_id, runtime in runtimes.items()}
    latent_seconds = perf_counter() - latent_started
    oracle_dir = output / "oracles"; oracle_dir.mkdir()
    latent_records = {}
    for task_id, table in sorted(latent_tables.items()):
        target = oracle_dir / f"latent_{task_id}.json"
        target.write_text(json.dumps(table.serializable(), sort_keys=True, indent=2) + "\n")
        latent_records[task_id] = {"path": str(target.relative_to(output)), "sha256": _sha256(target),
                                   "table_sha256": table.table_sha256}
    sequential_started = perf_counter(); sequential_sets = {}
    for condition, mask in masks.items():
        sequential_sets[condition] = build_sequential_observable_oracle(
            runtime_tables=((runtimes[task_id], table) for task_id, table in latent_tables.items()),
            mask=mask, policy=policy, utility=utility,
            benchmark_hash=_sha256(benchmark_path),
            max_information_states=oracle_limits["max_information_states"],
            max_information_transitions=oracle_limits["max_information_transitions"],
            max_members_per_belief=oracle_limits["max_members_per_belief"])
    sequential_seconds = perf_counter() - sequential_started
    sequential_records = {}
    for condition, oracle_set in sequential_sets.items():
        folder = oracle_dir / f"sequential_{condition.lower()}"; folder.mkdir()
        entries = {}
        for root_id, table in oracle_set.tables.items():
            target = folder / f"{root_id}.json"
            target.write_text(json.dumps(table.serializable(), sort_keys=True, indent=2) + "\n")
            entries[root_id] = {"path": str(target.relative_to(output)), "sha256": _sha256(target),
                                "table_sha256": table.table_sha256}
        sequential_records[condition] = {"set_sha256": oracle_set.table_sha256, "tables": entries}
    controller_started = perf_counter(); runs = {}
    decompositions = {}
    for condition, mask in masks.items():
        runs[condition] = run_condition(
            benchmark=benchmark, condition=condition, controller=MatchedMetareasoningController(),
            mask=mask, policy=policy, utility=utility, latent_tables=latent_tables,
            oracle_set=sequential_sets[condition])
        decompositions[condition] = class_decomposition(
            runs=runs[condition], oracle_set=sequential_sets[condition], latent_tables=latent_tables)
    controller_seconds = perf_counter() - controller_started
    replay_started = perf_counter(); receipt_dir = output / "trajectory_receipts"; receipt_dir.mkdir()
    trajectory_records = {}; condition_metrics = {}
    for condition, run in runs.items():
        items = []
        for task_run in run:
            table = sequential_sets[condition].tables[task_run.initial_information_state_id]
            payload = trajectory_payload(run=task_run, table=table, condition=condition,
                                         policy_sha256=policy.sha256, utility_sha256=utility.sha256,
                                         controller_revision=MatchedMetareasoningController.algorithm_id)
            replay = replay_trajectory(benchmark=benchmark, task_id=task_run.task_id,
                                       traces=payload["steps"], policy=policy, utility=utility)
            if replay["trajectory_utility"] != task_run.realized_utility:
                raise RuntimeError("I3.2 trajectory replay utility mismatch")
            items.append(payload)
        target = receipt_dir / f"{condition.lower()}.jsonl"
        target.write_text("\n".join(json.dumps(item, sort_keys=True, default=str) for item in items) + "\n")
        trajectory_records[condition] = {"path": str(target.relative_to(output)), "sha256": _sha256(target)}
        condition_metrics[condition] = aggregate_metrics(
            runs=run, decomposition=decompositions[condition], oracle_set=sequential_sets[condition],
            benchmark=benchmark, mask=masks[condition])
    replay_seconds = perf_counter() - replay_started
    all_tables = [table for oracle_set in sequential_sets.values() for table in oracle_set.tables.values()]
    receipt = {
        "schema": DEVELOPMENT_RECEIPT_SCHEMA, "run_valid": True, "status": "DEVELOPMENT_NOT_QUALIFIED",
        "claim_boundary": "Sequential information-state methodology validation only; no learned controller or V2B scientific verdict.",
        "source_commit": _git("rev-parse", "HEAD"), "source_tree_hash": _git("rev-parse", "HEAD^{tree}"),
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "benchmark": {"path": str(benchmark_path.relative_to(ROOT)), "sha256": _sha256(benchmark_path),
                      "artifact_hashes": dict(benchmark.artifact_hashes)},
        "policy": {"path": str(policy_path.relative_to(ROOT)), "sha256": policy.sha256},
        "utility": {"path": str(utility_path.relative_to(ROOT)), "sha256": utility.sha256,
                    "action_cost_sha256": frozen_action_cost_hash()},
        "observation_masks": {"path": str(masks_path.relative_to(ROOT)), "sha256": _sha256(masks_path),
                              "masks": {key: value.sha256() for key, value in masks.items()}},
        "prior_definition": config["prior_definition"],
        "policy_feedback_visibility_sha256": policy_feedback_visibility_hash(),
        "latent_oracle_tables": latent_records, "sequential_observable_oracles": sequential_records,
        "trajectory_receipts": trajectory_records,
        "conditions": {key: {"metrics": value, "class_decomposition": decompositions[key]}
                       for key, value in condition_metrics.items()},
        "timing": {"latent_table_build_seconds": latent_seconds,
                   "sequential_observable_build_seconds": sequential_seconds,
                   "controller_evaluation_seconds": controller_seconds, "replay_seconds": replay_seconds,
                   "sequential_information_states": sum(len(table.information_states) for table in all_tables),
                   "sequential_information_transitions": sum(len(table.transitions) for table in all_tables),
                   "sequential_peak_memory_kib": max(int(table.build_metrics["belief_peak_resident_memory_delta_kib"])
                                                       for table in all_tables),
                   "latent_cache_hit_rate": cache.hit_rate},
    }
    target = output / f"{args.run_id}.json"
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    _log(receipt, target); print(target)


if __name__ == "__main__":
    main()
