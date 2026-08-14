#!/usr/bin/env python3
"""Run the V2B-I3.1 development-only oracle-efficiency protocol.

Every invocation is recorded through LitLogger, but the emitted receipt remains
explicitly development-only and cannot be used as a V2B scientific verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.metareasoning_controller import (
    MatchedMetareasoningController, load_observation_masks)
from hrm_adaptive_memory.executive.metareasoning_executor import initial_i3_runtime
from hrm_adaptive_memory.executive.metareasoning_i3_1 import (
    AGGREGATE_RECEIPT_SCHEMA, aggregate_metrics, trajectory_payload)
from hrm_adaptive_memory.executive.metareasoning_loop import V2BMetareasoningExperiment
from hrm_adaptive_memory.executive.metareasoning_observable_oracle import build_observable_oracle
from hrm_adaptive_memory.executive.metareasoning_transition_table import OracleTableCache
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility, frozen_action_cost_hash
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _safe_run_id(value: str) -> str:
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise ValueError("run-id must contain only letters, digits, underscores, or hyphens")
    return value


def _log(receipt: dict[str, object], path: Path) -> None:
    import litlogger
    experiment = litlogger.init(
        name=f"v2b-i3.1-oracle-efficiency-{str(receipt['source_commit'])[:7]}",
        teamspace="deep-gpu-acceleration-project",
        metadata={"protocol": "DAPH_V2B_I3_1", "status": receipt["status"],
                  "source_commit": receipt["source_commit"],
                  "source_tree_hash": receipt["source_tree_hash"],
                  "receipt_sha256": _sha256(path), "location": "US", "altitude": "1334"},
        print_url=True,
    )
    for condition, details in receipt["conditions"].items():  # type: ignore[index]
        for metric, value in details["metrics"].items():  # type: ignore[index]
            experiment[f"{str(condition).lower()}_{metric}"].append(value)
    for metric, value in receipt["oracle_complexity"].items():  # type: ignore[index]
        experiment[f"oracle_{metric}"].append(value)
    experiment["run_valid"] = "true"
    experiment["qualification"] = "NOT_QUALIFIED_DEVELOPMENT_ONLY"
    experiment.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="v2b-i3-1-development")
    args = parser.parse_args()
    _safe_run_id(args.run_id)
    if _git("status", "--porcelain"):
        raise RuntimeError("I3.1 receipts require a clean committed checkout")
    config_path = ROOT / "experiments/v2b_i3_1/configs/v2b_i3_1_development.json"
    config = json.loads(config_path.read_text())
    if config.get("status") != "DEVELOPMENT_EXPERIMENT_NOT_QUALIFIED":
        raise RuntimeError("I3.1 runner accepts only its development configuration")
    benchmark_path = ROOT / config["benchmark"]["path"]
    policy_path = ROOT / config["policy"]["path"]
    masks_path = ROOT / config["observation_masks"]["path"]
    utility_path = ROOT / config["utility"]["path"]
    benchmark = load_metareasoning_benchmark(benchmark_path)
    policy = load_frozen_policy(policy_path)
    masks = load_observation_masks(masks_path)
    utility = MetareasoningUtility.from_file(utility_path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("I3.1 output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    cache = OracleTableCache()
    runtimes = {task.task_id: initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
                for task in benchmark.tasks}
    tables = {task_id: cache.get_or_build(initial_runtime=runtime, policy=policy, utility=utility)
              for task_id, runtime in runtimes.items()}
    observable_tables = {
        condition: build_observable_oracle(runtime_tables=((runtimes[task_id], table)
                                                            for task_id, table in tables.items()), mask=mask)
        for condition, mask in masks.items()
    }
    protocol = V2BMetareasoningExperiment(
        benchmark=benchmark, policy=policy, utility=utility, oracle_table_cache=cache)
    runs = {
        condition: protocol.run_condition(
            condition=condition, controller=MatchedMetareasoningController(),
            store_root=output / "cognitive_logs", mask=mask)
        for condition, mask in masks.items()
    }
    oracle_dir = output / "oracles"
    oracle_dir.mkdir()
    table_records = {}
    for task_id, table in sorted(tables.items()):
        target = oracle_dir / f"{task_id}.json"
        target.write_text(json.dumps(table.serializable(), sort_keys=True, indent=2) + "\n")
        table_records[task_id] = {"path": str(target.relative_to(output)), "sha256": _sha256(target),
                                  "table_sha256": table.table_sha256, "identity_sha256": table.identity_sha256}
    observable_records = {}
    for condition, table in observable_tables.items():
        target = oracle_dir / f"observable_{condition.lower()}.json"
        target.write_text(json.dumps(table.serializable(), sort_keys=True, indent=2) + "\n")
        observable_records[condition] = {"path": str(target.relative_to(output)), "sha256": _sha256(target),
                                         "table_sha256": table.table_sha256, "identity_sha256": table.identity_sha256}
    trajectory_dir = output / "trajectory_receipts"
    trajectory_dir.mkdir()
    trajectory_records = {}
    for condition, run in runs.items():
        target = trajectory_dir / f"{condition.lower()}.jsonl"
        payloads = [trajectory_payload(
            run=task_run, table=tables[task_run.task_id], observable=observable_tables[condition],
            condition=condition, observation_mask_sha256=masks[condition].sha256(),
            controller_revision=run.controller_algorithm_id, policy_sha256=policy.sha256,
            utility_sha256=utility.sha256,
            budget_sha256=hashlib.sha256(json.dumps(
                benchmark.budget_for(next(task for task in benchmark.tasks if task.task_id == task_run.task_id)).__dict__,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()) for task_run in run.tasks]
        target.write_text("\n".join(json.dumps(item, sort_keys=True, default=str) for item in payloads) + "\n")
        trajectory_records[condition] = {"path": str(target.relative_to(output)), "sha256": _sha256(target)}
    condition_metrics = {
        condition: {**run.metrics, **aggregate_metrics(run_tasks=run.tasks, tables=tables,
                                                        observable=observable_tables[condition])}
        for condition, run in runs.items()
    }
    complexity = {
        "reachable_states": sum(len(table.states) for table in tables.values()),
        "reachable_transitions": sum(len(table.transitions) for table in tables.values()),
        "oracle_build_seconds": sum(float(table.build_metrics["oracle_build_seconds"]) for table in tables.values()),
        "peak_table_bytes": max(int(table.build_metrics["table_bytes"]) for table in tables.values()),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_hit_rate": cache.hit_rate,
    }
    receipt = {
        "schema": AGGREGATE_RECEIPT_SCHEMA, "run_valid": True,
        "status": "DEVELOPMENT_NOT_QUALIFIED",
        "claim_boundary": "I3.1 deterministic methodology validation only; no learned-controller or scientific V2B result.",
        "source_commit": _git("rev-parse", "HEAD"), "source_tree_hash": _git("rev-parse", "HEAD^{tree}"),
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "benchmark": {"path": str(benchmark_path.relative_to(ROOT)), "sha256": _sha256(benchmark_path),
                      "artifact_hashes": dict(benchmark.artifact_hashes)},
        "policy": {"path": str(policy_path.relative_to(ROOT)), "sha256": policy.sha256},
        "utility": {"path": str(utility_path.relative_to(ROOT)), "sha256": utility.sha256,
                    "action_cost_sha256": frozen_action_cost_hash()},
        "observation_masks": {"path": str(masks_path.relative_to(ROOT)), "sha256": _sha256(masks_path),
                              "masks": {condition: mask.sha256() for condition, mask in masks.items()}},
        "latent_oracle_tables": table_records, "observable_oracle_tables": observable_records,
        "oracle_complexity": complexity, "trajectory_receipts": trajectory_records,
        "conditions": {condition: {"controller_id": runs[condition].controller_id,
                                     "metrics": metrics} for condition, metrics in condition_metrics.items()},
    }
    target = output / f"{args.run_id}.json"
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    _log(receipt, target)
    print(target)


if __name__ == "__main__":
    main()

