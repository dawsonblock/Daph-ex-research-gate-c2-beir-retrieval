#!/usr/bin/env python3
"""Run V2B-I3's development-only metareasoning validity protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from hrm_adaptive_memory.executive.metareasoning_loop import (
    STATE_AWARE, STATE_BLIND, V2BMetareasoningExperiment)
from hrm_adaptive_memory.executive.policy import load_frozen_policy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _log_development_receipt(receipt: dict[str, object], receipt_path: Path) -> None:
    """Log execution provenance and metrics without upgrading scientific status."""
    import litlogger

    experiment = litlogger.init(
        name=f"v2b-i3-metareasoning-development-{receipt['source_commit'][:7]}",
        teamspace="deep-gpu-acceleration-project",
        metadata={
            "protocol": "DAPH_V2B_I3_METAREASONING",
            "status": str(receipt["status"]),
            "claim_boundary": str(receipt["claim_boundary"]),
            "source_commit": str(receipt["source_commit"]),
            "source_tree_hash": str(receipt["source_tree_hash"]),
            "receipt_schema": str(receipt["schema"]),
            "receipt_sha256": _sha256(receipt_path),
            "controller_algorithm_id": str(receipt["controller_algorithm_id"]),
            "location": "US",
            "altitude": "1334",
        },
        print_url=True,
    )
    for condition, details in receipt["conditions"].items():  # type: ignore[index]
        for metric, value in details["metrics"].items():  # type: ignore[index]
            experiment[f"{condition.lower()}_{metric}"].append(value)
    for split, condition_runs in receipt["split_conditions"].items():  # type: ignore[index]
        for condition, details in condition_runs.items():  # type: ignore[index]
            for metric, value in details["metrics"].items():  # type: ignore[index]
                experiment[f"{split}_{condition.lower()}_{metric}"].append(value)
    experiment["run_valid"] = str(receipt["run_valid"]).lower()
    experiment["qualification"] = "NOT_QUALIFIED_DEVELOPMENT_ONLY"
    experiment.finalize()


def _write_trajectory_receipts(*, output: Path, benchmark, runs, masks) -> dict[str, dict[str, str]]:
    """Persist replayable condition trajectories outside controller input artifacts."""
    task_by_id = {task.task_id: task for task in benchmark.tasks}
    receipt_dir = output / "trajectory_receipts"
    receipt_dir.mkdir(exist_ok=True)
    records: dict[str, dict[str, str]] = {}
    for condition, run in runs.items():
        path = receipt_dir / f"{condition.lower()}.jsonl"
        lines = []
        for task in run.tasks:
            steps = [asdict(trace) for trace in task.traces]
            payload = {
                "schema": "DAPH_V2B_I3_TRAJECTORY_RECEIPT_V1",
                "task_id": task.task_id,
                "split": task_by_id[task.task_id].split,
                "condition": condition,
                "budget_profile": task_by_id[task.task_id].budget_profile,
                "initial_state_hash": task.traces[0].pre_state_hash if task.traces else None,
                "observation_mask_sha256": masks[condition].sha256(),
                "oracle_value": task.optimal_utility,
                "steps": steps,
                "terminal_result": task.terminal_result,
                "total_action_cost": -sum(
                    trace.action_cost or 0.0 for trace in task.traces
                    if trace.execution_status == "EXECUTED"),
                "trajectory_utility": task.realized_utility,
                "trajectory_regret": task.trajectory_regret,
            }
            lines.append(json.dumps(payload, sort_keys=True))
        path.write_text("\n".join(lines) + "\n")
        records[condition] = {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="v2b-i3-development")
    args = parser.parse_args()
    if (not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                               for char in args.run_id)):
        raise ValueError("run-id must contain only letters, digits, underscores, or hyphens")
    if _git("status", "--porcelain"):
        raise RuntimeError("V2B-I3 development receipts require a clean committed checkout")
    configuration_path = ROOT / "experiments/v2b/configs/v2b_i3_development.json"
    configuration = json.loads(configuration_path.read_text())
    if configuration.get("status") != "DEVELOPMENT_EXPERIMENT_NOT_QUALIFIED":
        raise RuntimeError("V2B-I3 runner only accepts the development-only configuration")
    benchmark_path = ROOT / configuration["benchmark"]["path"]
    policy_path = ROOT / configuration["policy"]["path"]
    masks_path = ROOT / configuration["observation_masks"]["path"]
    benchmark = load_metareasoning_benchmark(benchmark_path)
    policy = load_frozen_policy(policy_path)
    masks = load_observation_masks(masks_path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("V2B-I3 output directory must be empty for one isolated run")
    output.mkdir(parents=True, exist_ok=True)
    protocol = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    runs = {
        condition: protocol.run_condition(
            condition=condition, controller=MatchedMetareasoningController(),
            store_root=output / "cognitive_logs", mask=mask)
        for condition, mask in masks.items()
    }
    blind = runs[STATE_BLIND]
    split_runs = {}
    for split in ("development", "validation", "held_out"):
        split_protocol = V2BMetareasoningExperiment(
            benchmark=benchmark.for_split(split), policy=policy)
        split_runs[split] = {
            condition: split_protocol.run_condition(
                condition=condition, controller=MatchedMetareasoningController(),
                store_root=output / "cognitive_logs" / split, mask=mask)
            for condition, mask in masks.items()
        }
    trajectory_receipts = _write_trajectory_receipts(
        output=output, benchmark=benchmark, runs=runs, masks=masks)
    receipt = {
        "schema": "DAPH_V2B_I3_DEVELOPMENT_RECEIPT_V1",
        "run_valid": True,
        "status": "DEVELOPMENT_NOT_QUALIFIED",
        "claim_boundary": (
            "Deterministic metareasoning protocol validation only; no learned-controller "
            "or scientific V2B result."),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree_hash": _git("rev-parse", "HEAD^{tree}"),
        "configuration": {"path": str(configuration_path.relative_to(ROOT)), "sha256": _sha256(configuration_path)},
        "benchmark": {"path": str(benchmark_path.relative_to(ROOT)), "sha256": _sha256(benchmark_path),
                      "artifact_hashes": dict(benchmark.artifact_hashes)},
        "policy": {"path": str(policy_path.relative_to(ROOT)), "sha256": policy.sha256},
        "observation_masks": {
            "path": str(masks_path.relative_to(ROOT)), "sha256": _sha256(masks_path),
            "masks": {condition: mask.sha256() for condition, mask in masks.items()},
        },
        "controller_algorithm_id": blind.controller_algorithm_id,
        "trajectory_receipts": trajectory_receipts,
        "conditions": {
            condition: {"controller_id": run.controller_id, "metrics": run.metrics}
            for condition, run in runs.items()
        },
        "split_conditions": {
            split: {
                condition: {"controller_id": run.controller_id, "metrics": run.metrics}
                for condition, run in condition_runs.items()
            }
            for split, condition_runs in split_runs.items()
        },
    }
    target = (output / f"{args.run_id}.json").resolve()
    if output not in target.parents:
        raise ValueError("V2B-I3 receipt path must remain under the output directory")
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    _log_development_receipt(receipt, target)
    print(target)


if __name__ == "__main__":
    main()
