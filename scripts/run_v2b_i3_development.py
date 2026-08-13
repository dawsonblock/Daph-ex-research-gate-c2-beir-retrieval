#!/usr/bin/env python3
"""Run V2B-I3's development-only metareasoning validity protocol."""
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
from hrm_adaptive_memory.executive.metareasoning_controller import MatchedMetareasoningController
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
    experiment["run_valid"] = str(receipt["run_valid"]).lower()
    experiment["qualification"] = "NOT_QUALIFIED_DEVELOPMENT_ONLY"
    experiment.finalize()


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
    benchmark = load_metareasoning_benchmark(benchmark_path)
    policy = load_frozen_policy(policy_path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("V2B-I3 output directory must be empty for one isolated run")
    output.mkdir(parents=True, exist_ok=True)
    protocol = V2BMetareasoningExperiment(benchmark=benchmark, policy=policy)
    blind = protocol.run_condition(
        condition=STATE_BLIND, controller=MatchedMetareasoningController(),
        store_root=output / "cognitive_logs")
    aware = protocol.run_condition(
        condition=STATE_AWARE, controller=MatchedMetareasoningController(),
        store_root=output / "cognitive_logs")
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
        "benchmark": {"path": str(benchmark_path.relative_to(ROOT)), "sha256": _sha256(benchmark_path)},
        "policy": {"path": str(policy_path.relative_to(ROOT)), "sha256": policy.sha256},
        "controller_algorithm_id": blind.controller_algorithm_id,
        "conditions": {
            STATE_BLIND: {"controller_id": blind.controller_id, "metrics": blind.metrics},
            STATE_AWARE: {"controller_id": aware.controller_id, "metrics": aware.metrics},
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
