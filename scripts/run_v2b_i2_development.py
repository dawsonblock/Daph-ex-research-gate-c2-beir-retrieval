#!/usr/bin/env python3
"""Run the deterministic V2B-I2 development harness; never a qualification run."""
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

from hrm_adaptive_memory.executive.benchmark import load_frozen_benchmark
from hrm_adaptive_memory.executive.controller import (
    DeterministicCognitiveStateController, FixedBaselineController)
from hrm_adaptive_memory.executive.loop import V2BExperimentLoop
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceBudget


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="v2b-i2-development")
    args = parser.parse_args()
    if (not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                               for character in args.run_id)):
        raise ValueError("run-id must contain only letters, digits, underscores, or hyphens")
    if _git("status", "--porcelain"):
        raise RuntimeError("V2B-I2 development receipts require a clean committed checkout")
    configuration_path = ROOT / "experiments/v2b/configs/v2b_i2_development.json"
    configuration = json.loads(configuration_path.read_text())
    if configuration.get("status") != "DEVELOPMENT_EXPERIMENT_NOT_QUALIFIED":
        raise RuntimeError("V2B-I2 runner only accepts the development-only configuration")
    benchmark_path = ROOT / configuration["benchmark"]["path"]
    policy_path = ROOT / configuration["policy"]["path"]
    benchmark, policy = load_frozen_benchmark(benchmark_path), load_frozen_policy(policy_path)
    budget = ResourceBudget(**configuration["resource_budget"])
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("V2B-I2 output directory must be empty for one isolated run")
    output.mkdir(parents=True, exist_ok=True)
    loop = V2BExperimentLoop(policy=policy, budget=budget)
    control = loop.run_condition(benchmark, condition="CONTROL", controller=FixedBaselineController(),
                                 store_root=output / "cognitive_logs")
    v2b = loop.run_condition(benchmark, condition="V2B", controller=DeterministicCognitiveStateController(),
                             store_root=output / "cognitive_logs")
    receipt = {
        "schema": "DAPH_V2B_I2_DEVELOPMENT_RECEIPT_V1",
        "run_valid": True,
        "status": "DEVELOPMENT_NOT_QUALIFIED",
        "claim_boundary": "Synthetic deterministic harness only; no scientific V2B result.",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree_hash": _git("rev-parse", "HEAD^{tree}"),
        "configuration": {"path": str(configuration_path.relative_to(ROOT)), "sha256": _sha256(configuration_path)},
        "benchmark": {"path": str(benchmark_path.relative_to(ROOT)), "sha256": _sha256(benchmark_path)},
        "policy": {"path": str(policy_path.relative_to(ROOT)), "sha256": policy.sha256},
        "conditions": {
            "CONTROL": {"controller_id": control.controller_id, "metrics": control.metrics},
            "V2B": {"controller_id": v2b.controller_id, "metrics": v2b.metrics},
        },
    }
    target = (output / f"{args.run_id}.json").resolve()
    if output not in target.parents:
        raise ValueError("V2B-I2 receipt path must remain under the output directory")
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
