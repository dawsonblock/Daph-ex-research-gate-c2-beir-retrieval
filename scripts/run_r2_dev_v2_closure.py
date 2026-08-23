#!/usr/bin/env python3
"""
R2-DEV-V2 Closure Rerun — 60 missing budget-variant trajectories only.

This is a targeted rerun of the 60 trajectories that failed due to the
ResourceState.consume_retrieval/consume_search API mismatch. The fix
constructs ResourceState with retrieval_calls_used/search_calls_used
pre-set to the maximum, simulating prior exhaustion.

The original 260 completed trajectories are preserved untouched.
The original 60 errors are quarantined in errors.jsonl.

This rerun writes to a separate closure/ directory. After verification,
the closure results are merged with the original 260 to produce the
final 320/320 dataset.

Usage:
    PYTHONPATH=scripts:. python3 scripts/run_r2_dev_v2_closure.py \
        --dataset /path/to/balanced_dataset.jsonl \
        --output /path/to/closure/ \
        --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --original-results /path/to/original/results.jsonl \
        --original-schedule /path/to/original/execution_schedule.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from r2_allowed_actions import ALL_ARMS, C0, D, E, DE, R2Arm
from r2_backend_identity import R2_POLICY_BACKEND_V2


def main():
    import argparse

    parser = argparse.ArgumentParser(description="R2-DEV-V2 Closure Rerun")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for closure results")
    parser.add_argument("--gguf-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--original-results", type=Path, required=True,
                        help="Path to original results.jsonl")
    parser.add_argument("--original-schedule", type=Path, required=True,
                        help="Path to original execution_schedule.json")
    args = parser.parse_args()

    # Compute identity
    dataset_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    backend_sha = R2_POLICY_BACKEND_V2.identity_sha256()

    # Load original completed keys
    completed_keys = set()
    with open(args.original_results) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "completed":
                completed_keys.add(r["trajectory_key"])

    print(f"Original completed: {len(completed_keys)}")

    # Load schedule
    with open(args.original_schedule) as f:
        schedule = json.load(f)

    # Find missing entries
    missing = []
    for entry in schedule["schedule"]:
        task_idx = entry["task_index"]
        task_id = entry["task_id"]
        arm = entry["arm"]
        key = f"{dataset_sha}|{task_id}|{arm}|{backend_sha}"
        if key not in completed_keys:
            missing.append((entry, key))

    print(f"Missing trajectories to rerun: {len(missing)}")
    print()

    # Load dataset
    with open(args.dataset) as f:
        task_records = [json.loads(line) for line in f]

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write closure manifest
    manifest = {
        "closure_type": "budget_variant_rerun",
        "reason": "ResourceState.consume_retrieval/consume_search API mismatch",
        "fix": "Construct ResourceState with retrieval_calls_used/search_calls_used pre-set",
        "original_completed": len(completed_keys),
        "closure_trajectories": len(missing),
        "dataset_sha256": dataset_sha,
        "backend_identity_sha256": backend_sha,
        "run_start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.output / "closure_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Initialize backend
    print("Initializing R2DirectLlamaBackend...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
    )
    print("Backend initialized.")

    # Load i3_7e
    print("Loading i3_7e...")
    from run_r2_dev_v2 import _load_i3_7e, run_trajectory, IncrementalWriter
    i3_7e = _load_i3_7e()
    print("i3_7e loaded.")

    # Open incremental writers for closure
    writer = IncrementalWriter(args.output)

    # Run missing trajectories
    print(f"\nStarting closure rerun: {len(missing)} trajectories")
    print("=" * 60)

    completed = 0
    failed = 0
    start_time = time.time()

    arm_lookup = {"C0": C0, "D": D, "E": E, "DE": DE}

    for i, (entry, traj_key) in enumerate(missing):
        task_idx = entry["task_index"]
        task_id = entry["task_id"]
        arm_name = entry["arm"]
        arm = arm_lookup[arm_name]
        task_record = task_records[task_idx]

        elapsed = time.time() - start_time
        if completed > 0:
            eta = elapsed / completed * (len(missing) - completed)
        else:
            eta = 0
        print(f"[{i+1}/{len(missing)}] task={task_id} arm={arm_name} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        try:
            result = run_trajectory(
                task_record=task_record,
                arm=arm,
                backend=backend,
                writer=writer,
                trajectory_key=traj_key,
                max_tokens=args.max_tokens,
                i3_7e=i3_7e,
                task_id=task_id,
            )
            completed += 1
            if result.status != "completed":
                failed += 1
                print(f"  → {result.status}: {result.terminal_result}")
            else:
                print(f"  → utility={result.realized_utility}, "
                      f"terminal={result.terminal_action}, steps={result.steps}")
        except Exception as exc:
            failed += 1
            print(f"  → ERROR: {type(exc).__name__}: {exc}")
            error_record = {
                "trajectory_key": traj_key,
                "task_id": task_id,
                "arm": arm_name,
                "error": type(exc).__name__,
                "error_message": str(exc),
            }
            writer.write_error(error_record)

        writer.write_progress({
            "completed": completed,
            "failed": failed,
            "total": len(missing),
            "remaining": len(missing) - completed - failed,
            "elapsed_seconds": time.time() - start_time,
        })

    writer.close()

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("CLOSURE RERUN COMPLETE")
    print("=" * 60)
    print(f"  Total:    {len(missing)}")
    print(f"  Completed: {completed}")
    print(f"  Failed:   {failed}")
    print(f"  Time:     {total_time:.0f}s")

    # Update manifest
    manifest["run_end_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["closure_completed"] = completed
    manifest["closure_failed"] = failed
    manifest["total_time_seconds"] = total_time

    with open(args.output / "closure_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\nClosure results in: {args.output}")


if __name__ == "__main__":
    main()
