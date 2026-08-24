#!/usr/bin/env python3
"""DAPH I3.4d — Freeze B1 and generate held-out task set.

This script performs two critical operations:

1. Freeze the B1 phase×action table from the R2-DEV-V2 transition corpus.
   The frozen table is serialized to a JSON file with a SHA256 receipt.
   The held-out runner will LOAD this file, never re-fit.

2. Generate a new held-out task set with seed=9137 that has ZERO task-ID
   overlap with the R2 training corpus. This includes verifying that
   base task IDs (stripped of __budget_variant suffixes) also don't overlap.

3. Write a leakage receipt proving the intersection is empty.

Usage:
    PYTHONPATH=scripts:. python3 scripts/i3_4d_freeze_b1_and_generate_heldout.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph.value.empirical import GlobalActionMean, PhaseActionTable
from daph.value.dataset import load_transitions, get_action_value_target


def freeze_b1() -> tuple[str, str]:
    """Freeze B1 from R2 transitions. Returns (file_sha256, model_sha256)."""
    transitions_path = REPO_ROOT / "experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl"
    with open(transitions_path) as f:
        transitions = [json.loads(line) for line in f]

    print(f"Loading {len(transitions)} transitions from {transitions_path}")

    b0 = GlobalActionMean()
    b0.fit(transitions, get_action_value_target)

    b1 = PhaseActionTable(min_samples=3, fallback=b0)
    b1.fit(transitions, get_action_value_target)

    model_sha = b1.sha256()
    print(f"B1 model SHA256: {model_sha}")

    output_path = REPO_ROOT / "experiments/i3_4/value/frozen_b1_table.json"
    file_sha = b1.save(output_path)
    print(f"Frozen B1 written to: {output_path}")
    print(f"File SHA256: {file_sha}")

    # Also save the training task IDs for the leakage receipt
    training_task_ids = set()
    training_base_ids = set()
    for t in transitions:
        tid = t.get("task_id", "")
        if tid:
            training_task_ids.add(tid)
            training_base_ids.add(tid.split("__")[0])

    return file_sha, model_sha, training_task_ids, training_base_ids


def generate_heldout_dataset(
    training_base_ids: set[str],
    seed: int = 9137,
) -> tuple[list[dict], str, str]:
    """Generate a held-out task set with zero task-ID overlap.

    Returns (task_records, dataset_sha256, source_dataset_sha256).
    """
    from r2_dataset_generator import generate_r2_dataset
    from r2_dev_v2_dataset_selector import select_balanced_dataset

    # Generate the full R2 dataset with the new seed
    # Use id_prefix to avoid synthetic ID collision
    # Use exclude_task_ids to filter out any overlapping i3_15c IDs
    print(f"\nGenerating held-out dataset with seed={seed}...")
    all_tasks = generate_r2_dataset(
        n_per_stratum=40, seed=seed,
        id_prefix="ho_",
        exclude_task_ids=training_base_ids,
    )
    print(f"  Generated {len(all_tasks)} tasks")

    # Write to a temporary source file
    source_path = REPO_ROOT / f"experiments/i3_4/datasets/heldout_source_seed{seed}.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    from r2_dataset_generator import write_dataset
    write_dataset(all_tasks, source_path)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    print(f"  Source written to: {source_path}")
    print(f"  Source SHA256: {source_sha}")

    # Select balanced subset
    task_records = select_balanced_dataset(source_path, seed=seed)
    print(f"  Balanced subset: {len(task_records)} tasks")

    # Verify zero overlap
    heldout_task_ids = set(t["task_id"] for t in task_records)
    heldout_base_ids = set(t["task_id"].split("__")[0] for t in task_records)

    task_intersection = training_base_ids & heldout_task_ids
    base_intersection = training_base_ids & heldout_base_ids

    print(f"\nLeakage check:")
    print(f"  Training base IDs: {len(training_base_ids)}")
    print(f"  Held-out task IDs: {len(heldout_task_ids)}")
    print(f"  Held-out base IDs: {len(heldout_base_ids)}")
    print(f"  Task ID intersection: {len(task_intersection)}")
    print(f"  Base ID intersection: {len(base_intersection)}")

    if task_intersection or base_intersection:
        print("\n*** LEAKAGE DETECTED ***")
        if task_intersection:
            print(f"  Overlapping task IDs: {sorted(task_intersection)[:10]}")
        if base_intersection:
            print(f"  Overlapping base IDs: {sorted(base_intersection)[:10]}")
        sys.exit(1)

    print("  ✓ Zero overlap confirmed")

    # Write the held-out balanced dataset
    output_path = REPO_ROOT / f"experiments/i3_4/datasets/heldout_balanced_seed{seed}.jsonl"
    with open(output_path, "w") as f:
        for task in task_records:
            f.write(json.dumps(task, sort_keys=True) + "\n")
    dataset_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"\n  Held-out dataset written to: {output_path}")
    print(f"  Dataset SHA256: {dataset_sha}")

    return task_records, dataset_sha, source_sha


def write_leakage_receipt(
    b1_file_sha: str,
    b1_model_sha: str,
    training_task_ids: set[str],
    training_base_ids: set[str],
    heldout_task_records: list[dict],
    dataset_sha: str,
    source_sha: str,
    seed: int,
):
    """Write a frozen receipt proving no leakage."""
    heldout_task_ids = set(t["task_id"] for t in heldout_task_records)
    heldout_base_ids = set(t["task_id"].split("__")[0] for t in heldout_task_records)

    receipt = {
        "experiment": "i3_4d_heldout",
        "b1_training_task_sha": hashlib.sha256(
            json.dumps(sorted(training_task_ids)).encode()
        ).hexdigest(),
        "b1_training_base_task_sha": hashlib.sha256(
            json.dumps(sorted(training_base_ids)).encode()
        ).hexdigest(),
        "evaluation_task_sha": hashlib.sha256(
            json.dumps(sorted(heldout_task_ids)).encode()
        ).hexdigest(),
        "evaluation_base_task_sha": hashlib.sha256(
            json.dumps(sorted(heldout_base_ids)).encode()
        ).hexdigest(),
        "intersection_count": len(training_base_ids & heldout_task_ids),
        "base_intersection_count": len(training_base_ids & heldout_base_ids),
        "b1_model_sha256": b1_model_sha,
        "b1_file_sha256": b1_file_sha,
        "heldout_dataset_sha256": dataset_sha,
        "heldout_source_sha256": source_sha,
        "heldout_seed": seed,
        "training_n_tasks": len(training_task_ids),
        "training_n_base_tasks": len(training_base_ids),
        "evaluation_n_tasks": len(heldout_task_ids),
        "evaluation_n_base_tasks": len(heldout_base_ids),
    }

    receipt_path = REPO_ROOT / "experiments/i3_4/value/leakage_receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(f"\nLeakage receipt written to: {receipt_path}")
    print(f"  Intersection: {receipt['intersection_count']}")
    print(f"  Base intersection: {receipt['base_intersection_count']}")
    return receipt


def main():
    print("=" * 60)
    print("I3.4d — Freeze B1 and Generate Held-Out Task Set")
    print("=" * 60)

    # Step 1: Freeze B1
    print("\n--- Step 1: Freeze B1 from R2 transitions ---")
    b1_file_sha, b1_model_sha, training_task_ids, training_base_ids = freeze_b1()

    # Step 2: Generate held-out dataset
    print("\n--- Step 2: Generate held-out dataset ---")
    seed = 9137
    task_records, dataset_sha, source_sha = generate_heldout_dataset(
        training_base_ids, seed=seed
    )

    # Step 3: Write leakage receipt
    print("\n--- Step 3: Write leakage receipt ---")
    receipt = write_leakage_receipt(
        b1_file_sha, b1_model_sha,
        training_task_ids, training_base_ids,
        task_records, dataset_sha, source_sha, seed,
    )

    print("\n" + "=" * 60)
    print("DONE — B1 frozen and held-out dataset generated")
    print("=" * 60)
    print(f"\nNext: run scripts/run_i3_4_heldout.py with:")
    print(f"  --dataset experiments/i3_4/datasets/heldout_balanced_seed{seed}.jsonl")
    print(f"  --b1-table experiments/i3_4/value/frozen_b1_table.json")
    print(f"  --generator-seed {seed}")


if __name__ == "__main__":
    main()
