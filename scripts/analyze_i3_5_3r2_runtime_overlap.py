#!/usr/bin/env python3
"""Analyze overlap between runtime replay disagreements and training fork dataset.

Determines how many runtime states exactly overlap the training/fork corpus.
This closes the question of whether the positive runtime predictions are
simply known training patterns or new state-distribution variants.

Canonical state/pair identity:
  task_id + step_id + base_action + governor_action

Usage:
    PYTHONPATH=. python scripts/analyze_i3_5_3r2_runtime_overlap.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def state_pair_key(task_id: str, step_id: int, base_action: str, gov_action: str) -> str:
    """Canonical identity for a state/pair observation."""
    return f"{task_id}|step{step_id}|{base_action}->{gov_action}"


def main():
    parser = argparse.ArgumentParser(description="Analyze runtime/training overlap")
    parser.add_argument(
        "--replay-evaluations",
        default="experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c/replay/gate_evaluations_tau5_margin5.jsonl",
    )
    parser.add_argument(
        "--fork-dataset",
        default="experiments/v2b_i3_5_2/development/i353r1/expanded_fork_dataset_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/v2b_i3_5_2/development/i353r2_closure/runtime_overlap_analysis.json",
    )
    args = parser.parse_args()

    # Load runtime replay evaluations (disagreements only)
    print(f"Loading replay evaluations from {args.replay_evaluations}...")
    runtime_evals: list[dict[str, Any]] = []
    with open(args.replay_evaluations) as f:
        for line in f:
            ev = json.loads(line)
            if ev["decision"] != "SKIP_SAME_ACTION":
                runtime_evals.append(ev)
    print(f"  Runtime disagreements: {len(runtime_evals)}")

    # Load training fork dataset
    print(f"Loading fork dataset from {args.fork_dataset}...")
    training_records: list[dict[str, Any]] = []
    with open(args.fork_dataset) as f:
        for line in f:
            training_records.append(json.loads(line))
    print(f"  Training disagreements: {len(training_records)}")

    # Build canonical key sets
    runtime_keys = set()
    runtime_positive_keys = set()
    for ev in runtime_evals:
        key = state_pair_key(ev["task_id"], ev["step_id"],
                             ev["base_action"], ev["governor_action"])
        runtime_keys.add(key)
        if ev["predicted_delta_q_pi"] > 0:
            runtime_positive_keys.add(key)

    training_keys = set()
    training_positive_keys = set()
    for rec in training_records:
        key = state_pair_key(rec["task_id"], rec["step_id"],
                             rec["base_action"], rec["gov_action"])
        training_keys.add(key)
        if rec["delta_q_pi"] > 0:
            training_positive_keys.add(key)

    # Overlap
    exact_overlap = runtime_keys & training_keys
    runtime_only = runtime_keys - training_keys
    training_only = training_keys - runtime_keys

    # Positive runtime predictions: seen in training vs OOD
    positive_runtime_seen_in_training = runtime_positive_keys & training_keys
    positive_runtime_ood = runtime_positive_keys - training_keys

    print(f"\n{'='*78}")
    print("RUNTIME / TRAINING OVERLAP ANALYSIS")
    print(f"{'='*78}")
    print(f"  Runtime disagreements:                    {len(runtime_keys)}")
    print(f"  Training disagreements:                   {len(training_keys)}")
    print(f"  Exact pair overlap:                       {len(exact_overlap)}")
    print(f"  Runtime-only:                             {len(runtime_only)}")
    print(f"  Training-only:                            {len(training_only)}")
    print(f"  Positive runtime predictions:             {len(runtime_positive_keys)}")
    print(f"    seen in training:                       {len(positive_runtime_seen_in_training)}")
    print(f"    OOD (not in training):                  {len(positive_runtime_ood)}")

    # By action pair
    print(f"\n--- By action pair ---")
    pair_stats = defaultdict(lambda: {"runtime": 0, "training": 0, "overlap": 0,
                                       "runtime_positive": 0, "overlap_positive": 0})
    for ev in runtime_evals:
        pair = f"{ev['base_action']}->{ev['governor_action']}"
        key = state_pair_key(ev["task_id"], ev["step_id"],
                             ev["base_action"], ev["governor_action"])
        pair_stats[pair]["runtime"] += 1
        if ev["predicted_delta_q_pi"] > 0:
            pair_stats[pair]["runtime_positive"] += 1
        if key in training_keys:
            pair_stats[pair]["overlap"] += 1
        if key in training_keys and ev["predicted_delta_q_pi"] > 0:
            pair_stats[pair]["overlap_positive"] += 1

    for rec in training_records:
        pair = f"{rec['base_action']}->{rec['gov_action']}"
        pair_stats[pair]["training"] += 1

    for pair, stats in sorted(pair_stats.items(), key=lambda x: -x[1]["runtime"]):
        print(f"  {pair}: runtime={stats['runtime']}, training={stats['training']}, "
              f"overlap={stats['overlap']}, "
              f"runtime_pos={stats['runtime_positive']}, "
              f"overlap_pos={stats['overlap_positive']}")

    # Compute provenance
    def file_sha(path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            return "MISSING"
        return hashlib.sha256(p.read_bytes()).hexdigest()

    replay_sha = file_sha(args.replay_evaluations)
    fork_sha = file_sha(args.fork_dataset)
    script_sha = file_sha(__file__)

    import subprocess
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        source_commit = "UNKNOWN"

    # Save
    result = {
        "schema": "DAPH_V2B_I3_5_3R2_RUNTIME_OVERLAP_V1",
        "runtime_disagreements": len(runtime_keys),
        "training_disagreements": len(training_keys),
        "exact_pair_overlap": len(exact_overlap),
        "runtime_only": len(runtime_only),
        "training_only": len(training_only),
        "positive_runtime_predictions": len(runtime_positive_keys),
        "positive_runtime_predictions_seen_in_training": len(positive_runtime_seen_in_training),
        "positive_runtime_predictions_ood": len(positive_runtime_ood),
        "by_action_pair": {
            pair: {k: v for k, v in stats.items()}
            for pair, stats in sorted(pair_stats.items())
        },
        "provenance": {
            "replay_evaluations_sha256": replay_sha,
            "fork_dataset_sha256": fork_sha,
            "analysis_script_sha256": script_sha,
            "source_commit": source_commit,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
