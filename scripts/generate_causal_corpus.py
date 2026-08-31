#!/usr/bin/env python3
"""Generate the first DAPH-X causal training corpus.

For each task in the OOD pool, creates a checkpoint and evaluates
all candidate actions exhaustively. Produces CausalActionRecordV1 records.

Usage:
    python scripts/generate_causal_corpus.py [--n-tasks 140] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.receipts.checkpoint import checkpoint_from_task_and_runtime
from daph_x.receipts.causal_dataset import (
    build_causal_dataset, write_causal_dataset, group_by_checkpoint,
)
from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task
from build_structural_ood_pool import OOD_DOMAIN_TEMPLATES, generate_ood_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = REPO_ROOT / "experiments/daph_x/causal_corpus"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "causal_corpus_v1.jsonl"

    all_records = []
    task_count = 0

    # Generate tasks from each template
    n_per_template = args.n_tasks // len(OOD_DOMAIN_TEMPLATES)
    remainder = args.n_tasks % len(OOD_DOMAIN_TEMPLATES)

    print(f"Generating causal corpus: {args.n_tasks} tasks, seed={args.seed}")
    print(f"  Templates: {len(OOD_DOMAIN_TEMPLATES)}")
    print(f"  Tasks per template: {n_per_template}")

    for t_idx, template in enumerate(OOD_DOMAIN_TEMPLATES):
        n = n_per_template + (1 if t_idx < remainder else 0)
        for i in range(n):
            task = generate_ood_candidate(template, i)
            checkpoint = checkpoint_from_task_and_runtime(task, None, seed=args.seed)
            graph = build_graph_from_evidence_task(task)
            candidates = generate_and_prune(graph)

            if not candidates:
                print(f"  WARNING: No candidates for {task.task_id}")
                continue

            records = build_causal_dataset(checkpoint, candidates, seed=args.seed)
            all_records.extend(records)
            task_count += 1

            if task_count % 20 == 0:
                print(f"  Progress: {task_count} tasks, {len(all_records)} records")

    # Write
    write_causal_dataset(all_records, output_path)
    print(f"\nWrote {len(all_records)} records from {task_count} tasks to {output_path}")

    # Summary
    groups = group_by_checkpoint([r.to_dict() for r in all_records])
    print(f"\nSummary:")
    print(f"  Tasks: {task_count}")
    print(f"  Records: {len(all_records)}")
    print(f"  Counterfactual groups: {len(groups)}")
    print(f"  Avg actions per group: {len(all_records) / max(1, len(groups)):.1f}")

    # Utility stats
    utilities = [r.utility for r in all_records]
    regrets = [r.regret for r in all_records]
    near_opt = sum(1 for r in all_records if r.is_near_optimal)
    print(f"\n  Utility: min={min(utilities):.1f}, max={max(utilities):.1f}, mean={sum(utilities)/len(utilities):.1f}")
    print(f"  Regret: min={min(regrets):.1f}, max={max(regrets):.1f}, mean={sum(regrets)/len(regrets):.1f}")
    print(f"  Near-optimal: {near_opt}/{len(all_records)} ({100*near_opt/len(all_records):.1f}%)")

    # Action type breakdown
    from collections import Counter
    action_types = Counter(r.action_type for r in all_records)
    print(f"\n  Action types:")
    for at, count in sorted(action_types.items()):
        print(f"    {at}: {count}")

    # Success by action type
    print(f"\n  Success by action type:")
    for at in sorted(action_types.keys()):
        type_records = [r for r in all_records if r.action_type == at]
        n_success = sum(1 for r in type_records if r.success)
        print(f"    {at}: {n_success}/{len(type_records)} ({100*n_success/max(1,len(type_records)):.1f}%)")


if __name__ == "__main__":
    main()
