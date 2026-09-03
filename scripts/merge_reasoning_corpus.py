#!/usr/bin/env python3
"""Merge reasoning corpus v2 (120 tasks) + batch3 (80 tasks) → 200-task corpus."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REASONING_DIR = REPO_ROOT / "experiments/daph_x/reasoning"
CV_DIR = REPO_ROOT / "experiments/daph_x/cross_verification"

def load_jsonl(path):
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks

def main():
    v2 = load_jsonl(REASONING_DIR / "reasoning_corpus_v2.jsonl")
    batch3 = load_jsonl(REASONING_DIR / "reasoning_corpus_batch3.jsonl")

    print(f"V2 corpus: {len(v2)} tasks")
    print(f"Batch3 corpus: {len(batch3)} tasks")

    # Check for overlap
    v2_ids = {t["task_id"] for t in v2}
    batch3_ids = {t["task_id"] for t in batch3}
    overlap = v2_ids & batch3_ids
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping tasks: {sorted(overlap)[:5]}")

    merged = v2 + batch3
    print(f"Merged: {len(merged)} tasks")

    output_path = CV_DIR / "input_corpus_200.jsonl"
    with open(output_path, "w") as f:
        for task in merged:
            f.write(json.dumps(task, default=str) + "\n")
    print(f"Written to {output_path}")

if __name__ == "__main__":
    main()
