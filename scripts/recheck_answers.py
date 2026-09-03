#!/usr/bin/env python3
"""Re-check all candidates in existing corpora with the fixed answer checker.

The fix normalizes numeric answers so "2" and "2.0" are equivalent.
This re-evaluates is_correct for all candidates in:
  - reasoning_corpus_v2.jsonl (120 tasks)
  - reasoning_corpus_batch3.jsonl (80 tasks)
  - cross_verification/cv_corpus_v2.jsonl (200 tasks with verification)

Also recomputes task-level fields (base_correct, any_correct, etc.)
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import get_reasoning_task, check_answer


def recheck_corpus(path: Path):
    print(f"\nRe-checking {path}")
    if not path.exists():
        print(f"  File not found, skipping")
        return

    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))

    n_changed = 0
    n_cands_changed = 0

    for task in tasks:
        task_obj = get_reasoning_task(task["task_id"])
        if task_obj is None:
            continue

        for cand in task["candidates"]:
            old = cand["is_correct"]
            new = check_answer(cand["answer"], task_obj.answer, task_obj.answer_type)
            if old != new:
                n_cands_changed += 1
                cand["is_correct"] = new

        # Recompute task-level fields
        cands = task["candidates"]
        task["base_correct"] = cands[0]["is_correct"]
        task["any_correct"] = any(c["is_correct"] for c in cands)
        task["rescue_available"] = (not cands[0]["is_correct"]) and task["any_correct"]

        # Recompute majority
        from collections import Counter
        answers = [c["answer"] for c in cands]
        mv = Counter(answers).most_common(1)[0][0]
        task["majority_answer"] = mv
        task["majority_correct"] = any(c["is_correct"] for c in cands if c["answer"] == mv)
        task["n_unique_answers"] = len(set(answers))
        task["agreement_rate"] = max(Counter(answers).values()) / len(answers)

    # Write back
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, default=str) + "\n")

    print(f"  Tasks: {len(tasks)}, candidates changed: {n_cands_changed}")


def main():
    base = REPO_ROOT / "experiments/daph_x"

    corpora = [
        base / "reasoning/reasoning_corpus_v2.jsonl",
        base / "reasoning/reasoning_corpus_batch3.jsonl",
        base / "cross_verification/cv_corpus_v2.jsonl",
        base / "cross_verification/input_corpus_200.jsonl",
    ]

    for path in corpora:
        recheck_corpus(path)

    print("\nDone. All corpora re-checked with normalized answer matching.")


if __name__ == "__main__":
    main()
