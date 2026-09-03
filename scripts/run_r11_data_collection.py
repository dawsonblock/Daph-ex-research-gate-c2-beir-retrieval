#!/usr/bin/env python3
"""Generate extra candidates (7-12) for all 200 reasoning tasks.

Preserves candidate order: a_1, ..., a_6 (existing), a_7, ..., a_12 (new).
Uses varied temperatures for diversity:
  7: temp=0.3, 8: temp=0.5, 9: temp=0.7, 10: temp=0.9, 11: temp=1.0, 12: temp=1.2

Output: experiments/daph_x/r11/r11_corpus_12.jsonl

Usage:
    python scripts/run_r11_data_collection.py \\
        --corpus experiments/daph_x/cross_verification/cv_corpus_v2.jsonl \\
        --n_tasks 200 --n_extra 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.reasoning_tasks import get_reasoning_task, check_answer

R11_DIR = REPO_ROOT / "experiments/daph_x/r11"

# Import extract_answer from collection script
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_reasoning_collection import extract_answer


def load_corpus(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def estimate_confidence(response: str) -> float:
    """Extract confidence from response text."""
    match = re.search(r'(\d{1,3})\s*%', response)
    if match:
        return float(match.group(1))
    # Look for "confident" / "not sure" language
    resp_lower = response.lower()
    if "very confident" in resp_lower or "certain" in resp_lower:
        return 90.0
    if "confident" in resp_lower:
        return 70.0
    if "not sure" in resp_lower or "uncertain" in resp_lower:
        return 30.0
    if "guess" in resp_lower:
        return 20.0
    return 50.0


# Temperature schedule for extra candidates
EXTRA_TEMPS = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R11_DIR / "r11_corpus_12.jsonl"))
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--n_extra", type=int, default=6)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    R11_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    tasks = tasks[args.start_idx:args.start_idx + args.n_tasks]
    print(f"R11 data collection: generating {args.n_extra} extra candidates")
    print(f"  Tasks: {len(tasks)} (from idx {args.start_idx})")
    print(f"  Target: {6 + args.n_extra} candidates per task")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else R11_DIR / "r11_corpus_12.jsonl"

    # Check if output exists and resume — only skip tasks that ALREADY have 12 candidates
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                t = json.loads(line)
                existing[t["task_id"]] = t
        n_done = sum(1 for t in existing.values() if len(t["candidates"]) >= 6 + args.n_extra)
        print(f"  Resuming: {n_done} tasks already have {6 + args.n_extra} candidates")

    # Process tasks: update in memory, rewrite entire file after each task
    n_to_do = sum(1 for task in tasks
                  if not (task["task_id"] in existing
                          and len(existing[task["task_id"]]["candidates"]) >= 6 + args.n_extra))
    print(f"  Need to generate extra candidates for {n_to_do} tasks")

    for idx, task in enumerate(tasks):
        task_id = task["task_id"]

        # Skip only if already done (has enough candidates)
        if task_id in existing and len(existing[task_id]["candidates"]) >= 6 + args.n_extra:
            continue

        # Start from existing task if it has some candidates, else use corpus task
        if task_id in existing:
            task = existing[task_id]

        task_obj = get_reasoning_task(task_id)
        if task_obj is None:
            print(f"  SKIP {task_id}: task not found")
            continue

        cands = task["candidates"]
        n_current = len(cands)
        n_target = 6 + args.n_extra

        print(f"[{idx+1}/{len(tasks)}] {task_id}: {n_current} → {n_target} candidates", flush=True)

        t0 = time.monotonic()

        # Generate extra candidates
        for i in range(args.n_extra):
            temp = EXTRA_TEMPS[i % len(EXTRA_TEMPS)]
            seed = args.seed + idx * 10000 + i * 137 + 700

            try:
                response = model.generate_raw(
                    prompt=task_obj.prompt,
                    temperature=temp,
                    max_tokens=300,
                    seed=seed,
                )

                answer = extract_answer(response)
                is_correct = check_answer(answer, task_obj.answer, task_obj.answer_type)
                conf = estimate_confidence(response)

                cands.append({
                    "answer": answer,
                    "is_correct": is_correct,
                    "self_confidence": conf,
                    "response": response,
                    "temperature": temp,
                    "seed": seed,
                    "candidate_id": f"cand_{n_current + i}",
                    "task_id": task_id,
                })
            except Exception as e:
                print(f"    Error generating candidate {n_current + i}: {e}", flush=True)
                cands.append({
                    "answer": "",
                    "is_correct": False,
                    "self_confidence": 0.0,
                    "response": f"Error: {e}",
                    "temperature": temp,
                    "seed": seed,
                    "candidate_id": f"cand_{n_current + i}",
                    "task_id": task_id,
                })

        task["candidates"] = cands
        task["n_candidates"] = len(cands)
        existing[task_id] = task

        elapsed = time.monotonic() - t0
        n_correct = sum(1 for c in cands if c["is_correct"])
        print(f"  Done in {elapsed:.1f}s, {n_correct}/{len(cands)} correct", flush=True)

        # Rewrite entire file with all tasks (avoids duplicates)
        with open(output_path, "w") as f:
            for t in existing.values():
                f.write(json.dumps(t, default=str) + "\n")

    print(f"\nOutput: {output_path} ({len(existing)} tasks)")


if __name__ == "__main__":
    main()
