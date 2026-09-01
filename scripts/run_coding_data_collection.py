#!/usr/bin/env python3
"""Collect coding experiment data for training coding-specific authority models.

For each task:
  1. Generate N candidates at different temperatures
  2. Execute ALL candidates (not just base and DAPH-X picks)
  3. Record features, execution results, and pairwise ΔU labels

This produces the labeled data needed to train:
  - Q_res (residual value correction on coding features)
  - Pairwise advantage model (ΔU prediction)
  - Risk model (is_harmful prediction)
  - Conformal calibrator

Output: experiments/daph_x/coding/coding_corpus.jsonl

Usage:
    python scripts/run_coding_data_collection.py \\
        --model_path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        --n_candidates 4 \\
        --n_tasks 80 \\
        --max_tokens 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.tasks import get_all_tasks, CodingTask
from daph_x.coding.code_executor import execute_solution, ExecutionResult
from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb

OUTPUT_DIR = REPO_ROOT / "experiments/daph_x/coding"


def collect_task_data(
    model: CodingModelInterface,
    task: CodingTask,
    n_candidates: int,
    max_tokens: int,
) -> dict:
    """Collect data for a single task: generate candidates, execute all, record everything."""
    print(f"\n{'='*60}")
    print(f"  Task: {task.task_id} ({task.difficulty})")
    print(f"  {task.description}")
    print(f"{'='*60}")

    # Generate candidates
    print(f"  Generating {n_candidates} candidates...")
    t0 = time.monotonic()
    candidates = model.generate_candidates(
        task=task,
        n_candidates=n_candidates,
        max_tokens=max_tokens,
    )
    gen_time = time.monotonic() - t0
    print(f"  Generation took {gen_time:.1f}s ({gen_time/n_candidates:.1f}s/candidate)")

    # Execute all candidates
    print(f"  Executing all {len(candidates)} candidates...")
    records = []
    for cand in candidates:
        result = execute_solution(task, cand.solution_code, timeout_seconds=10.0)
        features = extract_code_features(cand.solution_code, task)
        q_mb = compute_q_mb(features, task)

        record = {
            "task_id": task.task_id,
            "task_description": task.description,
            "difficulty": task.difficulty,
            "function_name": task.function_name,
            "candidate_id": cand.candidate_id,
            "temperature": cand.temperature,
            "prompt_variant": cand.prompt_variant,
            "solution_code": cand.solution_code,
            "features": features,
            "q_mb": q_mb,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "pass_rate": result.pass_rate,
            "utility": result.utility,
            "execution_time_ms": result.execution_time_ms,
            "error": result.error,
            "latency_ms": cand.latency_ms,
        }
        records.append(record)

        status = f"{result.tests_passed}/{result.tests_total}"
        if result.error:
            print(f"    {cand.candidate_id}: {status} tests, ERROR: {result.error[:60]}")
        else:
            print(f"    {cand.candidate_id}: {status} tests, utility={result.utility:.2f}")

    # Identify base action (first candidate, temp=0)
    base = records[0]
    base_utility = base["utility"]

    # Compute pairwise ΔU for each candidate vs base
    for r in records:
        r["delta_u_vs_base"] = r["utility"] - base_utility
        r["is_better_than_base"] = r["delta_u_vs_base"] > 0.5
        r["is_worse_than_base"] = r["delta_u_vs_base"] < -0.5
        r["is_harmful"] = r["is_worse_than_base"]  # For risk model

    # Find best candidate (oracle)
    best = max(records, key=lambda r: r["utility"])
    base_rank = sorted(records, key=lambda r: -r["utility"]).index(base) + 1

    # Disagreement: does any non-base candidate have higher utility?
    disagreements = [r for r in records[1:] if r["delta_u_vs_base"] > 0.5]
    n_disagree = len(disagreements)

    # DAPH-X pick (highest Q_MB, since we don't have Q_res yet)
    daphx_pick = max(records, key=lambda r: r["q_mb"])
    daphx_disagrees = daphx_pick["candidate_id"] != base["candidate_id"]

    task_summary = {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "n_candidates": len(records),
        "base_utility": base_utility,
        "base_tests": f"{base['tests_passed']}/{base['tests_total']}",
        "best_utility": best["utility"],
        "best_candidate_id": best["candidate_id"],
        "base_rank_by_utility": base_rank,
        "n_disagreements": n_disagree,
        "daphx_picks_base": not daphx_disagrees,
        "daphx_candidate_id": daphx_pick["candidate_id"],
        "daphx_utility": daphx_pick["utility"],
        "daphx_delta_u": daphx_pick["utility"] - base_utility,
        "gen_time_s": gen_time,
        "candidates": records,
    }

    print(f"\n  Summary:")
    print(f"    Base utility: {base_utility:.2f} (rank {base_rank} by utility)")
    print(f"    Best utility: {best['utility']:.2f} ({best['candidate_id']})")
    print(f"    DAPH-X pick:  {daphx_pick['candidate_id']} (Q_MB={daphx_pick['q_mb']:.1f})")
    print(f"    Disagreements (ΔU > 0.5): {n_disagree}")
    if daphx_disagrees:
        du = daphx_pick["utility"] - base_utility
        outcome = "RESCUE" if du > 0.5 else ("BREAK" if du < -0.5 else "TIE")
        print(f"    DAPH-X vs base: ΔU={du:+.2f} → {outcome}")

    return task_summary


def main():
    parser = argparse.ArgumentParser(description="Collect coding experiment data")
    parser.add_argument(
        "--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    )
    parser.add_argument("--n_candidates", type=int, default=4)
    parser.add_argument("--n_tasks", type=int, default=80)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks
    all_tasks = get_all_tasks()
    if args.difficulty:
        all_tasks = [t for t in all_tasks if t.difficulty == args.difficulty]
    tasks = all_tasks[args.start_idx:args.start_idx + args.n_tasks]
    print(f"Tasks: {len(tasks)} (start_idx={args.start_idx}, difficulty={args.difficulty or 'all'})")
    print(f"Candidates per task: {args.n_candidates}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Model: {args.model_path}")
    print()

    # Initialize model
    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    # Collect data
    all_summaries = []
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "coding_corpus.jsonl"

    # Write JSONL (one line per task, flush after each)
    with open(output_path, "w") as f:
        for task_idx, task in enumerate(tasks):
            print(f"\n[Task {task_idx+1}/{len(tasks)}]")
            try:
                summary = collect_task_data(
                    model=model,
                    task=task,
                    n_candidates=args.n_candidates,
                    max_tokens=args.max_tokens,
                )
                all_summaries.append(summary)
                # Write to JSONL
                f.write(json.dumps(summary, default=str) + "\n")
                f.flush()
            except Exception as e:
                print(f"  ERROR collecting task {task.task_id}: {e}")
                import traceback
                traceback.print_exc()

    # Overall summary
    total_tasks = len(all_summaries)
    total_candidates = sum(s["n_candidates"] for s in all_summaries)
    total_disagree = sum(s["n_disagreements"] for s in all_summaries)
    daphx_disagree = sum(1 for s in all_summaries if not s["daphx_picks_base"])
    daphx_rescues = sum(1 for s in all_summaries if s["daphx_delta_u"] > 0.5)
    daphx_breaks = sum(1 for s in all_summaries if s["daphx_delta_u"] < -0.5)

    print(f"\n{'='*60}")
    print(f"  DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {total_tasks}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Total disagreements (ΔU > 0.5): {total_disagree}")
    print(f"  DAPH-X disagreements: {daphx_disagree}")
    print(f"  DAPH-X rescues: {daphx_rescues}")
    print(f"  DAPH-X breaks: {daphx_breaks}")
    print(f"  DAPH-X ties: {daphx_disagree - daphx_rescues - daphx_breaks}")
    print(f"\n  Per-difficulty:")
    for diff in ["easy", "medium", "hard"]:
        diff_tasks = [s for s in all_summaries if s["difficulty"] == diff]
        if diff_tasks:
            d_disagree = sum(1 for s in diff_tasks if not s["daphx_picks_base"])
            d_rescue = sum(1 for s in diff_tasks if s["daphx_delta_u"] > 0.5)
            d_break = sum(1 for s in diff_tasks if s["daphx_delta_u"] < -0.5)
            print(f"    {diff}: {len(diff_tasks)} tasks, {d_disagree} disagree, {d_rescue} rescue, {d_break} break")

    # Save summary
    summary_path = OUTPUT_DIR / "coding_corpus_summary.json"
    summary = {
        "model": model.model_name,
        "model_hash": model.model_hash,
        "n_tasks": total_tasks,
        "n_candidates_per_task": args.n_candidates,
        "total_candidates": total_candidates,
        "total_disagreements": total_disagree,
        "daphx_disagreements": daphx_disagree,
        "daphx_rescues": daphx_rescues,
        "daphx_breaks": daphx_breaks,
        "corpus_path": str(output_path),
        "task_summaries": [
            {
                "task_id": s["task_id"],
                "difficulty": s["difficulty"],
                "base_utility": s["base_utility"],
                "best_utility": s["best_utility"],
                "base_rank": s["base_rank_by_utility"],
                "n_disagreements": s["n_disagreements"],
                "daphx_picks_base": s["daphx_picks_base"],
                "daphx_delta_u": s["daphx_delta_u"],
            }
            for s in all_summaries
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Corpus: {output_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
