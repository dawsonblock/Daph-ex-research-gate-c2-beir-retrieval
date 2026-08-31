#!/usr/bin/env python3
"""DAPH-X coding agent experiment.

The single experiment that turns DAPH-X from "interesting research
infrastructure" into "something potentially useful":

  1. Take a real model (Qwen2.5-7B-Instruct-Q4_K_M) on real coding tasks
  2. Generate multiple candidate solutions per task
  3. DAPH-X independently ranks candidates using Q_MB + Q_res
  4. When DAPH-X's top choice ≠ model's first choice → fork
  5. Execute both solutions, run unit tests
  6. Measure ΔU = U_DAPH - U_base

Reports:
  - Number of disagreements
  - Rescue rate (DAPH-X better)
  - Break rate (DAPH-X worse)
  - Force precision
  - Per-task breakdown

Usage:
    python scripts/run_coding_experiment.py \\
        --model_path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        --n_candidates 4 \\
        --n_tasks 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.tasks import get_all_tasks, CodingTask
from daph_x.coding.code_executor import execute_solution, ExecutionResult
from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.daphx_ranker import (
    rank_candidates,
    identify_fork,
    extract_code_features,
    compute_q_mb,
    ForkDecision,
)

M4_DIR = REPO_ROOT / "experiments/daph_x/m4"
OUTPUT_DIR = REPO_ROOT / "experiments/daph_x/coding"


def run_experiment(
    model_path: str,
    n_candidates: int = 4,
    n_tasks: int = 20,
    n_gpu_layers: int = -1,
    seed: int = 42,
    start_idx: int = 0,
    difficulty: str | None = None,
):
    """Run the coding agent experiment."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks
    all_tasks = get_all_tasks()
    # Filter by difficulty if specified
    if difficulty:
        all_tasks = [t for t in all_tasks if t.difficulty == difficulty]
    tasks = all_tasks[start_idx:start_idx + n_tasks]
    print(f"Tasks: {len(tasks)} (start_idx={start_idx}, difficulty={difficulty or 'all'})")
    print(f"Candidates per task: {n_candidates}")
    print(f"Model: {model_path}")
    print()

    # Load Q_res model if available
    q_res_model = None
    q_res_feature_keys = None
    q_res_path = M4_DIR / "q_res_m4.pkl"
    if q_res_path.exists():
        import joblib
        q_res_data = joblib.load(q_res_path)
        q_res_model = q_res_data["model"]
        q_res_feature_keys = q_res_data["feature_keys"]
        print(f"Loaded Q_res model from {q_res_path}")
    else:
        print("No Q_res model found — using Q_MB only")

    # Initialize model interface
    model = CodingModelInterface(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
    )

    # Run experiment
    all_results = []
    disagreements = 0
    rescues = 0
    breaks = 0
    ties = 0

    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"  Task {task_idx+1}/{len(tasks)}: {task.task_id}")
        print(f"  {task.description}")
        print(f"{'='*60}")

        # Generate candidates
        print(f"  Generating {n_candidates} candidates...")
        candidates = model.generate_candidates(
            task=task,
            n_candidates=n_candidates,
            max_tokens=512,
        )

        print(f"  Generated {len(candidates)} candidates:")
        for c in candidates:
            code_preview = c.solution_code[:80].replace("\n", " ") if c.solution_code else "(empty)"
            print(f"    {c.candidate_id}: temp={c.temperature}, variant={c.prompt_variant}, code={code_preview}...")

        # Rank candidates
        rankings = rank_candidates(candidates, task, q_res_model, q_res_feature_keys)

        print(f"\n  DAPH-X rankings:")
        for r in rankings:
            print(f"    {r.candidate_id}: Q_MB={r.q_mb:.1f}, Q_res={r.q_res:.1f}, Q_X={r.q_x:.1f}, rank={r.rank}")

        # Identify fork
        fork = identify_fork(candidates, rankings)

        print(f"\n  Base action: {fork.base_candidate_id} (Q_X={fork.base_q_x:.1f})")
        print(f"  DAPH-X action: {fork.daphx_candidate_id} (Q_X={fork.daphx_q_x:.1f})")
        print(f"  Disagreement: {fork.disagreement}")

        if not fork.disagreement:
            # No disagreement — just execute the agreed-upon solution
            base_candidate = candidates[0]
            result = execute_solution(task, base_candidate.solution_code)
            print(f"  Agreed solution: {result.tests_passed}/{result.tests_total} tests, utility={result.utility:.2f}")
            all_results.append({
                "task_id": task.task_id,
                "description": task.description,
                "difficulty": task.difficulty,
                "disagreement": False,
                "base_candidate_id": fork.base_candidate_id,
                "daphx_candidate_id": fork.daphx_candidate_id,
                "base_utility": result.utility,
                "daphx_utility": result.utility,
                "delta_u": 0.0,
                "base_tests": result.tests_passed,
                "daphx_tests": result.tests_passed,
                "total_tests": result.tests_total,
                "base_error": result.error,
                "daphx_error": result.error,
                "rankings": [
                    {
                        "candidate_id": r.candidate_id,
                        "q_mb": r.q_mb,
                        "q_res": r.q_res,
                        "q_x": r.q_x,
                        "rank": r.rank,
                    }
                    for r in rankings
                ],
            })
            continue

        # FORK: execute both base and DAPH-X solutions
        disagreements += 1

        base_candidate = next(c for c in candidates if c.candidate_id == fork.base_candidate_id)
        daphx_candidate = next(c for c in candidates if c.candidate_id == fork.daphx_candidate_id)

        print(f"\n  Forking: executing both solutions...")

        base_result = execute_solution(task, base_candidate.solution_code)
        daphx_result = execute_solution(task, daphx_candidate.solution_code)

        delta_u = daphx_result.utility - base_result.utility

        print(f"  Base:   {base_result.tests_passed}/{base_result.tests_total} tests, utility={base_result.utility:.2f}")
        print(f"  DAPH-X: {daphx_result.tests_passed}/{daphx_result.tests_total} tests, utility={daphx_result.utility:.2f}")
        print(f"  ΔU = {delta_u:+.2f}")

        if delta_u > 0.5:
            rescues += 1
            outcome = "RESCUE"
        elif delta_u < -0.5:
            breaks += 1
            outcome = "BREAK"
        else:
            ties += 1
            outcome = "TIE"

        print(f"  Outcome: {outcome}")

        if base_result.error:
            print(f"  Base error: {base_result.error[:100]}")
        if daphx_result.error:
            print(f"  DAPH-X error: {daphx_result.error[:100]}")

        all_results.append({
            "task_id": task.task_id,
            "description": task.description,
            "difficulty": task.difficulty,
            "disagreement": True,
            "base_candidate_id": fork.base_candidate_id,
            "daphx_candidate_id": fork.daphx_candidate_id,
            "base_utility": base_result.utility,
            "daphx_utility": daphx_result.utility,
            "delta_u": delta_u,
            "base_tests": base_result.tests_passed,
            "daphx_tests": daphx_result.tests_passed,
            "total_tests": base_result.tests_total,
            "base_error": base_result.error,
            "daphx_error": daphx_result.error,
            "outcome": outcome,
            "rankings": [
                {
                    "candidate_id": r.candidate_id,
                    "q_mb": r.q_mb,
                    "q_res": r.q_res,
                    "q_x": r.q_x,
                    "rank": r.rank,
                }
                for r in rankings
            ],
        })

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Disagreements: {disagreements}")
    print(f"  Rescues (ΔU > 0): {rescues}")
    print(f"  Breaks (ΔU < 0): {breaks}")
    print(f"  Ties (|ΔU| ≤ 0.5): {ties}")

    if disagreements > 0:
        precision = rescues / disagreements
        break_rate = breaks / disagreements
        print(f"  Force precision: {precision:.4f}")
        print(f"  Break rate: {break_rate:.4f}")

        # Rule of three
        if breaks == 0 and disagreements > 0:
            break_upper_95 = 3.0 / disagreements
            print(f"  Break rate 95% upper bound: {break_upper_95:.4f}")

    # Per-difficulty breakdown
    print(f"\n  Per-difficulty:")
    for difficulty in ["easy", "medium", "hard"]:
        diff_results = [r for r in all_results if r["difficulty"] == difficulty and r["disagreement"]]
        if diff_results:
            d_rescues = sum(1 for r in diff_results if r["delta_u"] > 0.5)
            d_breaks = sum(1 for r in diff_results if r["delta_u"] < -0.5)
            print(f"    {difficulty}: {len(diff_results)} disagreements, {d_rescues} rescues, {d_breaks} breaks")

    # Save results
    output = {
        "experiment": "daph_x_coding_agent",
        "model": model.MODEL_NAME,
        "model_hash": model.model_hash,
        "n_tasks": len(tasks),
        "n_candidates": n_candidates,
        "n_disagreements": disagreements,
        "n_rescues": rescues,
        "n_breaks": breaks,
        "n_ties": ties,
        "force_precision": rescues / max(disagreements, 1),
        "break_rate": breaks / max(disagreements, 1),
        "results": all_results,
    }

    output_path = OUTPUT_DIR / "coding_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="DAPH-X coding agent experiment")
    parser.add_argument(
        "--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        help="Path to the GGUF model file",
    )
    parser.add_argument("--n_candidates", type=int, default=4, help="Candidates per task")
    parser.add_argument("--n_tasks", type=int, default=20, help="Number of tasks")
    parser.add_argument("--start_idx", type=int, default=0, help="Start task index (0=first, 20=hard tasks start)")
    parser.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty (easy/medium/hard)")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="GPU layers (-1 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_experiment(
        model_path=args.model_path,
        n_candidates=args.n_candidates,
        n_tasks=args.n_tasks,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        start_idx=args.start_idx,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
