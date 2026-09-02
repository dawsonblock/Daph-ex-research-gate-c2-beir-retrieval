#!/usr/bin/env python3
"""DAPH-X R1: Real Coding Authority Qualification.

Properly controlled experiment with matched probe budgets.

Four systems compared:
  1. Base: model's first candidate (temp=0), no probes
  2. Probe baseline: run probes on all candidates, pick best probe pass rate
  3. Simple reranker: Q_MB + probe pass rate (no learned authority)
  4. DAPH-X: full authority stack (Q_res + pairwise + risk + conformal)

All systems except Base get the same probe budget (first K tests).
All systems operate on the same tasks with the same candidates.

Frozen splits:
  - Development (60%): train Q_res, pairwise, risk
  - Calibration (15%): conformal calibration, threshold tuning
  - Confirmation (25%): untouched final evaluation

Target: 60+ effective interventions in confirmation for 5% upper bound.

Usage:
    python scripts/run_r1_experiment.py \\
        --model_path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        --n_candidates 4 \\
        --n_probe_tests 2 \\
        --max_tokens 400
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

from daph_x.coding.tasks import get_all_tasks, get_task, CodingTask
from daph_x.coding.misleading_probe_tasks import get_misleading_probe_tasks
from daph_x.coding.code_executor import execute_solution
from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb

CODING_DIR = REPO_ROOT / "experiments/daph_x/coding"
R1_DIR = REPO_ROOT / "experiments/daph_x/r1"


def run_probe_tests(task: CodingTask, solution_code: str, n_probe: int) -> dict:
    """Run candidate on first n_probe tests only."""
    probe_task = CodingTask(
        task_id=task.task_id,
        description=task.description,
        function_name=task.function_name,
        signature=task.signature,
        docstring=task.docstring,
        difficulty=task.difficulty,
        tests=task.tests[:n_probe],
        imports=task.imports,
        common_errors=task.common_errors,
    )
    result = execute_solution(probe_task, solution_code, timeout_seconds=5.0)
    return {
        "probe_pass_rate": result.pass_rate,
        "probe_n_passed": result.tests_passed,
        "probe_n_total": result.tests_total,
        "probe_has_error": 1.0 if result.error else 0.0,
        "probe_error": result.error,
    }


def run_full_tests(task: CodingTask, solution_code: str) -> dict:
    """Run candidate on ALL tests (ground truth utility)."""
    result = execute_solution(task, solution_code, timeout_seconds=10.0)
    return {
        "tests_passed": result.tests_passed,
        "tests_total": result.tests_total,
        "pass_rate": result.pass_rate,
        "utility": result.utility,
        "error": result.error,
    }


def collect_task_data(
    model: CodingModelInterface,
    task: CodingTask,
    n_candidates: int,
    n_probe: int,
    max_tokens: int,
) -> dict:
    """Collect complete data for one task: candidates, probes, full tests."""
    print(f"\n{'='*60}")
    print(f"  Task: {task.task_id} ({task.difficulty})")
    print(f"  {task.description}")
    print(f"{'='*60}")

    # Generate candidates
    t0 = time.monotonic()
    candidates = model.generate_candidates(
        task=task, n_candidates=n_candidates, max_tokens=max_tokens,
    )
    gen_time = time.monotonic() - t0
    print(f"  Generated {len(candidates)} candidates in {gen_time:.1f}s")

    # For each candidate: extract features, run probes, run full tests
    records = []
    for cand in candidates:
        features = extract_code_features(cand.solution_code, task)
        q_mb = compute_q_mb(features, task)
        probe = run_probe_tests(task, cand.solution_code, n_probe)
        full = run_full_tests(task, cand.solution_code)

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
            "probe": probe,
            "full": full,
            "latency_ms": cand.latency_ms,
        }
        records.append(record)

        p_str = f"probe={probe['probe_n_passed']}/{probe['probe_n_total']}"
        f_str = f"full={full['tests_passed']}/{full['tests_total']}"
        print(f"    {cand.candidate_id}: {p_str}, {f_str}, util={full['utility']:.1f}")

    # Identify base (first candidate, temp=0)
    base = records[0]
    base_utility = base["full"]["utility"]

    # Compute labels
    for r in records:
        r["delta_u_vs_base"] = r["full"]["utility"] - base_utility
        r["is_better_than_base"] = r["delta_u_vs_base"] > 0.5
        r["is_worse_than_base"] = r["delta_u_vs_base"] < -0.5

    # Summary
    best = max(records, key=lambda r: r["full"]["utility"])
    rescue_available = best["full"]["utility"] > base_utility + 0.5

    # Check if probes are informative but imperfect
    probe_perfect = all(r["probe"]["probe_pass_rate"] == 1.0 for r in records)
    probe_uninformative = all(r["probe"]["probe_n_passed"] == 0 for r in records)

    summary = {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "n_candidates": len(records),
        "n_probe_tests": n_probe,
        "n_total_tests": len(task.tests),
        "base_utility": base_utility,
        "base_pass_rate": base["full"]["pass_rate"],
        "best_utility": best["full"]["utility"],
        "best_candidate_id": best["candidate_id"],
        "rescue_available": rescue_available,
        "probe_perfect": probe_perfect,
        "probe_uninformative": probe_uninformative,
        "gen_time_s": gen_time,
        "candidates": records,
    }

    print(f"\n  Base: util={base_utility:.1f} ({base['full']['tests_passed']}/{base['full']['tests_total']})")
    print(f"  Best: util={best['full']['utility']:.1f} ({best['candidate_id']})")
    print(f"  Rescue available: {rescue_available}")
    print(f"  Probe perfect: {probe_perfect}, uninformative: {probe_uninformative}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="DAPH-X R1 data collection")
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_candidates", type=int, default=4)
    parser.add_argument("--n_probe_tests", type=int, default=2)
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    R1_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks — use hard tasks + misleading-probe tasks for R1
    all_tasks = get_all_tasks()
    hard_tasks = [t for t in all_tasks if int(t.task_id.split("_")[1]) >= 50]
    misleading = get_misleading_probe_tasks()
    # Combine: hard tasks first, then misleading-probe tasks
    combined = hard_tasks + misleading
    if args.difficulty:
        combined = [t for t in combined if t.difficulty == args.difficulty]
    tasks = combined[args.start_idx:args.start_idx + args.n_tasks]

    print(f"R1 Data Collection")
    print(f"  Tasks: {len(tasks)} (from {len(hard_tasks)} available hard tasks)")
    print(f"  Candidates per task: {args.n_candidates}")
    print(f"  Probe tests: {args.n_probe_tests}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Model: {args.model_path}")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else R1_DIR / "r1_corpus.jsonl"

    all_summaries = []
    with open(output_path, "w") as f:
        for idx, task in enumerate(tasks):
            print(f"\n[Task {idx+1}/{len(tasks)}]")
            try:
                summary = collect_task_data(
                    model=model, task=task,
                    n_candidates=args.n_candidates,
                    n_probe=args.n_probe_tests,
                    max_tokens=args.max_tokens,
                )
                all_summaries.append(summary)
                f.write(json.dumps(summary, default=str) + "\n")
                f.flush()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    total = len(all_summaries)
    rescue_avail = sum(1 for s in all_summaries if s["rescue_available"])
    probe_perfect = sum(1 for s in all_summaries if s["probe_perfect"])
    base_pass = sum(1 for s in all_summaries if s["base_pass_rate"] == 1.0)

    print(f"\n{'='*60}")
    print(f"  R1 DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {total}")
    print(f"  Base fully passes: {base_pass} ({base_pass/max(total,1)*100:.0f}%)")
    print(f"  Rescue available: {rescue_avail} ({rescue_avail/max(total,1)*100:.0f}%)")
    print(f"  Probe perfect (all candidates pass probes): {probe_perfect}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
