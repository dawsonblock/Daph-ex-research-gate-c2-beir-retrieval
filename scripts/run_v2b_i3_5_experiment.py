#!/usr/bin/env python3
"""Run the I3.5 development experiment with governor-enhanced trajectories.

Runs the governor-enhanced runner on structure_dev_v2 (300 tasks).
Requires DeepSeek API access via the model backend.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.i3_5_full_runner import (
    I35FullExperimentRunner, save_governor_results, save_governor_receipts,
    score_governor_results)
from hrm_adaptive_memory.executive.i3_4_full_runner import run_statistical_analysis
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.i3_4_generation_config import FROZEN_CONFIG


BENCHMARK_MANIFEST = ROOT / "experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json"
ORACLE_VIEWS_PATH = ROOT / "experiments/v2b_i3_5/oracle_tables/v2b_i3_5_observable_oracle_views_v1.json"
LATENT_ORACLE_PATH = ROOT / "experiments/v2b_i3_5/oracle_tables/v2b_i3_5_latent_oracles_v1.jsonl.gz"
OUTPUT_DIR = ROOT / "experiments/v2b_i3_5/results"
UTILITY_PATH = ROOT / "configs/v2b_i3_1_utility_v1.json"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="structure_dev_v2",
                        help="Split to run (default: structure_dev_v2)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Maximum tasks to run (for testing)")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    # Load benchmark
    print(f"Loading benchmark from {BENCHMARK_MANIFEST}...", flush=True)
    benchmark = load_metareasoning_benchmark(BENCHMARK_MANIFEST, verify_oracle_cache=True)
    print(f"Loaded {len(benchmark.tasks)} tasks", flush=True)

    # Load utility
    utility = MetareasoningUtility.from_file(UTILITY_PATH)

    # Create backend (reads DEEPSEEK_API_KEY from environment)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    backend = DeepSeekBackend(
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    # Create runner
    runner = I35FullExperimentRunner(
        backend=backend,
        utility=utility,
        experiment_id="v2b_i3_5_dev_experiment_v1",
        max_steps=24,
        strict_json=True,
        temperature=FROZEN_CONFIG.temperature,
        max_tokens=FROZEN_CONFIG.max_tokens,
    )

    # Run the split
    print(f"\nRunning {args.split} ({args.max_tasks or 'all'} tasks)...", flush=True)
    started = perf_counter()
    results = runner.run_split(
        benchmark, args.split,
        max_tasks=args.max_tasks,
        progress_every=args.progress_every)
    elapsed = perf_counter() - started

    print(f"\nCompleted {len(results)} pairs in {elapsed:.1f}s", flush=True)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"v2b_i3_5_{args.split}_results_v1.json"
    results_sha = save_governor_results(results, results_path)
    print(f"Results saved: {results_path} (SHA-256: {results_sha[:16]}...)", flush=True)

    # Save receipts
    receipts_path = OUTPUT_DIR / f"v2b_i3_5_{args.split}_receipts_v1.jsonl"
    receipts_sha = save_governor_receipts(runner.all_receipts(), receipts_path)
    print(f"Receipts saved: {receipts_path} (SHA-256: {receipts_sha[:16]}...)", flush=True)

    # Print runner summary
    summary = runner.runner_summary()
    print(f"\n=== RUNNER SUMMARY ===")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value}")

    # Score results
    print(f"\nScoring results...", flush=True)
    utility_weights = {
        "correct_answer": utility.correct_answer,
        "incorrect_answer": utility.incorrect_answer,
        "correct_defer": utility.correct_defer,
        "correct_stop": utility.correct_stop,
    }
    contributions, deltas = score_governor_results(
        results, benchmark, ORACLE_VIEWS_PATH, LATENT_ORACLE_PATH, utility_weights)
    print(f"Computed {len(contributions)} contributions and {len(deltas)} paired deltas", flush=True)

    # Run statistical analysis
    stats = run_statistical_analysis(deltas)
    print(f"\n=== STATISTICAL ANALYSIS ===")
    print(f"  N paired tasks: {stats['n_paired_tasks']}")
    print(f"  Mean ΔDG: {stats['mean_delta_dg']:.4f}")
    if stats.get("task_level_bootstrap"):
        tb = stats["task_level_bootstrap"]
        print(f"  Task bootstrap: {tb['point_estimate']:.4f} "
              f"CI=[{tb['ci_lower']:.4f}, {tb['ci_upper']:.4f}]")

    # Save scores and stats
    scores_path = OUTPUT_DIR / f"v2b_i3_5_{args.split}_scores_v1.json"
    stats_path = OUTPUT_DIR / f"v2b_i3_5_{args.split}_stats_v1.json"
    scores_path.write_text(json.dumps({
        "contributions": [c.as_dict() if hasattr(c, 'as_dict') else c.__dict__
                          for c in contributions],
        "deltas": [d.as_dict() if hasattr(d, 'as_dict') else d.__dict__
                   for d in deltas],
    }, indent=2, sort_keys=True) + "\n")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"\nScores saved: {scores_path}")
    print(f"Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
