#!/usr/bin/env python3
"""Run the I3.5.1 factorial experiment (4 arms x N tasks).

Usage:
    python scripts/run_v2b_i3_5_1_experiment.py --split development [--max-tasks 10]

Requires DEEPSEEK_API_KEY environment variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    MetareasoningBenchmark, load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import (
    FactorialExperimentRunner, save_results,
)
from hrm_adaptive_memory.executive.i3_5_1.factorial_scheduler import (
    schedule_block, check_balance,
)
from hrm_adaptive_memory.executive.i3_5_1.scoring import (
    score_factorial_results, save_scores, verify_identity_invariant,
)
from hrm_adaptive_memory.executive.i3_5_1.statistics import (
    compute_factorial_stats, save_stats,
)
from hrm_adaptive_memory.executive.i3_5_1.experiment_identity import (
    build_experiment_identity, save_experiment_identity,
)
from hrm_adaptive_memory.executive.i3_5_1.receipts import ReceiptLedger
from hrm_adaptive_memory.executive.i3_5_1.replay import replay_all_trajectories
from hrm_adaptive_memory.executive.i3_5_1.report import build_factorial_report, save_report


def main():
    parser = argparse.ArgumentParser(description="Run I3.5.1 factorial experiment")
    parser.add_argument("--split", default="structure_dev_v2",
                        help="Benchmark split to run")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks (for testing)")
    parser.add_argument("--benchmark-manifest",
                        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json")
    parser.add_argument("--oracle-views",
                        default="experiments/v2b_i3_5/oracle_tables/v2b_i3_5_observable_oracle_views_v1.json")
    parser.add_argument("--latent-oracles",
                        default="experiments/v2b_i3_5/oracle_tables/v2b_i3_5_latent_oracles_v1.jsonl.gz")
    parser.add_argument("--output-dir",
                        default="experiments/v2b_i3_5_1/development")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    # Load benchmark
    print(f"Loading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=True)
    split_benchmark = benchmark.for_split(args.split)
    tasks = split_benchmark.tasks
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    print(f"Loaded {len(tasks)} tasks for split '{args.split}'")

    # Build experiment identity
    criteria_path = ROOT / "experiments/v2b_i3_5_1/configs/v2b_i3_5_1_scientific_criteria_v1.json"
    criteria_sha = hashlib.sha256(criteria_path.read_bytes()).hexdigest()

    oracle_manifest_path = ROOT / "experiments/v2b_i3_5/oracle_tables/v2b_i3_5_oracle_cache_manifest_v1.json"
    oracle_manifest_sha = hashlib.sha256(oracle_manifest_path.read_bytes()).hexdigest()

    views_path = ROOT / args.oracle_views
    views_sha = hashlib.sha256(views_path.read_bytes()).hexdigest()

    tasks_path = ROOT / "experiments/v2b_i3_5/private/v2b_i3_5_tasks_v2.json"
    tasks_sha = hashlib.sha256(tasks_path.read_bytes()).hexdigest()

    identity = build_experiment_identity(
        root=str(ROOT),
        benchmark_identity="v2b_i3_5_benchmark_v2",
        split_identity="v2b_i3_5_splits_v2",
        task_corpus_sha256=tasks_sha,
        scientific_criteria_sha256=criteria_sha,
        oracle_manifest_sha256=oracle_manifest_sha,
        observable_oracle_views_sha256=views_sha,
    )

    output_dir = Path(args.output_dir) / identity.sha256()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save identity
    identity_path = output_dir / "experiment_identity.json"
    save_experiment_identity(identity, identity_path)
    print(f"Experiment identity: {identity.sha256()}")
    print(f"Output directory: {output_dir}")

    # Schedule blocks
    schedules = [schedule_block(t.task_id) for t in tasks]
    balance = check_balance(schedules)
    print(f"Schedule balance: {balance['total_blocks']} blocks")

    # Set up backend
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    backend = DeepSeekBackend()

    # Set up utility
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")

    # Create runner
    runner = FactorialExperimentRunner(
        backend=backend,
        utility=utility,
        experiment_id="v2b_i3_5_1_experiment_v1",
        experiment_identity_sha256=identity.sha256(),
        max_steps=24,
        strict_json=True,
        temperature=0.0,
        max_tokens=2048,
    )

    # Run all task blocks
    print(f"\nRunning {args.split} ({len(tasks)} tasks, 4 conditions each)...")
    results = []
    for i, (task, schedule) in enumerate(zip(tasks, schedules)):
        budget = split_benchmark.budget_for(task)
        block_result = runner.run_block(task, budget, schedule.condition_order)
        results.append(block_result)

        if (i + 1) % args.progress_every == 0:
            trajs = block_result["trajectories"]
            print(f"  [{i+1}/{len(tasks)}] {task.task_id}: "
                  f"B_NO_G={trajs['BLIND_NO_GOVERNOR']['task_success']}, "
                  f"B_G={trajs['BLIND_GOVERNOR']['task_success']}, "
                  f"A_NO_G={trajs['AWARE_NO_GOVERNOR']['task_success']}, "
                  f"A_G={trajs['AWARE_GOVERNOR']['task_success']}")

    # Save receipts
    receipts_path = output_dir / "receipts.jsonl"
    receipts_sha = runner.receipt_ledger.save(receipts_path)
    print(f"\nReceipts saved: {receipts_path} (SHA-256: {receipts_sha[:16]}...)")

    # Save results
    results_path = output_dir / "results.json"
    results_sha = save_results(
        results, results_path,
        experiment_identity_sha256=identity.sha256(),
        receipt_chain_root=runner.receipt_ledger.receipt_chain_root,
        source_receipts_sha256=receipts_sha,
    )
    print(f"Results saved: {results_path} (SHA-256: {results_sha[:16]}...)")

    # Score
    print("\nScoring results...")
    contributions = score_factorial_results(
        results, args.oracle_views, args.latent_oracles)

    # Verify IG/DG/TR identity
    identity_failures = 0
    for c in contributions:
        if not verify_identity_invariant(c):
            identity_failures += 1
    print(f"IG/DG/TR identity check: {len(contributions) - identity_failures}/{len(contributions)} pass")

    scores_path = output_dir / "scores.json"
    scores_sha = save_scores(
        contributions, scores_path,
        experiment_identity_sha256=identity.sha256(),
        source_results_sha256=results_sha,
    )
    print(f"Scores saved: {scores_path} (SHA-256: {scores_sha[:16]}...)")

    # Statistics
    print("\nComputing statistics...")
    stats = compute_factorial_stats(contributions)

    stats_path = output_dir / "stats.json"
    stats_sha = save_stats(
        stats, stats_path,
        experiment_identity_sha256=identity.sha256(),
        source_results_sha256=results_sha,
        source_scores_sha256=scores_sha,
        statistics_implementation_sha256=hashlib.sha256(
            Path(__file__).resolve().parent.parent.joinpath(
                "hrm_adaptive_memory/executive/i3_5_1/statistics.py"
            ).read_bytes()
        ).hexdigest(),
    )
    print(f"Stats saved: {stats_path}")

    # Print summary
    print(f"\n=== STATISTICAL ANALYSIS ===")
    print(f"  N tasks: {stats.n_tasks}")
    print(f"  ΔDG_gov|aware: {stats.mean_delta_dg_gov_aware:.4f} "
          f"CI={stats.ci_gov_aware}")
    print(f"  ΔDG_gov|blind: {stats.mean_delta_dg_gov_blind:.4f} "
          f"CI={stats.ci_gov_blind}")
    print(f"  ΔDG_state|no-gov: {stats.mean_delta_dg_state_no_gov:.4f} "
          f"CI={stats.ci_state_no_gov}")
    print(f"  ΔDG_state|gov: {stats.mean_delta_dg_state_gov:.4f} "
          f"CI={stats.ci_state_gov}")
    print(f"  Δ_interaction: {stats.mean_delta_interaction:.4f} "
          f"CI={stats.ci_interaction}")

    # Runner summary
    summary = runner.runner_summary()
    print(f"\n=== RUNNER SUMMARY ===")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")

    # Replay verification
    print("\nVerifying replay...")
    replay_result = replay_all_trajectories(results, benchmark, utility=utility, split=args.split)
    print(f"  Replay: {replay_result['full_matches']}/{replay_result['total_trajectories']} fully match")
    if not replay_result["all_match"]:
        print(f"  FAILURES: {len(replay_result['failures'])}")
        for f in replay_result["failures"][:5]:
            print(f"    {f}")

    # Build report
    print("\nBuilding report...")
    report = build_factorial_report(
        n_tasks=len(results),
        contributions=[c.as_dict() for c in contributions],
        stats=stats.as_dict(),
        results=results,
        experiment_identity_sha256=identity.sha256(),
        source_stats_sha256=stats_sha,
        source_results_sha256=results_sha,
        source_run_id=runner.receipt_ledger.run_id,
    )
    report_path = output_dir / "report.json"
    report_sha = save_report(report, report_path)
    print(f"Report saved: {report_path}")

    print(f"\nDone. All artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
