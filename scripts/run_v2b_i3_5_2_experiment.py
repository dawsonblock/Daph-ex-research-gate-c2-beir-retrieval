#!/usr/bin/env python3
"""Run the I3.5.2 Selective Governor Comparison Experiment.

Compares:
  - AWARE_OFF: No governor (clean base packet)
  - AWARE_ALWAYS_ON: Full governor always injected
  - AWARE_SELECTIVE: SelectiveGovernorGate decides intervention per step
  - AWARE_SHADOW_SELECTIVE: Base packet executed, gate evaluates silently

Usage:
    python scripts/run_v2b_i3_5_2_experiment.py --split structure_dev_v2 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.i3_5_1.receipts import ReceiptLedger
from hrm_adaptive_memory.executive.i3_5_2.modes import GovernorMode
from hrm_adaptive_memory.executive.i3_5_2.trajectory_runner import (
    I352FactorialRunner,
)
from hrm_adaptive_memory.executive.selective_governor import (
    SelectiveGovernorGate,
    RuleBasedInterventionPredictor,
    compute_gate_identity,
)


def main():
    parser = argparse.ArgumentParser(description="Run I3.5.2 Selective Governor Experiment")
    parser.add_argument("--split", default="structure_dev_v2")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Loading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    tasks = split_bm.tasks
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    print(f"Loaded {len(tasks)} tasks for split '{args.split}'")

    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")
    gate_id_info = compute_gate_identity()
    gate_sha = gate_id_info["gate_identity_sha256"]
    print(f"Selective Gate Identity: {gate_sha}")

    output_dir = Path(args.output_dir) / gate_sha[:12]
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = (
        GovernorMode.OFF,
        GovernorMode.ALWAYS_ON,
        GovernorMode.SELECTIVE,
    )

    print(f"\nRunning I3.5.2 ({len(tasks)} tasks, modes={[m.value for m in modes]}, {args.workers} workers)...")

    def run_one_task(item):
        idx, task = item
        budget = split_bm.budget_for(task)
        worker_backend = DeepSeekBackend()
        worker_gate = SelectiveGovernorGate(predictor=RuleBasedInterventionPredictor())
        worker_runner = I352FactorialRunner(
            backend=worker_backend,
            gate=worker_gate,
            utility=utility,
            experiment_id="v2b_i3_5_2_experiment_v1",
            experiment_identity_sha256=gate_sha,
            max_steps=24,
            strict_json=True,
            temperature=0.0,
            max_tokens=2048,
        )
        block_result, block_receipts = worker_runner.run_comparison_block_standalone(
            task, budget, modes=modes,
        )
        return idx, block_result, block_receipts

    results = []
    all_receipts = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_task, (i, t)): i
            for i, t in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx, block_result, block_receipts = future.result()
            results.append((idx, block_result, block_receipts))
            completed += 1
            if completed % args.progress_every == 0:
                trajs = block_result["trajectories"]
                print(f"  [{completed}/{len(tasks)}] {block_result['task_id']}: "
                      f"OFF_succ={trajs['OFF']['task_success']} (u={trajs['OFF']['realized_utility']:.1f}), "
                      f"ALWAYS_succ={trajs['ALWAYS_ON']['task_success']} (u={trajs['ALWAYS_ON']['realized_utility']:.1f}), "
                      f"SELECT_succ={trajs['SELECTIVE']['task_success']} (u={trajs['SELECTIVE']['realized_utility']:.1f}, "
                      f"interv={trajs['SELECTIVE']['interventions_approved']})")

    results.sort(key=lambda x: x[0])
    block_results = [r[1] for r in results]
    for r in results:
        all_receipts.extend(r[2])

    run_id = f"run_i352_{gate_sha[:12]}"
    ledger = ReceiptLedger.build_chain_from_receipts(all_receipts, run_id=run_id)
    print(f"\nBuilt receipt chain: {ledger.receipt_count} receipts")
    assert ledger.verify_chain(), "Receipt chain verification failed!"

    receipts_path = output_dir / "receipts.jsonl"
    receipts_sha = ledger.save(receipts_path)
    print(f"Receipts saved: {receipts_path} (SHA-256: {receipts_sha[:16]}...)")

    results_path = output_dir / "results.json"
    results_payload = {
        "schema": "DAPH_V2B_I3_5_2_RESULTS_V1",
        "schema_version": 1,
        "gate_identity_sha256": gate_sha,
        "receipt_chain_root": ledger.receipt_chain_root,
        "source_receipts_sha256": receipts_sha,
        "results": block_results,
    }
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True) + "\n")
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
