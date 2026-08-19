#!/usr/bin/env python3
"""I3.7e-r1 repair check — A vs M_clean on same 50 tasks.

Two arms only:
  A       = semantic evidence baseline (unchanged)
  M_clean = leakage-clean minimal decision state

Counterbalanced.  No action override.

The crucial question: does the 5-rescue pattern survive the leakage repair?

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_7e_r1_repair_check.py \\
        --n-tasks 50 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse everything from the main I3.7e script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
i3_7e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_7e)

from hrm_adaptive_memory.executive.evidence_benchmark import (
    load_evidence_benchmark, EvidenceTask, EvidenceExecutor,
    initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility


def counterbalance_2arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    return ["A", "M"] if int(h[:8], 16) % 2 == 0 else ["M", "A"]


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    fork_order = counterbalance_2arm(task.task_id)

    arm_modes = {
        "A": "BASELINE",
        "M": "MINIMAL_DECISION_STATE",
    }

    results: dict[str, dict] = {}
    for arm_id in fork_order:
        results[arm_id] = i3_7e.run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=arm_modes[arm_id], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    u_a = results["A"]["realized_utility"]
    u_m = results["M"]["realized_utility"]

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "oracle_path": list(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a": u_a,
        "u_m": u_m,
        "m_gain": round(u_m - u_a, 4),
        "a_success": results["A"]["success"],
        "m_success": results["M"]["success"],
        "fork_a": results["A"],
        "fork_m": results["M"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="experiments/v2b_i3_7/manifests/i3_7_evidence_benchmark_v1.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_7/development/i3_7e_r1",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evidence benchmark from {args.benchmark}...")
    benchmark = load_evidence_benchmark(args.benchmark)
    tasks = benchmark.tasks[:args.n_tasks]
    budget = benchmark.budget_profiles["STANDARD"]
    print(f"  Loaded {len(tasks)} tasks")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks with {args.workers} workers...")
    print(f"  A=baseline, M_clean=leakage-clean minimal decision state")

    all_results: list[dict[str, Any]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_task, task, budget, utility, api_key): task
                   for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 5 == 0:
                    print(f"  Completed {completed}/{len(tasks)} tasks...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_results)} tasks")

    results_path = output_dir / "repair_check_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Summary
    n = len(all_results)
    if n == 0:
        print("No tasks completed!")
        return

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    u_a_mean = sum(r["u_a"] for r in all_results) / n
    u_m_mean = sum(r["u_m"] for r in all_results) / n
    a_success = sum(1 for r in all_results if r["a_success"])
    m_success = sum(1 for r in all_results if r["m_success"])

    classes = Counter(classify(r["a_success"], r["m_success"]) for r in all_results)
    rescues = classes.get("RESCUE", 0)
    breaks = classes.get("BREAK", 0)

    # Redundant actions
    a_redundant = sum(r["fork_a"]["redundant_action_count"] for r in all_results)
    m_redundant = sum(r["fork_m"]["redundant_action_count"] for r in all_results)
    a_steps = sum(r["fork_a"]["steps"] for r in all_results)
    m_steps = sum(r["fork_m"]["steps"] for r in all_results)

    # ACS metrics
    a_acs = sum(1 for r in all_results if r["fork_a"]["answer_condition_satisfied_before_terminal"])
    m_acs = sum(1 for r in all_results if r["fork_m"]["answer_condition_satisfied_before_terminal"])
    a_match = sum(1 for r in all_results if r["fork_a"]["terminal_action_matches_condition"])
    m_match = sum(1 for r in all_results if r["fork_m"]["terminal_action_matches_condition"])
    a_led = sum(1 for r in all_results if r["fork_a"]["condition_led_to_success"])
    m_led = sum(1 for r in all_results if r["fork_m"]["condition_led_to_success"])

    gates = {
        "success_ge_baseline": m_success >= a_success,
        "breaks_le_1": breaks <= 1,
        "rescues_ge_1": rescues >= 1,
        "mean_u_ge_baseline": u_m_mean >= u_a_mean,
        "rescues_gt_breaks": rescues > breaks,
    }

    summary = {
        "schema": "DAPH_V2B_I3_7E_R1_REPAIR_CHECK_V1",
        "n_tasks": n,
        "arms": {
            "A": "semantic evidence baseline",
            "M": "leakage-clean minimal decision state",
        },
        "mean_u": {"A": round(u_a_mean, 4), "M": round(u_m_mean, 4)},
        "success": {"A": f"{a_success}/{n}", "M": f"{m_success}/{n}"},
        "classification": dict(classes),
        "rescues": rescues,
        "breaks": breaks,
        "redundant_actions": {"A": a_redundant, "M": m_redundant},
        "redundant_rate": {
            "A": round(a_redundant / max(a_steps, 1), 4),
            "M": round(m_redundant / max(m_steps, 1), 4),
        },
        "mean_steps": {"A": round(a_steps / n, 2), "M": round(m_steps / n, 2)},
        "acs_metrics": {
            "A": {"acs_before": a_acs, "terminal_matches": a_match, "led_to_success": a_led},
            "M": {"acs_before": m_acs, "terminal_matches": m_match, "led_to_success": m_led},
        },
        "gates": gates,
    }

    summary_path = output_dir / "repair_check_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*78}")
    print("I3.7e-r1 REPAIR CHECK: A vs M_clean")
    print(f"{'='*78}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A={u_a_mean:+.4f}  M={u_m_mean:+.4f}")
    print(f"  Success:       A={a_success}/{n}  M={m_success}/{n}")
    print(f"\n  Classification: {dict(classes)}")
    print(f"  Rescues: {rescues}  Breaks: {breaks}")
    print(f"\n  Redundant actions: A={a_redundant}  M={m_redundant}")
    print(f"  Redundant rate:    A={a_redundant/max(a_steps,1):.4f}  M={m_redundant/max(m_steps,1):.4f}")
    print(f"  Mean steps:        A={a_steps/n:.2f}  M={m_steps/n:.2f}")
    print(f"\n  ACS metrics:")
    print(f"    A: acs_before={a_acs}, terminal_matches={a_match}, led_to_success={a_led}")
    print(f"    M: acs_before={m_acs}, terminal_matches={m_match}, led_to_success={m_led}")
    print(f"\n  GATES:")
    for gate, passed in gates.items():
        print(f"    {gate}: {'PASS' if passed else 'FAIL'}")
    print(f"  Total: {sum(gates.values())}/{len(gates)}")

    # Rescue details
    rescue_tasks = [r for r in all_results if classify(r["a_success"], r["m_success"]) == "RESCUE"]
    if rescue_tasks:
        print(f"\n  RESCUE DETAILS ({len(rescue_tasks)}):")
        for r in rescue_tasks:
            print(f"    {r['task_id']}: cat={r['category']}")
            print(f"      A: {r['fork_a']['continuation_actions']}  U={r['u_a']:+.2f}")
            print(f"      M: {r['fork_m']['continuation_actions']}  U={r['u_m']:+.2f}")

    # Break details
    break_tasks = [r for r in all_results if classify(r["a_success"], r["m_success"]) == "BREAK"]
    if break_tasks:
        print(f"\n  BREAK DETAILS ({len(break_tasks)}):")
        for r in break_tasks:
            print(f"    {r['task_id']}: cat={r['category']}")
            print(f"      A: {r['fork_a']['continuation_actions']}  U={r['u_a']:+.2f}")
            print(f"      M: {r['fork_m']['continuation_actions']}  U={r['u_m']:+.2f}")


if __name__ == "__main__":
    main()
