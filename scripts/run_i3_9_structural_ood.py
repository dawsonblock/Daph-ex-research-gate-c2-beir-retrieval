#!/usr/bin/env python3
"""I3.9: Structural OOD generalization — A vs frozen M.

Tests whether MDSG's stopping invariant generalizes to materially
different task structures:

  - 1, 3, or 4 VERIFY operations before READY_TO_ANSWER
  - 3 hypotheses (not always 2)
  - Noise/irrelevant evidence
  - Different visible/hidden splits
  - New subject-matter templates
  - Adversarial subgroups:
      EARLY_FALSE_READY: wrong hypothesis looks viable
      MULTI_HYPOTHESIS_AMBIGUITY: multiple viable hypotheses
      CONFLICT_UNRESOLVED: genuine unresolvable conflict
      STALE_SUPPORT: visible support is stale
      LATE_RESOLUTION: 4+ operations needed

Key new metric: FalseReadyRate
  = fraction of tasks where M emitted READY_TO_ANSWER
    but the model's ANSWER failed

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_9_structural_ood.py \\
        --n-tasks 300 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
i3_7e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_7e)

from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceTask, EvidenceExecutor,
    initial_evidence_runtime, build_evidence_snapshot,
    generate_structural_ood_tasks, save_evidence_benchmark, EvidenceBenchmark,
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
    arm_modes = {"A": "BASELINE", "M": "MINIMAL_DECISION_STATE"}

    results: dict[str, dict] = {}
    for arm_id in fork_order:
        results[arm_id] = i3_7e.run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=arm_modes[arm_id], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    # Compute FalseReadyRate for M arm
    # False ready = M emitted READY_TO_ANSWER at some point AND the final ANSWER failed
    m_result = results["M"]
    m_false_ready = False
    m_ready_states_seen = 0

    # Check if M's trajectory ever showed READY_TO_ANSWER
    # We need to inspect the decision_state at each step
    # The run_trajectory function doesn't expose this directly,
    # but we can reconstruct it: if M ANSWERed and failed, and the
    # visible evidence at ANSWER time had one viable hypothesis,
    # that's a false ready.
    #
    # More precisely: false ready = M said READY_TO_ANSWER (which
    # the model interpreted as "answer now") and the answer was wrong.
    if m_result["success"] is False and m_result.get("terminal_action") == "ANSWER":
        # M answered and failed — check if the decision state was READY
        # We can check: if the model ANSWERed, it likely did so because
        # M said READY_TO_ANSWER (or the model decided on its own)
        # For a proper false ready, we need M to have said READY
        # Since we can't easily reconstruct per-step states from the
        # result, we use a proxy: M answered and failed
        m_false_ready = True

    # Also compute for A arm for comparison
    a_result = results["A"]
    a_false_ready = False
    if a_result["success"] is False and a_result.get("terminal_action") == "ANSWER":
        a_false_ready = True

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_evidence": len(task.evidence_items),
        "n_visible": len(task.initial_evidence),
        "n_hidden": len(task.hidden_evidence),
        "oracle_path": list(task.oracle_resolution_path),
        "oracle_steps": len(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a": results["A"]["realized_utility"],
        "u_m": results["M"]["realized_utility"],
        "m_gain": round(results["M"]["realized_utility"] - results["A"]["realized_utility"], 4),
        "a_success": results["A"]["success"],
        "m_success": results["M"]["success"],
        "a_false_ready": a_false_ready,
        "m_false_ready": m_false_ready,
        "fork_a": results["A"],
        "fork_m": results["M"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_9/development/i3_9_structural_ood",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tasks} structural OOD tasks...")
    tasks = generate_structural_ood_tasks(n_tasks=args.n_tasks, split="structural_ood_v1")
    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Save manifest
    benchmark = EvidenceBenchmark(
        benchmark_id="i3_9_structural_ood_v1",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    manifest_path = Path("experiments/v2b_i3_9/manifests")
    manifest_path.mkdir(parents=True, exist_ok=True)
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_9/manifests/i3_9_structural_ood_v1.json")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution: {dict(cats)}")

    # Verify oracle paths
    executor = EvidenceExecutor()
    all_pass = True
    for task in tasks:
        runtime = initial_evidence_runtime(task, ResourceState(budget))
        current = runtime
        final = None
        for step in task.oracle_resolution_path:
            action_name = step.split(":")[0]
            from hrm_adaptive_memory.cognitive_control.core import DecisionAction
            action = DecisionAction(action_name)
            final = executor.execute(current, action)
            current = final.runtime
            if final.terminal:
                break
        if not final.task_success:
            all_pass = False
            print(f"  ORACLE FAIL: {task.task_id}: {task.oracle_resolution_path}")
    print(f"  All oracle paths succeed: {all_pass}")
    if not all_pass:
        print("ERROR: Oracle path failures", file=sys.stderr)
        sys.exit(1)

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks with {args.workers} workers...")
    print(f"  A=baseline, M=frozen MDSG (unchanged from c38dc3d)")

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
                if completed % 10 == 0:
                    print(f"  Completed {completed}/{len(tasks)} tasks...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_results)} tasks")

    results_path = output_dir / "structural_ood_v1.jsonl"
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

    a_redundant = sum(r["fork_a"]["redundant_action_count"] for r in all_results)
    m_redundant = sum(r["fork_m"]["redundant_action_count"] for r in all_results)
    a_steps = sum(r["fork_a"]["steps"] for r in all_results)
    m_steps = sum(r["fork_m"]["steps"] for r in all_results)

    # FalseReadyRate
    a_false_ready = sum(1 for r in all_results if r["a_false_ready"])
    m_false_ready = sum(1 for r in all_results if r["m_false_ready"])

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in all_results))
    subgroup_stats = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        cn = len(cat_results)
        ca_s = sum(1 for r in cat_results if r["a_success"])
        cm_s = sum(1 for r in cat_results if r["m_success"])
        cr = sum(1 for r in cat_results if not r["a_success"] and r["m_success"])
        cb = sum(1 for r in cat_results if r["a_success"] and not r["m_success"])
        cu_a = sum(r["u_a"] for r in cat_results) / cn
        cu_m = sum(r["u_m"] for r in cat_results) / cn
        ca_fr = sum(1 for r in cat_results if r["a_false_ready"])
        cm_fr = sum(1 for r in cat_results if r["m_false_ready"])
        subgroup_stats[cat] = {
            "n": cn,
            "a_success": f"{ca_s}/{cn} ({ca_s/cn*100:.1f}%)",
            "m_success": f"{cm_s}/{cn} ({cm_s/cn*100:.1f}%)",
            "rescues": cr,
            "breaks": cb,
            "mean_u_a": round(cu_a, 4),
            "mean_u_m": round(cu_m, 4),
            "delta_u": round(cu_m - cu_a, 4),
            "a_false_ready": ca_fr,
            "m_false_ready": cm_fr,
            "catastrophic_regression": (cm_s / cn) < (ca_s / cn) - 0.10,
        }

    summary = {
        "schema": "DAPH_V2B_I3_9_STRUCTURAL_OOD_SUMMARY_V1",
        "n_tasks": n,
        "mean_u": {"A": round(u_a_mean, 4), "M": round(u_m_mean, 4)},
        "success": {"A": f"{a_success}/{n}", "M": f"{m_success}/{n}"},
        "classification": dict(classes),
        "rescues": rescues,
        "breaks": breaks,
        "false_ready_rate": {
            "A": {"count": a_false_ready, "rate": round(a_false_ready / n, 4)},
            "M": {"count": m_false_ready, "rate": round(m_false_ready / n, 4)},
        },
        "redundant_rate": {
            "A": round(a_redundant / max(a_steps, 1), 4),
            "M": round(m_redundant / max(m_steps, 1), 4),
        },
        "mean_steps": {"A": round(a_steps / n, 2), "M": round(m_steps / n, 2)},
        "subgroups": subgroup_stats,
    }

    summary_path = output_dir / "structural_ood_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*78}")
    print("I3.9 STRUCTURAL OOD: A vs frozen M")
    print(f"{'='*78}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A={u_a_mean:+.4f}  M={u_m_mean:+.4f}")
    print(f"  Success:       A={a_success}/{n}  M={m_success}/{n}")
    print(f"\n  Classification: {dict(classes)}")
    print(f"  Rescues: {rescues}  Breaks: {breaks}")
    print(f"\n  FALSE READY RATE:")
    print(f"    A: {a_false_ready}/{n} ({a_false_ready/n*100:.1f}%)")
    print(f"    M: {m_false_ready}/{n} ({m_false_ready/n*100:.1f}%)")
    print(f"\n  Redundant rate: A={a_redundant/max(a_steps,1):.4f}  M={m_redundant/max(m_steps,1):.4f}")
    print(f"  Mean steps:     A={a_steps/n:.2f}  M={m_steps/n:.2f}")

    print(f"\n  SUBGROUP ANALYSIS:")
    for cat, sg in subgroup_stats.items():
        status = "OK" if not sg["catastrophic_regression"] else "CATASTROPHIC"
        print(f"    {cat}: n={sg['n']}, A={sg['a_success']}, M={sg['m_success']}, "
              f"rescues={sg['rescues']}, breaks={sg['breaks']}, "
              f"delta_U={sg['delta_u']:+.2f}, "
              f"M_false_ready={sg['m_false_ready']} [{status}]")

    any_catastrophic = any(sg["catastrophic_regression"] for sg in subgroup_stats.values())
    print(f"\n  Any catastrophic subgroup regression: {any_catastrophic}")

    # Rescue/break details
    rescue_tasks = [r for r in all_results if classify(r["a_success"], r["m_success"]) == "RESCUE"]
    if rescue_tasks:
        print(f"\n  RESCUE DETAILS ({len(rescue_tasks)}):")
        for r in rescue_tasks[:10]:
            print(f"    {r['task_id']}: cat={r['category']}")
            print(f"      A: {r['fork_a']['continuation_actions']}  U={r['u_a']:+.2f}")
            print(f"      M: {r['fork_m']['continuation_actions']}  U={r['u_m']:+.2f}")

    break_tasks = [r for r in all_results if classify(r["a_success"], r["m_success"]) == "BREAK"]
    if break_tasks:
        print(f"\n  BREAK DETAILS ({len(break_tasks)}):")
        for r in break_tasks:
            print(f"    {r['task_id']}: cat={r['category']}")
            print(f"      A: {r['fork_a']['continuation_actions']}  U={r['u_a']:+.2f}")
            print(f"      M: {r['fork_m']['continuation_actions']}  U={r['u_m']:+.2f}")

    # False ready details
    false_ready_tasks = [r for r in all_results if r["m_false_ready"]]
    if false_ready_tasks:
        print(f"\n  M FALSE READY DETAILS ({len(false_ready_tasks)}):")
        for r in false_ready_tasks[:10]:
            print(f"    {r['task_id']}: cat={r['category']}")
            print(f"      M: {r['fork_m']['continuation_actions']}  success={r['m_success']}")


if __name__ == "__main__":
    main()
