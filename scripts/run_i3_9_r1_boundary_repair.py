#!/usr/bin/env python3
"""I3.9-r1: MDSG boundary repair — A vs M0 vs M1 on structural OOD v2.

Three arms:
  A  = baseline (semantic evidence)
  M0 = frozen I3.8 MDSG (with operation recommendations)
  M1 = MDSG-StateOnly (no operation recs, conservative READY)

True FalseReadyRate: records per-step MDSG decision_state, then checks
  FalseReady = M said READY_TO_ANSWER AND model ANSWERed AND ANSWER failed

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_9_r1_boundary_repair.py \\
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


def counterbalance_3arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    import itertools
    arms = ["A", "M0", "M1"]
    perms = list(itertools.permutations(arms))
    return list(perms[int(h[:8], 16) % len(perms)])


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    fork_order = counterbalance_3arm(task.task_id)
    arm_modes = {
        "A": "BASELINE",
        "M0": "MINIMAL_DECISION_STATE",
        "M1": "MDSG_STATE_ONLY",
    }

    results: dict[str, dict] = {}
    for arm_id in fork_order:
        results[arm_id] = i3_7e.run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=arm_modes[arm_id], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    # Compute true FalseReadyRate for M0 and M1
    def compute_false_ready(result: dict) -> bool:
        """True false ready: M said READY_TO_ANSWER at some step,
        model ANSWERed, and ANSWER failed."""
        if result["success"] or result.get("terminal_action") != "ANSWER":
            return False
        # Check if any step had READY_TO_ANSWER
        for entry in result.get("decision_state_log", []):
            if entry.get("decision_state") == "READY_TO_ANSWER":
                return True
        return False

    def compute_provisional_false_ready(result: dict) -> bool:
        """Model ANSWERed and failed after seeing PROVISIONALLY_READY
        (but never READY_TO_ANSWER). This is a softer failure."""
        if result["success"] or result.get("terminal_action") != "ANSWER":
            return False
        has_ready = any(
            e.get("decision_state") == "READY_TO_ANSWER"
            for e in result.get("decision_state_log", [])
        )
        has_provisional = any(
            e.get("decision_state") == "PROVISIONALLY_READY"
            for e in result.get("decision_state_log", [])
        )
        return has_provisional and not has_ready

    m0_false_ready = compute_false_ready(results["M0"])
    m1_false_ready = compute_false_ready(results["M1"])
    m1_provisional_fail = compute_provisional_false_ready(results["M1"])

    # Failed terminal answer rate for baseline (no MDSG state)
    a_failed_answer = (
        not results["A"]["success"]
        and results["A"].get("terminal_action") == "ANSWER"
    )

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_evidence": len(task.evidence_items),
        "n_visible": len(task.initial_evidence),
        "n_hidden": len(task.hidden_evidence),
        "oracle_steps": len(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a": results["A"]["realized_utility"],
        "u_m0": results["M0"]["realized_utility"],
        "u_m1": results["M1"]["realized_utility"],
        "m0_gain": round(results["M0"]["realized_utility"] - results["A"]["realized_utility"], 4),
        "m1_gain": round(results["M1"]["realized_utility"] - results["A"]["realized_utility"], 4),
        "m1_m0_delta": round(results["M1"]["realized_utility"] - results["M0"]["realized_utility"], 4),
        "a_success": results["A"]["success"],
        "m0_success": results["M0"]["success"],
        "m1_success": results["M1"]["success"],
        "a_failed_answer": a_failed_answer,
        "m0_false_ready": m0_false_ready,
        "m1_false_ready": m1_false_ready,
        "m1_provisional_fail": m1_provisional_fail,
        "fork_a": results["A"],
        "fork_m0": results["M0"],
        "fork_m1": results["M1"],
    }


def paired_bootstrap_ci(deltas, n_iterations=10000, seed=42):
    import random
    rng = random.Random(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    return boot_means[int(0.025 * n_iterations)], boot_means[int(0.975 * n_iterations)]


def mcnemar(a_success, b_success):
    from math import comb
    b = sum(1 for a, m in zip(a_success, b_success) if a and not m)
    c = sum(1 for a, m in zip(a_success, b_success) if not a and m)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    larger = max(b, c)
    tail = sum(comb(n, k) * 0.5**k * 0.5**(n-k) for k in range(larger, n + 1))
    p = min(2 * tail, 1.0)
    return {"b": b, "c": c, "p": round(p, 8)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_9/development/i3_9_r1_boundary_repair",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tasks} structural OOD v3 tasks...")
    tasks = generate_structural_ood_tasks(n_tasks=args.n_tasks, split="structural_ood_v3")
    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    benchmark = EvidenceBenchmark(
        benchmark_id="i3_9_structural_ood_v3",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_9/manifests/i3_9_structural_ood_v3.json")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution: {dict(cats)}")

    # Verify oracle paths
    executor = EvidenceExecutor()
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    all_pass = True
    for task in tasks:
        runtime = initial_evidence_runtime(task, ResourceState(budget))
        current = runtime
        final = None
        for step in task.oracle_resolution_path:
            parts = step.split(":")
            action = DecisionAction(parts[0])
            target = parts[1] if len(parts) > 1 else None
            final = executor.execute(current, action, target_evidence_id=target)
            current = final.runtime
            if final.terminal:
                break
        if not final.task_success:
            all_pass = False
            print(f"  ORACLE FAIL: {task.task_id}")
    print(f"  All oracle paths succeed: {all_pass}")
    if not all_pass:
        sys.exit(1)

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks with {args.workers} workers...")
    print(f"  A=baseline, M0=frozen MDSG, M1=MDSG-StateOnly")

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

    results_path = output_dir / "boundary_repair_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Compute statistics
    n = len(all_results)
    a_s = sum(1 for r in all_results if r["a_success"])
    m0_s = sum(1 for r in all_results if r["m0_success"])
    m1_s = sum(1 for r in all_results if r["m1_success"])

    u_a = sum(r["u_a"] for r in all_results) / n
    u_m0 = sum(r["u_m0"] for r in all_results) / n
    u_m1 = sum(r["u_m1"] for r in all_results) / n

    # Bootstrap CIs
    m0_deltas = [r["m0_gain"] for r in all_results]
    m1_deltas = [r["m1_gain"] for r in all_results]
    m1_m0_deltas = [r["m1_m0_delta"] for r in all_results]

    m0_ci = paired_bootstrap_ci(m0_deltas)
    m1_ci = paired_bootstrap_ci(m1_deltas)
    m1_m0_ci = paired_bootstrap_ci(m1_m0_deltas)

    # McNemar tests
    mcnemar_m0_a = mcnemar([r["a_success"] for r in all_results],
                            [r["m0_success"] for r in all_results])
    mcnemar_m1_a = mcnemar([r["a_success"] for r in all_results],
                            [r["m1_success"] for r in all_results])
    mcnemar_m1_m0 = mcnemar([r["m0_success"] for r in all_results],
                             [r["m1_success"] for r in all_results])

    # Rescues and breaks
    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    m0_classes = Counter(classify(r["a_success"], r["m0_success"]) for r in all_results)
    m1_classes = Counter(classify(r["a_success"], r["m1_success"]) for r in all_results)
    m1_m0_classes = Counter(classify(r["m0_success"], r["m1_success"]) for r in all_results)

    # True FalseReadyRate
    m0_false_ready = sum(1 for r in all_results if r["m0_false_ready"])
    m1_false_ready = sum(1 for r in all_results if r["m1_false_ready"])
    m1_provisional_fail = sum(1 for r in all_results if r["m1_provisional_fail"])
    a_failed_answer = sum(1 for r in all_results if r["a_failed_answer"])

    # Redundant actions and steps
    a_redundant = sum(r["fork_a"]["redundant_action_count"] for r in all_results)
    m0_redundant = sum(r["fork_m0"]["redundant_action_count"] for r in all_results)
    m1_redundant = sum(r["fork_m1"]["redundant_action_count"] for r in all_results)
    a_steps = sum(r["fork_a"]["steps"] for r in all_results)
    m0_steps = sum(r["fork_m0"]["steps"] for r in all_results)
    m1_steps = sum(r["fork_m1"]["steps"] for r in all_results)

    # Subgroup analysis
    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        ca = sum(1 for r in cr if r["a_success"])
        cm0 = sum(1 for r in cr if r["m0_success"])
        cm1 = sum(1 for r in cr if r["m1_success"])
        cu_a = sum(r["u_a"] for r in cr) / cn
        cu_m0 = sum(r["u_m0"] for r in cr) / cn
        cu_m1 = sum(r["u_m1"] for r in cr) / cn
        m0_r = sum(1 for r in cr if not r["a_success"] and r["m0_success"])
        m0_b = sum(1 for r in cr if r["a_success"] and not r["m0_success"])
        m1_r = sum(1 for r in cr if not r["a_success"] and r["m1_success"])
        m1_b = sum(1 for r in cr if r["a_success"] and not r["m1_success"])
        m0_fr = sum(1 for r in cr if r["m0_false_ready"])
        m1_fr = sum(1 for r in cr if r["m1_false_ready"])

        subgroups[cat] = {
            "n": cn,
            "a_success": f"{ca}/{cn} ({ca/cn*100:.1f}%)",
            "m0_success": f"{cm0}/{cn} ({cm0/cn*100:.1f}%)",
            "m1_success": f"{cm1}/{cn} ({cm1/cn*100:.1f}%)",
            "mean_u_a": round(cu_a, 4),
            "mean_u_m0": round(cu_m0, 4),
            "mean_u_m1": round(cu_m1, 4),
            "delta_u_m0_a": round(cu_m0 - cu_a, 4),
            "delta_u_m1_a": round(cu_m1 - cu_a, 4),
            "delta_u_m1_m0": round(cu_m1 - cu_m0, 4),
            "m0_rescues": m0_r, "m0_breaks": m0_b,
            "m1_rescues": m1_r, "m1_breaks": m1_b,
            "m0_false_ready": m0_fr,
            "m1_false_ready": m1_fr,
            "m1_catastrophic_vs_a": (cm1 / cn) < (ca / cn) - 0.10,
            "m1_catastrophic_vs_m0": (cm1 / cn) < (cm0 / cn) - 0.10,
        }

    summary = {
        "schema": "DAPH_V2B_I3_9_R1_BOUNDARY_REPAIR_V1",
        "n_tasks": n,
        "arms": {
            "A": "semantic evidence baseline",
            "M0": "frozen I3.8 MDSG (with operation recommendations)",
            "M1": "MDSG-StateOnly (no operation recs, conservative READY)",
        },
        "overall": {
            "mean_u": {"A": round(u_a, 4), "M0": round(u_m0, 4), "M1": round(u_m1, 4)},
            "success": {"A": f"{a_s}/{n}", "M0": f"{m0_s}/{n}", "M1": f"{m1_s}/{n}"},
            "bootstrap_ci_m0_a": [round(m0_ci[0], 4), round(m0_ci[1], 4)],
            "bootstrap_ci_m1_a": [round(m1_ci[0], 4), round(m1_ci[1], 4)],
            "bootstrap_ci_m1_m0": [round(m1_m0_ci[0], 4), round(m1_m0_ci[1], 4)],
            "mcnemar_m0_a": mcnemar_m0_a,
            "mcnemar_m1_a": mcnemar_m1_a,
            "mcnemar_m1_m0": mcnemar_m1_m0,
            "m0_classification": dict(m0_classes),
            "m1_classification": dict(m1_classes),
            "m1_m0_classification": dict(m1_m0_classes),
            "m0_rescues": m0_classes.get("RESCUE", 0),
            "m0_breaks": m0_classes.get("BREAK", 0),
            "m1_rescues": m1_classes.get("RESCUE", 0),
            "m1_breaks": m1_classes.get("BREAK", 0),
            "true_false_ready_rate": {
                "A_failed_answer": {"count": a_failed_answer, "rate": round(a_failed_answer / n, 4)},
                "M0_false_ready": {"count": m0_false_ready, "rate": round(m0_false_ready / n, 4)},
                "M1_false_ready": {"count": m1_false_ready, "rate": round(m1_false_ready / n, 4)},
                "M1_provisional_fail": {"count": m1_provisional_fail, "rate": round(m1_provisional_fail / n, 4)},
            },
            "redundant_rate": {
                "A": round(a_redundant / max(a_steps, 1), 4),
                "M0": round(m0_redundant / max(m0_steps, 1), 4),
                "M1": round(m1_redundant / max(m1_steps, 1), 4),
            },
            "mean_steps": {
                "A": round(a_steps / n, 2),
                "M0": round(m0_steps / n, 2),
                "M1": round(m1_steps / n, 2),
            },
        },
        "subgroups": subgroups,
        "frozen_claims": {
            "C1_m1_lower_endpoint_95ci_positive": m1_ci[0] > 0,
            "C2_m1_success_gt_a": m1_s > a_s,
            "C3_m1_rescues_gt_breaks": m1_classes.get("RESCUE", 0) > m1_classes.get("BREAK", 0),
            "C4_m1_redundant_lt_a": (m1_redundant / max(m1_steps, 1)) < (a_redundant / max(a_steps, 1)),
            "C5_m1_steps_lt_a": (m1_steps / n) < (a_steps / n),
            "safety_m1_breaks_le_rescues": m1_classes.get("BREAK", 0) <= m1_classes.get("RESCUE", 0),
            "adversarial_m1_false_ready_lt_5pct": (m1_false_ready / n) < 0.05,
            "generalization_no_catastrophic": not any(sg["m1_catastrophic_vs_a"] for sg in subgroups.values()),
            "retain_m0_gain_multi_hyp": subgroups.get("multi_hypothesis_ambiguity", {}).get("m1_rescues", 0) > 15,
            "stale_support_m1_not_collapse": subgroups.get("stale_support", {}).get("m1_success", "0/30") != "0/30 (0.0%)",
        },
    }

    summary_path = output_dir / "boundary_repair_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*78}")
    print("I3.9-r1 BOUNDARY REPAIR: A vs M0 vs M1")
    print(f"{'='*78}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A={u_a:+.4f}  M0={u_m0:+.4f}  M1={u_m1:+.4f}")
    print(f"  Success:       A={a_s}/{n}  M0={m0_s}/{n}  M1={m1_s}/{n}")
    print(f"\n  Bootstrap 95% CI:")
    print(f"    M0-A: [{m0_ci[0]:+.4f}, {m0_ci[1]:+.4f}]")
    print(f"    M1-A: [{m1_ci[0]:+.4f}, {m1_ci[1]:+.4f}]")
    print(f"    M1-M0: [{m1_m0_ci[0]:+.4f}, {m1_m0_ci[1]:+.4f}]")
    print(f"\n  McNemar:")
    print(f"    M0-A: b={mcnemar_m0_a['b']}, c={mcnemar_m0_a['c']}, p={mcnemar_m0_a['p']}")
    print(f"    M1-A: b={mcnemar_m1_a['b']}, c={mcnemar_m1_a['c']}, p={mcnemar_m1_a['p']}")
    print(f"    M1-M0: b={mcnemar_m1_m0['b']}, c={mcnemar_m1_m0['c']}, p={mcnemar_m1_m0['p']}")
    print(f"\n  Rescues/Breaks:")
    print(f"    M0: rescues={m0_classes.get('RESCUE',0)}, breaks={m0_classes.get('BREAK',0)}")
    print(f"    M1: rescues={m1_classes.get('RESCUE',0)}, breaks={m1_classes.get('BREAK',0)}")
    print(f"\n  TRUE FALSE READY RATE:")
    print(f"    A failed answer: {a_failed_answer}/{n} ({a_failed_answer/n*100:.1f}%)")
    print(f"    M0 false ready:  {m0_false_ready}/{n} ({m0_false_ready/n*100:.1f}%)")
    print(f"    M1 false ready:  {m1_false_ready}/{n} ({m1_false_ready/n*100:.1f}%)")
    print(f"    M1 provisional fail: {m1_provisional_fail}/{n} ({m1_provisional_fail/n*100:.1f}%)")
    print(f"\n  Redundant rate: A={a_redundant/max(a_steps,1):.4f}  M0={m0_redundant/max(m0_steps,1):.4f}  M1={m1_redundant/max(m1_steps,1):.4f}")
    print(f"  Mean steps:     A={a_steps/n:.2f}  M0={m0_steps/n:.2f}  M1={m1_steps/n:.2f}")

    print(f"\n  SUBGROUP ANALYSIS:")
    print(f"    {'Category':<30} {'n':>3} {'A%':>6} {'M0%':>6} {'M1%':>6} {'M0_R':>5} {'M0_B':>5} {'M1_R':>5} {'M1_B':>5} {'M0_FR':>6} {'M1_FR':>6}")
    for cat, sg in subgroups.items():
        a_pct = sg["a_success"].split("(")[1].rstrip(")")
        m0_pct = sg["m0_success"].split("(")[1].rstrip(")")
        m1_pct = sg["m1_success"].split("(")[1].rstrip(")")
        print(f"    {cat:<30} {sg['n']:>3} {a_pct:>6} {m0_pct:>6} {m1_pct:>6} "
              f"{sg['m0_rescues']:>5} {sg['m0_breaks']:>5} {sg['m1_rescues']:>5} {sg['m1_breaks']:>5} "
              f"{sg['m0_false_ready']:>6} {sg['m1_false_ready']:>6}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in summary["frozen_claims"].items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
