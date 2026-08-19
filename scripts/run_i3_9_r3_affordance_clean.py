#!/usr/bin/env python3
"""I3.9-r3: Affordance-clean MDSG — A0 vs A1 vs M1 vs M3 on structural OOD v5.

Four arms with clean causal decomposition:
  A0 = original baseline (no MDSG, no affordances)
  A1 = baseline + public affordances (can_retrieve/can_search/can_verify)
  M1 = MDSG-StateOnly (no hints, conservative READY, SUPPORTED_BUT_UNRESOLVED)
  M3 = MDSG-StateWithAffordances (state + clean affordances)

Primary comparison: M3-A1 (both have identical affordance info)
  This isolates the value of MDSG state compression.

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_9_r3_affordance_clean.py \\
        --n-tasks 300 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
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


def counterbalance_4arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    arms = ["A0", "A1", "M1", "M3"]
    perms = list(itertools.permutations(arms))
    return list(perms[int(h[:8], 16) % len(perms)])


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    fork_order = counterbalance_4arm(task.task_id)
    arm_modes = {
        "A0": "BASELINE",
        "A1": "BASELINE_WITH_AFFORDANCES",
        "M1": "MDSG_STATE_ONLY",
        "M3": "MDSG_STATE_WITH_AFFORDANCES",
    }

    results: dict[str, dict] = {}
    for arm_id in fork_order:
        results[arm_id] = i3_7e.run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=arm_modes[arm_id], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    def compute_false_ready(result: dict) -> bool:
        if result["success"] or result.get("terminal_action") != "ANSWER":
            return False
        for entry in result.get("decision_state_log", []):
            if entry.get("decision_state") == "READY_TO_ANSWER":
                return True
        return False

    def compute_unresolved_fail(result: dict) -> bool:
        """Model ANSWERed and failed after seeing SUPPORTED_BUT_UNRESOLVED."""
        if result["success"] or result.get("terminal_action") != "ANSWER":
            return False
        has_ready = any(e.get("decision_state") == "READY_TO_ANSWER"
                        for e in result.get("decision_state_log", []))
        has_unresolved = any(e.get("decision_state") == "SUPPORTED_BUT_UNRESOLVED"
                             for e in result.get("decision_state_log", []))
        return has_unresolved and not has_ready

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": len(task.hidden_evidence),
        "oracle_steps": len(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a0": results["A0"]["realized_utility"],
        "u_a1": results["A1"]["realized_utility"],
        "u_m1": results["M1"]["realized_utility"],
        "u_m3": results["M3"]["realized_utility"],
        "m3_a1_gain": round(results["M3"]["realized_utility"] - results["A1"]["realized_utility"], 4),
        "m3_a0_gain": round(results["M3"]["realized_utility"] - results["A0"]["realized_utility"], 4),
        "a1_a0_gain": round(results["A1"]["realized_utility"] - results["A0"]["realized_utility"], 4),
        "m3_m1_delta": round(results["M3"]["realized_utility"] - results["M1"]["realized_utility"], 4),
        "a0_success": results["A0"]["success"],
        "a1_success": results["A1"]["success"],
        "m1_success": results["M1"]["success"],
        "m3_success": results["M3"]["success"],
        "a0_failed_answer": (not results["A0"]["success"]
                             and results["A0"].get("terminal_action") == "ANSWER"),
        "m1_false_ready": compute_false_ready(results["M1"]),
        "m3_false_ready": compute_false_ready(results["M3"]),
        "m1_unresolved_fail": compute_unresolved_fail(results["M1"]),
        "m3_unresolved_fail": compute_unresolved_fail(results["M3"]),
        "fork_a0": results["A0"],
        "fork_a1": results["A1"],
        "fork_m1": results["M1"],
        "fork_m3": results["M3"],
    }


def paired_bootstrap_ci(deltas, n_iterations=10000, seed=42):
    rng = random.Random(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    return boot_means[int(0.025 * n_iterations)], boot_means[int(0.975 * n_iterations)]


def mcnemar(a_success, b_success):
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
        default="experiments/v2b_i3_9/development/i3_9_r3_affordance_clean",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tasks} structural OOD v5 tasks...")
    tasks = generate_structural_ood_tasks(n_tasks=args.n_tasks, split="structural_ood_v5")
    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    benchmark = EvidenceBenchmark(
        benchmark_id="i3_9_structural_ood_v5",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_9/manifests/i3_9_structural_ood_v5.json")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution: {dict(cats)}")

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
    print(f"  A0=baseline, A1=baseline+affordances, M1=StateOnly, M3=State+Affordances")

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

    results_path = output_dir / "affordance_clean_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    n = len(all_results)
    a0_s = sum(1 for r in all_results if r["a0_success"])
    a1_s = sum(1 for r in all_results if r["a1_success"])
    m1_s = sum(1 for r in all_results if r["m1_success"])
    m3_s = sum(1 for r in all_results if r["m3_success"])

    u_a0 = sum(r["u_a0"] for r in all_results) / n
    u_a1 = sum(r["u_a1"] for r in all_results) / n
    u_m1 = sum(r["u_m1"] for r in all_results) / n
    u_m3 = sum(r["u_m3"] for r in all_results) / n

    m3_a1_deltas = [r["m3_a1_gain"] for r in all_results]
    m3_a0_deltas = [r["m3_a0_gain"] for r in all_results]
    a1_a0_deltas = [r["a1_a0_gain"] for r in all_results]
    m3_m1_deltas = [r["m3_m1_delta"] for r in all_results]

    m3_a1_ci = paired_bootstrap_ci(m3_a1_deltas)
    m3_a0_ci = paired_bootstrap_ci(m3_a0_deltas)
    a1_a0_ci = paired_bootstrap_ci(a1_a0_deltas)
    m3_m1_ci = paired_bootstrap_ci(m3_m1_deltas)

    mc_m3_a1 = mcnemar([r["a1_success"] for r in all_results], [r["m3_success"] for r in all_results])
    mc_m3_a0 = mcnemar([r["a0_success"] for r in all_results], [r["m3_success"] for r in all_results])
    mc_a1_a0 = mcnemar([r["a0_success"] for r in all_results], [r["a1_success"] for r in all_results])
    mc_m3_m1 = mcnemar([r["m1_success"] for r in all_results], [r["m3_success"] for r in all_results])

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    m3_a1_cl = Counter(classify(r["a1_success"], r["m3_success"]) for r in all_results)
    m3_a0_cl = Counter(classify(r["a0_success"], r["m3_success"]) for r in all_results)
    a1_a0_cl = Counter(classify(r["a0_success"], r["a1_success"]) for r in all_results)
    m3_m1_cl = Counter(classify(r["m1_success"], r["m3_success"]) for r in all_results)

    m1_fr = sum(1 for r in all_results if r["m1_false_ready"])
    m3_fr = sum(1 for r in all_results if r["m3_false_ready"])
    m1_uf = sum(1 for r in all_results if r["m1_unresolved_fail"])
    m3_uf = sum(1 for r in all_results if r["m3_unresolved_fail"])
    a0_fa = sum(1 for r in all_results if r["a0_failed_answer"])

    a0_red = sum(r["fork_a0"]["redundant_action_count"] for r in all_results)
    a1_red = sum(r["fork_a1"]["redundant_action_count"] for r in all_results)
    m1_red = sum(r["fork_m1"]["redundant_action_count"] for r in all_results)
    m3_red = sum(r["fork_m3"]["redundant_action_count"] for r in all_results)
    a0_steps = sum(r["fork_a0"]["steps"] for r in all_results)
    a1_steps = sum(r["fork_a1"]["steps"] for r in all_results)
    m1_steps = sum(r["fork_m1"]["steps"] for r in all_results)
    m3_steps = sum(r["fork_m3"]["steps"] for r in all_results)

    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        ca0 = sum(1 for r in cr if r["a0_success"])
        ca1 = sum(1 for r in cr if r["a1_success"])
        cm1 = sum(1 for r in cr if r["m1_success"])
        cm3 = sum(1 for r in cr if r["m3_success"])

        subgroups[cat] = {
            "n": cn,
            "a0_success": f"{ca0}/{cn} ({ca0/cn*100:.1f}%)",
            "a1_success": f"{ca1}/{cn} ({ca1/cn*100:.1f}%)",
            "m1_success": f"{cm1}/{cn} ({cm1/cn*100:.1f}%)",
            "m3_success": f"{cm3}/{cn} ({cm3/cn*100:.1f}%)",
            "mean_u_a0": round(sum(r["u_a0"] for r in cr) / cn, 4),
            "mean_u_a1": round(sum(r["u_a1"] for r in cr) / cn, 4),
            "mean_u_m1": round(sum(r["u_m1"] for r in cr) / cn, 4),
            "mean_u_m3": round(sum(r["u_m3"] for r in cr) / cn, 4),
            "delta_u_m3_a1": round(sum(r["u_m3"] for r in cr) / cn - sum(r["u_a1"] for r in cr) / cn, 4),
            "delta_u_m3_a0": round(sum(r["u_m3"] for r in cr) / cn - sum(r["u_a0"] for r in cr) / cn, 4),
            "delta_u_a1_a0": round(sum(r["u_a1"] for r in cr) / cn - sum(r["u_a0"] for r in cr) / cn, 4),
            "m3_a1_rescues": sum(1 for r in cr if not r["a1_success"] and r["m3_success"]),
            "m3_a1_breaks": sum(1 for r in cr if r["a1_success"] and not r["m3_success"]),
            "m3_a0_rescues": sum(1 for r in cr if not r["a0_success"] and r["m3_success"]),
            "m3_a0_breaks": sum(1 for r in cr if r["a0_success"] and not r["m3_success"]),
            "m3_false_ready": sum(1 for r in cr if r["m3_false_ready"]),
            "m3_catastrophic_vs_a1": (cm3 / cn) < (ca1 / cn) - 0.10,
        }

    summary = {
        "schema": "DAPH_V2B_I3_9_R3_AFFORDANCE_CLEAN_V1",
        "n_tasks": n,
        "arms": {
            "A0": "original baseline (no MDSG, no affordances)",
            "A1": "baseline + public affordances (can_retrieve/can_search/can_verify)",
            "M1": "MDSG-StateOnly (SUPPORTED_BUT_UNRESOLVED, no affordances)",
            "M3": "MDSG-StateWithAffordances (state + clean affordances)",
        },
        "overall": {
            "mean_u": {"A0": round(u_a0, 4), "A1": round(u_a1, 4),
                        "M1": round(u_m1, 4), "M3": round(u_m3, 4)},
            "success": {"A0": f"{a0_s}/{n}", "A1": f"{a1_s}/{n}",
                         "M1": f"{m1_s}/{n}", "M3": f"{m3_s}/{n}"},
            "bootstrap_ci_m3_a1": [round(m3_a1_ci[0], 4), round(m3_a1_ci[1], 4)],
            "bootstrap_ci_m3_a0": [round(m3_a0_ci[0], 4), round(m3_a0_ci[1], 4)],
            "bootstrap_ci_a1_a0": [round(a1_a0_ci[0], 4), round(a1_a0_ci[1], 4)],
            "bootstrap_ci_m3_m1": [round(m3_m1_ci[0], 4), round(m3_m1_ci[1], 4)],
            "mcnemar_m3_a1": mc_m3_a1, "mcnemar_m3_a0": mc_m3_a0,
            "mcnemar_a1_a0": mc_a1_a0, "mcnemar_m3_m1": mc_m3_m1,
            "m3_a1_classification": dict(m3_a1_cl),
            "m3_a0_classification": dict(m3_a0_cl),
            "a1_a0_classification": dict(a1_a0_cl),
            "m3_m1_classification": dict(m3_m1_cl),
            "m3_a1_rescues": m3_a1_cl.get("RESCUE", 0),
            "m3_a1_breaks": m3_a1_cl.get("BREAK", 0),
            "true_false_ready_rate": {
                "A0_failed_answer": {"count": a0_fa, "rate": round(a0_fa / n, 4)},
                "M1_false_ready": {"count": m1_fr, "rate": round(m1_fr / n, 4)},
                "M3_false_ready": {"count": m3_fr, "rate": round(m3_fr / n, 4)},
                "M1_unresolved_fail": {"count": m1_uf, "rate": round(m1_uf / n, 4)},
                "M3_unresolved_fail": {"count": m3_uf, "rate": round(m3_uf / n, 4)},
            },
            "redundant_rate": {
                "A0": round(a0_red / max(a0_steps, 1), 4),
                "A1": round(a1_red / max(a1_steps, 1), 4),
                "M1": round(m1_red / max(m1_steps, 1), 4),
                "M3": round(m3_red / max(m3_steps, 1), 4),
            },
            "mean_steps": {
                "A0": round(a0_steps / n, 2), "A1": round(a1_steps / n, 2),
                "M1": round(m1_steps / n, 2), "M3": round(m3_steps / n, 2),
            },
        },
        "subgroups": subgroups,
        "frozen_claims": {
            "C1_m3_a1_lower_endpoint_95ci_positive": m3_a1_ci[0] > 0,
            "C2_m3_success_gt_a1": m3_s > a1_s,
            "C3_m3_a1_rescues_gt_breaks": m3_a1_cl.get("RESCUE", 0) > m3_a1_cl.get("BREAK", 0),
            "C4_m3_redundant_lt_a1": (m3_red / max(m3_steps, 1)) < (a1_red / max(a1_steps, 1)),
            "C5_m3_steps_lt_a1": (m3_steps / n) < (a1_steps / n),
            "safety_m3_breaks_le_rescues": m3_a1_cl.get("BREAK", 0) <= m3_a1_cl.get("RESCUE", 0),
            "adversarial_m3_false_ready_lt_5pct": (m3_fr / n) < 0.05,
            "generalization_no_catastrophic_vs_a1": not any(sg["m3_catastrophic_vs_a1"] for sg in subgroups.values()),
            "early_false_ready_m3_within_10pp_a1": not subgroups.get("early_false_ready", {}).get("m3_catastrophic_vs_a1", False),
            "stale_support_m3_no_collapse": subgroups.get("stale_support", {}).get("m3_success", "0/30") != "0/30 (0.0%)",
            "multi_hyp_m3_retain": subgroups.get("multi_hypothesis_ambiguity", {}).get("m3_a1_rescues", 0) > 10,
            "conflict_unresolved_m3_retain": subgroups.get("conflict_unresolved", {}).get("m3_a1_rescues", 0) > 10,
            "m3_gt_m1_ci_positive": m3_m1_ci[0] > 0,
        },
    }

    summary_path = output_dir / "affordance_clean_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*82}")
    print("I3.9-r3 AFFORDANCE-CLEAN: A0 vs A1 vs M1 vs M3")
    print(f"{'='*82}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A0={u_a0:+.4f}  A1={u_a1:+.4f}  M1={u_m1:+.4f}  M3={u_m3:+.4f}")
    print(f"  Success:       A0={a0_s}/{n}  A1={a1_s}/{n}  M1={m1_s}/{n}  M3={m3_s}/{n}")
    print(f"\n  Bootstrap 95% CI (PRIMARY comparisons):")
    print(f"    M3-A1: [{m3_a1_ci[0]:+.4f}, {m3_a1_ci[1]:+.4f}]  <-- PRIMARY")
    print(f"    M3-A0: [{m3_a0_ci[0]:+.4f}, {m3_a0_ci[1]:+.4f}]")
    print(f"    A1-A0: [{a1_a0_ci[0]:+.4f}, {a1_a0_ci[1]:+.4f}]  <-- affordance value alone")
    print(f"    M3-M1: [{m3_m1_ci[0]:+.4f}, {m3_m1_ci[1]:+.4f}]  <-- affordance value to MDSG")
    print(f"\n  McNemar:")
    print(f"    M3-A1: b={mc_m3_a1['b']}, c={mc_m3_a1['c']}, p={mc_m3_a1['p']}")
    print(f"    M3-A0: b={mc_m3_a0['b']}, c={mc_m3_a0['c']}, p={mc_m3_a0['p']}")
    print(f"    A1-A0: b={mc_a1_a0['b']}, c={mc_a1_a0['c']}, p={mc_a1_a0['p']}")
    print(f"    M3-M1: b={mc_m3_m1['b']}, c={mc_m3_m1['c']}, p={mc_m3_m1['p']}")
    print(f"\n  M3 vs A1: rescues={m3_a1_cl.get('RESCUE',0)}, breaks={m3_a1_cl.get('BREAK',0)}")
    print(f"\n  TRUE FALSE READY RATE:")
    print(f"    A0 failed answer:  {a0_fa}/{n} ({a0_fa/n*100:.1f}%)")
    print(f"    M1 false ready:    {m1_fr}/{n} ({m1_fr/n*100:.1f}%)")
    print(f"    M3 false ready:    {m3_fr}/{n} ({m3_fr/n*100:.1f}%)")
    print(f"    M1 unresolved fail:{m1_uf}/{n} ({m1_uf/n*100:.1f}%)")
    print(f"    M3 unresolved fail:{m3_uf}/{n} ({m3_uf/n*100:.1f}%)")
    print(f"\n  Redundant: A0={a0_red/max(a0_steps,1):.4f} A1={a1_red/max(a1_steps,1):.4f} M1={m1_red/max(m1_steps,1):.4f} M3={m3_red/max(m3_steps,1):.4f}")
    print(f"  Steps:     A0={a0_steps/n:.2f} A1={a1_steps/n:.2f} M1={m1_steps/n:.2f} M3={m3_steps/n:.2f}")

    print(f"\n  SUBGROUP ANALYSIS:")
    print(f"    {'Category':<30} {'n':>3} {'A0%':>6} {'A1%':>6} {'M1%':>6} {'M3%':>6} {'M3-A1_R':>7} {'M3-A1_B':>7} {'M3FR':>5}")
    for cat, sg in subgroups.items():
        a0p = sg["a0_success"].split("(")[1].rstrip(")")
        a1p = sg["a1_success"].split("(")[1].rstrip(")")
        m1p = sg["m1_success"].split("(")[1].rstrip(")")
        m3p = sg["m3_success"].split("(")[1].rstrip(")")
        print(f"    {cat:<30} {sg['n']:>3} {a0p:>6} {a1p:>6} {m1p:>6} {m3p:>6} "
              f"{sg['m3_a1_rescues']:>7} {sg['m3_a1_breaks']:>7} {sg['m3_false_ready']:>5}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in summary["frozen_claims"].items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
