#!/usr/bin/env python3
"""I3.9-r2: MDSG action-hint repair — A vs M0 vs M1 vs M2 on structural OOD v4.

Four arms:
  A  = baseline (semantic evidence)
  M0 = frozen I3.8 MDSG (with operation recommendations)
  M1 = MDSG-StateOnly (no hints, conservative READY)
  M2 = MDSG-StateWithHints (conservative READY + action availability hints)

M2 addresses the fundamental tension from I3.9-r1:
  M0's operation recommendations: harmful when wrong, helpful when right
  M1's no hints: safe but model can't choose RETRIEVE vs SEARCH_MORE
  M2's availability hints: safe + model knows which operations are viable

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_9_r2_action_hints.py \\
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
    arms = ["A", "M0", "M1", "M2"]
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
        "A": "BASELINE",
        "M0": "MINIMAL_DECISION_STATE",
        "M1": "MDSG_STATE_ONLY",
        "M2": "MDSG_STATE_WITH_HINTS",
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

    def compute_provisional_fail(result: dict) -> bool:
        if result["success"] or result.get("terminal_action") != "ANSWER":
            return False
        has_ready = any(e.get("decision_state") == "READY_TO_ANSWER"
                        for e in result.get("decision_state_log", []))
        has_provisional = any(e.get("decision_state") == "PROVISIONALLY_READY"
                              for e in result.get("decision_state_log", []))
        return has_provisional and not has_ready

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": len(task.hidden_evidence),
        "oracle_steps": len(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a": results["A"]["realized_utility"],
        "u_m0": results["M0"]["realized_utility"],
        "u_m1": results["M1"]["realized_utility"],
        "u_m2": results["M2"]["realized_utility"],
        "m0_gain": round(results["M0"]["realized_utility"] - results["A"]["realized_utility"], 4),
        "m1_gain": round(results["M1"]["realized_utility"] - results["A"]["realized_utility"], 4),
        "m2_gain": round(results["M2"]["realized_utility"] - results["A"]["realized_utility"], 4),
        "m2_m0_delta": round(results["M2"]["realized_utility"] - results["M0"]["realized_utility"], 4),
        "m2_m1_delta": round(results["M2"]["realized_utility"] - results["M1"]["realized_utility"], 4),
        "a_success": results["A"]["success"],
        "m0_success": results["M0"]["success"],
        "m1_success": results["M1"]["success"],
        "m2_success": results["M2"]["success"],
        "a_failed_answer": (not results["A"]["success"]
                            and results["A"].get("terminal_action") == "ANSWER"),
        "m0_false_ready": compute_false_ready(results["M0"]),
        "m1_false_ready": compute_false_ready(results["M1"]),
        "m2_false_ready": compute_false_ready(results["M2"]),
        "m1_provisional_fail": compute_provisional_fail(results["M1"]),
        "m2_provisional_fail": compute_provisional_fail(results["M2"]),
        "fork_a": results["A"],
        "fork_m0": results["M0"],
        "fork_m1": results["M1"],
        "fork_m2": results["M2"],
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
        default="experiments/v2b_i3_9/development/i3_9_r2_action_hints",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tasks} structural OOD v4 tasks...")
    tasks = generate_structural_ood_tasks(n_tasks=args.n_tasks, split="structural_ood_v4")
    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    benchmark = EvidenceBenchmark(
        benchmark_id="i3_9_structural_ood_v4",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_9/manifests/i3_9_structural_ood_v4.json")

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
    print(f"  A=baseline, M0=frozen MDSG, M1=StateOnly, M2=StateWithHints")

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

    results_path = output_dir / "action_hints_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    n = len(all_results)
    a_s = sum(1 for r in all_results if r["a_success"])
    m0_s = sum(1 for r in all_results if r["m0_success"])
    m1_s = sum(1 for r in all_results if r["m1_success"])
    m2_s = sum(1 for r in all_results if r["m2_success"])

    u_a = sum(r["u_a"] for r in all_results) / n
    u_m0 = sum(r["u_m0"] for r in all_results) / n
    u_m1 = sum(r["u_m1"] for r in all_results) / n
    u_m2 = sum(r["u_m2"] for r in all_results) / n

    m0_deltas = [r["m0_gain"] for r in all_results]
    m1_deltas = [r["m1_gain"] for r in all_results]
    m2_deltas = [r["m2_gain"] for r in all_results]
    m2_m0_deltas = [r["m2_m0_delta"] for r in all_results]
    m2_m1_deltas = [r["m2_m1_delta"] for r in all_results]

    m0_ci = paired_bootstrap_ci(m0_deltas)
    m1_ci = paired_bootstrap_ci(m1_deltas)
    m2_ci = paired_bootstrap_ci(m2_deltas)
    m2_m0_ci = paired_bootstrap_ci(m2_m0_deltas)
    m2_m1_ci = paired_bootstrap_ci(m2_m1_deltas)

    mc_m0_a = mcnemar([r["a_success"] for r in all_results], [r["m0_success"] for r in all_results])
    mc_m1_a = mcnemar([r["a_success"] for r in all_results], [r["m1_success"] for r in all_results])
    mc_m2_a = mcnemar([r["a_success"] for r in all_results], [r["m2_success"] for r in all_results])
    mc_m2_m0 = mcnemar([r["m0_success"] for r in all_results], [r["m2_success"] for r in all_results])
    mc_m2_m1 = mcnemar([r["m1_success"] for r in all_results], [r["m2_success"] for r in all_results])

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    m0_cl = Counter(classify(r["a_success"], r["m0_success"]) for r in all_results)
    m1_cl = Counter(classify(r["a_success"], r["m1_success"]) for r in all_results)
    m2_cl = Counter(classify(r["a_success"], r["m2_success"]) for r in all_results)
    m2_m0_cl = Counter(classify(r["m0_success"], r["m2_success"]) for r in all_results)
    m2_m1_cl = Counter(classify(r["m1_success"], r["m2_success"]) for r in all_results)

    m0_fr = sum(1 for r in all_results if r["m0_false_ready"])
    m1_fr = sum(1 for r in all_results if r["m1_false_ready"])
    m2_fr = sum(1 for r in all_results if r["m2_false_ready"])
    m1_pf = sum(1 for r in all_results if r["m1_provisional_fail"])
    m2_pf = sum(1 for r in all_results if r["m2_provisional_fail"])
    a_fa = sum(1 for r in all_results if r["a_failed_answer"])

    a_red = sum(r["fork_a"]["redundant_action_count"] for r in all_results)
    m0_red = sum(r["fork_m0"]["redundant_action_count"] for r in all_results)
    m1_red = sum(r["fork_m1"]["redundant_action_count"] for r in all_results)
    m2_red = sum(r["fork_m2"]["redundant_action_count"] for r in all_results)
    a_steps = sum(r["fork_a"]["steps"] for r in all_results)
    m0_steps = sum(r["fork_m0"]["steps"] for r in all_results)
    m1_steps = sum(r["fork_m1"]["steps"] for r in all_results)
    m2_steps = sum(r["fork_m2"]["steps"] for r in all_results)

    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        ca = sum(1 for r in cr if r["a_success"])
        cm0 = sum(1 for r in cr if r["m0_success"])
        cm1 = sum(1 for r in cr if r["m1_success"])
        cm2 = sum(1 for r in cr if r["m2_success"])
        cu_a = sum(r["u_a"] for r in cr) / cn
        cu_m0 = sum(r["u_m0"] for r in cr) / cn
        cu_m1 = sum(r["u_m1"] for r in cr) / cn
        cu_m2 = sum(r["u_m2"] for r in cr) / cn

        subgroups[cat] = {
            "n": cn,
            "a_success": f"{ca}/{cn} ({ca/cn*100:.1f}%)",
            "m0_success": f"{cm0}/{cn} ({cm0/cn*100:.1f}%)",
            "m1_success": f"{cm1}/{cn} ({cm1/cn*100:.1f}%)",
            "m2_success": f"{cm2}/{cn} ({cm2/cn*100:.1f}%)",
            "mean_u_a": round(cu_a, 4), "mean_u_m0": round(cu_m0, 4),
            "mean_u_m1": round(cu_m1, 4), "mean_u_m2": round(cu_m2, 4),
            "delta_u_m0_a": round(cu_m0 - cu_a, 4),
            "delta_u_m1_a": round(cu_m1 - cu_a, 4),
            "delta_u_m2_a": round(cu_m2 - cu_a, 4),
            "delta_u_m2_m0": round(cu_m2 - cu_m0, 4),
            "delta_u_m2_m1": round(cu_m2 - cu_m1, 4),
            "m0_rescues": sum(1 for r in cr if not r["a_success"] and r["m0_success"]),
            "m0_breaks": sum(1 for r in cr if r["a_success"] and not r["m0_success"]),
            "m1_rescues": sum(1 for r in cr if not r["a_success"] and r["m1_success"]),
            "m1_breaks": sum(1 for r in cr if r["a_success"] and not r["m1_success"]),
            "m2_rescues": sum(1 for r in cr if not r["a_success"] and r["m2_success"]),
            "m2_breaks": sum(1 for r in cr if r["a_success"] and not r["m2_success"]),
            "m0_false_ready": sum(1 for r in cr if r["m0_false_ready"]),
            "m1_false_ready": sum(1 for r in cr if r["m1_false_ready"]),
            "m2_false_ready": sum(1 for r in cr if r["m2_false_ready"]),
            "m2_catastrophic_vs_a": (cm2 / cn) < (ca / cn) - 0.10,
        }

    summary = {
        "schema": "DAPH_V2B_I3_9_R2_ACTION_HINTS_V1",
        "n_tasks": n,
        "arms": {
            "A": "semantic evidence baseline",
            "M0": "frozen I3.8 MDSG (with operation recommendations)",
            "M1": "MDSG-StateOnly (no hints, conservative READY)",
            "M2": "MDSG-StateWithHints (conservative READY + action availability hints)",
        },
        "overall": {
            "mean_u": {"A": round(u_a, 4), "M0": round(u_m0, 4),
                        "M1": round(u_m1, 4), "M2": round(u_m2, 4)},
            "success": {"A": f"{a_s}/{n}", "M0": f"{m0_s}/{n}",
                         "M1": f"{m1_s}/{n}", "M2": f"{m2_s}/{n}"},
            "bootstrap_ci_m0_a": [round(m0_ci[0], 4), round(m0_ci[1], 4)],
            "bootstrap_ci_m1_a": [round(m1_ci[0], 4), round(m1_ci[1], 4)],
            "bootstrap_ci_m2_a": [round(m2_ci[0], 4), round(m2_ci[1], 4)],
            "bootstrap_ci_m2_m0": [round(m2_m0_ci[0], 4), round(m2_m0_ci[1], 4)],
            "bootstrap_ci_m2_m1": [round(m2_m1_ci[0], 4), round(m2_m1_ci[1], 4)],
            "mcnemar_m0_a": mc_m0_a, "mcnemar_m1_a": mc_m1_a, "mcnemar_m2_a": mc_m2_a,
            "mcnemar_m2_m0": mc_m2_m0, "mcnemar_m2_m1": mc_m2_m1,
            "m0_classification": dict(m0_cl), "m1_classification": dict(m1_cl),
            "m2_classification": dict(m2_cl),
            "m2_m0_classification": dict(m2_m0_cl),
            "m2_m1_classification": dict(m2_m1_cl),
            "m0_rescues": m0_cl.get("RESCUE", 0), "m0_breaks": m0_cl.get("BREAK", 0),
            "m1_rescues": m1_cl.get("RESCUE", 0), "m1_breaks": m1_cl.get("BREAK", 0),
            "m2_rescues": m2_cl.get("RESCUE", 0), "m2_breaks": m2_cl.get("BREAK", 0),
            "true_false_ready_rate": {
                "A_failed_answer": {"count": a_fa, "rate": round(a_fa / n, 4)},
                "M0_false_ready": {"count": m0_fr, "rate": round(m0_fr / n, 4)},
                "M1_false_ready": {"count": m1_fr, "rate": round(m1_fr / n, 4)},
                "M2_false_ready": {"count": m2_fr, "rate": round(m2_fr / n, 4)},
                "M1_provisional_fail": {"count": m1_pf, "rate": round(m1_pf / n, 4)},
                "M2_provisional_fail": {"count": m2_pf, "rate": round(m2_pf / n, 4)},
            },
            "redundant_rate": {
                "A": round(a_red / max(a_steps, 1), 4),
                "M0": round(m0_red / max(m0_steps, 1), 4),
                "M1": round(m1_red / max(m1_steps, 1), 4),
                "M2": round(m2_red / max(m2_steps, 1), 4),
            },
            "mean_steps": {
                "A": round(a_steps / n, 2), "M0": round(m0_steps / n, 2),
                "M1": round(m1_steps / n, 2), "M2": round(m2_steps / n, 2),
            },
        },
        "subgroups": subgroups,
        "frozen_claims": {
            "C1_m2_lower_endpoint_95ci_positive": m2_ci[0] > 0,
            "C2_m2_success_gt_a": m2_s > a_s,
            "C3_m2_rescues_gt_breaks": m2_cl.get("RESCUE", 0) > m2_cl.get("BREAK", 0),
            "C4_m2_redundant_lt_a": (m2_red / max(m2_steps, 1)) < (a_red / max(a_steps, 1)),
            "C5_m2_steps_lt_a": (m2_steps / n) < (a_steps / n),
            "safety_m2_breaks_le_rescues": m2_cl.get("BREAK", 0) <= m2_cl.get("RESCUE", 0),
            "adversarial_m2_false_ready_lt_5pct": (m2_fr / n) < 0.05,
            "generalization_no_catastrophic": not any(sg["m2_catastrophic_vs_a"] for sg in subgroups.values()),
            "retain_m0_multi_hyp": subgroups.get("multi_hypothesis_ambiguity", {}).get("m2_rescues", 0) > 15,
            "stale_support_m2_repair": subgroups.get("stale_support", {}).get("m2_success", "0/30") != "0/30 (0.0%)",
            "retain_m1_conflict_unresolved": subgroups.get("conflict_unresolved", {}).get("m2_rescues", 0) > 15,
            "m2_gt_m0_ci_positive": m2_m0_ci[0] > 0,
            "m2_gt_m1_ci_positive": m2_m1_ci[0] > 0,
        },
    }

    summary_path = output_dir / "action_hints_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*82}")
    print("I3.9-r2 ACTION HINTS: A vs M0 vs M1 vs M2")
    print(f"{'='*82}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A={u_a:+.4f}  M0={u_m0:+.4f}  M1={u_m1:+.4f}  M2={u_m2:+.4f}")
    print(f"  Success:       A={a_s}/{n}  M0={m0_s}/{n}  M1={m1_s}/{n}  M2={m2_s}/{n}")
    print(f"\n  Bootstrap 95% CI:")
    print(f"    M0-A:  [{m0_ci[0]:+.4f}, {m0_ci[1]:+.4f}]")
    print(f"    M1-A:  [{m1_ci[0]:+.4f}, {m1_ci[1]:+.4f}]")
    print(f"    M2-A:  [{m2_ci[0]:+.4f}, {m2_ci[1]:+.4f}]")
    print(f"    M2-M0: [{m2_m0_ci[0]:+.4f}, {m2_m0_ci[1]:+.4f}]")
    print(f"    M2-M1: [{m2_m1_ci[0]:+.4f}, {m2_m1_ci[1]:+.4f}]")
    print(f"\n  McNemar:")
    print(f"    M0-A:  b={mc_m0_a['b']}, c={mc_m0_a['c']}, p={mc_m0_a['p']}")
    print(f"    M1-A:  b={mc_m1_a['b']}, c={mc_m1_a['c']}, p={mc_m1_a['p']}")
    print(f"    M2-A:  b={mc_m2_a['b']}, c={mc_m2_a['c']}, p={mc_m2_a['p']}")
    print(f"    M2-M0: b={mc_m2_m0['b']}, c={mc_m2_m0['c']}, p={mc_m2_m0['p']}")
    print(f"    M2-M1: b={mc_m2_m1['b']}, c={mc_m2_m1['c']}, p={mc_m2_m1['p']}")
    print(f"\n  Rescues/Breaks:")
    print(f"    M0: rescues={m0_cl.get('RESCUE',0)}, breaks={m0_cl.get('BREAK',0)}")
    print(f"    M1: rescues={m1_cl.get('RESCUE',0)}, breaks={m1_cl.get('BREAK',0)}")
    print(f"    M2: rescues={m2_cl.get('RESCUE',0)}, breaks={m2_cl.get('BREAK',0)}")
    print(f"\n  TRUE FALSE READY RATE:")
    print(f"    A failed answer:  {a_fa}/{n} ({a_fa/n*100:.1f}%)")
    print(f"    M0 false ready:   {m0_fr}/{n} ({m0_fr/n*100:.1f}%)")
    print(f"    M1 false ready:   {m1_fr}/{n} ({m1_fr/n*100:.1f}%)")
    print(f"    M2 false ready:   {m2_fr}/{n} ({m2_fr/n*100:.1f}%)")
    print(f"    M1 provisional:   {m1_pf}/{n} ({m1_pf/n*100:.1f}%)")
    print(f"    M2 provisional:   {m2_pf}/{n} ({m2_pf/n*100:.1f}%)")
    print(f"\n  Redundant rate: A={a_red/max(a_steps,1):.4f}  M0={m0_red/max(m0_steps,1):.4f}  M1={m1_red/max(m1_steps,1):.4f}  M2={m2_red/max(m2_steps,1):.4f}")
    print(f"  Mean steps:     A={a_steps/n:.2f}  M0={m0_steps/n:.2f}  M1={m1_steps/n:.2f}  M2={m2_steps/n:.2f}")

    print(f"\n  SUBGROUP ANALYSIS:")
    print(f"    {'Category':<30} {'n':>3} {'A%':>6} {'M0%':>6} {'M1%':>6} {'M2%':>6} {'M0_R':>5} {'M1_R':>5} {'M2_R':>5} {'M0_B':>5} {'M1_B':>5} {'M2_B':>5} {'M0FR':>5} {'M1FR':>5} {'M2FR':>5}")
    for cat, sg in subgroups.items():
        a_pct = sg["a_success"].split("(")[1].rstrip(")")
        m0_pct = sg["m0_success"].split("(")[1].rstrip(")")
        m1_pct = sg["m1_success"].split("(")[1].rstrip(")")
        m2_pct = sg["m2_success"].split("(")[1].rstrip(")")
        print(f"    {cat:<30} {sg['n']:>3} {a_pct:>6} {m0_pct:>6} {m1_pct:>6} {m2_pct:>6} "
              f"{sg['m0_rescues']:>5} {sg['m1_rescues']:>5} {sg['m2_rescues']:>5} "
              f"{sg['m0_breaks']:>5} {sg['m1_breaks']:>5} {sg['m2_breaks']:>5} "
              f"{sg['m0_false_ready']:>5} {sg['m1_false_ready']:>5} {sg['m2_false_ready']:>5}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in summary["frozen_claims"].items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
