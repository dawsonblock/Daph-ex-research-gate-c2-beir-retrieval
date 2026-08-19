#!/usr/bin/env python3
"""I3.10b: Efficient MDSG — A1 vs M3 vs M4a vs M4b on efficiency corpus.

Four arms:
  A1  = baseline + public affordances
  M3  = frozen MDSG-StateWithAffordances (efficiency FAIL)
  M4a = M3 + observed_resolution_strength
  M4b = M3 + evidence_pipeline_state

Primary comparison: M4x vs M3 (can we improve efficiency without sacrificing correctness?)

Fresh efficiency-development corpus with adversarial safeguards:
  Overweights SBU bulk-retrieval cases but includes cases where
  hidden evidence genuinely must be acquired.

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_10b_efficient_mdsg.py \\
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


def generate_efficiency_corpus(n_tasks: int = 300, split: str = "efficiency_dev_v1"):
    """Generate a fresh efficiency-development corpus.

    Overweights SBU bulk-retrieval cases but includes adversarial safeguards
    where hidden evidence genuinely must be acquired.

    Target distribution (300 tasks):
      25% SBU bulk-retrieval cases (single_verify_ready, triple_verify_ready, noise_evidence)
      20% single_verify_ready
      15% varying_visible_split
      10% late_resolution
      10% early_false_ready (adversarial: hidden evidence must be checked)
      10% conflict_unresolved (adversarial: must DEFER)
      10% multi_hypothesis_ambiguity + stale_support
    """
    # Generate extra tasks from each category and sample the target distribution
    target_counts = {
        "single_verify_ready": 75,      # 25% SBU bulk-retrieval
        "triple_verify_ready": 30,      # 10% SBU bulk-retrieval
        "noise_evidence": 30,           # 10% SBU bulk-retrieval
        "varying_visible_split": 45,    # 15%
        "late_resolution": 30,          # 10%
        "early_false_ready": 30,        # 10% adversarial
        "conflict_unresolved": 30,      # 10% adversarial
        "multi_hypothesis_ambiguity": 15,  # 5%
        "stale_support": 15,            # 5%
    }

    # Generate enough tasks per category
    total_needed = sum(target_counts.values())
    tasks_per_cat = max(target_counts.values()) + 10

    all_tasks = []
    for cat, count in target_counts.items():
        cat_tasks = generate_structural_ood_tasks(
            n_tasks=tasks_per_cat,
            split=f"{split}_{cat}",
            category_filter=cat,
        )
        # Take the first 'count' tasks
        all_tasks.extend(cat_tasks[:count])

    # Renumber task IDs to be sequential
    for i, task in enumerate(all_tasks):
        object.__setattr__(task, 'task_id', f"{split}_{i:04d}")

    return all_tasks[:total_needed]


def counterbalance_4arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    arms = ["A1", "M3", "M4a", "M4b"]
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
        "A1": "BASELINE_WITH_AFFORDANCES",
        "M3": "MDSG_STATE_WITH_AFFORDANCES",
        "M4a": "MDSG_OBSERVED_RESOLUTION",
        "M4b": "MDSG_PIPELINE_STATE",
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

    def compute_premature_termination(result: dict) -> dict:
        """Model ANSWERs from any non-READY state AND task fails."""
        if result["success"] or result.get("terminal_action") != "ANSWER":
            return {"premature": False, "state": None}
        log = result.get("decision_state_log", [])
        actions = result.get("continuation_actions", [])
        answer_step = len(actions) - 1
        if answer_step < len(log):
            state = log[answer_step].get("decision_state", "UNKNOWN")
            if state != "READY_TO_ANSWER":
                return {"premature": True, "state": state}
        return {"premature": False, "state": None}

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": len(task.hidden_evidence),
        "oracle_steps": len(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a1": results["A1"]["realized_utility"],
        "u_m3": results["M3"]["realized_utility"],
        "u_m4a": results["M4a"]["realized_utility"],
        "u_m4b": results["M4b"]["realized_utility"],
        "m4a_m3_delta": round(results["M4a"]["realized_utility"] - results["M3"]["realized_utility"], 4),
        "m4b_m3_delta": round(results["M4b"]["realized_utility"] - results["M3"]["realized_utility"], 4),
        "m4a_a1_gain": round(results["M4a"]["realized_utility"] - results["A1"]["realized_utility"], 4),
        "m4b_a1_gain": round(results["M4b"]["realized_utility"] - results["A1"]["realized_utility"], 4),
        "a1_success": results["A1"]["success"],
        "m3_success": results["M3"]["success"],
        "m4a_success": results["M4a"]["success"],
        "m4b_success": results["M4b"]["success"],
        "m3_false_ready": compute_false_ready(results["M3"]),
        "m4a_false_ready": compute_false_ready(results["M4a"]),
        "m4b_false_ready": compute_false_ready(results["M4b"]),
        "m3_premature": compute_premature_termination(results["M3"]),
        "m4a_premature": compute_premature_termination(results["M4a"]),
        "m4b_premature": compute_premature_termination(results["M4b"]),
        "fork_a1": results["A1"],
        "fork_m3": results["M3"],
        "fork_m4a": results["M4a"],
        "fork_m4b": results["M4b"],
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
        default="experiments/v2b_i3_10/development/i3_10b_efficient",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tasks} efficiency-development tasks...")
    tasks = generate_efficiency_corpus(n_tasks=args.n_tasks, split="efficiency_dev_v1")
    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    benchmark = EvidenceBenchmark(
        benchmark_id="i3_10_efficiency_dev_v1",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_9/manifests/i3_10_efficiency_dev_v1.json")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution: {dict(cats)}")

    # Oracle validation
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
    print(f"  A1=baseline+affordances, M3=frozen, M4a=observed_resolution, M4b=pipeline_state")

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

    results_path = output_dir / "efficient_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    n = len(all_results)
    a1_s = sum(1 for r in all_results if r["a1_success"])
    m3_s = sum(1 for r in all_results if r["m3_success"])
    m4a_s = sum(1 for r in all_results if r["m4a_success"])
    m4b_s = sum(1 for r in all_results if r["m4b_success"])

    u_a1 = sum(r["u_a1"] for r in all_results) / n
    u_m3 = sum(r["u_m3"] for r in all_results) / n
    u_m4a = sum(r["u_m4a"] for r in all_results) / n
    u_m4b = sum(r["u_m4b"] for r in all_results) / n

    m4a_m3_deltas = [r["m4a_m3_delta"] for r in all_results]
    m4b_m3_deltas = [r["m4b_m3_delta"] for r in all_results]
    m4a_a1_deltas = [r["m4a_a1_gain"] for r in all_results]
    m4b_a1_deltas = [r["m4b_a1_gain"] for r in all_results]

    m4a_m3_ci = paired_bootstrap_ci(m4a_m3_deltas)
    m4b_m3_ci = paired_bootstrap_ci(m4b_m3_deltas)
    m4a_a1_ci = paired_bootstrap_ci(m4a_a1_deltas)
    m4b_a1_ci = paired_bootstrap_ci(m4b_a1_deltas)

    mc_m4a_m3 = mcnemar([r["m3_success"] for r in all_results], [r["m4a_success"] for r in all_results])
    mc_m4b_m3 = mcnemar([r["m3_success"] for r in all_results], [r["m4b_success"] for r in all_results])

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    m4a_m3_cl = Counter(classify(r["m3_success"], r["m4a_success"]) for r in all_results)
    m4b_m3_cl = Counter(classify(r["m3_success"], r["m4b_success"]) for r in all_results)

    m3_fr = sum(1 for r in all_results if r["m3_false_ready"])
    m4a_fr = sum(1 for r in all_results if r["m4a_false_ready"])
    m4b_fr = sum(1 for r in all_results if r["m4b_false_ready"])

    m3_pt = sum(1 for r in all_results if r["m3_premature"]["premature"])
    m4a_pt = sum(1 for r in all_results if r["m4a_premature"]["premature"])
    m4b_pt = sum(1 for r in all_results if r["m4b_premature"]["premature"])

    a1_red = sum(r["fork_a1"]["redundant_action_count"] for r in all_results)
    m3_red = sum(r["fork_m3"]["redundant_action_count"] for r in all_results)
    m4a_red = sum(r["fork_m4a"]["redundant_action_count"] for r in all_results)
    m4b_red = sum(r["fork_m4b"]["redundant_action_count"] for r in all_results)
    a1_steps = sum(r["fork_a1"]["steps"] for r in all_results)
    m3_steps = sum(r["fork_m3"]["steps"] for r in all_results)
    m4a_steps = sum(r["fork_m4a"]["steps"] for r in all_results)
    m4b_steps = sum(r["fork_m4b"]["steps"] for r in all_results)

    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        ca1 = sum(1 for r in cr if r["a1_success"])
        cm3 = sum(1 for r in cr if r["m3_success"])
        cm4a = sum(1 for r in cr if r["m4a_success"])
        cm4b = sum(1 for r in cr if r["m4b_success"])

        subgroups[cat] = {
            "n": cn,
            "a1_success": f"{ca1}/{cn} ({ca1/cn*100:.1f}%)",
            "m3_success": f"{cm3}/{cn} ({cm3/cn*100:.1f}%)",
            "m4a_success": f"{cm4a}/{cn} ({cm4a/cn*100:.1f}%)",
            "m4b_success": f"{cm4b}/{cn} ({cm4b/cn*100:.1f}%)",
            "m3_steps": round(sum(r["fork_m3"]["steps"] for r in cr) / cn, 2),
            "m4a_steps": round(sum(r["fork_m4a"]["steps"] for r in cr) / cn, 2),
            "m4b_steps": round(sum(r["fork_m4b"]["steps"] for r in cr) / cn, 2),
            "m4a_m3_rescues": sum(1 for r in cr if not r["m3_success"] and r["m4a_success"]),
            "m4a_m3_breaks": sum(1 for r in cr if r["m3_success"] and not r["m4a_success"]),
            "m4b_m3_rescues": sum(1 for r in cr if not r["m3_success"] and r["m4b_success"]),
            "m4b_m3_breaks": sum(1 for r in cr if r["m3_success"] and not r["m4b_success"]),
            "m4a_false_ready": sum(1 for r in cr if r["m4a_false_ready"]),
            "m4b_false_ready": sum(1 for r in cr if r["m4b_false_ready"]),
            "m4a_premature": sum(1 for r in cr if r["m4a_premature"]["premature"]),
            "m4b_premature": sum(1 for r in cr if r["m4b_premature"]["premature"]),
            "m4a_catastrophic_vs_m3": (cm4a / cn) < (cm3 / cn) - 0.10,
            "m4b_catastrophic_vs_m3": (cm4b / cn) < (cm3 / cn) - 0.10,
        }

    summary = {
        "schema": "DAPH_V2B_I3_10B_EFFICIENT_MDSG_V1",
        "n_tasks": n,
        "arms": {
            "A1": "baseline + public affordances",
            "M3": "frozen MDSG-StateWithAffordances (efficiency FAIL)",
            "M4a": "M3 + observed_resolution_strength",
            "M4b": "M3 + evidence_pipeline_state",
        },
        "overall": {
            "mean_u": {"A1": round(u_a1, 4), "M3": round(u_m3, 4),
                        "M4a": round(u_m4a, 4), "M4b": round(u_m4b, 4)},
            "success": {"A1": f"{a1_s}/{n}", "M3": f"{m3_s}/{n}",
                         "M4a": f"{m4a_s}/{n}", "M4b": f"{m4b_s}/{n}"},
            "bootstrap_ci_m4a_m3": [round(m4a_m3_ci[0], 4), round(m4a_m3_ci[1], 4)],
            "bootstrap_ci_m4b_m3": [round(m4b_m3_ci[0], 4), round(m4b_m3_ci[1], 4)],
            "bootstrap_ci_m4a_a1": [round(m4a_a1_ci[0], 4), round(m4a_a1_ci[1], 4)],
            "bootstrap_ci_m4b_a1": [round(m4b_a1_ci[0], 4), round(m4b_a1_ci[1], 4)],
            "mcnemar_m4a_m3": mc_m4a_m3,
            "mcnemar_m4b_m3": mc_m4b_m3,
            "m4a_m3_classification": dict(m4a_m3_cl),
            "m4b_m3_classification": dict(m4b_m3_cl),
            "false_ready_rate": {
                "M3": {"count": m3_fr, "rate": round(m3_fr / n, 4)},
                "M4a": {"count": m4a_fr, "rate": round(m4a_fr / n, 4)},
                "M4b": {"count": m4b_fr, "rate": round(m4b_fr / n, 4)},
            },
            "premature_termination_rate": {
                "M3": {"count": m3_pt, "rate": round(m3_pt / n, 4)},
                "M4a": {"count": m4a_pt, "rate": round(m4a_pt / n, 4)},
                "M4b": {"count": m4b_pt, "rate": round(m4b_pt / n, 4)},
            },
            "redundant_rate": {
                "A1": round(a1_red / max(a1_steps, 1), 4),
                "M3": round(m3_red / max(m3_steps, 1), 4),
                "M4a": round(m4a_red / max(m4a_steps, 1), 4),
                "M4b": round(m4b_red / max(m4b_steps, 1), 4),
            },
            "mean_steps": {
                "A1": round(a1_steps / n, 2), "M3": round(m3_steps / n, 2),
                "M4a": round(m4a_steps / n, 2), "M4b": round(m4b_steps / n, 2),
            },
        },
        "subgroups": subgroups,
        "frozen_claims": {
            # Primary: M4x must beat M3 on utility
            "C1_m4a_m3_ci_positive": m4a_m3_ci[0] > 0,
            "C1_m4b_m3_ci_positive": m4b_m3_ci[0] > 0,
            # Success: M4x must not lose more than 1pp vs M3
            "C2_m4a_success_within_1pp_m3": m4a_s >= m3_s - 3,  # 1pp of 300
            "C2_m4b_success_within_1pp_m3": m4b_s >= m3_s - 3,
            # Efficiency: M4x must take fewer steps than M3
            "C3_m4a_steps_lt_m3": (m4a_steps / n) < (m3_steps / n),
            "C3_m4b_steps_lt_m3": (m4b_steps / n) < (m3_steps / n),
            # Redundancy: M4x must have lower redundant rate than M3
            "C4_m4a_redundant_lt_m3": (m4a_red / max(m4a_steps, 1)) < (m3_red / max(m3_steps, 1)),
            "C4_m4b_redundant_lt_m3": (m4b_red / max(m4b_steps, 1)) < (m3_red / max(m3_steps, 1)),
            # Safety: false ready < 5%
            "C5_m4a_false_ready_lt_5pct": (m4a_fr / n) < 0.05,
            "C5_m4b_false_ready_lt_5pct": (m4b_fr / n) < 0.05,
            # Safety: premature termination not worse than M3 + 1pp
            "C6_m4a_premature_le_m3_plus_1pp": m4a_pt <= m3_pt + 3,
            "C6_m4b_premature_le_m3_plus_1pp": m4b_pt <= m3_pt + 3,
            # Safety: breaks <= rescues
            "C7_m4a_breaks_le_rescues": m4a_m3_cl.get("BREAK", 0) <= m4a_m3_cl.get("RESCUE", 0),
            "C7_m4b_breaks_le_rescues": m4b_m3_cl.get("BREAK", 0) <= m4b_m3_cl.get("RESCUE", 0),
            # Generalization: no catastrophic subgroup regression vs M3
            "C8_m4a_no_catastrophic_vs_m3": not any(sg["m4a_catastrophic_vs_m3"] for sg in subgroups.values()),
            "C8_m4b_no_catastrophic_vs_m3": not any(sg["m4b_catastrophic_vs_m3"] for sg in subgroups.values()),
            # Architecture: M4x must still beat A1
            "C9_m4a_a1_ci_positive": m4a_a1_ci[0] > 0,
            "C9_m4b_a1_ci_positive": m4b_a1_ci[0] > 0,
        },
    }

    summary_path = output_dir / "efficient_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*82}")
    print("I3.10b EFFICIENT MDSG: A1 vs M3 vs M4a vs M4b")
    print(f"{'='*82}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A1={u_a1:+.4f}  M3={u_m3:+.4f}  M4a={u_m4a:+.4f}  M4b={u_m4b:+.4f}")
    print(f"  Success:       A1={a1_s}/{n}  M3={m3_s}/{n}  M4a={m4a_s}/{n}  M4b={m4b_s}/{n}")
    print(f"\n  Bootstrap 95% CI:")
    print(f"    M4a-M3: [{m4a_m3_ci[0]:+.4f}, {m4a_m3_ci[1]:+.4f}]  <-- PRIMARY")
    print(f"    M4b-M3: [{m4b_m3_ci[0]:+.4f}, {m4b_m3_ci[1]:+.4f}]  <-- PRIMARY")
    print(f"    M4a-A1: [{m4a_a1_ci[0]:+.4f}, {m4a_a1_ci[1]:+.4f}]")
    print(f"    M4b-A1: [{m4b_a1_ci[0]:+.4f}, {m4b_a1_ci[1]:+.4f}]")
    print(f"\n  McNemar:")
    print(f"    M4a-M3: b={mc_m4a_m3['b']}, c={mc_m4a_m3['c']}, p={mc_m4a_m3['p']}")
    print(f"    M4b-M3: b={mc_m4b_m3['b']}, c={mc_m4b_m3['c']}, p={mc_m4b_m3['p']}")
    print(f"\n  M4a vs M3: rescues={m4a_m3_cl.get('RESCUE',0)}, breaks={m4a_m3_cl.get('BREAK',0)}")
    print(f"  M4b vs M3: rescues={m4b_m3_cl.get('RESCUE',0)}, breaks={m4b_m3_cl.get('BREAK',0)}")
    print(f"\n  FALSE READY RATE:")
    print(f"    M3={m3_fr}/{n} ({m3_fr/n*100:.1f}%)  M4a={m4a_fr}/{n} ({m4a_fr/n*100:.1f}%)  M4b={m4b_fr}/{n} ({m4b_fr/n*100:.1f}%)")
    print(f"\n  PREMATURE TERMINATION RATE:")
    print(f"    M3={m3_pt}/{n} ({m3_pt/n*100:.1f}%)  M4a={m4a_pt}/{n} ({m4a_pt/n*100:.1f}%)  M4b={m4b_pt}/{n} ({m4b_pt/n*100:.1f}%)")
    print(f"\n  Redundant: A1={a1_red/max(a1_steps,1):.4f} M3={m3_red/max(m3_steps,1):.4f} M4a={m4a_red/max(m4a_steps,1):.4f} M4b={m4b_red/max(m4b_steps,1):.4f}")
    print(f"  Steps:     A1={a1_steps/n:.2f} M3={m3_steps/n:.2f} M4a={m4a_steps/n:.2f} M4b={m4b_steps/n:.2f}")

    print(f"\n  SUBGROUP ANALYSIS:")
    print(f"    {'Category':<30} {'n':>3} {'A1%':>6} {'M3%':>6} {'M4a%':>6} {'M4b%':>6} {'M3st':>5} {'M4ast':>6} {'M4bst':>6} {'M4aR':>5} {'M4bR':>5} {'M4aB':>5} {'M4bB':>5}")
    for cat, sg in subgroups.items():
        a1p = sg["a1_success"].split("(")[1].rstrip(")")
        m3p = sg["m3_success"].split("(")[1].rstrip(")")
        m4ap = sg["m4a_success"].split("(")[1].rstrip(")")
        m4bp = sg["m4b_success"].split("(")[1].rstrip(")")
        print(f"    {cat:<30} {sg['n']:>3} {a1p:>6} {m3p:>6} {m4ap:>6} {m4bp:>6} "
              f"{sg['m3_steps']:>5.1f} {sg['m4a_steps']:>6.1f} {sg['m4b_steps']:>6.1f} "
              f"{sg['m4a_m3_rescues']:>5} {sg['m4b_m3_rescues']:>5} "
              f"{sg['m4a_m3_breaks']:>5} {sg['m4b_m3_breaks']:>5}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in summary["frozen_claims"].items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
