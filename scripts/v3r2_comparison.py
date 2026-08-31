#!/usr/bin/env python3
"""Phase 16: Direct V3R2 comparison.

Runs the same tasks through both V3R2 (baseline) and DAPH-X executives.
Compares success, utility, regret, tool cost, intervention rate, etc.

Usage:
    python scripts/v3r2_comparison.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.authority.executive import select_action, ExecutiveConfig
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task
from daph_x.receipts.checkpoint import checkpoint_from_task_and_runtime
from daph_x.receipts.fork_engine import evaluate_all_actions, compute_oracle_action, compute_regret
from build_structural_ood_pool import OOD_DOMAIN_TEMPLATES, generate_ood_candidate


def run_v3r2_baseline(task, seed=42):
    """Simulate V3R2 baseline behavior.

    V3R2 uses terminal certificates:
    - If unique verified support for ANSWER → ANSWER
    - If competing verified support → DEFER
    - Otherwise → DEFER (conservative)
    """
    # Check if unique verified support exists
    verified_support = {}
    for e in task.evidence_items:
        if e.verification_state.value == "SUFFICIENT":
            for h_id in e.supports:
                verified_support[h_id] = True

    # Find unique supported hypothesis
    supported = [h.hypothesis_id for h in task.hypotheses
                 if h.hypothesis_id in verified_support]

    if len(supported) == 1:
        # V3R2 would ANSWER
        return {
            "action": "ANSWER",
            "target": supported[0],
            "authority": "FORCE",
            "correct": supported[0] == task.correct_hypothesis_id,
        }
    elif len(supported) > 1:
        # V3R2 would DEFER (competing support)
        return {
            "action": "DEFER",
            "target": None,
            "authority": "FORCE",
            "correct": task.expected_terminal.value == "DEFER",
        }
    else:
        # V3R2 would DEFER (no support)
        return {
            "action": "DEFER",
            "target": None,
            "authority": "FORCE",
            "correct": task.expected_terminal.value == "DEFER",
        }


def run_daph_x(task, seed=42):
    """Run DAPH-X executive."""
    checkpoint = checkpoint_from_task_and_runtime(task, None, seed=seed)
    graph = build_graph_from_evidence_task(task)
    candidates = generate_and_prune(graph)

    if not candidates:
        return {
            "action": "STOP",
            "target": None,
            "authority": "ABSTAIN",
            "correct": False,
            "score": 0.0,
        }

    decision = select_action(graph, config=ExecutiveConfig())

    # Evaluate for ground truth
    results = evaluate_all_actions(checkpoint, candidates, seed=seed)
    oracle_action, oracle_utility = compute_oracle_action(results)

    return {
        "action": str(decision.selected_action),
        "target": decision.selected_action.target,
        "authority": decision.authority_mode.value,
        "correct": str(decision.selected_action) == oracle_action,
        "score": decision.expected_value,
        "regret": compute_regret(results, str(decision.selected_action)),
    }


def main():
    print("=" * 80)
    print("PHASE 16: DIRECT V3R2 COMPARISON")
    print("=" * 80)

    # Generate tasks from all templates
    all_results = []
    for template in OOD_DOMAIN_TEMPLATES:
        for i in range(10):  # 10 tasks per template
            task = generate_ood_candidate(template, i)

            v3r2 = run_v3r2_baseline(task)
            daphx = run_daph_x(task)

            all_results.append({
                "task_id": task.task_id,
                "category": template["category"],
                "v3r2": v3r2,
                "daphx": daphx,
            })

    # Aggregate metrics
    print(f"\nTotal tasks: {len(all_results)}")

    # Success rates
    v3r2_correct = sum(1 for r in all_results if r["v3r2"]["correct"])
    daphx_correct = sum(1 for r in all_results if r["daphx"]["correct"])
    print(f"\nSuccess rate:")
    print(f"  V3R2:   {v3r2_correct}/{len(all_results)} ({100*v3r2_correct/len(all_results):.1f}%)")
    print(f"  DAPH-X: {daphx_correct}/{len(all_results)} ({100*daphx_correct/len(all_results):.1f}%)")

    # By category
    print(f"\nBy category:")
    categories = defaultdict(list)
    for r in all_results:
        categories[r["category"]].append(r)

    for cat, cat_results in sorted(categories.items()):
        v3r2_cat = sum(1 for r in cat_results if r["v3r2"]["correct"])
        daphx_cat = sum(1 for r in cat_results if r["daphx"]["correct"])
        print(f"  {cat:>40}: V3R2={v3r2_cat}/{len(cat_results)}, DAPH-X={daphx_cat}/{len(cat_results)}")

    # Disagreement analysis
    disagreements = [r for r in all_results
                     if r["v3r2"]["action"] != r["daphx"]["action"].split("(")[0]]
    print(f"\nDisagreements: {len(disagreements)}/{len(all_results)}")

    # Where DAPH-X is better
    daphx_better = [r for r in all_results
                    if r["daphx"]["correct"] and not r["v3r2"]["correct"]]
    print(f"DAPH-X better: {len(daphx_better)}")
    for r in daphx_better[:5]:
        print(f"  {r['task_id']}: V3R2={r['v3r2']['action']}, DAPH-X={r['daphx']['action']}")

    # Where V3R2 is better
    v3r2_better = [r for r in all_results
                   if r["v3r2"]["correct"] and not r["daphx"]["correct"]]
    print(f"V3R2 better: {len(v3r2_better)}")
    for r in v3r2_better[:5]:
        print(f"  {r['task_id']}: V3R2={r['v3r2']['action']}, DAPH-X={r['daphx']['action']}")

    # Regret comparison
    daphx_regrets = [r["daphx"].get("regret", 0) for r in all_results]
    print(f"\nDAPH-X regret: mean={np.mean(daphx_regrets):.2f}, max={np.max(daphx_regrets):.2f}")

    # Authority modes used by DAPH-X
    authority_counts = defaultdict(int)
    for r in all_results:
        authority_counts[r["daphx"]["authority"]] += 1
    print(f"\nDAPH-X authority modes:")
    for mode, count in sorted(authority_counts.items()):
        print(f"  {mode}: {count}")

    # Save
    output_path = REPO_ROOT / "experiments/daph_x/v3r2_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
