#!/usr/bin/env python3
"""Generate and freeze the I3.5 State-Discrimination Benchmark.

Creates 120 one_live tasks across 5 balanced subtypes:
  OL-A (answer):    24 tasks — correct first action: ANSWER
  OL-D (defer):     24 tasks — correct first action: DEFER
  OL-R (retrieve):  24 tasks — correct first action: RETRIEVE
  OL-V (verify):    24 tasks — correct first action: VERIFY
  OL-S (search):    24 tasks — correct first action: SEARCH_MORE

A trivial "always DEFER" policy scores ~20% (only OL-D).
A trivial "always ANSWER" policy scores ~20% (only OL-A).
This forces the controller to discriminate between states.

Output:
  experiments/i3_5/datasets/state_discrimination_v1.jsonl
  experiments/i3_5/datasets/state_discrimination_v1_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_state_discrimination_generator import (
    generate_i3_5_state_discrimination_benchmark,
    benchmark_summary,
    benchmark_sha256,
    CORRECT_FIRST_ACTION,
)


def main():
    output_dir = REPO_ROOT / "experiments/i3_5/datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the benchmark
    n_per_subtype = 24
    seed = 9137
    tasks = generate_i3_5_state_discrimination_benchmark(
        n_per_subtype=n_per_subtype,
        seed=seed,
    )

    # Compute summary
    summary = benchmark_summary(tasks)
    sha = benchmark_sha256(tasks)

    print(f"Generated {len(tasks)} tasks")
    print(f"SHA256: {sha}")
    print(f"Categories: {summary['categories']}")
    print(f"Expected terminals: {summary['expected_terminals']}")
    print(f"Correct first actions: {summary['correct_first_actions']}")
    print(f"Balanced: {summary['balanced']}")

    # Write JSONL with gold labels
    output_path = output_dir / "state_discrimination_v1.jsonl"
    with open(output_path, "w") as f:
        for task in tasks:
            cfa = CORRECT_FIRST_ACTION.get(task.category)
            cfa_str = cfa.value if hasattr(cfa, "value") else str(cfa)

            # Determine budget cases
            if task.category == "ol_answer":
                retrieval_budget_case = "available"
                search_budget_case = "available"
            elif task.category == "ol_defer":
                retrieval_budget_case = "exhausted"
                search_budget_case = "exhausted"
            elif task.category == "ol_retrieve":
                retrieval_budget_case = "available"
                search_budget_case = "available"
            elif task.category == "ol_verify":
                retrieval_budget_case = "exhausted"  # no retrieval needed
                search_budget_case = "available"
            elif task.category == "ol_search":
                retrieval_budget_case = "exhausted"  # local retrieval exhausted
                search_budget_case = "available"
            else:
                retrieval_budget_case = "available"
                search_budget_case = "available"

            record = {
                "task_id": task.task_id,
                "stratum": task.category,
                "category": task.category,
                "expected_terminal": task.expected_terminal.value,
                "correct_first_action": cfa_str,
                "retrieval_budget_case": retrieval_budget_case,
                "search_budget_case": search_budget_case,
                "gold_n_live": 1,  # all one_live tasks
                "gold_n_eliminated": 1,
                "gold_t2": False,
                "gold_all_eliminated": False,
                "gold_verify_epistemically_relevant": task.category in ("ol_verify", "ol_retrieve", "ol_search"),
                "gold_should_gate_verify": False,
                "semantic_error_class": None,
                "budget_profile": task.budget_profile,
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # Write manifest
    manifest = {
        "benchmark_name": "i3_5_state_discrimination_v1",
        "seed": seed,
        "n_per_subtype": n_per_subtype,
        "n_total": len(tasks),
        "sha256": sha,
        "subtypes": {
            "ol_answer": {"n": n_per_subtype, "correct_first_action": "ANSWER"},
            "ol_defer": {"n": n_per_subtype, "correct_first_action": "DEFER"},
            "ol_retrieve": {"n": n_per_subtype, "correct_first_action": "RETRIEVE"},
            "ol_verify": {"n": n_per_subtype, "correct_first_action": "VERIFY"},
            "ol_search": {"n": n_per_subtype, "correct_first_action": "SEARCH_MORE"},
        },
        "trivial_policy_expected_scores": {
            "always_ANSWER": f"~{n_per_subtype}/{len(tasks)} = ~20%",
            "always_DEFER": f"~{n_per_subtype}/{len(tasks)} = ~20%",
            "always_RETRIEVE": f"~{n_per_subtype}/{len(tasks)} = ~20%",
            "always_VERIFY": f"~{n_per_subtype}/{len(tasks)} = ~20%",
            "always_SEARCH_MORE": f"~{n_per_subtype}/{len(tasks)} = ~20%",
        },
        "gate_requirement": "Success(simple heuristic) < 70%",
        "summary": summary,
    }
    manifest_path = output_dir / "state_discrimination_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\nWritten: {output_path}")
    print(f"Written: {manifest_path}")

    # Verify trivial policy scores
    print("\n=== Trivial Policy Check ===")
    for policy in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE"]:
        score = sum(1 for t in tasks
                    if CORRECT_FIRST_ACTION.get(t.category, "").value == policy
                    if hasattr(CORRECT_FIRST_ACTION.get(t.category, ""), "value"))
        print(f"  always_{policy}: {score}/{len(tasks)} = {score/len(tasks)*100:.1f}%")


if __name__ == "__main__":
    main()
