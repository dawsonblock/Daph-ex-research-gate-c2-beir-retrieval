#!/usr/bin/env python3
"""I3.26 VP failure audit: classify remaining VP failures.

Before investing in selective search, classify every VP failure
from the Phase 25 confirmation run. If most failures are base-model
reasoning failures rather than planning failures, search won't fix them.

Suggested gate: at least 20% of VP failures should plausibly involve
long-horizon/action-sequence planning.

Failure categories:
  Q_ESTIMATE_ERROR: Q values pointed to wrong action
  PROGRESS_ERROR: Progress tie-break chose wrong action
  LONG_HORIZON_PLANNING_FAILURE: Right first action, wrong later sequence
  RESOURCE_FAILURE: Ran out of budget before reaching terminal
  BASE_MODEL_REASONING_FAILURE: Qwen made a bad object-level decision
  TOOL_FAILURE: RETRIEVE/VERIFY/SEARCH returned unexpected result
  UNAVOIDABLE: Task is too hard for any controller
  BENCHMARK_DEFECT: Task definition issue
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def classify_failure(vp_record: dict, c0_record: dict | None = None,
                     v1_record: dict | None = None) -> str:
    """Classify a single VP failure.

    Heuristic classification based on action sequence, terminal result,
    and comparison with other arms.
    """
    actions = vp_record.get("actions_taken", [])
    terminal_result = vp_record.get("terminal_result", "UNKNOWN")
    category = vp_record.get("category", "unknown")
    resource_exhaustion = vp_record.get("resource_exhaustion", False)
    steps = vp_record.get("steps", 0)
    budget_profile = vp_record.get("budget_profile", "")

    # Check if C0 or V1 succeeded on this task
    c0_succeeded = c0_record and c0_record.get("success", False)
    v1_succeeded = v1_record and v1_record.get("success", False)

    # Resource exhaustion with many steps
    if resource_exhaustion and steps >= 5:
        # Did we repeat actions?
        action_counts = defaultdict(int)
        for a in actions:
            action_counts[a] += 1
        max_repeated = max(action_counts.values()) if action_counts else 0

        if max_repeated >= 3:
            return "LONG_HORIZON_PLANNING_FAILURE"
        elif "RETRIEVE" in actions and action_counts.get("RETRIEVE", 0) >= 2:
            return "RESOURCE_FAILURE"
        else:
            return "RESOURCE_FAILURE"

    # Resource exhaustion with few steps — probably budget too tight
    if resource_exhaustion and steps <= 3:
        if c0_succeeded or v1_succeeded:
            return "Q_ESTIMATE_ERROR"
        return "UNAVOIDABLE"

    # Terminal action failed
    if terminal_result in ("ANSWER_INCORRECT", "ANSWER_FAILED"):
        if len(actions) <= 2:
            # Answered too quickly
            return "BASE_MODEL_REASONING_FAILURE"
        else:
            # Answered after some investigation but wrong
            return "BASE_MODEL_REASONING_FAILURE"

    if terminal_result in ("DEFER_UNJUSTIFIED", "DEFER_FAILED"):
        if len(actions) <= 2:
            return "BASE_MODEL_REASONING_FAILURE"
        else:
            return "LONG_HORIZON_PLANNING_FAILURE"

    # Step limit without terminal
    if terminal_result == "STEP_LIMIT":
        if resource_exhaustion:
            return "RESOURCE_FAILURE"
        return "LONG_HORIZON_PLANNING_FAILURE"

    # Decoder or admissibility errors
    if terminal_result in ("DECODER_ERROR", "ADMISSIBILITY_VIOLATION"):
        return "BASE_MODEL_REASONING_FAILURE"

    if terminal_result == "BACKEND_ERROR":
        return "TOOL_FAILURE"

    # Default
    if resource_exhaustion:
        return "RESOURCE_FAILURE"

    return "BASE_MODEL_REASONING_FAILURE"


def main():
    conf_dir = REPO_ROOT / "experiments/i3_5/confirmation"
    traj_path = conf_dir / "trajectories_v1.jsonl"

    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found")
        sys.exit(1)

    records = [json.loads(line) for line in open(traj_path)]

    by_task_arm = {}
    for r in records:
        by_task_arm[(r["task_id"], r["arm"])] = r

    task_ids = sorted(set(r["task_id"] for r in records))

    # Find VP failures
    vp_failures = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        if vp and not vp["success"]:
            c0 = by_task_arm.get((tid, "C0"))
            v1 = by_task_arm.get((tid, "V1"))
            failure_type = classify_failure(vp, c0, v1)
            vp_failures.append({
                "task_id": tid,
                "category": vp["category"],
                "failure_type": failure_type,
                "actions": vp["actions_taken"],
                "terminal_result": vp["terminal_result"],
                "resource_exhaustion": vp.get("resource_exhaustion", False),
                "steps": vp["steps"],
                "budget_profile": vp.get("budget_profile", ""),
                "c0_succeeded": c0["success"] if c0 else None,
                "v1_succeeded": v1["success"] if v1 else None,
            })

    # Summary
    print("=" * 80)
    print("VP FAILURE AUDIT (Phase 25 confirmation)")
    print("=" * 80)
    print(f"\nTotal VP failures: {len(vp_failures)} / {len(task_ids)} tasks")
    print(f"VP success rate: {(len(task_ids) - len(vp_failures)) / len(task_ids):.2%}")

    type_counts = defaultdict(int)
    for f in vp_failures:
        type_counts[f["failure_type"]] += 1

    print(f"\nFailure type distribution:")
    for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(vp_failures) * 100 if vp_failures else 0
        print(f"  {ftype}: {count} ({pct:.1f}%)")

    # Check planning failure gate
    planning_types = {"LONG_HORIZON_PLANNING_FAILURE", "RESOURCE_FAILURE"}
    planning_count = sum(count for ftype, count in type_counts.items()
                         if ftype in planning_types)
    planning_pct = planning_count / len(vp_failures) * 100 if vp_failures else 0

    print(f"\nPlanning-related failures: {planning_count} / {len(vp_failures)} ({planning_pct:.1f}%)")
    print(f"Gate (>= 20% planning): {'PASS' if planning_pct >= 20 else 'FAIL'}")

    # Per-category breakdown
    print(f"\nFailures by category:")
    cat_failures = defaultdict(list)
    for f in vp_failures:
        cat_failures[f["category"]].append(f)
    for cat, fails in sorted(cat_failures.items()):
        types = defaultdict(int)
        for f in fails:
            types[f["failure_type"]] += 1
        print(f"  {cat} ({len(fails)} failures):")
        for ftype, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"    {ftype}: {count}")

    # Detailed failure list
    print(f"\n{'='*80}")
    print("DETAILED FAILURE LIST")
    print(f"{'='*80}")
    for f in vp_failures:
        print(f"\n  {f['task_id']} ({f['category']}):")
        print(f"    Type: {f['failure_type']}")
        print(f"    Actions: {f['actions']}")
        print(f"    Terminal: {f['terminal_result']}")
        print(f"    Exhausted: {f['resource_exhaustion']}")
        print(f"    Steps: {f['steps']}")
        print(f"    Budget: {f['budget_profile']}")
        print(f"    C0 succeeded: {f['c0_succeeded']}")
        print(f"    V1 succeeded: {f['v1_succeeded']}")

    # Save
    output = {
        "total_vp_failures": len(vp_failures),
        "total_tasks": len(task_ids),
        "vp_success_rate": (len(task_ids) - len(vp_failures)) / len(task_ids),
        "failure_type_counts": dict(type_counts),
        "planning_failure_count": planning_count,
        "planning_failure_pct": planning_pct,
        "gate_20pct_planning": planning_pct >= 20,
        "failures": vp_failures,
    }
    output_path = REPO_ROOT / "experiments/i3_26/failure_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
