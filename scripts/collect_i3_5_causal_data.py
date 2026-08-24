#!/usr/bin/env python3
"""Collect causal action data by forcing all scheduled actions from checkpoints.

For each intervention in the frozen schedule:
  1. Restore the checkpoint
  2. Force the specified action
  3. For terminal actions (ANSWER, DEFER, STOP): record outcome immediately
  4. For non-terminal actions (RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE):
     continue with an oracle policy that follows the task's oracle_resolution_path
  5. Record U(s,a), success, premature_defer, premature_answer, etc.

This produces the causal dataset:
    Q*(s,a) ≈ E[U | do(a), s]

Output:
  experiments/i3_5/causal/causal_actions_v1.jsonl
  experiments/i3_5/causal/causal_actions_v1_manifest.json
  experiments/i3_5/causal/causal_receipts_v1.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState
from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_state_discrimination_generator import (
    generate_i3_5_state_discrimination_benchmark,
    CORRECT_FIRST_ACTION,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceRuntime, EvidenceTask, initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState, ResourceExhausted

from daph.intervention.checkpoint import StateCheckpoint, create_checkpoint
from daph.intervention.restore import restore_runtime
from daph.intervention.force_action import force_action, ForcedActionResult
from daph.intervention.schedule import load_schedule, InterventionSchedule
from daph.intervention.receipts import create_receipt, InterventionReceipt


def get_budget_for_category(category: str, budget: ResourceBudget) -> tuple[int, int]:
    """Return (retrieval_used, search_used) for a task category."""
    if category in ("ol_defer", "tl_defer"):
        return budget.max_retrieval_calls, budget.max_search_calls
    elif category in ("ol_verify", "ol_search", "tl_verify", "tl_search"):
        return budget.max_retrieval_calls, 0
    else:
        return 0, 0


def load_checkpoints(path: Path) -> list[dict]:
    """Load checkpoint records from JSONL."""
    checkpoints = []
    with open(path) as f:
        for line in f:
            checkpoints.append(json.loads(line))
    return checkpoints


def find_verify_target(checkpoint_rec: dict) -> str | None:
    """Find the first unverified visible evidence item for VERIFY targeting."""
    for ev in checkpoint_rec["evidence"]:
        if ev["retrieved"] and ev["verification_state"] == "UNVERIFIED":
            return ev["evidence_id"]
    return None


def oracle_policy_rollout(
    runtime: EvidenceRuntime,
    task: EvidenceTask,
    forced_action: DecisionAction,
    max_steps: int = 10,
) -> tuple[float, bool, str | None, list[str], bool, bool, bool, bool]:
    """Continue with oracle policy after a forced non-terminal action.

    The oracle policy follows the task's oracle_resolution_path from the
    point after the forced action. This gives us the causal outcome:
    "if we force action X, then follow the optimal path, what is U(s,X)?"

    Returns:
        (terminal_utility, success, terminal_action, downstream_actions,
         premature_defer, premature_answer, resource_exhaustion, loop)
    """
    executor = EvidenceExecutor()
    downstream: list[str] = []
    current = runtime

    # Build remaining oracle path based on what's already been done
    oracle_path = list(task.oracle_resolution_path)
    # Skip oracle steps that were already accomplished by the forced action
    # For simplicity, we just follow the oracle from where we are

    steps = 0
    while steps < max_steps:
        # Try to follow oracle path
        acted = False
        for i, step_spec in enumerate(oracle_path):
            if ":" in step_spec:
                action_name, target = step_spec.split(":")
                action = DecisionAction(action_name)
                # Check if this step is still needed
                if action_name == "VERIFY":
                    # Check if target is already verified
                    target_ev = None
                    for ev in current.evidence:
                        if ev.evidence_id == target:
                            target_ev = ev
                            break
                    if target_ev and target_ev.verification_state != VerificationState.UNVERIFIED:
                        continue  # already verified, skip
                elif action_name == "RETRIEVE":
                    # Check if target is already retrieved
                    target_ev = None
                    for ev in current.evidence:
                        if ev.evidence_id == target:
                            target_ev = ev
                            break
                    if target_ev and target_ev.retrieved:
                        continue  # already retrieved, skip
                elif action_name == "SEARCH_MORE":
                    if current.searched:
                        continue  # already searched
            else:
                action = DecisionAction(step_spec)

            # Try to execute this oracle step
            target_eid = None
            if ":" in step_spec:
                _, target_eid = step_spec.split(":")

            try:
                result = executor.execute(current, action, target_evidence_id=target_eid)
                current = result.runtime
                downstream.append(action.value)
                steps += 1
                acted = True

                if result.terminal:
                    utility = 1.0 if result.task_success else -1.0
                    pd = False
                    pa = False
                    if action is DecisionAction.DEFER and not result.task_success:
                        pd = True
                    if action is DecisionAction.ANSWER and not result.task_success:
                        pa = True
                    return (utility, result.task_success, action.value, downstream,
                            pd, pa, result.outcome_code == "RESOURCE_EXHAUSTED", False)
                break  # restart oracle path scan from beginning
            except (ResourceExhausted, ValueError):
                continue

        if not acted:
            # No oracle step could execute, try terminal action
            expected = task.expected_terminal
            try:
                result = executor.execute(current, expected)
                current = result.runtime
                downstream.append(expected.value)
                utility = 1.0 if result.task_success else -1.0
                pd = expected is DecisionAction.DEFER and not result.task_success
                pa = expected is DecisionAction.ANSWER and not result.task_success
                return (utility, result.task_success, expected.value, downstream,
                        pd, pa, result.outcome_code == "RESOURCE_EXHAUSTED", False)
            except (ResourceExhausted, ValueError):
                break

    # Step limit
    return (-0.5, False, None, downstream, False, False, False, len(downstream) >= 3 and len(set(downstream)) == 1)


def main():
    output_dir = REPO_ROOT / "experiments/i3_5/causal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark tasks
    tasks = generate_i3_5_state_discrimination_benchmark(
        n_per_subtype=24, n_per_two_live_subtype=20, seed=9137,
    )
    task_lookup = {t.task_id: t for t in tasks}

    # Load checkpoints
    checkpoints_path = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
    checkpoint_recs = load_checkpoints(checkpoints_path)
    print(f"Loaded {len(checkpoint_recs)} checkpoints")

    # Load schedule
    schedule_path = REPO_ROOT / "experiments/i3_5/datasets/intervention_schedule_v1.json"
    schedule = load_schedule(schedule_path)
    print(f"Loaded schedule: {schedule.schedule_id[:16]}... ({len(schedule.interventions)} interventions)")

    # Backend identity (frozen executor, no LLM)
    backend_sha = hashlib.sha256(b"EvidenceExecutor:v1:no_llm:oracle_rollout").hexdigest()

    # Collect causal data
    results: list[dict] = []
    receipts: list[InterventionReceipt] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    executor = EvidenceExecutor()
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )

    # Group interventions by checkpoint
    by_checkpoint: dict[str, list] = {}
    for iv in schedule.interventions:
        by_checkpoint.setdefault(iv.checkpoint_id, []).append(iv)

    n_total = len(schedule.interventions)
    n_done = 0
    n_success = 0
    n_terminal = 0
    n_premature_defer = 0
    n_premature_answer = 0

    for cp_rec in checkpoint_recs:
        cp_id = cp_rec["checkpoint_id"]
        task_id = cp_rec["task_id"]
        task = task_lookup.get(task_id)
        if task is None:
            print(f"WARNING: task {task_id} not found")
            continue

        interventions = by_checkpoint.get(cp_id, [])

        for iv in interventions:
            action = DecisionAction(iv.action)
            target_eid = iv.target_evidence_id

            # Reconstruct checkpoint object
            from daph.intervention.checkpoint import StateCheckpoint
            cp = StateCheckpoint(
                checkpoint_id=cp_rec["checkpoint_id"],
                task_id=cp_rec["task_id"],
                step=cp_rec["step"],
                phase=cp_rec["phase"],
                hypotheses=tuple(cp_rec["hypotheses"]),
                evidence=tuple(cp_rec["evidence"]),
                state_features=cp_rec["state_features"],
                resources=cp_rec["resources"],
                legal_actions=tuple(cp_rec["legal_actions"]),
                state_sha256=cp_rec["state_sha256"],
                prior_actions=tuple(cp_rec["prior_actions"]),
                prior_outcomes=tuple(cp_rec["prior_outcomes"]),
            )

            # Force the action
            forced_result, post_runtime = force_action(cp, task, action, target_evidence_id=target_eid)

            # If non-terminal, continue with oracle policy
            if not forced_result.terminal:
                (terminal_utility, success, terminal_action, downstream,
                 pd, pa, exhausted, loop) = oracle_policy_rollout(
                    post_runtime, task, action, max_steps=8,
                )
                forced_result = ForcedActionResult(
                    checkpoint_id=forced_result.checkpoint_id,
                    action=forced_result.action,
                    intervention_type=forced_result.intervention_type,
                    immediate_utility=forced_result.immediate_utility,
                    terminal_utility=terminal_utility,
                    success=success,
                    terminal=terminal_action is not None,
                    terminal_action=terminal_action,
                    steps_to_terminal=len(downstream) + 1,
                    outcome_code=forced_result.outcome_code,
                    premature_defer=pd or forced_result.premature_defer,
                    premature_answer=pa or forced_result.premature_answer,
                    resource_exhaustion=exhausted,
                    loop=loop,
                    forced_action_execution=forced_result.forced_action_execution,
                    downstream_actions=tuple(downstream),
                )

            # Build causal record
            record = {
                "checkpoint_id": cp_id,
                "task_id": task_id,
                "category": cp_rec["category"],
                "correct_first_action": cp_rec["correct_first_action"],
                "expected_terminal": cp_rec["expected_terminal"],
                "forced_action": iv.action,
                "target_evidence_id": iv.target_evidence_id,
                "intervention_type": iv.intervention_type,
                "state_features": cp_rec["state_features"],
                "state_sha256": cp_rec["state_sha256"],
                "immediate_utility": forced_result.immediate_utility,
                "terminal_utility": forced_result.terminal_utility,
                "success": forced_result.success,
                "terminal": forced_result.terminal,
                "terminal_action": forced_result.terminal_action,
                "steps_to_terminal": forced_result.steps_to_terminal,
                "outcome_code": forced_result.outcome_code,
                "premature_defer": forced_result.premature_defer,
                "premature_answer": forced_result.premature_answer,
                "resource_exhaustion": forced_result.resource_exhaustion,
                "loop": forced_result.loop,
                "downstream_actions": list(forced_result.downstream_actions),
                "schedule_id": schedule.schedule_id,
            }
            results.append(record)

            # Create receipt
            receipt = create_receipt(
                checkpoint_id=cp_id,
                action=iv.action,
                intervention_type=iv.intervention_type,
                result=forced_result.as_dict(),
                state_sha_before=cp_rec["state_sha256"],
                state_sha_after=None,
                backend_identity_sha256=backend_sha,
                timestamp=timestamp,
            )
            receipts.append(receipt)

            n_done += 1
            if forced_result.success:
                n_success += 1
            if forced_result.terminal:
                n_terminal += 1
            if forced_result.premature_defer:
                n_premature_defer += 1
            if forced_result.premature_answer:
                n_premature_answer += 1

            if n_done % 100 == 0:
                print(f"  Progress: {n_done}/{n_total}")

    # Write causal data
    causal_path = output_dir / "causal_actions_v1.jsonl"
    with open(causal_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nWritten {len(results)} causal records to {causal_path}")

    # Write receipts
    receipts_path = output_dir / "causal_receipts_v1.jsonl"
    with open(receipts_path, "w") as f:
        for r in receipts:
            f.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")
    print(f"Written {len(receipts)} receipts to {receipts_path}")

    # Compute manifest
    causal_content = json.dumps(results, sort_keys=True)
    causal_sha = hashlib.sha256(causal_content.encode()).hexdigest()

    # Per-action Q estimates
    by_action: dict[str, list[float]] = {}
    for r in results:
        by_action.setdefault(r["forced_action"], []).append(r["terminal_utility"])

    action_stats = {}
    for action, utils in by_action.items():
        n = len(utils)
        mean = sum(utils) / n if n > 0 else 0
        n_succ = sum(1 for r in results if r["forced_action"] == action and r["success"])
        action_stats[action] = {
            "n": n,
            "mean_utility": round(mean, 4),
            "success_rate": round(n_succ / n, 4) if n > 0 else 0,
        }

    # Per-category × action Q estimates
    by_cat_action: dict[str, list[float]] = {}
    for r in results:
        key = f"{r['category']}:{r['forced_action']}"
        by_cat_action.setdefault(key, []).append(r["terminal_utility"])

    cat_action_stats = {}
    for key, utils in by_cat_action.items():
        n = len(utils)
        mean = sum(utils) / n if n > 0 else 0
        cat_action_stats[key] = {"n": n, "mean_utility": round(mean, 4)}

    manifest = {
        "causal_data_sha256": causal_sha,
        "n_records": len(results),
        "n_checkpoints": len(checkpoint_recs),
        "schedule_id": schedule.schedule_id,
        "backend_identity_sha256": backend_sha,
        "timestamp": timestamp,
        "overall_success_rate": round(n_success / n_done, 4) if n_done > 0 else 0,
        "terminal_rate": round(n_terminal / n_done, 4) if n_done > 0 else 0,
        "premature_defer_rate": round(n_premature_defer / n_done, 4) if n_done > 0 else 0,
        "premature_answer_rate": round(n_premature_answer / n_done, 4) if n_done > 0 else 0,
        "action_stats": action_stats,
        "category_action_stats": cat_action_stats,
    }
    manifest_path = output_dir / "causal_actions_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Written manifest to {manifest_path}")

    # Summary
    print(f"\n=== Causal Data Summary ===")
    print(f"  Records: {len(results)}")
    print(f"  Overall success rate: {manifest['overall_success_rate']}")
    print(f"  Terminal rate: {manifest['terminal_rate']}")
    print(f"  Premature DEFER rate: {manifest['premature_defer_rate']}")
    print(f"  Premature ANSWER rate: {manifest['premature_answer_rate']}")
    print(f"\n  Per-action Q estimates:")
    for action, stats in sorted(action_stats.items()):
        print(f"    {action:15s}: n={stats['n']:4d}, Q={stats['mean_utility']:+.4f}, success={stats['success_rate']:.4f}")
    print(f"\n  Per-category × action Q estimates (correct action marked with *):")
    for key in sorted(cat_action_stats.keys()):
        cat, action = key.split(":")
        cfa = task_lookup.get(f"i3_5_{cat}_0000")
        # Use the checkpoint records to find correct first action
        correct = None
        for r in results:
            if r["category"] == cat:
                correct = r["correct_first_action"]
                break
        marker = " *" if correct == action else "  "
        stats = cat_action_stats[key]
        print(f"    {marker}{cat:15s} × {action:15s}: n={stats['n']:3d}, Q={stats['mean_utility']:+.4f}")


if __name__ == "__main__":
    main()
