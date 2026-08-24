#!/usr/bin/env python3
"""Generate and freeze the I3.5 intervention schedule.

For each of the 220 benchmark tasks, creates a checkpoint at step 0
(initial state) and schedules all legal actions to be forced from that
checkpoint. This produces the causal dataset:

    Q*(s,a) ≈ E[U | do(a), s]

The schedule is frozen before execution with a SHA256 hash for provenance.

Output:
  experiments/i3_5/datasets/checkpoints_v1.jsonl
  experiments/i3_5/datasets/intervention_schedule_v1.json
  experiments/i3_5/datasets/intervention_schedule_v1_manifest.json
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
    CORRECT_FIRST_ACTION,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

from daph.intervention.checkpoint import create_checkpoint, StateCheckpoint
from daph.intervention.schedule import (
    build_intervention_schedule, save_schedule, InterventionSchedule,
)


def get_budget_for_category(category: str, budget: ResourceBudget) -> tuple[int, int]:
    """Return (retrieval_used, search_used) for a task category."""
    if category in ("ol_defer", "tl_defer"):
        return budget.max_retrieval_calls, budget.max_search_calls
    elif category in ("ol_verify", "ol_search", "tl_verify", "tl_search"):
        return budget.max_retrieval_calls, 0
    else:
        return 0, 0


def main():
    output_dir = REPO_ROOT / "experiments/i3_5/datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the benchmark
    tasks = generate_i3_5_state_discrimination_benchmark(
        n_per_subtype=24,
        n_per_two_live_subtype=20,
        seed=9137,
    )
    print(f"Generated {len(tasks)} tasks")

    # Create checkpoints for each task at step 0
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )

    checkpoints: list[StateCheckpoint] = []
    checkpoint_records: list[dict] = []

    for task in tasks:
        retrieval_used, search_used = get_budget_for_category(task.category, budget)
        resources = ResourceState(
            budget,
            retrieval_calls_used=retrieval_used,
            search_calls_used=search_used,
        )
        runtime = initial_evidence_runtime(task, resources)
        cp = create_checkpoint(runtime, step=0, phase="INITIAL")
        checkpoints.append(cp)

        checkpoint_records.append({
            **cp.as_dict(),
            "category": task.category,
            "correct_first_action": CORRECT_FIRST_ACTION[task.category].value
                if hasattr(CORRECT_FIRST_ACTION[task.category], "value")
                else str(CORRECT_FIRST_ACTION[task.category]),
            "expected_terminal": task.expected_terminal.value,
        })

    # Write checkpoints
    checkpoints_path = output_dir / "checkpoints_v1.jsonl"
    with open(checkpoints_path, "w") as f:
        for rec in checkpoint_records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"Written {len(checkpoint_records)} checkpoints to {checkpoints_path}")

    # Build intervention schedule: for each checkpoint, force all legal actions
    actions_per_checkpoint: dict[str, list[str]] = {}
    for cp in checkpoints:
        legal = list(cp.legal_actions)
        # For VERIFY, add target evidence ID if there's a clear target
        # The first unverified visible evidence item
        verify_targets = []
        for ev_dict in cp.evidence:
            if ev_dict["retrieved"] and ev_dict["verification_state"] == "UNVERIFIED":
                verify_targets.append(ev_dict["evidence_id"])

        actions = []
        for action in legal:
            if action == "VERIFY" and verify_targets:
                actions.append(f"VERIFY:{verify_targets[0]}")
            else:
                actions.append(action)
        actions_per_checkpoint[cp.checkpoint_id] = actions

    schedule = build_intervention_schedule(
        [cp.checkpoint_id for cp in checkpoints],
        actions_per_checkpoint,
        description="I3.5 causal intervention schedule: all legal actions from step-0 checkpoints",
        created_at="2025-01-24T00:00:00Z",
    )

    # Save schedule
    schedule_path = output_dir / "intervention_schedule_v1.json"
    save_schedule(schedule, schedule_path)
    print(f"Written schedule to {schedule_path}")

    # Compute schedule manifest
    total_interventions = len(schedule.interventions)
    by_type = {}
    by_action = {}
    for iv in schedule.interventions:
        by_type[iv.intervention_type] = by_type.get(iv.intervention_type, 0) + 1
        by_action[iv.action] = by_action.get(iv.action, 0) + 1

    # Checkpoint SHA
    checkpoint_content = json.dumps(
        [cp.as_dict() for cp in checkpoints], sort_keys=True
    )
    checkpoints_sha = hashlib.sha256(checkpoint_content.encode()).hexdigest()

    manifest = {
        "schedule_id": schedule.schedule_id,
        "n_checkpoints": len(checkpoints),
        "n_interventions": total_interventions,
        "checkpoints_sha256": checkpoints_sha,
        "by_intervention_type": by_type,
        "by_action": by_action,
        "benchmark_sha256": "9cae0ffba718634b380ddc10cf4a48b263e19338e21492f2da10ceaafd338124",
        "budget": {
            "max_executive_steps": 10,
            "max_retrieval_calls": 3,
            "max_search_calls": 2,
            "max_verification_calls": 5,
        },
    }
    manifest_path = output_dir / "intervention_schedule_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Written manifest to {manifest_path}")

    # Summary
    print(f"\n=== Schedule Summary ===")
    print(f"  Checkpoints: {len(checkpoints)}")
    print(f"  Total interventions: {total_interventions}")
    print(f"  By type: {by_type}")
    print(f"  By action: {by_action}")
    print(f"  Schedule ID: {schedule.schedule_id[:16]}...")
    print(f"  Checkpoints SHA: {checkpoints_sha[:16]}...")

    # Verify: each checkpoint has the correct first action in its legal actions
    print(f"\n=== Correct First Action Availability ===")
    missing = 0
    for cp, rec in zip(checkpoints, checkpoint_records):
        cfa = rec["correct_first_action"]
        legal = set(cp.legal_actions)
        if cfa not in legal:
            print(f"  WARNING: {cp.task_id} ({rec['category']}): correct action {cfa} not in legal {legal}")
            missing += 1
    if missing == 0:
        print(f"  All {len(checkpoints)} checkpoints have their correct first action available")
    else:
        print(f"  {missing} checkpoints missing correct first action!")


if __name__ == "__main__":
    main()
