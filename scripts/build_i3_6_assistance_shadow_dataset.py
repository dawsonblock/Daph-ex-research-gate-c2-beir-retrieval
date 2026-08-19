#!/usr/bin/env python3
"""Build the I3.6 assistance shadow dataset.

Replays the frozen OFF development trajectories and, for every state,
builds an ExecutionAssistanceFrame. No counterfactual action is executed.

This characterizes where assistance would be issued, what type it would be,
and what the distribution looks like — before running any expensive
model calls.

Outputs:
  experiments/v2b_i3_6/development/shadow/
  ├── assistance_states_v1.jsonl      (per-state assistance records)
  ├── assistance_distribution_v1.json (aggregate statistics)
  └── assistance_examples_v1.json     (representative examples per type)

Usage:
    PYTHONPATH=. python scripts/build_i3_6_assistance_shadow_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.state import DecisionSummary
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.governor.bottlenecks import detect_bottlenecks
from hrm_adaptive_memory.executive.governor.state import build_governor_state
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, initial_runtime as init_task_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, initial_i3_runtime,
)
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.execution_governor import (
    ExecutionGovernor,
    assistance_frame_sha256,
    assert_no_evaluator_leakage,
    validate_assistance_frame,
)
from hrm_adaptive_memory.executive.execution_governor.identity import (
    compute_assistance_identity,
)


def features_from_step(
    task: Any,
    i3_runtime: Any,
    t_runtime: Any,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    remaining_steps: int,
):
    """Extract controller-visible features at a trajectory state."""
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
    observation = build_observation(
        t_runtime, task, cond,
        tuple(prior_decisions), tuple(prior_outcomes))
    prior_action_strs = tuple(
        d.selected_action if isinstance(d.selected_action, str)
        else d.selected_action.value for d in prior_decisions)
    return observation, prior_action_strs


def replay_off_trajectory(
    task: Any,
    budget: Any,
    off_steps: list[dict[str, Any]],
    exec_governor: ExecutionGovernor,
    max_steps: int = 24,
) -> list[dict[str, Any]]:
    """Replay one OFF trajectory and build assistance frames at every state."""
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    resources = ResourceState(budget)
    i3_runtime = initial_i3_runtime(task, resources)
    adapter = _I3TaskAdapter(task)
    t_runtime = init_task_runtime(adapter, ResourceState(budget))

    prior_decisions: list[DecisionSummary] = []
    prior_outcomes: list[str] = []
    records: list[dict[str, Any]] = []

    for step_idx, step_data in enumerate(off_steps):
        a_b_str = step_data["executed_action"]
        remaining = max_steps - step_idx

        observation, prior_action_strs = features_from_step(
            task, i3_runtime, t_runtime,
            prior_decisions, prior_outcomes, remaining)

        # Build assistance frame
        assistance_frame = exec_governor.plan(
            observation=observation,
            remaining_steps=remaining,
            prior_actions=prior_action_strs,
            prior_outcomes=tuple(prior_outcomes),
        )

        if assistance_frame is not None:
            # Validate
            assert_no_evaluator_leakage(assistance_frame)
            issues = validate_assistance_frame(assistance_frame)
            if issues:
                print(f"  WARNING: validation issues for {task.task_id}:step{step_idx}: {issues}")

            frame_sha = assistance_frame_sha256(assistance_frame)
            assistance_type = f"{assistance_frame.recommended_action}_{assistance_frame.bottleneck_type}"

            records.append({
                "task_id": task.task_id,
                "step_id": step_idx,
                "base_action": a_b_str,
                "governor_action": assistance_frame.recommended_action,
                "agreement": a_b_str == assistance_frame.recommended_action,
                "assistance_type": assistance_type,
                "assistance_frame": assistance_frame.as_dict(),
                "assistance_frame_sha256": frame_sha,
                "max_assisted_steps": assistance_frame.max_assisted_steps,
                "bottleneck_type": assistance_frame.bottleneck_type,
                "objective": assistance_frame.objective,
                "target_description": assistance_frame.target_description,
            })
        else:
            records.append({
                "task_id": task.task_id,
                "step_id": step_idx,
                "base_action": a_b_str,
                "governor_action": None,
                "agreement": True,
                "assistance_type": "NO_ASSISTANCE",
                "assistance_frame": None,
                "assistance_frame_sha256": None,
                "max_assisted_steps": 0,
                "bottleneck_type": "NONE",
                "objective": None,
                "target_description": None,
            })

        # Step forward using the executed action
        from hrm_adaptive_memory.cognitive_control.core import DecisionAction
        a_exec = DecisionAction(a_b_str)
        exec_res = oracle_executor.execute(i3_runtime, a_exec)
        i3_runtime = exec_res.runtime
        t_runtime = task_executor.execute(t_runtime, a_exec).runtime
        prior_decisions.append(DecisionSummary(
            f"{task.task_id}:step:{step_idx}", a_b_str,
            step_data.get("reason_code", ""), exec_res.outcome_code))
        prior_outcomes.append(exec_res.outcome_code)

        if exec_res.terminal:
            break

    return records


def main():
    parser = argparse.ArgumentParser(description="Build I3.6 assistance shadow dataset")
    parser.add_argument(
        "--results",
        default="experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c/results.json",
        help="I3.5.3-r1 results file (contains OFF trajectories)",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_6/development/shadow",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute assistance identity
    print("Computing assistance identity...")
    identity = compute_assistance_identity(
        benchmark_manifest_path=args.benchmark_manifest,
        utility_config_path=args.utility,
        policy_config_path=args.policy,
    )
    print(f"  Assistance identity: {identity['assistance_identity_sha256'][:16]}...")

    # Load benchmark
    print(f"\nLoading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split("structure_dev_v2")
    task_map = {t.task_id: t for t in split_bm.tasks}

    # Load results
    print(f"Loading results from {args.results}...")
    results_data = json.loads(Path(args.results).read_text())
    blocks = results_data["results"]
    print(f"Loaded {len(blocks)} task blocks")

    # Find OFF trajectories
    off_key = None
    if blocks:
        traj_keys = list(blocks[0]["trajectories"].keys())
        print(f"  Available trajectory keys: {traj_keys}")
        # Look for OFF or BLIND_NO_GOVERNOR
        for key in ("OFF", "BLIND_NO_GOVERNOR", "AWARE_NO_GOVERNOR"):
            if key in traj_keys:
                off_key = key
                break
    if off_key is None:
        print("ERROR: No OFF trajectory found in results")
        sys.exit(1)
    print(f"  Using OFF trajectory key: {off_key}")

    # Build execution governor
    exec_governor = ExecutionGovernor()

    # Replay all OFF trajectories
    print(f"\nReplaying OFF trajectories and building assistance frames...")
    all_records: list[dict[str, Any]] = []

    for i, block in enumerate(blocks):
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        off_traj = block["trajectories"].get(off_key)
        if off_traj is None:
            continue
        off_steps = off_traj.get("steps", [])
        if not off_steps:
            continue

        records = replay_off_trajectory(
            task=task, budget=budget, off_steps=off_steps,
            exec_governor=exec_governor)
        all_records.extend(records)

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(blocks)}] tasks, "
                  f"{len(all_records)} states so far")

    print(f"\nTotal states: {len(all_records)}")

    # Save per-state records
    states_path = output_dir / "assistance_states_v1.jsonl"
    with open(states_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"Saved: {states_path}")

    # Compute distribution statistics
    n_total = len(all_records)
    n_assisted = sum(1 for r in all_records if r["assistance_frame"] is not None)
    n_no_assistance = n_total - n_assisted
    n_agreement = sum(1 for r in all_records if r["agreement"])
    n_disagreement = n_total - n_agreement

    # Assistance type distribution
    type_counts = Counter(r["assistance_type"] for r in all_records)

    # Action pair distribution
    pair_counts = Counter(
        (r["base_action"], r["governor_action"])
        for r in all_records if r["governor_action"] is not None)

    # Step distribution
    step_counts = Counter(r["step_id"] for r in all_records if r["assistance_frame"] is not None)

    # Budget distribution
    budget_counts = Counter(
        r["max_assisted_steps"] for r in all_records if r["assistance_frame"] is not None)

    # Frame sizes (JSON string length)
    frame_sizes = [
        len(json.dumps(r["assistance_frame"], sort_keys=True))
        for r in all_records if r["assistance_frame"] is not None]

    # Duplicate frame rate
    frame_shas = [
        r["assistance_frame_sha256"]
        for r in all_records if r["assistance_frame_sha256"] is not None]
    unique_shas = set(frame_shas)
    dup_rate = 1.0 - (len(unique_shas) / max(len(frame_shas), 1))

    # Bottleneck type distribution
    bottleneck_counts = Counter(
        r["bottleneck_type"] for r in all_records if r["assistance_frame"] is not None)

    distribution = {
        "schema": "DAPH_V2B_I3_6_SHADOW_DISTRIBUTION_V1",
        "assistance_identity_sha256": identity["assistance_identity_sha256"],
        "n_total_states": n_total,
        "n_assisted": n_assisted,
        "n_no_assistance": n_no_assistance,
        "n_agreement": n_agreement,
        "n_disagreement": n_disagreement,
        "assistance_rate": n_assisted / max(n_total, 1),
        "disagreement_rate": n_disagreement / max(n_total, 1),
        "type_distribution": dict(type_counts.most_common()),
        "action_pair_distribution": {
            f"{b}->{g}": c for (b, g), c in pair_counts.most_common()},
        "step_distribution": dict(sorted(step_counts.items())),
        "budget_distribution": dict(sorted(budget_counts.items())),
        "bottleneck_distribution": dict(bottleneck_counts.most_common()),
        "frame_size": {
            "min": min(frame_sizes) if frame_sizes else 0,
            "max": max(frame_sizes) if frame_sizes else 0,
            "mean": sum(frame_sizes) / max(len(frame_sizes), 1),
        },
        "duplicate_frame_rate": dup_rate,
        "unique_frames": len(unique_shas),
        "total_frames": len(frame_shas),
    }

    dist_path = output_dir / "assistance_distribution_v1.json"
    dist_path.write_text(json.dumps(distribution, indent=2, sort_keys=True) + "\n")
    print(f"\nDistribution saved: {dist_path}")

    # Print summary
    print(f"\n{'='*78}")
    print("ASSISTANCE SHADOW DATASET SUMMARY")
    print(f"{'='*78}")
    print(f"  Total states:              {n_total}")
    print(f"  Assisted states:           {n_assisted} ({n_assisted/max(n_total,1):.1%})")
    print(f"  No assistance:             {n_no_assistance}")
    print(f"  Agreement (a_B == a_G):    {n_agreement} ({n_agreement/max(n_total,1):.1%})")
    print(f"  Disagreement:              {n_disagreement}")
    print(f"  Unique frames:             {len(unique_shas)}/{len(frame_shas)}")
    print(f"  Duplicate rate:            {dup_rate:.1%}")
    print(f"\n  Assistance types:")
    for t, c in type_counts.most_common(10):
        print(f"    {t}: {c}")
    print(f"\n  Action pairs (base -> gov):")
    for (b, g), c in pair_counts.most_common(10):
        print(f"    {b} -> {g}: {c}")
    print(f"\n  Bottleneck types:")
    for bt, c in bottleneck_counts.most_common(10):
        print(f"    {bt}: {c}")
    print(f"\n  Budget distribution:")
    for budget, c in sorted(budget_counts.items()):
        print(f"    max_steps={budget}: {c}")
    print(f"\n  Frame sizes: min={distribution['frame_size']['min']}, "
          f"max={distribution['frame_size']['max']}, "
          f"mean={distribution['frame_size']['mean']:.0f}")

    # Save representative examples per type
    examples: dict[str, dict] = {}
    for rec in all_records:
        if rec["assistance_frame"] is None:
            continue
        atype = rec["assistance_type"]
        if atype not in examples:
            examples[atype] = rec

    examples_path = output_dir / "assistance_examples_v1.json"
    examples_path.write_text(json.dumps(examples, indent=2, sort_keys=True) + "\n")
    print(f"\nExamples saved: {examples_path}")


if __name__ == "__main__":
    main()
