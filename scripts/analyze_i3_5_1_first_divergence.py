#!/usr/bin/env python3
"""Analyze first divergence between AWARE_NO_GOVERNOR and AWARE_GOVERNOR.

Extracts paired trajectories from I3.5.1 experiment results, identifies the
first step where the governor caused the model to take a different action,
captures the exact controller-visible state at that step, and computes the
resulting utility delta and intervention outcome (HELP / NEUTRAL / HARM).

Usage:
    python scripts/analyze_i3_5_1_first_divergence.py \
        --results experiments/v2b_i3_5_1/development/e21f63ff4fa9/results.json \
        --benchmark experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json \
        --split structure_dev_v2 \
        --output experiments/v2b_i3_5_1/development/e21f63ff4fa9/divergence_analysis_v1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary, VerificationState
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, initial_runtime,
)
from hrm_adaptive_memory.executive.governor.chain_progress import extract_chain_progress
from hrm_adaptive_memory.executive.governor.resources import normalize_resources
from hrm_adaptive_memory.executive.governor.state import build_governor_state
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.metareasoning_benchmark import load_metareasoning_benchmark
from hrm_adaptive_memory.executive.resources import ResourceState

# Frozen intervention classification thresholds (Step 3)
POSITIVE_MARGIN = 5.0
NEGATIVE_MARGIN = 5.0


def extract_controller_features(
    obs: Any,
    remaining_steps: int,
    prior_actions: tuple[str, ...],
    prior_outcomes: tuple[str, ...],
) -> dict[str, Any]:
    """Extract strictly controller-visible features from observation and history.

    Never accesses latent oracle, topology ID, future outcomes, or evaluator labels.
    """
    gov_state = build_governor_state(
        observation=obs,
        remaining_steps=remaining_steps,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )
    cs = obs.cognitive_state
    norm_res = normalize_resources(obs.resource_state)

    # Count memories / evidence
    evidence_count = len(cs.relevant_memories) if cs and cs.relevant_memories else 0
    
    # Verification details
    verif_state_str = "NONE"
    verified_count = 0
    if cs and cs.verification_states:
        for v in cs.verification_states:
            st = v.state.value if hasattr(v.state, "value") else str(v.state)
            if st == "SUFFICIENT":
                verified_count += 1
            verif_state_str = st

    # Temporal status
    temporal_str = "NONE"
    if cs and cs.temporal_status:
        temporal_str = (
            cs.temporal_status.value
            if hasattr(cs.temporal_status, "value")
            else str(cs.temporal_status)
        )

    # Conflicts
    conflict_count = len(cs.unresolved_conflicts) if cs and cs.unresolved_conflicts else 0

    # Chain progress
    cp = gov_state.chain_progress
    chain_stage = cp.stages_completed
    chain_started = cp.is_started
    chain_completed = cp.is_complete
    chain_length = len(cp.stage_outcomes)

    return {
        "remaining_steps": remaining_steps,
        "prior_action_count": len(prior_actions),
        "last_action": prior_actions[-1] if prior_actions else None,
        "last_outcome": prior_outcomes[-1] if prior_outcomes else None,
        "repeated_no_gain": gov_state.repeated_no_gain,
        "has_cognitive_state": gov_state.has_cognitive_state,
        "evidence_count": evidence_count,
        "verified_count": verified_count,
        "verification_state": verif_state_str,
        "temporal_status": temporal_str,
        "conflict_count": conflict_count,
        "reasoning_depth": chain_stage,
        "retrieval_budget_remaining": norm_res.retrieval_remaining,
        "verification_budget_remaining": norm_res.verification_remaining,
        "search_budget_remaining": norm_res.search_remaining,
        "reasoning_budget_remaining": norm_res.reasoning_tokens_remaining,
        "chain_started": chain_started,
        "chain_completed": chain_completed,
        "chain_length": chain_length,
        "chain_stage": chain_stage,
    }


def classify_outcome(delta_u: float) -> str:
    if delta_u > POSITIVE_MARGIN:
        return "HELP"
    elif delta_u < -NEGATIVE_MARGIN:
        return "HARM"
    return "NEUTRAL"


def main():
    parser = argparse.ArgumentParser(description="Analyze I3.5.1 first divergence")
    parser.add_argument(
        "--results",
        default="experiments/v2b_i3_5_1/development/e21f63ff4fa9/results.json",
        help="Path to results.json",
    )
    parser.add_argument(
        "--benchmark",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
        help="Path to benchmark manifest",
    )
    parser.add_argument(
        "--split",
        default="structure_dev_v2",
        help="Split name",
    )
    parser.add_argument(
        "--output",
        default="experiments/v2b_i3_5_1/development/e21f63ff4fa9/divergence_analysis_v1.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    results_data = json.loads(Path(args.results).read_text())
    benchmark = load_metareasoning_benchmark(args.benchmark, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    task_map = {t.task_id: t for t in split_bm.tasks}

    blocks = results_data["results"]
    print(f"Loaded {len(blocks)} task blocks from {args.results}")

    executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    task_records: list[dict[str, Any]] = []
    all_decision_states: list[dict[str, Any]] = []
    substitution_counts = Counter()
    substitution_deltas = defaultdict(list)
    step_distribution = Counter()
    label_distribution = Counter()

    for block in blocks:
        task_id = block["task_id"]
        task = task_map.get(task_id)
        if task is None:
            continue
        budget = split_bm.budget_for(task)

        b_traj = block["trajectories"]["AWARE_NO_GOVERNOR"]
        g_traj = block["trajectories"]["AWARE_GOVERNOR"]
        b_steps = b_traj["steps"]
        g_steps = g_traj["steps"]

        b_util = b_traj["realized_utility"]
        g_util = g_traj["realized_utility"]
        delta_u = g_util - b_util
        b_succ = b_traj["task_success"]
        g_succ = g_traj["task_success"]

        # Step-by-step comparison to find first divergence
        first_div = None
        min_len = min(len(b_steps), len(g_steps))
        for i in range(min_len):
            if b_steps[i]["executed_action"] != g_steps[i]["executed_action"]:
                first_div = i
                break
        if first_div is None and len(b_steps) != len(g_steps):
            first_div = min_len

        # Replay the trajectory up to first divergence to capture exact observation state
        resources = ResourceState(budget)
        runtime = initial_runtime(_I3TaskAdapter(task), resources)
        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        # We will collect features for all steps up to divergence
        divergence_features = None
        max_step_to_replay = (first_div + 1) if first_div is not None else min_len

        for step_idx in range(max_step_to_replay):
            remaining_steps = 24 - step_idx
            obs = build_observation(
                runtime, task, cond,
                tuple(prior_decisions), tuple(prior_outcomes),
            )
            p_actions = tuple(d.selected_action for d in prior_decisions)
            p_outcomes = tuple(prior_outcomes)

            features = extract_controller_features(
                obs, remaining_steps, p_actions, p_outcomes,
            )

            is_div_step = (step_idx == first_div)
            if is_div_step:
                divergence_features = features

            # Record step-level dataset entry
            step_b_act = b_steps[step_idx]["executed_action"] if step_idx < len(b_steps) else "TERMINATED"
            step_g_act = g_steps[step_idx]["executed_action"] if step_idx < len(g_steps) else "TERMINATED"
            gov_top = g_steps[step_idx]["governor_top_action"] if step_idx < len(g_steps) else None
            gov_agrees = g_steps[step_idx]["governor_agrees"] if step_idx < len(g_steps) else None

            all_decision_states.append({
                "task_id": task_id,
                "step_id": step_idx,
                "is_first_divergence": is_div_step,
                "baseline_action": step_b_act,
                "governor_action": step_g_act,
                "governor_top_action": gov_top,
                "governor_agrees": gov_agrees,
                "delta_utility": delta_u if is_div_step else 0.0,
                "outcome_label": classify_outcome(delta_u) if is_div_step else "NEUTRAL",
                "features": features,
            })

            # Advance runtime using the shared prefix action
            if step_idx < min_len and b_steps[step_idx]["executed_action"] == g_steps[step_idx]["executed_action"]:
                act = DecisionAction(b_steps[step_idx]["executed_action"])
                exec_res = executor.execute(runtime, act)
                runtime = exec_res.runtime
                prior_decisions.append(
                    DecisionSummary(
                        f"{task.task_id}:step:{step_idx}",
                        act.value,
                        b_steps[step_idx]["reason_code"],
                        exec_res.outcome_code,
                    )
                )
                prior_outcomes.append(exec_res.outcome_code)
            else:
                break

        if first_div is not None:
            b_act = b_steps[first_div]["executed_action"] if first_div < len(b_steps) else "TERMINATED"
            g_act = g_steps[first_div]["executed_action"] if first_div < len(g_steps) else "TERMINATED"
            gov_top = g_steps[first_div]["governor_top_action"] if first_div < len(g_steps) else None
            gov_agrees = g_steps[first_div]["governor_agrees"] if first_div < len(g_steps) else None
            label = classify_outcome(delta_u)

            substitution_counts[(b_act, g_act)] += 1
            substitution_deltas[(b_act, g_act)].append(delta_u)
            step_distribution[first_div] += 1
            label_distribution[label] += 1

            record = {
                "task_id": task_id,
                "first_divergence_step": first_div,
                "baseline_action": b_act,
                "governor_action": g_act,
                "governor_top_action": gov_top,
                "governor_agrees": gov_agrees,
                "baseline_utility": b_util,
                "governor_utility": g_util,
                "utility_delta": delta_u,
                "baseline_success": b_succ,
                "governor_success": g_succ,
                "outcome_label": label,
                "state_before_divergence": divergence_features,
            }
        else:
            label_distribution["NEUTRAL"] += 1
            record = {
                "task_id": task_id,
                "first_divergence_step": None,
                "baseline_action": None,
                "governor_action": None,
                "governor_top_action": None,
                "governor_agrees": None,
                "baseline_utility": b_util,
                "governor_utility": g_util,
                "utility_delta": 0.0,
                "baseline_success": b_succ,
                "governor_success": g_succ,
                "outcome_label": "NEUTRAL",
                "state_before_divergence": None,
            }

        task_records.append(record)

    # Summary statistics
    n_total = len(task_records)
    n_diverged = sum(1 for r in task_records if r["first_divergence_step"] is not None)
    n_identical = n_total - n_diverged

    substitution_matrix = []
    for (b_act, g_act), cnt in substitution_counts.most_common():
        deltas = substitution_deltas[(b_act, g_act)]
        substitution_matrix.append({
            "baseline_action": b_act,
            "governor_action": g_act,
            "count": cnt,
            "percentage_of_divergences": cnt / n_diverged if n_diverged > 0 else 0.0,
            "mean_delta_utility": statistics.mean(deltas),
            "min_delta_utility": min(deltas),
            "max_delta_utility": max(deltas),
        })

    output_payload = {
        "schema": "DAPH_V2B_I3_5_1_FIRST_DIVERGENCE_ANALYSIS_V1",
        "schema_version": 1,
        "split": args.split,
        "total_tasks": n_total,
        "diverged_tasks": n_diverged,
        "identical_tasks": n_identical,
        "divergence_rate": n_diverged / n_total if n_total > 0 else 0.0,
        "outcome_labels": dict(label_distribution),
        "first_divergence_step_distribution": {
            str(k): v for k, v in sorted(step_distribution.items())
        },
        "substitution_matrix": substitution_matrix,
        "task_records": task_records,
        "decision_states": all_decision_states,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2) + "\n")
    print(f"\nSaved divergence analysis to {out_path}")
    print(f"  Total tasks: {n_total}")
    print(f"  Diverged tasks: {n_diverged} ({n_diverged/n_total:.1%})")
    print(f"  Identical tasks: {n_identical} ({n_identical/n_total:.1%})")
    print(f"  Outcome labels: {dict(label_distribution)}")
    print(f"  First divergence step distribution: {dict(step_distribution)}")
    print(f"  Total decision states extracted: {len(all_decision_states)}")


if __name__ == "__main__":
    main()
