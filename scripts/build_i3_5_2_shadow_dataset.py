#!/usr/bin/env python3
"""Build State-Level Counterfactual Dataset & Q-Advantage Analysis for V2B-I3.5.2a.

For every decision state s_t encountered along the baseline (AWARE_NO_GOVERNOR)
trajectory in development (300 tasks):
1. Captures controller-visible state features x(s_t).
2. Obtains baseline action a_B(s_t).
3. Obtains governor advisory recommendation a_G(s_t).
4. Computes exact oracle Q(s_t, a) for all valid actions using the OraclePolicyTable.
5. Computes action-level intervention advantage:
       ΔQ(s_t) = Q(s_t, a_G) - Q(s_t, a_B)
6. Labels each individual decision state as HELP (ΔQ > +5.0), NEUTRAL (|ΔQ| <= 5.0), or HARM (ΔQ < -5.0).
7. Outputs structured state ledger (.jsonl) and comprehensive diagnostic summary (.json).

Usage:
    python scripts/build_i3_5_2_shadow_dataset.py \
        --split structure_dev_v2 \
        --results experiments/v2b_i3_5_1/development/e21f63ff4fa9/results.json \
        --output-dir experiments/v2b_i3_5_2/development
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor,
    initial_runtime as init_task_runtime,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    I3BenchmarkTask,
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor,
    initial_i3_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    OraclePolicyTable,
    build_oracle_policy_table,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.selective_governor.features import (
    InterventionFeatures,
    extract_features,
)

# Standard action list
VALID_ACTIONS = tuple(
    a for a in DecisionAction
    if a.value in ("ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP")
)


def get_action_q(table: OraclePolicyTable, state_id: str, action: DecisionAction) -> float:
    """Look up exact Q(s, a) from oracle table with standard failure penalties."""
    q = table.q_values.get((state_id, action))
    if q is not None:
        return q
    pq = table.proposal_q_values.get((state_id, action))
    if pq is not None:
        return pq
    # Rejection or illegal action penalty
    if action == DecisionAction.ANSWER:
        return -125.11
    if action in (DecisionAction.DEFER, DecisionAction.STOP):
        return -30.11
    return -125.0


def classify_delta_q(delta_q: float, threshold: float = 5.0) -> str:
    if delta_q > threshold:
        return "HELP"
    elif delta_q < -threshold:
        return "HARM"
    return "NEUTRAL"


def main():
    parser = argparse.ArgumentParser(description="Build I3.5.2a Shadow Counterfactual Dataset")
    parser.add_argument("--split", default="structure_dev_v2")
    parser.add_argument(
        "--results",
        default="experiments/v2b_i3_5_1/development/e21f63ff4fa9/results.json",
        help="Path to baseline I3.5.1 results.json",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument(
        "--policy",
        default="configs/v2b_i3_policy_v1.json",
    )
    parser.add_argument(
        "--utility",
        default="configs/v2b_i3_1_utility_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_5_2/development",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    task_map = {t.task_id: t for t in split_bm.tasks}

    results_data = json.loads(Path(args.results).read_text())
    blocks = results_data["results"]
    print(f"Loaded {len(blocks)} task blocks from {args.results}")

    policy = load_frozen_policy(args.policy)
    utility = MetareasoningUtility.from_file(args.utility)
    governor = GeneralGovernor()
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    state_records: list[dict[str, Any]] = []
    task_summary_list: list[dict[str, Any]] = []

    substitution_counter = Counter()
    substitution_deltas = defaultdict(list)
    label_counter = Counter()

    print(f"\nProcessing all {len(blocks)} tasks in '{args.split}'...")

    for i, block in enumerate(blocks):
        task_id = block["task_id"]
        task = task_map[task_id]
        budget = split_bm.budget_for(task)
        table = build_oracle_policy_table(task=task, policy=policy, utility=utility, budget=budget)

        b_traj = block["trajectories"]["AWARE_NO_GOVERNOR"]
        b_steps = b_traj["steps"]

        # Replay baseline trajectory to visit each baseline state s_t
        resources = ResourceState(budget)
        i3_runtime = initial_i3_runtime(task, resources)
        adapter = _I3TaskAdapter(task)
        t_runtime = init_task_runtime(adapter, ResourceState(budget))

        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        task_states: list[dict[str, Any]] = []

        for step_idx, step_data in enumerate(b_steps):
            state_id = table.state_id_for(i3_runtime)
            v_star = table.state_values.get(state_id, 0.0)

            a_base_str = step_data["executed_action"]
            a_base = DecisionAction(a_base_str)

            # Build observation and assess governor
            obs = build_observation(
                t_runtime, task, cond,
                tuple(prior_decisions), tuple(prior_outcomes),
            )
            p_actions = tuple(d.selected_action for d in prior_decisions)
            p_outcomes = tuple(prior_outcomes)

            frame = governor.assess(
                observation=obs,
                remaining_steps=24 - step_idx,
                prior_actions=p_actions,
                prior_outcomes=p_outcomes,
            )
            gov_top_str = frame.governor_top_action or a_base_str
            gov_top = DecisionAction(gov_top_str)

            # Extract strictly controller-visible features
            features = extract_features(
                obs,
                remaining_steps=24 - step_idx,
                prior_actions=p_actions,
                prior_outcomes=p_outcomes,
            )

            # Compute Q-values for all valid actions
            all_q_values = {
                act.value: round(get_action_q(table, state_id, act), 4)
                for act in VALID_ACTIONS
            }

            q_base = all_q_values.get(a_base_str, -125.0)
            q_gov = all_q_values.get(gov_top_str, -125.0)
            delta_q = round(q_gov - q_base, 4)
            label = classify_delta_q(delta_q)

            same_action = (a_base_str == gov_top_str)
            substitution_counter[(a_base_str, gov_top_str)] += 1
            substitution_deltas[(a_base_str, gov_top_str)].append(delta_q)
            label_counter[label] += 1

            record = {
                "task_id": task_id,
                "topology_id": task.semantic_structure_coarse,
                "step_id": step_idx,
                "state_id": state_id,
                "v_star": round(v_star, 4),
                "base_action": a_base_str,
                "governor_top_action": gov_top_str,
                "same_action": same_action,
                "q_base": q_base,
                "q_gov": q_gov,
                "delta_q": delta_q,
                "outcome_label": label,
                "governor_reason_code": frame.governor_reason_code,
                "all_q_values": all_q_values,
                "features": features.as_dict(),
            }
            state_records.append(record)
            task_states.append(record)

            # Step the baseline environment
            exec_res = oracle_executor.execute(i3_runtime, a_base)
            i3_runtime = exec_res.runtime
            t_runtime = task_executor.execute(t_runtime, a_base).runtime

            prior_decisions.append(
                DecisionSummary(
                    f"{task_id}:step:{step_idx}",
                    a_base.value,
                    step_data["reason_code"],
                    exec_res.outcome_code,
                )
            )
            prior_outcomes.append(exec_res.outcome_code)
            if exec_res.terminal:
                break

        task_summary_list.append({
            "task_id": task_id,
            "topology_id": task.semantic_structure_coarse,
            "steps_count": len(task_states),
            "baseline_success": b_traj["task_success"],
            "baseline_utility": b_traj["realized_utility"],
            "max_step_delta_q": max(s["delta_q"] for s in task_states),
            "min_step_delta_q": min(s["delta_q"] for s in task_states),
            "mean_step_delta_q": statistics.mean(s["delta_q"] for s in task_states),
            "has_positive_intervention": any(s["delta_q"] > 5.0 for s in task_states),
            "has_harmful_intervention": any(s["delta_q"] < -5.0 for s in task_states),
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(blocks)}] tasks... ({len(state_records)} states collected)")

    # 1. Save JSONL state ledger
    states_path = out_dir / "intervention_states_v1.jsonl"
    with open(states_path, "w") as f:
        for r in state_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nSaved state dataset: {states_path} ({len(state_records)} records)")

    # 2. Build and save Advantage Analysis
    substitution_summary = []
    for (b_act, g_act), cnt in substitution_counter.most_common():
        deltas = substitution_deltas[(b_act, g_act)]
        substitution_summary.append({
            "base_action": b_act,
            "governor_action": g_act,
            "count": cnt,
            "percentage": round(cnt / len(state_records), 4),
            "mean_delta_q": round(statistics.mean(deltas), 4),
            "min_delta_q": round(min(deltas), 4),
            "max_delta_q": round(max(deltas), 4),
            "help_count": sum(1 for d in deltas if d > 5.0),
            "harm_count": sum(1 for d in deltas if d < -5.0),
            "neutral_count": sum(1 for d in deltas if -5.0 <= d <= 5.0),
        })

    total_states = len(state_records)
    pos_states = [s for s in state_records if s["delta_q"] > 5.0]
    neu_states = [s for s in state_records if -5.0 <= s["delta_q"] <= 5.0]
    neg_states = [s for s in state_records if s["delta_q"] < -5.0]

    tasks_with_pos = sum(1 for t in task_summary_list if t["has_positive_intervention"])
    tasks_with_harm = sum(1 for t in task_summary_list if t["has_harmful_intervention"])

    advantage_analysis = {
        "schema": "DAPH_V2B_I3_5_2_INTERVENTION_ADVANTAGE_V1",
        "schema_version": 1,
        "split": args.split,
        "total_tasks": len(blocks),
        "total_decision_states": total_states,
        "mean_trajectory_length": round(total_states / len(blocks), 2),
        "outcome_distribution": {
            "HELP": {"count": len(pos_states), "rate": round(len(pos_states) / total_states, 4)},
            "NEUTRAL": {"count": len(neu_states), "rate": round(len(neu_states) / total_states, 4)},
            "HARM": {"count": len(neg_states), "rate": round(len(neg_states) / total_states, 4)},
        },
        "task_level_opportunity": {
            "tasks_with_at_least_one_help_step": tasks_with_pos,
            "tasks_with_at_least_one_help_step_rate": round(tasks_with_pos / len(blocks), 4),
            "tasks_with_at_least_one_harm_step": tasks_with_harm,
            "tasks_with_at_least_one_harm_step_rate": round(tasks_with_harm / len(blocks), 4),
        },
        "substitution_matrix": substitution_summary,
    }

    advantage_path = out_dir / "intervention_advantage_v1.json"
    advantage_path.write_text(json.dumps(advantage_analysis, indent=2) + "\n")
    print(f"Saved advantage analysis: {advantage_path}")

    # 3. Build Feature Analysis (P(HELP | features), E[ΔQ | features])
    # Group by key feature dimensions
    feature_dimensions: dict[str, dict[str, list[float]]] = {
        "prior_action_count": defaultdict(list),
        "verification_state": defaultdict(list),
        "temporal_status": defaultdict(list),
        "conflict_count": defaultdict(list),
        "last_action": defaultdict(list),
        "repeated_no_gain": defaultdict(list),
        "remaining_steps_bracket": defaultdict(list),
    }

    for s in state_records:
        f = s["features"]
        dq = s["delta_q"]

        step_bracket = "low (<=4)" if f["remaining_steps"] <= 4 else "mid (5-16)" if f["remaining_steps"] <= 16 else "high (17-24)"

        feature_dimensions["prior_action_count"][str(f["prior_action_count"])].append(dq)
        feature_dimensions["verification_state"][str(f["verification_state"])].append(dq)
        feature_dimensions["temporal_status"][str(f["temporal_status"])].append(dq)
        feature_dimensions["conflict_count"][str(f["conflict_count"])].append(dq)
        feature_dimensions["last_action"][str(f["last_action"])].append(dq)
        feature_dimensions["repeated_no_gain"][str(f["repeated_no_gain"])].append(dq)
        feature_dimensions["remaining_steps_bracket"][step_bracket].append(dq)

    opportunity_map = {}
    for dim_name, groups in feature_dimensions.items():
        dim_summary = []
        for grp_val, deltas in sorted(groups.items()):
            n = len(deltas)
            mean_dq = statistics.mean(deltas)
            help_n = sum(1 for d in deltas if d > 5.0)
            harm_n = sum(1 for d in deltas if d < -5.0)
            dim_summary.append({
                "value": grp_val,
                "n": n,
                "percentage_of_states": round(n / total_states, 4),
                "mean_delta_q": round(mean_dq, 4),
                "help_rate": round(help_n / n, 4),
                "harm_rate": round(harm_n / n, 4),
            })
        opportunity_map[dim_name] = dim_summary

    feature_analysis_payload = {
        "schema": "DAPH_V2B_I3_5_2_FEATURE_ANALYSIS_V1",
        "schema_version": 1,
        "split": args.split,
        "total_states": total_states,
        "opportunity_map": opportunity_map,
    }

    feat_path = out_dir / "intervention_feature_analysis_v1.json"
    feat_path.write_text(json.dumps(feature_analysis_payload, indent=2) + "\n")
    print(f"Saved feature analysis: {feat_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("V2B-I3.5.2a STATE-LEVEL COUNTERFACTUAL INTERVENTION OPPORTUNITY MAP")
    print("=" * 70)
    print(f"Total Decision States Analyzed: {total_states} (across {len(blocks)} tasks)")
    print(f"  HELP (ΔQ > +5.0):      {len(pos_states):>4} ({len(pos_states)/total_states:>5.1%})")
    print(f"  NEUTRAL (|ΔQ| <= 5.0): {len(neu_states):>4} ({len(neu_states)/total_states:>5.1%})")
    print(f"  HARM (ΔQ < -5.0):      {len(neg_states):>4} ({len(neg_states)/total_states:>5.1%})")
    print(f"\nTasks with at least 1 positive intervention step: {tasks_with_pos}/{len(blocks)} ({tasks_with_pos/len(blocks):.1%})")

    print("\n--- Key Opportunity Slices ---")
    for dim in ["verification_state", "prior_action_count", "temporal_status", "last_action"]:
        print(f"\n[{dim}]")
        for row in opportunity_map[dim]:
            print(f"  {row['value']:<15}: N={row['n']:>3} ({row['percentage_of_states']:>5.1%}) | mean ΔQ={row['mean_delta_q']:>8.2f} | HelpRate={row['help_rate']:>5.1%} | HarmRate={row['harm_rate']:>5.1%}")


if __name__ == "__main__":
    main()
