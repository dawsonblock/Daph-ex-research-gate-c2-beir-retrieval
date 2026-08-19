#!/usr/bin/env python3
"""I3.6e-a — Resolution Frame Completeness Audit.

Evaluates whether resolution frames contain task-resolving information,
not merely valid structure.

For every disagreement state, generates a resolution frame and checks:

  1. hypothesis_count: >= 2 competing hypotheses
  2. evidence_items_mapped_to_hypotheses: evidence has supports/contradicts links
  3. unmapped_evidence_count: evidence without hypothesis links
  4. discriminator_count: >= 1 discriminator
  5. discriminator_has_two_distinct_outcomes: if_true != if_false
  6. answer_conditions_cover_all_viable_hypotheses: every viable H has a condition
  7. search_spec_is_hypothesis_discriminating: search targets a discriminator
  8. decision_mapping_complete: every viable H has an answer path

ResolutionCompletenessScore (RCS) = count of passed components / 8

Also computes RescuePathExists(frame, state):
  True only if:
    - at least one unresolved hypothesis can be eliminated
    - at least one discriminator corresponds to an available legal operation
    - resolving it can satisfy an answer condition
    - that condition leads to a terminal action before budget exhaustion

No model calls. Pure structural analysis.

Usage:
    PYTHONPATH=. python scripts/audit_i3_6e_a_frame_completeness.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, TaskRuntime, initial_runtime,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    I3BenchmarkTask, load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, initial_i3_runtime,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.resolution_governor import (
    ResolutionGovernor,
    ResolutionAssistanceFrame,
    compute_resolution_identity,
)
from hrm_adaptive_memory.executive.resolution_governor.schema import (
    HYPOTHESIS_SUPPORTED, HYPOTHESIS_WEAK, HYPOTHESIS_CONTRADICTED,
    HYPOTHESIS_UNRESOLVED, HYPOTHESIS_ELIMINATED,
    VERIFICATION_SUFFICIENT, VERIFICATION_MISSING, VERIFICATION_UNVERIFIED,
)


def evaluate_frame_completeness(
    frame: ResolutionAssistanceFrame,
    legal_actions: tuple[str, ...],
    remaining_steps: int,
) -> dict[str, Any]:
    """Evaluate 8 completeness components for a resolution frame.

    Returns pass/fail for each component plus RCS score.
    """
    components = {}

    # 1. hypothesis_count >= 2
    components["hypothesis_count"] = len(frame.candidate_hypotheses)
    components["hypothesis_count_pass"] = len(frame.candidate_hypotheses) >= 2

    # 2. evidence_items_mapped_to_hypotheses
    mapped = sum(
        1 for e in frame.current_evidence
        if e.supports or e.contradicts
    )
    components["evidence_mapped"] = mapped
    components["evidence_mapped_pass"] = mapped >= 1

    # 3. unmapped_evidence_count
    unmapped = sum(
        1 for e in frame.current_evidence
        if not e.supports and not e.contradicts
    )
    components["unmapped_evidence"] = unmapped
    components["unmapped_evidence_pass"] = unmapped == 0 or len(frame.current_evidence) == 0

    # 4. discriminator_count >= 1
    components["discriminator_count"] = len(frame.discriminating_evidence)
    components["discriminator_count_pass"] = len(frame.discriminating_evidence) >= 1

    # 5. discriminator_has_two_distinct_outcomes
    distinct_outcomes = False
    if frame.discriminating_evidence:
        d = frame.discriminating_evidence[0]
        distinct_outcomes = d.if_true_supports != d.if_false_supports
    components["discriminator_distinct_outcomes"] = distinct_outcomes
    components["discriminator_distinct_outcomes_pass"] = distinct_outcomes

    # 6. answer_conditions_cover_all_viable_hypotheses
    viable = [
        h for h in frame.candidate_hypotheses
        if h.current_status not in (HYPOTHESIS_ELIMINATED, HYPOTHESIS_CONTRADICTED)
    ]
    viable_ids = {h.hypothesis_id for h in viable}
    covered_ids = {ac.hypothesis_id for ac in frame.answer_conditions}
    components["viable_hypotheses"] = len(viable)
    components["covered_hypotheses"] = len(viable_ids & covered_ids)
    components["answer_coverage_pass"] = viable_ids <= covered_ids if viable_ids else True

    # 7. search_spec_is_hypothesis_discriminating
    search_discriminating = False
    if frame.search_specification and frame.discriminating_evidence:
        d = frame.discriminating_evidence[0]
        must_disambig = set(frame.search_specification.must_disambiguate)
        search_discriminating = (
            d.if_true_supports in must_disambig and
            d.if_false_supports in must_disambig
        )
    components["search_discriminating"] = search_discriminating
    components["search_discriminating_pass"] = (
        search_discriminating or frame.search_specification is None
    )

    # 8. decision_mapping_complete: every viable H has an answer path
    # An answer path means: there exists an answer_condition for that H
    # AND the execution_plan has a step that could lead to it
    decision_complete = True
    for h in viable:
        has_condition = any(
            ac.hypothesis_id == h.hypothesis_id
            for ac in frame.answer_conditions
        )
        if not has_condition:
            decision_complete = False
            break
        # Check if execution plan could lead to the answer condition
        # (at least one step exists or the action is terminal)
        if not frame.execution_plan and frame.recommended_action not in ("ANSWER", "DEFER", "STOP"):
            decision_complete = False
            break
    components["decision_mapping_complete"] = decision_complete
    components["decision_mapping_pass"] = decision_complete

    # RCS score
    pass_count = sum(
        1 for k, v in components.items()
        if k.endswith("_pass") and v
    )
    total_components = sum(1 for k in components if k.endswith("_pass"))
    components["rcs_score"] = pass_count / total_components if total_components > 0 else 0.0
    components["rcs_pass_count"] = pass_count
    components["rcs_total"] = total_components

    return components


def rescue_path_exists(
    frame: ResolutionAssistanceFrame,
    legal_actions: tuple[str, ...],
    remaining_steps: int,
) -> dict[str, Any]:
    """Check whether a frame structurally contains a path to task rescue.

    True only if:
      1. At least one unresolved hypothesis can be eliminated
      2. At least one discriminator corresponds to an available legal operation
      3. Resolving it can satisfy an answer condition
      4. That condition leads to a terminal action before budget exhaustion
    """
    # 1. At least one unresolved hypothesis can be eliminated
    unresolved = [
        h for h in frame.candidate_hypotheses
        if h.current_status in (HYPOTHESIS_UNRESOLVED, HYPOTHESIS_WEAK)
    ]
    has_eliminable = len(unresolved) >= 1
    # Also need at least 2 hypotheses total for elimination to matter
    has_competition = len(frame.candidate_hypotheses) >= 2

    # 2. At least one discriminator corresponds to an available legal operation
    # The discriminator's evidence_target implies a search/verify/retrieve operation
    disc_has_legal_op = False
    if frame.discriminating_evidence:
        for d in frame.discriminating_evidence:
            # Check if the recommended action or any legal action could resolve this
            if d.verification_required and "VERIFY" in legal_actions:
                disc_has_legal_op = True
                break
            if not d.verification_required and "SEARCH_MORE" in legal_actions:
                disc_has_legal_op = True
                break
            # SEARCH_MORE or RETRIEVE can also find discriminating evidence
            if "SEARCH_MORE" in legal_actions or "RETRIEVE" in legal_actions:
                disc_has_legal_op = True
                break

    # 3. Resolving a discriminator can satisfy an answer condition
    can_satisfy_condition = False
    if frame.discriminating_evidence and frame.answer_conditions:
        for d in frame.discriminating_evidence:
            # If discriminator resolves, it supports if_true_supports
            # Check if that hypothesis has an answer condition
            for ac in frame.answer_conditions:
                if ac.hypothesis_id == d.if_true_supports:
                    can_satisfy_condition = True
                    break
            if can_satisfy_condition:
                break

    # 4. That condition leads to a terminal action before budget exhaustion
    leads_to_terminal = False
    if can_satisfy_condition:
        for ac in frame.answer_conditions:
            if ac.terminal_action in ("ANSWER", "DEFER", "STOP"):
                # Check budget: max_additional_actions must be <= remaining_steps
                if frame.max_additional_actions <= remaining_steps:
                    leads_to_terminal = True
                    break

    rescue_path = (
        has_eliminable and has_competition and
        disc_has_legal_op and can_satisfy_condition and leads_to_terminal
    )

    return {
        "rescue_path_exists": rescue_path,
        "has_eliminable_hypothesis": has_eliminable,
        "has_competition": has_competition,
        "discriminator_has_legal_op": disc_has_legal_op,
        "can_satisfy_answer_condition": can_satisfy_condition,
        "leads_to_terminal": leads_to_terminal,
    }


def main():
    parser = argparse.ArgumentParser(description="I3.6e-a frame completeness audit")
    parser.add_argument(
        "--results",
        default="experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c/results.json",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_6/development/i3_6e_a",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing resolution identity...")
    identity = compute_resolution_identity(
        benchmark_manifest_path=args.benchmark_manifest,
        utility_config_path=args.utility,
        policy_config_path=args.policy,
    )
    print(f"  Resolution identity: {identity['resolution_identity_sha256'][:16]}...")

    print(f"\nLoading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split("structure_dev_v2")
    task_map = {t.task_id: t for t in split_bm.tasks}

    print(f"Loading results from {args.results}...")
    results_data = json.loads(Path(args.results).read_text())
    blocks = results_data["results"]

    general_governor = GeneralGovernor()
    res_governor = ResolutionGovernor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    print("\nReplaying OFF trajectories to find disagreement states...")
    fork_states: list[dict] = []
    n_tasks_processed = 0

    for i, block in enumerate(blocks):
        if n_tasks_processed >= args.n_tasks:
            break
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        off_traj = block["trajectories"].get("OFF")
        if off_traj is None:
            continue
        off_steps = off_traj.get("steps", [])
        if not off_steps:
            continue

        resources = ResourceState(budget)
        i3_runtime = initial_i3_runtime(task, resources)
        adapter = _I3TaskAdapter(task)
        t_runtime = initial_runtime(adapter, ResourceState(budget))

        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        for step_idx, step_data in enumerate(off_steps):
            a_b_str = step_data["executed_action"]
            remaining = 24 - step_idx

            observation = build_observation(
                t_runtime, task, cond,
                tuple(prior_decisions), tuple(prior_outcomes))
            prior_action_strs = tuple(
                d.selected_action if isinstance(d.selected_action, str)
                else d.selected_action.value for d in prior_decisions)

            gov_frame = general_governor.assess(
                observation=observation,
                remaining_steps=remaining,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )
            a_g_str = gov_frame.governor_top_action or a_b_str

            if a_b_str != a_g_str:
                fork_states.append({
                    "task_id": task_id,
                    "step_idx": step_idx,
                    "a_base": a_b_str,
                    "a_gov": a_g_str,
                    "observation": observation,
                    "remaining_steps": remaining,
                    "prior_actions": prior_action_strs,
                    "prior_outcomes": tuple(prior_outcomes),
                    "legal_actions": tuple(
                        a.value for a in observation.allowed_actions),
                    "task": task,
                })

            a_exec = DecisionAction(a_b_str)
            exec_res = oracle_executor.execute(i3_runtime, a_exec)
            t_exec_res = task_executor.execute(t_runtime, a_exec)
            i3_runtime = exec_res.runtime
            t_runtime = t_exec_res.runtime
            prior_decisions.append(DecisionSummary(
                f"{task_id}:step:{step_idx}", a_b_str,
                step_data.get("reason_code", ""), exec_res.outcome_code))
            prior_outcomes.append(exec_res.outcome_code)

            if exec_res.terminal:
                break

        n_tasks_processed += 1

    print(f"Found {len(fork_states)} disagreement states across {n_tasks_processed} tasks")

    # Generate frames and audit
    print(f"\nAuditing {len(fork_states)} resolution frames...")

    all_audits: list[dict[str, Any]] = []
    rcs_scores: list[float] = []
    rescue_paths: list[bool] = []

    for fs in fork_states:
        frame = res_governor.plan(
            observation=fs["observation"],
            remaining_steps=fs["remaining_steps"],
            prior_actions=fs["prior_actions"],
            prior_outcomes=fs["prior_outcomes"],
        )

        if frame is None:
            all_audits.append({
                "task_id": fs["task_id"],
                "step_id": fs["step_idx"],
                "base_action": fs["a_base"],
                "gov_action": fs["a_gov"],
                "frame_generated": False,
                "rcs_score": 0.0,
                "rescue_path_exists": False,
                "reason": "STOP recommended, no frame",
            })
            rcs_scores.append(0.0)
            rescue_paths.append(False)
            continue

        completeness = evaluate_frame_completeness(
            frame, fs["legal_actions"], fs["remaining_steps"])

        rescue = rescue_path_exists(
            frame, fs["legal_actions"], fs["remaining_steps"])

        all_audits.append({
            "task_id": fs["task_id"],
            "step_id": fs["step_idx"],
            "base_action": fs["a_base"],
            "gov_action": fs["a_gov"],
            "frame_generated": True,
            "recommended_action": frame.recommended_action,
            "rcs_score": completeness["rcs_score"],
            "rcs_pass_count": completeness["rcs_pass_count"],
            "rcs_total": completeness["rcs_total"],
            "components": completeness,
            "rescue_path_exists": rescue["rescue_path_exists"],
            "rescue_path_details": rescue,
            "hypothesis_count": len(frame.candidate_hypotheses),
            "evidence_count": len(frame.current_evidence),
            "discriminator_count": len(frame.discriminating_evidence),
            "answer_condition_count": len(frame.answer_conditions),
            "has_search_spec": frame.search_specification is not None,
        })
        rcs_scores.append(completeness["rcs_score"])
        rescue_paths.append(rescue["rescue_path_exists"])

    # Summary
    n = len(all_audits)
    frames_generated = sum(1 for a in all_audits if a["frame_generated"])
    mean_rcs = sum(rcs_scores) / n if n > 0 else 0.0
    rescue_path_count = sum(rescue_paths)

    # Component pass rates
    component_pass_rates = {}
    if frames_generated > 0:
        for component in ["hypothesis_count", "evidence_mapped", "unmapped_evidence",
                          "discriminator_count", "discriminator_distinct_outcomes",
                          "answer_coverage", "search_discriminating", "decision_mapping"]:
            key = f"{component}_pass"
            passed = sum(
                1 for a in all_audits
                if a.get("components", {}).get(key, False)
            )
            component_pass_rates[component] = {
                "passed": passed,
                "total": frames_generated,
                "rate": passed / frames_generated if frames_generated > 0 else 0.0,
            }

    # RCS distribution
    rcs_high = sum(1 for s in rcs_scores if s >= 0.75)
    rcs_medium = sum(1 for s in rcs_scores if 0.5 <= s < 0.75)
    rcs_low = sum(1 for s in rcs_scores if s < 0.5)

    summary = {
        "schema": "DAPH_V2B_I3_6E_A_FRAME_COMPLETENESS_V1",
        "resolution_identity_sha256": identity["resolution_identity_sha256"],
        "n_states": n,
        "n_tasks": n_tasks_processed,
        "frames_generated": frames_generated,
        "mean_rcs": round(mean_rcs, 4),
        "rescue_path_count": rescue_path_count,
        "rescue_path_rate": round(rescue_path_count / n, 4) if n > 0 else 0.0,
        "rcs_distribution": {
            "high_>=0.75": rcs_high,
            "medium_0.5-0.75": rcs_medium,
            "low_<0.5": rcs_low,
        },
        "component_pass_rates": component_pass_rates,
    }

    summary_path = output_dir / "frame_completeness_audit_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Save per-state audits
    audits_path = output_dir / "frame_completeness_audits_v1.jsonl"
    with open(audits_path, "w") as f:
        for a in all_audits:
            f.write(json.dumps(a, sort_keys=True) + "\n")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {audits_path}")

    print(f"\n{'='*78}")
    print("I3.6e-a FRAME COMPLETENESS AUDIT")
    print(f"{'='*78}")
    print(f"  States:                  {n}")
    print(f"  Frames generated:        {frames_generated}/{n}")
    print(f"  Mean RCS:                {mean_rcs:.4f}")
    print(f"  Rescue path exists:      {rescue_path_count}/{n} ({rescue_path_count/n:.1%})" if n > 0 else "")
    print(f"\n  RCS distribution:")
    print(f"    High (>=0.75):         {rcs_high}")
    print(f"    Medium (0.5-0.75):     {rcs_medium}")
    print(f"    Low (<0.5):            {rcs_low}")
    print(f"\n  Component pass rates:")
    for comp, stats in component_pass_rates.items():
        print(f"    {comp}: {stats['passed']}/{stats['total']} ({stats['rate']:.1%})")

    # Key diagnostic
    print(f"\n  KEY DIAGNOSTIC:")
    if rescue_path_count / n < 0.25:
        print(f"    FAIL: Only {rescue_path_count}/{n} frames have a rescue path.")
        print(f"    Most frames cannot structurally produce a rescue.")
        print(f"    REDESIGN frame generation before spending model calls.")
    elif rescue_path_count / n < 0.50:
        print(f"    MARGINAL: {rescue_path_count}/{n} frames have a rescue path.")
        print(f"    Some frames could produce rescues; investigate failures.")
    else:
        print(f"    PASS: {rescue_path_count}/{n} frames have a rescue path.")
        print(f"    Frames structurally support rescues; proceed to utilization probe.")


if __name__ == "__main__":
    main()
