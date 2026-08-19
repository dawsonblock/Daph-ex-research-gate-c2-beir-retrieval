#!/usr/bin/env python3
"""I3.6e-b — Packet Utilization Probe.

Tests whether the model actually follows the resolution frame's
discriminators and answer conditions, or ignores them.

For each disagreement state, sends the resolution packet and asks
the model to return:

  {
    "action": "...",
    "reason_code": "...",
    "active_hypothesis": "...",
    "selected_discriminator": "...",
    "expected_resolution": "...",
    "answer_condition_target": "..."
  }

Then measures:
  P(model action executes active discriminator)
  P(model terminal decision matches answer condition)
  P(model references correct active hypothesis)

This distinguishes:
  - bad resolution frame (frame doesn't contain useful info)
  - good frame ignored by model (model doesn't use the structure)

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_6e_b_utilization_probe.py \\
        --n-tasks 50 --workers 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from hrm_adaptive_memory.executive.executor import initial_runtime
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
from hrm_adaptive_memory.executive.executor import DeterministicActionExecutor
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.resolution_governor import (
    ResolutionGovernor,
    serialize_resolution_packet,
    assert_no_evaluator_leakage,
    compute_resolution_identity,
)
from hrm_adaptive_memory.executive.resolution_governor.serializer import (
    packet_json as resolution_packet_json,
)


# Extended system prompt that asks for resolution-aware output
UTILIZATION_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

You are given a resolution assistance frame with:
  - candidate_hypotheses: competing explanations
  - current_evidence: evidence with hypothesis relationships
  - discriminating_evidence: questions whose answers would change the decision
  - execution_plan: bounded steps with decision consequences
  - answer_conditions: explicit hypothesis to answer mappings
  - defer_condition: when to give up

You MUST respond with a JSON object containing exactly these fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "short reason for the action",
  "active_hypothesis": "the hypothesis_id you are currently evaluating",
  "selected_discriminator": "the discriminator question you are addressing, or null",
  "expected_resolution": "what you expect to happen if this action succeeds",
  "answer_condition_target": "the answer_condition you are working toward, or null"
}

The word json appears in this prompt to satisfy the API requirement."""


def probe_one_state(
    fs: dict,
    general_governor: GeneralGovernor,
    res_governor: ResolutionGovernor,
    api_key: str,
) -> dict[str, Any]:
    """Send resolution packet to model and capture utilization response."""
    observation = fs["observation"]
    remaining = fs["remaining_steps"]
    prior_actions = fs["prior_actions"]
    prior_outcomes = fs["prior_outcomes"]

    # Generate governor and resolution frames
    gov_frame = general_governor.assess(
        observation=observation,
        remaining_steps=remaining,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )
    res_frame = res_governor.plan(
        observation=observation,
        remaining_steps=remaining,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )

    if res_frame is None:
        return {
            "task_id": fs["task_id"],
            "step_id": fs["step_idx"],
            "frame_generated": False,
            "error": "STOP recommended, no frame",
        }

    # Build packet
    packet = serialize_resolution_packet(
        observation, gov_frame, res_frame, context=None,
        mode="RESOLUTION_ASSIST")
    assert_no_evaluator_leakage(packet)
    user_prompt = resolution_packet_json(packet)

    # Call model with extended prompt
    backend = DeepSeekBackend()
    backend.task_id = fs["task_id"]
    backend.condition = "i3_6e_b_probe"
    backend.pair_id = f"i3_6e_b:{fs['task_id']}:{fs['step_idx']}"

    try:
        call_result = backend.generate(
            system_prompt=UTILIZATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0, max_tokens=2048)
    except Exception as e:
        return {
            "task_id": fs["task_id"],
            "step_id": fs["step_idx"],
            "frame_generated": True,
            "error": f"backend error: {e}",
        }

    raw_output = call_result.raw_output

    # Parse the model's response
    try:
        response = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "task_id": fs["task_id"],
            "step_id": fs["step_idx"],
            "frame_generated": True,
            "raw_output": raw_output,
            "parse_error": True,
            "recommended_action": res_frame.recommended_action,
        }

    # Extract model's claimed utilization
    model_action = response.get("action", "")
    model_hypothesis = response.get("active_hypothesis", "")
    model_discriminator = response.get("selected_discriminator", "")
    model_expected = response.get("expected_resolution", "")
    model_answer_target = response.get("answer_condition_target", "")

    # Frame's expected values
    frame_recommended = res_frame.recommended_action
    frame_hypotheses = [h.hypothesis_id for h in res_frame.candidate_hypotheses]
    frame_discriminators = [d.question for d in res_frame.discriminating_evidence]
    frame_answer_conditions = [
        f"{ac.hypothesis_id}->{ac.terminal_action}"
        for ac in res_frame.answer_conditions
    ]

    # Check utilization
    # 1. Does model action match frame's recommended action?
    action_matches = model_action == frame_recommended

    # 2. Does model reference a valid hypothesis?
    hypothesis_valid = model_hypothesis in frame_hypotheses if model_hypothesis else False

    # 3. Does model reference a valid discriminator?
    discriminator_valid = (
        any(model_discriminator in d or d in model_discriminator
            for d in frame_discriminators)
        if model_discriminator and frame_discriminators else False
    )

    # 4. Does model's action correspond to the active discriminator?
    # If there's a discriminator, the action should be one that could resolve it
    action_follows_discriminator = False
    if res_frame.discriminating_evidence:
        d = res_frame.discriminating_evidence[0]
        if d.verification_required:
            action_follows_discriminator = model_action in ("VERIFY", "SEARCH_MORE", "RETRIEVE")
        else:
            action_follows_discriminator = model_action in ("SEARCH_MORE", "RETRIEVE", "REASON_MORE")
    else:
        # No discriminator — any action is fine
        action_follows_discriminator = True

    # 5. Does model reference an answer condition?
    answer_condition_referenced = (
        any(ac_id in model_answer_target for ac_id in frame_answer_conditions)
        if model_answer_target and frame_answer_conditions else False
    )

    return {
        "task_id": fs["task_id"],
        "step_id": fs["step_idx"],
        "frame_generated": True,
        "raw_output": raw_output[:500],  # truncate for storage
        "parse_error": False,
        # Model response
        "model_action": model_action,
        "model_hypothesis": model_hypothesis,
        "model_discriminator": model_discriminator,
        "model_expected": model_expected,
        "model_answer_target": model_answer_target,
        # Frame expectations
        "frame_recommended": frame_recommended,
        "frame_hypotheses": frame_hypotheses,
        "frame_discriminators": frame_discriminators,
        "frame_answer_conditions": frame_answer_conditions,
        # Utilization metrics
        "action_matches": action_matches,
        "hypothesis_valid": hypothesis_valid,
        "discriminator_valid": discriminator_valid,
        "action_follows_discriminator": action_follows_discriminator,
        "answer_condition_referenced": answer_condition_referenced,
        # Frame details
        "hypothesis_count": len(res_frame.candidate_hypotheses),
        "evidence_count": len(res_frame.current_evidence),
        "discriminator_count": len(res_frame.discriminating_evidence),
        "answer_condition_count": len(res_frame.answer_conditions),
    }


def main():
    parser = argparse.ArgumentParser(description="I3.6e-b utilization probe")
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_6/development/i3_6e_b",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

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
                    "observation": observation,
                    "remaining_steps": remaining,
                    "prior_actions": prior_action_strs,
                    "prior_outcomes": tuple(prior_outcomes),
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

    print(f"\nProbing {len(fork_states)} states with {args.workers} workers...")

    all_probes: list[dict[str, Any]] = []
    completed = 0

    def probe_wrapper(fs):
        return probe_one_state(fs, general_governor, res_governor, api_key)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe_wrapper, fs): fs for fs in fork_states}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_probes.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Completed {completed}/{len(fork_states)} probes...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_probes)} probes")

    # Save results
    probes_path = output_dir / "utilization_probes_v1.jsonl"
    with open(probes_path, "w") as f:
        for p in all_probes:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    print(f"Saved: {probes_path}")

    # Compute summary
    n = len(all_probes)
    valid = [p for p in all_probes if p.get("frame_generated") and not p.get("parse_error") and not p.get("error")]
    n_valid = len(valid)

    if n_valid == 0:
        print("No valid probes!")
        return

    action_matches = sum(1 for p in valid if p["action_matches"])
    hypothesis_valid = sum(1 for p in valid if p["hypothesis_valid"])
    discriminator_valid = sum(1 for p in valid if p["discriminator_valid"])
    action_follows = sum(1 for p in valid if p["action_follows_discriminator"])
    answer_ref = sum(1 for p in valid if p["answer_condition_referenced"])

    # Action distribution
    action_dist = Counter(p["model_action"] for p in valid)

    summary = {
        "schema": "DAPH_V2B_I3_6E_B_UTILIZATION_PROBE_V1",
        "resolution_identity_sha256": identity["resolution_identity_sha256"],
        "n_probes": n,
        "n_valid": n_valid,
        "n_parse_errors": sum(1 for p in all_probes if p.get("parse_error")),
        "n_backend_errors": sum(1 for p in all_probes if p.get("error")),
        "utilization_rates": {
            "action_matches_recommended": {
                "count": action_matches,
                "rate": round(action_matches / n_valid, 4),
            },
            "hypothesis_valid": {
                "count": hypothesis_valid,
                "rate": round(hypothesis_valid / n_valid, 4),
            },
            "discriminator_valid": {
                "count": discriminator_valid,
                "rate": round(discriminator_valid / n_valid, 4),
            },
            "action_follows_discriminator": {
                "count": action_follows,
                "rate": round(action_follows / n_valid, 4),
            },
            "answer_condition_referenced": {
                "count": answer_ref,
                "rate": round(answer_ref / n_valid, 4),
            },
        },
        "action_distribution": dict(action_dist),
    }

    summary_path = output_dir / "utilization_probe_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*78}")
    print("I3.6e-b UTILIZATION PROBE SUMMARY")
    print(f"{'='*78}")
    print(f"  Probes:                  {n}")
    print(f"  Valid:                   {n_valid}")
    print(f"  Parse errors:            {summary['n_parse_errors']}")
    print(f"  Backend errors:          {summary['n_backend_errors']}")
    print(f"\n  Utilization rates:")
    print(f"    Action matches recommended:  {action_matches}/{n_valid} ({action_matches/n_valid:.1%})")
    print(f"    Hypothesis valid:            {hypothesis_valid}/{n_valid} ({hypothesis_valid/n_valid:.1%})")
    print(f"    Discriminator valid:         {discriminator_valid}/{n_valid} ({discriminator_valid/n_valid:.1%})")
    print(f"    Action follows discriminator: {action_follows}/{n_valid} ({action_follows/n_valid:.1%})")
    print(f"    Answer condition referenced: {answer_ref}/{n_valid} ({answer_ref/n_valid:.1%})")
    print(f"\n  Action distribution:")
    for action, cnt in action_dist.most_common():
        print(f"    {action}: {cnt}")

    # Key diagnostic
    print(f"\n  KEY DIAGNOSTIC:")
    follow_rate = action_follows / n_valid
    if follow_rate < 0.25:
        print(f"    FAIL: Model follows discriminator in only {follow_rate:.1%} of states.")
        print(f"    The model is ignoring the resolution structure.")
        print(f"    REDESIGN the serializer/prompt interface before more experiments.")
    elif follow_rate < 0.50:
        print(f"    MARGINAL: Model follows discriminator in {follow_rate:.1%} of states.")
        print(f"    Some utilization; investigate why model ignores frame in many cases.")
    else:
        print(f"    PASS: Model follows discriminator in {follow_rate:.1%} of states.")
        print(f"    Model is utilizing the resolution structure.")
        print(f"    Proceed to context-only rescue forks.")


if __name__ == "__main__":
    main()
