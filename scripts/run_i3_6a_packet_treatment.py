#!/usr/bin/env python3
"""I3.6a — Packet treatment experiment.

At identical saved states, makes paired model calls with three packet types:
  BASE:           base packet (no governor)
  ACTION_ONLY:    existing governor packet (action recommendation)
  EXEC_ASSIST:    execution assistance packet (structured scaffold)

No actions are executed. This measures:
  - model action agreement across packets
  - action substitution rates
  - JSON validity
  - latency
  - tokens
  - assistance field utilization (structured output fields)

Usage:
    PYTHONPATH=. python scripts/run_i3_6a_packet_treatment.py \\
        --n-tasks 50 \\
        --output-dir experiments/v2b_i3_6/development/i3_6a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.state import DecisionSummary
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import (
    build_base_packet, build_governor_packet,
    packet_json, packet_sha256, assert_no_evaluator_leakage,
)
from hrm_adaptive_memory.executive.i3_5_1.model_prompt import SYSTEM_PROMPT
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
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.execution_governor import (
    ExecutionGovernor,
    serialize_assistance_packet,
)
from hrm_adaptive_memory.executive.execution_governor.serializer import (
    assert_no_evaluator_leakage as assert_assistance_no_leakage,
)
from hrm_adaptive_memory.executive.execution_governor.identity import (
    compute_assistance_identity,
    assistance_frame_sha256,
)


def features_from_step(task, i3_runtime, t_runtime, prior_decisions, prior_outcomes, remaining):
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
    observation = build_observation(
        t_runtime, task, cond,
        tuple(prior_decisions), tuple(prior_outcomes))
    prior_action_strs = tuple(
        d.selected_action if isinstance(d.selected_action, str)
        else d.selected_action.value for d in prior_decisions)
    return observation, prior_action_strs


def call_model(backend, system_prompt, user_prompt, temperature, max_tokens):
    """Make a model call and return (result, prompt_tokens, completion_tokens, latency_ms)."""
    t_start = time.monotonic()
    try:
        result = backend.generate(
            system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=temperature, max_tokens=max_tokens)
        latency_ms = (time.monotonic() - t_start) * 1000
        return result, result.prompt_tokens, result.completion_tokens, latency_ms, None
    except Exception as e:
        latency_ms = (time.monotonic() - t_start) * 1000
        return None, 0, 0, latency_ms, str(e)


def parse_model_response(response_text: str) -> dict | None:
    """Parse the model's JSON response."""
    if response_text is None:
        return None
    try:
        return json.loads(response_text)
    except Exception:
        # Try to extract JSON from the response
        import re
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
        return None


def run_packet_treatment(
    task: Any,
    budget: Any,
    off_steps: list[dict[str, Any]],
    general_governor: GeneralGovernor,
    exec_governor: ExecutionGovernor,
    backend: DeepSeekBackend,
    max_steps: int = 24,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Run packet treatment at each state of one OFF trajectory."""
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    resources = ResourceState(budget)
    i3_runtime = initial_i3_runtime(task, resources)
    adapter = _I3TaskAdapter(task)
    t_runtime = init_task_runtime(adapter, ResourceState(budget))

    prior_decisions: list[DecisionSummary] = []
    prior_outcomes: list[str] = []
    results: list[dict[str, Any]] = []

    for step_idx, step_data in enumerate(off_steps):
        a_b_str = step_data["executed_action"]
        remaining = max_steps - step_idx

        observation, prior_action_strs = features_from_step(
            task, i3_runtime, t_runtime,
            prior_decisions, prior_outcomes, remaining)

        # Build governor frame
        gov_frame = general_governor.assess(
            observation=observation,
            remaining_steps=remaining,
            prior_actions=prior_action_strs,
            prior_outcomes=tuple(prior_outcomes),
        )

        # Build assistance frame
        assist_frame = exec_governor.plan(
            observation=observation,
            remaining_steps=remaining,
            prior_actions=prior_action_strs,
            prior_outcomes=tuple(prior_outcomes),
        )

        # Only run treatment at disagreement states
        if assist_frame is None or a_b_str == assist_frame.recommended_action:
            # Step forward without making calls
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
            continue

        # Build three packets
        base_packet = build_base_packet(observation)
        action_packet = build_governor_packet(observation, gov_frame)
        assist_packet = serialize_assistance_packet(
            observation, gov_frame, assist_frame, mode="EXECUTION_ASSIST")

        # Verify no leakage
        assert_no_evaluator_leakage(base_packet)
        assert_no_evaluator_leakage(action_packet)
        assert_assistance_no_leakage(assist_packet)

        assist_sha = assistance_frame_sha256(assist_frame)

        # Make three paired model calls
        backend.task_id = task.task_id
        backend.condition = "I3_6A_PACKET_TREATMENT"
        backend.pair_id = f"{task.task_id}:step{step_idx}"

        # BASE call
        backend.pair_id = f"{task.task_id}:step{step_idx}:BASE"
        base_result, base_pt, base_ct, base_lat, base_err = call_model(
            backend, SYSTEM_PROMPT, packet_json(base_packet), temperature, max_tokens)

        # ACTION_ONLY call
        backend.pair_id = f"{task.task_id}:step{step_idx}:ACTION_ONLY"
        action_result, action_pt, action_ct, action_lat, action_err = call_model(
            backend, SYSTEM_PROMPT, packet_json(action_packet), temperature, max_tokens)

        # EXEC_ASSIST call
        backend.pair_id = f"{task.task_id}:step{step_idx}:EXEC_ASSIST"
        assist_result, assist_pt, assist_ct, assist_lat, assist_err = call_model(
            backend, SYSTEM_PROMPT, packet_json(assist_packet), temperature, max_tokens)

        # Parse responses
        base_parsed = parse_model_response(
            base_result.raw_output if base_result else None)
        action_parsed = parse_model_response(
            action_result.raw_output if action_result else None)
        assist_parsed = parse_model_response(
            assist_result.raw_output if assist_result else None)

        base_action = base_parsed.get("action") if base_parsed else None
        action_action = action_parsed.get("action") if action_parsed else None
        assist_action = assist_parsed.get("action") if assist_parsed else None

        # Assistance utilization fields
        assist_used = assist_parsed.get("assistance_used") if assist_parsed else None
        assist_step = assist_parsed.get("assistance_step") if assist_parsed else None
        success_cond = assist_parsed.get("success_condition_understood") if assist_parsed else None
        reason_code = assist_parsed.get("reason_code") if assist_parsed else None

        results.append({
            "task_id": task.task_id,
            "step_id": step_idx,
            "base_executed_action": a_b_str,
            "governor_action": assist_frame.recommended_action,
            "assistance_type": assist_frame.assistance_type if hasattr(assist_frame, 'assistance_type') else f"{assist_frame.recommended_action}_{assist_frame.bottleneck_type}",
            "assistance_sha256": assist_sha,
            "base": {
                "action": base_action,
                "prompt_tokens": base_pt,
                "completion_tokens": base_ct,
                "latency_ms": round(base_lat, 1),
                "error": base_err,
                "json_valid": base_parsed is not None,
            },
            "action_only": {
                "action": action_action,
                "prompt_tokens": action_pt,
                "completion_tokens": action_ct,
                "latency_ms": round(action_lat, 1),
                "error": action_err,
                "json_valid": action_parsed is not None,
            },
            "exec_assist": {
                "action": assist_action,
                "prompt_tokens": assist_pt,
                "completion_tokens": assist_ct,
                "latency_ms": round(assist_lat, 1),
                "error": assist_err,
                "json_valid": assist_parsed is not None,
                "assistance_used": assist_used,
                "assistance_step": assist_step,
                "success_condition_understood": success_cond,
                "reason_code": reason_code,
            },
            "agreement": {
                "base_vs_action": base_action == action_action,
                "base_vs_assist": base_action == assist_action,
                "action_vs_assist": action_action == assist_action,
                "base_follows_governor": base_action == assist_frame.recommended_action,
                "action_follows_governor": action_action == assist_frame.recommended_action,
                "assist_follows_governor": assist_action == assist_frame.recommended_action,
            },
        })

        # Step forward
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

    return results


def main():
    parser = argparse.ArgumentParser(description="I3.6a packet treatment experiment")
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
    parser.add_argument("--n-tasks", type=int, default=50, help="Number of tasks to process")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_6/development/i3_6a",
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

    # Initialize governors and backend
    general_governor = GeneralGovernor()
    exec_governor = ExecutionGovernor()
    backend = DeepSeekBackend()

    # Process tasks
    print(f"\nRunning I3.6a packet treatment on {args.n_tasks} tasks...")
    all_results: list[dict[str, Any]] = []
    n_processed = 0

    for i, block in enumerate(blocks):
        if n_processed >= args.n_tasks:
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

        print(f"  [{n_processed+1}/{args.n_tasks}] Task {task_id}...")
        results = run_packet_treatment(
            task=task, budget=budget, off_steps=off_steps,
            general_governor=general_governor,
            exec_governor=exec_governor,
            backend=backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        all_results.extend(results)
        n_processed += 1

    print(f"\nTotal treatment states: {len(all_results)}")

    # Save per-state results
    results_path = output_dir / "packet_treatment_v1.jsonl"
    with open(results_path, "w") as f:
        for rec in all_results:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Compute summary statistics
    n = len(all_results)
    if n == 0:
        print("No treatment states found!")
        return

    base_valid = sum(1 for r in all_results if r["base"]["json_valid"])
    action_valid = sum(1 for r in all_results if r["action_only"]["json_valid"])
    assist_valid = sum(1 for r in all_results if r["exec_assist"]["json_valid"])

    base_errors = sum(1 for r in all_results if r["base"]["error"])
    action_errors = sum(1 for r in all_results if r["action_only"]["error"])
    assist_errors = sum(1 for r in all_results if r["exec_assist"]["error"])

    base_follows = sum(1 for r in all_results if r["agreement"]["base_follows_governor"])
    action_follows = sum(1 for r in all_results if r["agreement"]["action_follows_governor"])
    assist_follows = sum(1 for r in all_results if r["agreement"]["assist_follows_governor"])

    base_vs_action = sum(1 for r in all_results if r["agreement"]["base_vs_action"])
    base_vs_assist = sum(1 for r in all_results if r["agreement"]["base_vs_assist"])
    action_vs_assist = sum(1 for r in all_results if r["agreement"]["action_vs_assist"])

    # Token costs
    base_pt = sum(r["base"]["prompt_tokens"] for r in all_results)
    action_pt = sum(r["action_only"]["prompt_tokens"] for r in all_results)
    assist_pt = sum(r["exec_assist"]["prompt_tokens"] for r in all_results)

    base_ct = sum(r["base"]["completion_tokens"] for r in all_results)
    action_ct = sum(r["action_only"]["completion_tokens"] for r in all_results)
    assist_ct = sum(r["exec_assist"]["completion_tokens"] for r in all_results)

    # Assistance utilization
    assist_used_count = sum(
        1 for r in all_results
        if r["exec_assist"].get("assistance_used") is True)
    success_understood = sum(
        1 for r in all_results
        if r["exec_assist"].get("success_condition_understood") is True)

    # Action substitution rates
    base_actions = Counter(r["base"]["action"] for r in all_results if r["base"]["action"])
    action_actions = Counter(r["action_only"]["action"] for r in all_results if r["action_only"]["action"])
    assist_actions = Counter(r["exec_assist"]["action"] for r in all_results if r["exec_assist"]["action"])

    summary = {
        "schema": "DAPH_V2B_I3_6A_PACKET_TREATMENT_V1",
        "assistance_identity_sha256": identity["assistance_identity_sha256"],
        "n_treatment_states": n,
        "n_tasks_processed": n_processed,
        "json_validity": {
            "base": base_valid,
            "action_only": action_valid,
            "exec_assist": assist_valid,
        },
        "backend_errors": {
            "base": base_errors,
            "action_only": action_errors,
            "exec_assist": assist_errors,
        },
        "governor_follow_rate": {
            "base": f"{base_follows}/{n} ({base_follows/n:.1%})",
            "action_only": f"{action_follows}/{n} ({action_follows/n:.1%})",
            "exec_assist": f"{assist_follows}/{n} ({assist_follows/n:.1%})",
        },
        "agreement_rates": {
            "base_vs_action": f"{base_vs_action}/{n} ({base_vs_action/n:.1%})",
            "base_vs_assist": f"{base_vs_assist}/{n} ({base_vs_assist/n:.1%})",
            "action_vs_assist": f"{action_vs_assist}/{n} ({action_vs_assist/n:.1%})",
        },
        "token_costs": {
            "base": {"prompt": base_pt, "completion": base_ct, "total": base_pt + base_ct},
            "action_only": {"prompt": action_pt, "completion": action_ct, "total": action_pt + action_ct},
            "exec_assist": {"prompt": assist_pt, "completion": assist_ct, "total": assist_pt + assist_ct},
        },
        "assistance_utilization": {
            "assistance_used": f"{assist_used_count}/{n}",
            "success_condition_understood": f"{success_understood}/{n}",
        },
        "action_distributions": {
            "base": dict(base_actions.most_common()),
            "action_only": dict(action_actions.most_common()),
            "exec_assist": dict(assist_actions.most_common()),
        },
    }

    summary_path = output_dir / "packet_treatment_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*78}")
    print("I3.6a PACKET TREATMENT SUMMARY")
    print(f"{'='*78}")
    print(f"  Treatment states:          {n}")
    print(f"  Tasks processed:           {n_processed}")
    print(f"\n  JSON validity:")
    print(f"    BASE:        {base_valid}/{n} ({base_valid/n:.1%})")
    print(f"    ACTION_ONLY: {action_valid}/{n} ({action_valid/n:.1%})")
    print(f"    EXEC_ASSIST: {assist_valid}/{n} ({assist_valid/n:.1%})")
    print(f"\n  Governor follow rate:")
    print(f"    BASE:        {base_follows}/{n} ({base_follows/n:.1%})")
    print(f"    ACTION_ONLY: {action_follows}/{n} ({action_follows/n:.1%})")
    print(f"    EXEC_ASSIST: {assist_follows}/{n} ({assist_follows/n:.1%})")
    print(f"\n  Agreement rates:")
    print(f"    BASE vs ACTION:     {base_vs_action}/{n} ({base_vs_action/n:.1%})")
    print(f"    BASE vs ASSIST:     {base_vs_assist}/{n} ({base_vs_assist/n:.1%})")
    print(f"    ACTION vs ASSIST:   {action_vs_assist}/{n} ({action_vs_assist/n:.1%})")
    print(f"\n  Token costs (total):")
    print(f"    BASE:        prompt={base_pt}, completion={base_ct}")
    print(f"    ACTION_ONLY: prompt={action_pt}, completion={action_ct}")
    print(f"    EXEC_ASSIST: prompt={assist_pt}, completion={assist_ct}")
    print(f"\n  Assistance utilization:")
    print(f"    assistance_used:              {assist_used_count}/{n}")
    print(f"    success_condition_understood: {success_understood}/{n}")


if __name__ == "__main__":
    main()
