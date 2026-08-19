#!/usr/bin/env python3
"""I3.6b — Single-intervention continuation forks.

For each governor/base disagreement state, fork into three continuations:

  Fork A: a_B + pi_B    (baseline action, baseline continuation)
  Fork B: a_G + pi_B    (governor action, baseline continuation)
  Fork C: a_G + pi_Assist (governor action, execution-assistance continuation)

The central I3.6 measurement:

  A_B = U(a_G + pi_B) - U(a_B + pi_B)    [action-only advantage]
  A_E = U(a_G + pi_Assist) - U(a_B + pi_B) [execution-assist advantage]
  ExecutionGain = A_E - A_B              [scaffold unlocks value?]

Also classifies each fork pair as:
  BOTH_SUCCESS, BOTH_FAIL, ASSIST_RESCUE, ASSIST_BREAK

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_6b_continuation_forks.py \\
        --n-tasks 50 --workers 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from hrm_adaptive_memory.executive.actions import ActionProposal
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, TaskRuntime, initial_runtime,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import (
    build_base_packet, build_governor_packet,
    packet_json, packet_sha256, assert_no_evaluator_leakage,
)
from hrm_adaptive_memory.executive.i3_5_1.model_prompt import SYSTEM_PROMPT
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    I3BenchmarkTask, load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, initial_i3_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState, ResourceExhausted
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.execution_governor import (
    ExecutionGovernor,
    serialize_assistance_packet,
)
from hrm_adaptive_memory.executive.execution_governor.serializer import (
    assert_no_evaluator_leakage as assert_assist_no_leakage,
)
from hrm_adaptive_memory.executive.execution_governor.identity import (
    compute_assistance_identity, assistance_frame_sha256,
)


def continue_with_model(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    t_runtime: TaskRuntime,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    backend: DeepSeekBackend,
    utility: MetareasoningUtility,
    general_governor: GeneralGovernor,
    exec_governor: ExecutionGovernor,
    mode: str,  # "BASE" or "EXEC_ASSIST"
    fork_label: str,
    start_step: int = 0,
    max_steps: int = 24,
) -> dict[str, Any]:
    """Continue a trajectory from the given state.

    mode="BASE": use base packet (no governor) for all continuation steps
    mode="EXEC_ASSIST": use execution-assist packet for all continuation steps
    """
    executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    backend_errors = 0

    for step_id in range(start_step, max_steps):
        observation = build_observation(
            t_runtime, task, cond,
            tuple(prior_decisions), tuple(prior_outcomes))

        prior_action_strs = tuple(
            d.selected_action if isinstance(d.selected_action, str)
            else d.selected_action.value for d in prior_decisions)

        if mode == "BASE":
            packet = build_base_packet(observation)
            assert_no_evaluator_leakage(packet)
        elif mode == "EXEC_ASSIST":
            gov_frame = general_governor.assess(
                observation=observation,
                remaining_steps=max_steps - step_id,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )
            assist_frame = exec_governor.plan(
                observation=observation,
                remaining_steps=max_steps - step_id,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )
            if assist_frame is not None:
                packet = serialize_assistance_packet(
                    observation, gov_frame, assist_frame, mode="EXECUTION_ASSIST")
                assert_assist_no_leakage(packet)
            else:
                # No assistance (e.g., STOP) — use governor packet
                packet = build_governor_packet(observation, gov_frame)
                assert_no_evaluator_leakage(packet)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        user_prompt = packet_json(packet)

        backend.task_id = task.task_id
        backend.condition = f"i3_6b_{mode}"
        backend.pair_id = f"i3_6b:{task.task_id}:{fork_label}:step{step_id}"

        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
        except Exception:
            backend_errors += 1
            proposal = BACKEND_ERROR_PROPOSAL
        else:
            outcome = decode_output(call_result.raw_output, strict=True)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = FAIL_CLOSED_PROPOSAL

        action = proposal.action
        resources_before = t_runtime.resources
        try:
            execution = executor.execute(t_runtime, action)
        except ResourceExhausted:
            execution = type(execution)(
                DecisionAction.DEFER, t_runtime, True, False, "RESOURCE_EXHAUSTED")

        resources_after = execution.runtime.resources
        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost
        if execution.terminal:
            realized += utility.terminal_reward(execution.action, bool(execution.task_success))
            success = bool(execution.task_success)
            terminal = True
            terminal_result = execution.outcome_code

        prior_decisions.append(DecisionSummary(
            f"{task.task_id}:{fork_label}:{step_id}", action.value,
            proposal.reason_code, execution.outcome_code))
        prior_outcomes.append(execution.outcome_code)

        t_runtime = execution.runtime
        steps_taken += 1

        if execution.terminal:
            break

    return {
        "realized_utility": round(realized, 4),
        "success": success,
        "steps": steps_taken,
        "model_calls": model_calls,
        "backend_errors": backend_errors,
        "terminal_result": terminal_result,
    }


def process_one_fork(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    step_idx: int,
    a_base: DecisionAction,
    a_gov: DecisionAction,
    i3_runtime_state: Any,
    t_runtime_state: Any,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    utility: MetareasoningUtility,
    general_governor: GeneralGovernor,
    exec_governor: ExecutionGovernor,
    api_key: str,
) -> dict[str, Any]:
    """Process one three-way fork: A (a_B + pi_B), B (a_G + pi_B), C (a_G + pi_Assist)."""
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Fork A: execute a_base, continue with BASE model
    fork_a_i3 = copy.deepcopy(i3_runtime_state)
    fork_a_t = copy.deepcopy(t_runtime_state)
    fork_a_pd = list(prior_decisions)
    fork_a_po = list(prior_outcomes)

    exec_a = oracle_executor.execute(fork_a_i3, a_base)
    exec_a_t = task_executor.execute(fork_a_t, a_base)
    fork_a_pd.append(DecisionSummary(
        f"{task.task_id}:forkA:{step_idx}", a_base.value,
        "FORK", exec_a.outcome_code))
    fork_a_po.append(exec_a.outcome_code)

    backend = DeepSeekBackend()

    if exec_a.terminal:
        cont_a = {
            "realized_utility": round(
                utility.terminal_reward(a_base, bool(exec_a.task_success)), 4),
            "success": bool(exec_a.task_success),
            "steps": 0, "model_calls": 0, "backend_errors": 0,
            "terminal_result": exec_a.outcome_code,
        }
    else:
        cont_a = continue_with_model(
            task=task, budget=budget, t_runtime=exec_a_t.runtime,
            prior_decisions=fork_a_pd, prior_outcomes=fork_a_po,
            backend=backend, utility=utility,
            general_governor=general_governor,
            exec_governor=exec_governor,
            mode="BASE", fork_label=f"forkA_step{step_idx}",
            start_step=step_idx + 1)

    # Fork B: execute a_gov, continue with BASE model
    fork_b_i3 = copy.deepcopy(i3_runtime_state)
    fork_b_t = copy.deepcopy(t_runtime_state)
    fork_b_pd = list(prior_decisions)
    fork_b_po = list(prior_outcomes)

    exec_b = oracle_executor.execute(fork_b_i3, a_gov)
    exec_b_t = task_executor.execute(fork_b_t, a_gov)
    fork_b_pd.append(DecisionSummary(
        f"{task.task_id}:forkB:{step_idx}", a_gov.value,
        "FORK", exec_b.outcome_code))
    fork_b_po.append(exec_b.outcome_code)

    if exec_b.terminal:
        cont_b = {
            "realized_utility": round(
                utility.terminal_reward(a_gov, bool(exec_b.task_success)), 4),
            "success": bool(exec_b.task_success),
            "steps": 0, "model_calls": 0, "backend_errors": 0,
            "terminal_result": exec_b.outcome_code,
        }
    else:
        cont_b = continue_with_model(
            task=task, budget=budget, t_runtime=exec_b_t.runtime,
            prior_decisions=fork_b_pd, prior_outcomes=fork_b_po,
            backend=backend, utility=utility,
            general_governor=general_governor,
            exec_governor=exec_governor,
            mode="BASE", fork_label=f"forkB_step{step_idx}",
            start_step=step_idx + 1)

    # Fork C: execute a_gov, continue with EXEC_ASSIST model
    fork_c_i3 = copy.deepcopy(i3_runtime_state)
    fork_c_t = copy.deepcopy(t_runtime_state)
    fork_c_pd = list(prior_decisions)
    fork_c_po = list(prior_outcomes)

    exec_c = oracle_executor.execute(fork_c_i3, a_gov)
    exec_c_t = task_executor.execute(fork_c_t, a_gov)
    fork_c_pd.append(DecisionSummary(
        f"{task.task_id}:forkC:{step_idx}", a_gov.value,
        "FORK", exec_c.outcome_code))
    fork_c_po.append(exec_c.outcome_code)

    if exec_c.terminal:
        cont_c = {
            "realized_utility": round(
                utility.terminal_reward(a_gov, bool(exec_c.task_success)), 4),
            "success": bool(exec_c.task_success),
            "steps": 0, "model_calls": 0, "backend_errors": 0,
            "terminal_result": exec_c.outcome_code,
        }
    else:
        cont_c = continue_with_model(
            task=task, budget=budget, t_runtime=exec_c_t.runtime,
            prior_decisions=fork_c_pd, prior_outcomes=fork_c_po,
            backend=backend, utility=utility,
            general_governor=general_governor,
            exec_governor=exec_governor,
            mode="EXEC_ASSIST", fork_label=f"forkC_step{step_idx}",
            start_step=step_idx + 1)

    # Compute advantages
    u_a = cont_a["realized_utility"]
    u_b = cont_b["realized_utility"]
    u_c = cont_c["realized_utility"]

    a_b = round(u_b - u_a, 4)  # action-only advantage
    a_e = round(u_c - u_a, 4)  # execution-assist advantage
    execution_gain = round(a_e - a_b, 4)

    # Rescue/break classification (A vs C)
    if cont_a["success"] and cont_c["success"]:
        rescue_class = "BOTH_SUCCESS"
    elif not cont_a["success"] and not cont_c["success"]:
        rescue_class = "BOTH_FAIL"
    elif not cont_a["success"] and cont_c["success"]:
        rescue_class = "ASSIST_RESCUE"
    else:
        rescue_class = "ASSIST_BREAK"

    # Also classify A vs B
    if cont_a["success"] and cont_b["success"]:
        action_class = "BOTH_SUCCESS"
    elif not cont_a["success"] and not cont_b["success"]:
        action_class = "BOTH_FAIL"
    elif not cont_a["success"] and cont_b["success"]:
        action_class = "ACTION_RESCUE"
    else:
        action_class = "ACTION_BREAK"

    return {
        "task_id": task.task_id,
        "step_id": step_idx,
        "base_action": a_base.value,
        "gov_action": a_gov.value,
        # Utilities
        "u_base": u_a,
        "u_action_only": u_b,
        "u_exec_assist": u_c,
        # Advantages
        "a_b": a_b,
        "a_e": a_e,
        "execution_gain": execution_gain,
        # Success
        "base_success": cont_a["success"],
        "action_success": cont_b["success"],
        "assist_success": cont_c["success"],
        # Classification
        "rescue_class": rescue_class,
        "action_class": action_class,
        # Steps and calls
        "base_steps": cont_a["steps"],
        "action_steps": cont_b["steps"],
        "assist_steps": cont_c["steps"],
        "base_model_calls": cont_a["model_calls"],
        "action_model_calls": cont_b["model_calls"],
        "assist_model_calls": cont_c["model_calls"],
        "backend_errors": cont_a["backend_errors"] + cont_b["backend_errors"] + cont_c["backend_errors"],
    }


def main():
    parser = argparse.ArgumentParser(description="I3.6b continuation forks")
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
    parser.add_argument("--max-forks", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_6/development/i3_6b",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

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

    # Load benchmark and results
    print(f"\nLoading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split("structure_dev_v2")
    task_map = {t.task_id: t for t in split_bm.tasks}

    print(f"Loading results from {args.results}...")
    results_data = json.loads(Path(args.results).read_text())
    blocks = results_data["results"]
    print(f"Loaded {len(blocks)} task blocks")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)
    general_governor = GeneralGovernor()
    exec_governor = ExecutionGovernor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Replay OFF trajectories to find disagreement states
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

        # Replay to find disagreement states
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
                    "a_base": DecisionAction(a_b_str),
                    "a_gov": DecisionAction(a_g_str),
                    "i3_runtime": copy.deepcopy(i3_runtime),
                    "t_runtime": copy.deepcopy(t_runtime),
                    "prior_decisions": list(prior_decisions),
                    "prior_outcomes": list(prior_outcomes),
                    "budget": budget,
                    "task": task,
                })

            # Step forward
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

    if args.max_forks:
        fork_states = fork_states[:args.max_forks]
        print(f"Limited to {len(fork_states)} forks")

    # Process forks in parallel
    print(f"\nProcessing {len(fork_states)} three-way forks with {args.workers} workers...")

    all_results: list[dict[str, Any]] = []
    completed = 0

    def fork_wrapper(fs):
        return process_one_fork(
            task=fs["task"],
            budget=fs["budget"],
            step_idx=fs["step_idx"],
            a_base=fs["a_base"],
            a_gov=fs["a_gov"],
            i3_runtime_state=fs["i3_runtime"],
            t_runtime_state=fs["t_runtime"],
            prior_decisions=fs["prior_decisions"],
            prior_outcomes=fs["prior_outcomes"],
            utility=utility,
            general_governor=general_governor,
            exec_governor=exec_governor,
            api_key=api_key,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fork_wrapper, fs): fs for fs in fork_states}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Completed {completed}/{len(fork_states)} forks...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_results)} forks")

    # Save results
    results_path = output_dir / "continuation_forks_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Compute summary
    n = len(all_results)
    if n == 0:
        print("No forks completed!")
        return

    u_base_mean = sum(r["u_base"] for r in all_results) / n
    u_action_mean = sum(r["u_action_only"] for r in all_results) / n
    u_assist_mean = sum(r["u_exec_assist"] for r in all_results) / n

    a_b_mean = sum(r["a_b"] for r in all_results) / n
    a_e_mean = sum(r["a_e"] for r in all_results) / n
    eg_mean = sum(r["execution_gain"] for r in all_results) / n

    # Success rates
    base_success = sum(1 for r in all_results if r["base_success"])
    action_success = sum(1 for r in all_results if r["action_success"])
    assist_success = sum(1 for r in all_results if r["assist_success"])

    # Rescue/break
    rescue_classes = Counter(r["rescue_class"] for r in all_results)
    action_classes = Counter(r["action_class"] for r in all_results)

    n_rescue = rescue_classes.get("ASSIST_RESCUE", 0)
    n_break = rescue_classes.get("ASSIST_BREAK", 0)
    n_action_rescue = action_classes.get("ACTION_RESCUE", 0)
    n_action_break = action_classes.get("ACTION_BREAK", 0)

    # Token costs (model calls as proxy)
    base_calls = sum(r["base_model_calls"] for r in all_results)
    action_calls = sum(r["action_model_calls"] for r in all_results)
    assist_calls = sum(r["assist_model_calls"] for r in all_results)

    summary = {
        "schema": "DAPH_V2B_I3_6B_CONTINUATION_FORKS_V1",
        "assistance_identity_sha256": identity["assistance_identity_sha256"],
        "n_forks": n,
        "n_tasks": n_tasks_processed,
        "utility": {
            "mean_u_base": round(u_base_mean, 4),
            "mean_u_action_only": round(u_action_mean, 4),
            "mean_u_exec_assist": round(u_assist_mean, 4),
        },
        "advantages": {
            "mean_a_b": round(a_b_mean, 4),
            "mean_a_e": round(a_e_mean, 4),
            "mean_execution_gain": round(eg_mean, 4),
        },
        "success": {
            "base": f"{base_success}/{n}",
            "action_only": f"{action_success}/{n}",
            "exec_assist": f"{assist_success}/{n}",
        },
        "rescue_classification": {
            "assist_rescue": n_rescue,
            "assist_break": n_break,
            "both_success": rescue_classes.get("BOTH_SUCCESS", 0),
            "both_fail": rescue_classes.get("BOTH_FAIL", 0),
            "action_rescue": n_action_rescue,
            "action_break": n_action_break,
        },
        "model_calls": {
            "base": base_calls,
            "action_only": action_calls,
            "exec_assist": assist_calls,
        },
        "backend_errors": sum(r["backend_errors"] for r in all_results),
    }

    summary_path = output_dir / "continuation_forks_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*78}")
    print("I3.6b CONTINUATION FORKS SUMMARY")
    print(f"{'='*78}")
    print(f"  Forks:                     {n}")
    print(f"  Tasks:                     {n_tasks_processed}")
    print(f"\n  Mean utility:")
    print(f"    BASE (a_B + pi_B):       {u_base_mean:+.4f}")
    print(f"    ACTION_ONLY (a_G + pi_B): {u_action_mean:+.4f}")
    print(f"    EXEC_ASSIST (a_G + pi_A): {u_assist_mean:+.4f}")
    print(f"\n  Mean advantages:")
    print(f"    A_B (action-only):       {a_b_mean:+.4f}")
    print(f"    A_E (execution-assist):  {a_e_mean:+.4f}")
    print(f"    ExecutionGain = A_E - A_B: {eg_mean:+.4f}")
    print(f"\n  Success rates:")
    print(f"    BASE:        {base_success}/{n} ({base_success/n:.1%})")
    print(f"    ACTION_ONLY: {action_success}/{n} ({action_success/n:.1%})")
    print(f"    EXEC_ASSIST: {assist_success}/{n} ({assist_success/n:.1%})")
    print(f"\n  Rescue/break (ASSIST vs BASE):")
    print(f"    ASSIST_RESCUE: {n_rescue}")
    print(f"    ASSIST_BREAK:  {n_break}")
    print(f"    BOTH_SUCCESS:  {rescue_classes.get('BOTH_SUCCESS', 0)}")
    print(f"    BOTH_FAIL:     {rescue_classes.get('BOTH_FAIL', 0)}")
    print(f"\n  Rescue/break (ACTION vs BASE):")
    print(f"    ACTION_RESCUE: {n_action_rescue}")
    print(f"    ACTION_BREAK:  {n_action_break}")
    print(f"\n  Model calls:")
    print(f"    BASE:        {base_calls}")
    print(f"    ACTION_ONLY: {action_calls}")
    print(f"    EXEC_ASSIST: {assist_calls}")
    print(f"  Backend errors: {sum(r['backend_errors'] for r in all_results)}")


if __name__ == "__main__":
    main()
