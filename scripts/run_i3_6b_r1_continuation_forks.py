#!/usr/bin/env python3
"""I3.6b-r1 — Execution Assistance Causal Qualification.

Four-way continuation forks with full traces, forced-action costs,
counterbalanced execution order, and one-shot vs persistent assistance
separation.

For each governor/base disagreement state, fork into four continuations:

  Fork A: a_B + pi_B           (baseline action, baseline continuation)
  Fork B: a_G + pi_B           (governor action, baseline continuation)
  Fork C: a_G + one-shot assist  (governor action, ONE assist frame, then BASE)
  Fork D: a_G + persistent assist (governor action, assist for ALL continuation steps)

Advantages:
  A_B = U_B - U_A    (action-only advantage)
  OneShotGain = U_C - U_B   (one-shot scaffold value over action-only)
  PersistentGain = U_D - U_B  (persistent scaffold value over action-only)
  PersistenceGain = U_D - U_C  (value of persisting beyond one shot)

All utilities include the forced first-action cost:
  U_A = -cost(a_B) + U(continuation | a_B executed)
  U_B = -cost(a_G) + U(continuation | a_G executed)
  U_C = -cost(a_G) + U(one-shot-assist continuation | a_G executed)
  U_D = -cost(a_G) + U(persistent-assist continuation | a_G executed)

Full continuation traces are persisted for mechanism analysis.

Fork execution order is counterbalanced using a deterministic permutation
of (A, B, C, D) derived from HMAC(task_id, step_id).

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_6b_r1_continuation_forks.py \\
        --n-tasks 50 --workers 4
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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


def counterbalance_order(task_id: str, step_id: int) -> list[str]:
    """Deterministically permute fork execution order (A, B, C, D).

    Uses HMAC-SHA256(task_id, step_id) to select one of 24 permutations.
    This removes fixed-order confounding from API state, caching, etc.
    """
    import itertools
    h = hashlib.sha256(f"{task_id}:{step_id}".encode()).hexdigest()
    perm_idx = int(h[:8], 16) % 24
    perms = list(itertools.permutations(["A", "B", "C", "D"]))
    return list(perms[perm_idx])


def continue_with_model(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    t_runtime: TaskRuntime,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    utility: MetareasoningUtility,
    general_governor: GeneralGovernor,
    exec_governor: ExecutionGovernor,
    mode: str,  # "BASE", "EXEC_ASSIST", or "ONE_SHOT_ASSIST"
    fork_label: str,
    api_key: str,
    start_step: int = 0,
    max_steps: int = 24,
) -> dict[str, Any]:
    """Continue a trajectory from the given state.

    mode="BASE": base packet for all continuation steps
    mode="EXEC_ASSIST": execution-assist packet for ALL continuation steps
    mode="ONE_SHOT_ASSIST": execution-assist for the FIRST continuation step only,
                             then BASE for all subsequent steps

    Returns full trace information including:
      - continuation_actions[], continuation_outcomes[]
      - assistance_types[], assistance_frame_sha256s[]
      - step_costs[], terminal_reward, terminal_action, total_action_cost
    """
    executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    # Create a fresh backend for each fork to avoid shared state
    backend = DeepSeekBackend()

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    terminal_action = None
    backend_errors = 0

    # Full trace
    continuation_actions: list[str] = []
    continuation_outcomes: list[str] = []
    assistance_types: list[str] = []
    assistance_frame_shas: list[str] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    for step_id in range(start_step, max_steps):
        observation = build_observation(
            t_runtime, task, cond,
            tuple(prior_decisions), tuple(prior_outcomes))

        prior_action_strs = tuple(
            d.selected_action if isinstance(d.selected_action, str)
            else d.selected_action.value for d in prior_decisions)

        # Determine packet mode for this step
        if mode == "BASE":
            step_mode = "BASE"
        elif mode == "EXEC_ASSIST":
            step_mode = "EXEC_ASSIST"
        elif mode == "ONE_SHOT_ASSIST":
            # First continuation step gets assist, rest get BASE
            step_mode = "EXEC_ASSIST" if steps_taken == 0 else "BASE"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        assist_type = None
        assist_sha = None

        if step_mode == "BASE":
            packet = build_base_packet(observation)
            assert_no_evaluator_leakage(packet)
        else:
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
                assist_type = f"{assist_frame.recommended_action}_{assist_frame.bottleneck_type}"
                assist_sha = assistance_frame_sha256(assist_frame)
            else:
                packet = build_governor_packet(observation, gov_frame)
                assert_no_evaluator_leakage(packet)

        user_prompt = packet_json(packet)

        backend.task_id = task.task_id
        backend.condition = f"i3_6b_r1_{mode}"
        backend.pair_id = f"i3_6b_r1:{task.task_id}:{fork_label}:step{step_id}"

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
        total_action_cost += step_cost
        step_costs.append(round(step_cost, 4))

        # Record trace
        action_str = action.value if hasattr(action, "value") else str(action)
        continuation_actions.append(action_str)
        continuation_outcomes.append(execution.outcome_code)
        assistance_types.append(assist_type)
        assistance_frame_shas.append(assist_sha)

        if execution.terminal:
            tr = utility.terminal_reward(execution.action, bool(execution.task_success))
            realized += tr
            terminal_reward = tr
            success = bool(execution.task_success)
            terminal = True
            terminal_result = execution.outcome_code
            terminal_action = action_str

        prior_decisions.append(DecisionSummary(
            f"{task.task_id}:{fork_label}:{step_id}", action_str,
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
        "terminal_action": terminal_action,
        "terminal_reward": round(terminal_reward, 4),
        "total_action_cost": round(total_action_cost, 4),
        # Full trace
        "continuation_actions": continuation_actions,
        "continuation_outcomes": continuation_outcomes,
        "assistance_types": assistance_types,
        "assistance_frame_sha256s": assistance_frame_shas,
        "step_costs": step_costs,
    }


def execute_fork(
    fork_id: str,
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    step_idx: int,
    forced_action: DecisionAction,
    i3_runtime_state: Any,
    t_runtime_state: Any,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    utility: MetareasoningUtility,
    general_governor: GeneralGovernor,
    exec_governor: ExecutionGovernor,
    api_key: str,
    continuation_mode: str,
) -> dict[str, Any]:
    """Execute one fork: force the first action, then continue with the given mode.

    Returns the full fork result including forced-action cost.
    """
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Deep copy state
    fork_i3 = copy.deepcopy(i3_runtime_state)
    fork_t = copy.deepcopy(t_runtime_state)
    fork_pd = list(prior_decisions)
    fork_po = list(prior_outcomes)

    # Execute forced action
    exec_res = oracle_executor.execute(fork_i3, forced_action)
    exec_t = task_executor.execute(fork_t, forced_action)

    # Compute forced action cost
    resources_before = t_runtime_state.resources
    resources_after = exec_t.runtime.resources
    forced_cost = utility.action_cost(resources_before, resources_after)

    fork_pd.append(DecisionSummary(
        f"{task.task_id}:fork{fork_id}:{step_idx}",
        forced_action.value, "FORCED", exec_res.outcome_code))
    fork_po.append(exec_res.outcome_code)

    if exec_res.terminal:
        # Forced action was terminal
        tr = utility.terminal_reward(forced_action, bool(exec_res.task_success))
        total_utility = round(-forced_cost + tr, 4)
        return {
            "fork_id": fork_id,
            "forced_action": forced_action.value,
            "forced_action_cost": round(forced_cost, 4),
            "continuation_utility": round(tr, 4),
            "total_utility": total_utility,
            "success": bool(exec_res.task_success),
            "steps": 0,
            "model_calls": 0,
            "backend_errors": 0,
            "terminal_result": exec_res.outcome_code,
            "terminal_action": forced_action.value,
            "terminal_reward": round(tr, 4),
            "total_action_cost": round(forced_cost, 4),
            "continuation_actions": [],
            "continuation_outcomes": [],
            "assistance_types": [],
            "assistance_frame_sha256s": [],
            "step_costs": [],
        }

    # Continue with model
    cont = continue_with_model(
        task=task, budget=budget, t_runtime=exec_t.runtime,
        prior_decisions=fork_pd, prior_outcomes=fork_po,
        utility=utility,
        general_governor=general_governor,
        exec_governor=exec_governor,
        mode=continuation_mode,
        fork_label=f"fork{fork_id}_step{step_idx}",
        api_key=api_key,
        start_step=step_idx + 1)

    total_utility = round(-forced_cost + cont["realized_utility"], 4)

    return {
        "fork_id": fork_id,
        "forced_action": forced_action.value,
        "forced_action_cost": round(forced_cost, 4),
        "continuation_utility": cont["realized_utility"],
        "total_utility": total_utility,
        "success": cont["success"],
        "steps": cont["steps"],
        "model_calls": cont["model_calls"],
        "backend_errors": cont["backend_errors"],
        "terminal_result": cont["terminal_result"],
        "terminal_action": cont["terminal_action"],
        "terminal_reward": cont["terminal_reward"],
        "total_action_cost": round(forced_cost + cont["total_action_cost"], 4),
        "continuation_actions": cont["continuation_actions"],
        "continuation_outcomes": cont["continuation_outcomes"],
        "assistance_types": cont["assistance_types"],
        "assistance_frame_sha256s": cont["assistance_frame_sha256s"],
        "step_costs": cont["step_costs"],
    }


def process_one_fork_set(
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
    """Process one four-way fork set: A, B, C, D.

    A: a_B + pi_B
    B: a_G + pi_B
    C: a_G + one-shot assist then pi_B
    D: a_G + persistent pi_Assist

    Fork execution order is counterbalanced.
    """
    # Determine counterbalanced execution order
    fork_order = counterbalance_order(task.task_id, step_idx)

    # Define fork configurations
    fork_configs = {
        "A": {"forced_action": a_base, "continuation_mode": "BASE"},
        "B": {"forced_action": a_gov, "continuation_mode": "BASE"},
        "C": {"forced_action": a_gov, "continuation_mode": "ONE_SHOT_ASSIST"},
        "D": {"forced_action": a_gov, "continuation_mode": "EXEC_ASSIST"},
    }

    # Execute forks in counterbalanced order
    results: dict[str, dict] = {}
    for fork_id in fork_order:
        cfg = fork_configs[fork_id]
        results[fork_id] = execute_fork(
            fork_id=fork_id,
            task=task, budget=budget, step_idx=step_idx,
            forced_action=cfg["forced_action"],
            i3_runtime_state=i3_runtime_state,
            t_runtime_state=t_runtime_state,
            prior_decisions=prior_decisions,
            prior_outcomes=prior_outcomes,
            utility=utility,
            general_governor=general_governor,
            exec_governor=exec_governor,
            api_key=api_key,
            continuation_mode=cfg["continuation_mode"],
        )

    # Extract utilities (including forced-action cost)
    u_a = results["A"]["total_utility"]
    u_b = results["B"]["total_utility"]
    u_c = results["C"]["total_utility"]
    u_d = results["D"]["total_utility"]

    # Compute advantages
    a_b = round(u_b - u_a, 4)          # action-only advantage
    one_shot_gain = round(u_c - u_b, 4)  # one-shot scaffold value
    persistent_gain = round(u_d - u_b, 4)  # persistent scaffold value
    persistence_gain = round(u_d - u_c, 4)  # value of persisting

    # Rescue/break classification (BASE vs each governor variant)
    def classify(base_ok: bool, treat_ok: bool) -> str:
        if base_ok and treat_ok:
            return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok:
            return "BOTH_FAIL"
        elif not base_ok and treat_ok:
            return "RESCUE"
        else:
            return "BREAK"

    base_ok = results["A"]["success"]
    action_class = classify(base_ok, results["B"]["success"])
    oneshot_class = classify(base_ok, results["C"]["success"])
    persistent_class = classify(base_ok, results["D"]["success"])

    return {
        "task_id": task.task_id,
        "step_id": step_idx,
        "base_action": a_base.value,
        "gov_action": a_gov.value,
        "fork_order": fork_order,
        # Utilities (with forced-action cost)
        "u_a": u_a,
        "u_b": u_b,
        "u_c": u_c,
        "u_d": u_d,
        # Advantages
        "a_b": a_b,
        "one_shot_gain": one_shot_gain,
        "persistent_gain": persistent_gain,
        "persistence_gain": persistence_gain,
        # Success
        "base_success": base_ok,
        "action_success": results["B"]["success"],
        "oneshot_success": results["C"]["success"],
        "persistent_success": results["D"]["success"],
        # Classification
        "action_class": action_class,
        "oneshot_class": oneshot_class,
        "persistent_class": persistent_class,
        # Fork details (full traces)
        "fork_a": results["A"],
        "fork_b": results["B"],
        "fork_c": results["C"],
        "fork_d": results["D"],
    }


def main():
    parser = argparse.ArgumentParser(description="I3.6b-r1 continuation forks")
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
        default="experiments/v2b_i3_6/development/i3_6b_r1",
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
    print(f"\nProcessing {len(fork_states)} four-way forks with {args.workers} workers...")
    print(f"  Fork modes: A=a_B+pi_B, B=a_G+pi_B, C=a_G+one-shot, D=a_G+persistent")

    all_results: list[dict[str, Any]] = []
    completed = 0

    def fork_wrapper(fs):
        return process_one_fork_set(
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
                if completed % 5 == 0:
                    print(f"  Completed {completed}/{len(fork_states)} fork sets...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_results)} fork sets")

    # Save results
    results_path = output_dir / "continuation_forks_r1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Compute summary
    n = len(all_results)
    if n == 0:
        print("No forks completed!")
        return

    u_a_mean = sum(r["u_a"] for r in all_results) / n
    u_b_mean = sum(r["u_b"] for r in all_results) / n
    u_c_mean = sum(r["u_c"] for r in all_results) / n
    u_d_mean = sum(r["u_d"] for r in all_results) / n

    a_b_mean = sum(r["a_b"] for r in all_results) / n
    osg_mean = sum(r["one_shot_gain"] for r in all_results) / n
    pg_mean = sum(r["persistent_gain"] for r in all_results) / n
    persg_mean = sum(r["persistence_gain"] for r in all_results) / n

    base_success = sum(1 for r in all_results if r["base_success"])
    action_success = sum(1 for r in all_results if r["action_success"])
    oneshot_success = sum(1 for r in all_results if r["oneshot_success"])
    persistent_success = sum(1 for r in all_results if r["persistent_success"])

    action_classes = Counter(r["action_class"] for r in all_results)
    oneshot_classes = Counter(r["oneshot_class"] for r in all_results)
    persistent_classes = Counter(r["persistent_class"] for r in all_results)

    # Model calls
    base_calls = sum(r["fork_a"]["model_calls"] for r in all_results)
    action_calls = sum(r["fork_b"]["model_calls"] for r in all_results)
    oneshot_calls = sum(r["fork_c"]["model_calls"] for r in all_results)
    persistent_calls = sum(r["fork_d"]["model_calls"] for r in all_results)

    summary = {
        "schema": "DAPH_V2B_I3_6B_R1_CONTINUATION_FORKS_V1",
        "assistance_identity_sha256": identity["assistance_identity_sha256"],
        "n_forks": n,
        "n_tasks": n_tasks_processed,
        "fork_modes": {
            "A": "a_B + pi_B (baseline)",
            "B": "a_G + pi_B (action-only)",
            "C": "a_G + one-shot assist then pi_B",
            "D": "a_G + persistent pi_Assist",
        },
        "utility": {
            "mean_u_a_base": round(u_a_mean, 4),
            "mean_u_b_action": round(u_b_mean, 4),
            "mean_u_c_oneshot": round(u_c_mean, 4),
            "mean_u_d_persistent": round(u_d_mean, 4),
        },
        "advantages": {
            "mean_a_b": round(a_b_mean, 4),
            "mean_one_shot_gain": round(osg_mean, 4),
            "mean_persistent_gain": round(pg_mean, 4),
            "mean_persistence_gain": round(persg_mean, 4),
        },
        "success": {
            "base": f"{base_success}/{n}",
            "action_only": f"{action_success}/{n}",
            "one_shot": f"{oneshot_success}/{n}",
            "persistent": f"{persistent_success}/{n}",
        },
        "classification_base_vs_action": dict(action_classes),
        "classification_base_vs_oneshot": dict(oneshot_classes),
        "classification_base_vs_persistent": dict(persistent_classes),
        "model_calls": {
            "base": base_calls,
            "action_only": action_calls,
            "one_shot": oneshot_calls,
            "persistent": persistent_calls,
        },
        "delta_metrics": {
            "delta_u_oneshot_vs_action": round(u_c_mean - u_b_mean, 4),
            "delta_u_persistent_vs_action": round(u_d_mean - u_b_mean, 4),
            "delta_calls_oneshot_vs_action": oneshot_calls - action_calls,
            "delta_calls_persistent_vs_action": persistent_calls - action_calls,
        },
    }

    summary_path = output_dir / "continuation_forks_r1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*78}")
    print("I3.6b-r1 CONTINUATION FORKS SUMMARY")
    print(f"{'='*78}")
    print(f"  Fork sets:                 {n}")
    print(f"  Tasks:                     {n_tasks_processed}")
    print(f"\n  Mean utility (with forced-action cost):")
    print(f"    A: a_B + pi_B:            {u_a_mean:+.4f}")
    print(f"    B: a_G + pi_B:            {u_b_mean:+.4f}")
    print(f"    C: a_G + one-shot:        {u_c_mean:+.4f}")
    print(f"    D: a_G + persistent:      {u_d_mean:+.4f}")
    print(f"\n  Mean advantages:")
    print(f"    A_B (action-only):        {a_b_mean:+.4f}")
    print(f"    OneShotGain (C-B):        {osg_mean:+.4f}")
    print(f"    PersistentGain (D-B):     {pg_mean:+.4f}")
    print(f"    PersistenceGain (D-C):    {persg_mean:+.4f}")
    print(f"\n  Success rates:")
    print(f"    BASE:       {base_success}/{n} ({base_success/n:.1%})")
    print(f"    ACTION_ONLY: {action_success}/{n} ({action_success/n:.1%})")
    print(f"    ONE_SHOT:   {oneshot_success}/{n} ({oneshot_success/n:.1%})")
    print(f"    PERSISTENT: {persistent_success}/{n} ({persistent_success/n:.1%})")
    print(f"\n  Classification (BASE vs ACTION):")
    for cls, cnt in action_classes.most_common():
        print(f"    {cls}: {cnt}")
    print(f"\n  Classification (BASE vs ONE_SHOT):")
    for cls, cnt in oneshot_classes.most_common():
        print(f"    {cls}: {cnt}")
    print(f"\n  Classification (BASE vs PERSISTENT):")
    for cls, cnt in persistent_classes.most_common():
        print(f"    {cls}: {cnt}")
    print(f"\n  Model calls:")
    print(f"    BASE:       {base_calls}")
    print(f"    ACTION_ONLY: {action_calls}")
    print(f"    ONE_SHOT:   {oneshot_calls}")
    print(f"    PERSISTENT: {persistent_calls}")
    print(f"\n  Delta metrics:")
    print(f"    dU oneshot vs action:    {summary['delta_metrics']['delta_u_oneshot_vs_action']:+.4f}")
    print(f"    dU persistent vs action: {summary['delta_metrics']['delta_u_persistent_vs_action']:+.4f}")
    print(f"    dCalls oneshot vs action:    {summary['delta_metrics']['delta_calls_oneshot_vs_action']}")
    print(f"    dCalls persistent vs action: {summary['delta_metrics']['delta_calls_persistent_vs_action']}")


if __name__ == "__main__":
    main()
