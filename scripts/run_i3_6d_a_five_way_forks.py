#!/usr/bin/env python3
"""I3.6d-a — Resolution Scaffold Five-Way Forks.

Five-way continuation forks testing whether resolution assistance
(hypotheses + evidence map + discriminators + answer conditions)
produces actual task rescues, not just terminal-penalty shifts.

For each governor/base disagreement state, fork into five continuations:

  A = a_B + pi_B              (baseline action, baseline continuation)
  B = a_G + pi_B              (governor action, baseline continuation)
  C = a_G + generic execution assist  (old I3.6a/b scaffold)
  D = a_G + resolution assist (new I3.6d scaffold)
  E = a_B + resolution assist (baseline action + resolution context)

Key contrasts:
  ResolutionGain = U_D - U_C   (resolution scaffold vs generic scaffold)
  ContextGain = U_E - U_D      (context value without action override)
  ActionOverrideGain = U_D - U_E  (value of governor action override)

STOP->ANSWER hazard is classified as UNSAFE_OVERRIDE_STOP_TO_ANSWER.

All utilities include forced first-action cost.
Full continuation traces are persisted.
Fork execution order is counterbalanced.
Task-clustered bootstrap CIs are computed.

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_6d_a_five_way_forks.py \\
        --n-tasks 50 --workers 4
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
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
    assistance_frame_sha256,
)
from hrm_adaptive_memory.executive.resolution_governor import (
    ResolutionGovernor,
    ResolutionContext,
    serialize_resolution_packet,
    assert_no_evaluator_leakage as assert_resolution_no_leakage,
    compute_resolution_identity,
    resolution_frame_sha256,
)
from hrm_adaptive_memory.executive.resolution_governor.serializer import (
    packet_json as resolution_packet_json,
    packet_sha256 as resolution_packet_sha256,
)


def counterbalance_order(task_id: str, step_id: int) -> list[str]:
    """Deterministically permute fork execution order (A, B, C, D, E)."""
    h = hashlib.sha256(f"{task_id}:{step_id}".encode()).hexdigest()
    # 5! = 120 permutations
    perm_idx = int(h[:8], 16) % 120
    perms = list(itertools.permutations(["A", "B", "C", "D", "E"]))
    return list(perms[perm_idx])


def classify_stop_answer_hazard(a_base: DecisionAction, a_gov: DecisionAction) -> str:
    """Classify STOP->ANSWER as a known hazard."""
    if a_base == DecisionAction.STOP and a_gov == DecisionAction.ANSWER:
        return "UNSAFE_OVERRIDE_STOP_TO_ANSWER"
    return "SAFE"


def continue_with_model(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    t_runtime: TaskRuntime,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    utility: MetareasoningUtility,
    general_governor: GeneralGovernor,
    exec_governor: ExecutionGovernor,
    res_governor: ResolutionGovernor,
    mode: str,  # "BASE", "GENERIC_ASSIST", "RESOLUTION_ASSIST"
    fork_label: str,
    api_key: str,
    start_step: int = 0,
    max_steps: int = 24,
) -> dict[str, Any]:
    """Continue a trajectory from the given state.

    mode="BASE": base packet for all continuation steps
    mode="GENERIC_ASSIST": old I3.6a/b execution-assist packet for all steps
    mode="RESOLUTION_ASSIST": new I3.6d resolution scaffold for all steps
    """
    executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    backend = DeepSeekBackend()

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    terminal_action = None
    backend_errors = 0

    continuation_actions: list[str] = []
    continuation_outcomes: list[str] = []
    assistance_types: list[str] = []
    assistance_frame_shas: list[str] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    # Resolution context (persistent for RESOLUTION_ASSIST)
    res_context: ResolutionContext | None = None

    for step_id in range(start_step, max_steps):
        observation = build_observation(
            t_runtime, task, cond,
            tuple(prior_decisions), tuple(prior_outcomes))

        prior_action_strs = tuple(
            d.selected_action if isinstance(d.selected_action, str)
            else d.selected_action.value for d in prior_decisions)

        assist_type = None
        assist_sha = None

        if mode == "BASE":
            packet = build_base_packet(observation)
            assert_no_evaluator_leakage(packet)
        elif mode == "GENERIC_ASSIST":
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
                assist_sha = assistance_frame_sha256(assist_frame) if hasattr(assist_frame, 'as_dict') else None
            else:
                packet = build_governor_packet(observation, gov_frame)
                assert_no_evaluator_leakage(packet)
        elif mode == "RESOLUTION_ASSIST":
            gov_frame = general_governor.assess(
                observation=observation,
                remaining_steps=max_steps - step_id,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )
            res_frame = res_governor.plan(
                observation=observation,
                remaining_steps=max_steps - step_id,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )
            if res_frame is not None:
                # Initialize or update context
                if res_context is None:
                    res_context = res_governor.init_context(
                        task_id=task.task_id,
                        observation=observation,
                        remaining_steps=max_steps - step_id,
                        prior_actions=prior_action_strs,
                        prior_outcomes=tuple(prior_outcomes),
                    )
                packet = serialize_resolution_packet(
                    observation, gov_frame, res_frame, context=res_context,
                    mode="RESOLUTION_ASSIST")
                assert_resolution_no_leakage(packet)
                assist_type = f"resolution_{res_frame.recommended_action}"
                assist_sha = resolution_frame_sha256(res_frame)
            else:
                packet = build_governor_packet(observation, gov_frame)
                assert_no_evaluator_leakage(packet)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        user_prompt = packet_json(packet) if mode in ("BASE", "GENERIC_ASSIST") else resolution_packet_json(packet)

        backend.task_id = task.task_id
        backend.condition = f"i3_6d_{mode}"
        backend.pair_id = f"i3_6d:{task.task_id}:{fork_label}:step{step_id}"

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

        action_str = action.value if hasattr(action, "value") else str(action)
        continuation_actions.append(action_str)
        continuation_outcomes.append(execution.outcome_code)
        assistance_types.append(assist_type)
        assistance_frame_shas.append(assist_sha)

        # Update resolution context if applicable
        if mode == "RESOLUTION_ASSIST" and res_context is not None:
            new_evidence_found = action_str in ("RETRIEVE", "SEARCH_MORE")
            evidence_verified = action_str == "VERIFY"
            res_context, _ = res_governor.update_context(
                context=res_context,
                action_taken=action_str,
                new_observation=observation,
                new_evidence_found=new_evidence_found,
                evidence_verified=evidence_verified,
            )

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
    res_governor: ResolutionGovernor,
    api_key: str,
    continuation_mode: str,
) -> dict[str, Any]:
    """Execute one fork: force the first action, then continue."""
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    fork_i3 = copy.deepcopy(i3_runtime_state)
    fork_t = copy.deepcopy(t_runtime_state)
    fork_pd = list(prior_decisions)
    fork_po = list(prior_outcomes)

    exec_res = oracle_executor.execute(fork_i3, forced_action)
    exec_t = task_executor.execute(fork_t, forced_action)

    resources_before = t_runtime_state.resources
    resources_after = exec_t.runtime.resources
    forced_cost = utility.action_cost(resources_before, resources_after)

    fork_pd.append(DecisionSummary(
        f"{task.task_id}:fork{fork_id}:{step_idx}",
        forced_action.value, "FORCED", exec_res.outcome_code))
    fork_po.append(exec_res.outcome_code)

    if exec_res.terminal:
        tr = utility.terminal_reward(forced_action, bool(exec_res.task_success))
        total_utility = round(-forced_cost + tr, 4)
        return {
            "fork_id": fork_id,
            "forced_action": forced_action.value,
            "forced_action_cost": round(forced_cost, 4),
            "continuation_utility": round(tr, 4),
            "total_utility": total_utility,
            "success": bool(exec_res.task_success),
            "steps": 0, "model_calls": 0, "backend_errors": 0,
            "terminal_result": exec_res.outcome_code,
            "terminal_action": forced_action.value,
            "terminal_reward": round(tr, 4),
            "total_action_cost": round(forced_cost, 4),
            "continuation_actions": [], "continuation_outcomes": [],
            "assistance_types": [], "assistance_frame_sha256s": [],
            "step_costs": [],
        }

    cont = continue_with_model(
        task=task, budget=budget, t_runtime=exec_t.runtime,
        prior_decisions=fork_pd, prior_outcomes=fork_po,
        utility=utility,
        general_governor=general_governor,
        exec_governor=exec_governor,
        res_governor=res_governor,
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
    res_governor: ResolutionGovernor,
    api_key: str,
) -> dict[str, Any]:
    """Process one five-way fork set: A, B, C, D, E.

    A = a_B + pi_B              (baseline)
    B = a_G + pi_B              (action-only)
    C = a_G + generic assist    (old scaffold)
    D = a_G + resolution assist (new scaffold)
    E = a_B + resolution assist (baseline action + resolution context)
    """
    fork_order = counterbalance_order(task.task_id, step_idx)
    hazard = classify_stop_answer_hazard(a_base, a_gov)

    fork_configs = {
        "A": {"forced_action": a_base, "continuation_mode": "BASE"},
        "B": {"forced_action": a_gov, "continuation_mode": "BASE"},
        "C": {"forced_action": a_gov, "continuation_mode": "GENERIC_ASSIST"},
        "D": {"forced_action": a_gov, "continuation_mode": "RESOLUTION_ASSIST"},
        "E": {"forced_action": a_base, "continuation_mode": "RESOLUTION_ASSIST"},
    }

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
            res_governor=res_governor,
            api_key=api_key,
            continuation_mode=cfg["continuation_mode"],
        )

    u_a = results["A"]["total_utility"]
    u_b = results["B"]["total_utility"]
    u_c = results["C"]["total_utility"]
    u_d = results["D"]["total_utility"]
    u_e = results["E"]["total_utility"]

    # Key contrasts
    a_b = round(u_b - u_a, 4)              # action-only advantage
    resolution_gain = round(u_d - u_c, 4)   # resolution vs generic scaffold
    context_gain = round(u_e - u_d, 4)      # context without action override
    action_override_gain = round(u_d - u_e, 4)  # value of governor action
    resolution_vs_base = round(u_d - u_a, 4)    # resolution vs baseline
    context_vs_base = round(u_e - u_a, 4)       # context-only vs baseline

    def classify(base_ok: bool, treat_ok: bool) -> str:
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    base_ok = results["A"]["success"]
    return {
        "task_id": task.task_id,
        "step_id": step_idx,
        "base_action": a_base.value,
        "gov_action": a_gov.value,
        "hazard_class": hazard,
        "fork_order": fork_order,
        # Utilities
        "u_a": u_a, "u_b": u_b, "u_c": u_c, "u_d": u_d, "u_e": u_e,
        # Key contrasts
        "a_b": a_b,
        "resolution_gain": resolution_gain,
        "context_gain": context_gain,
        "action_override_gain": action_override_gain,
        "resolution_vs_base": resolution_vs_base,
        "context_vs_base": context_vs_base,
        # Success
        "base_success": base_ok,
        "action_success": results["B"]["success"],
        "generic_success": results["C"]["success"],
        "resolution_success": results["D"]["success"],
        "context_success": results["E"]["success"],
        # Classification
        "action_class": classify(base_ok, results["B"]["success"]),
        "generic_class": classify(base_ok, results["C"]["success"]),
        "resolution_class": classify(base_ok, results["D"]["success"]),
        "context_class": classify(base_ok, results["E"]["success"]),
        # Fork details
        "fork_a": results["A"], "fork_b": results["B"],
        "fork_c": results["C"], "fork_d": results["D"],
        "fork_e": results["E"],
    }


def main():
    parser = argparse.ArgumentParser(description="I3.6d-a five-way forks")
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
        default="experiments/v2b_i3_6/development/i3_6d_a",
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
    print(f"Loaded {len(blocks)} task blocks")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)
    general_governor = GeneralGovernor()
    exec_governor = ExecutionGovernor()
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

    # Count hazards
    hazards = Counter(
        classify_stop_answer_hazard(fs["a_base"], fs["a_gov"])
        for fs in fork_states
    )
    print(f"  Hazard classification: {dict(hazards)}")

    if args.max_forks:
        fork_states = fork_states[:args.max_forks]
        print(f"Limited to {len(fork_states)} forks")

    print(f"\nProcessing {len(fork_states)} five-way forks with {args.workers} workers...")
    print(f"  A=a_B+pi_B, B=a_G+pi_B, C=a_G+generic, D=a_G+resolution, E=a_B+resolution")

    all_results: list[dict[str, Any]] = []
    completed = 0

    def fork_wrapper(fs):
        return process_one_fork_set(
            task=fs["task"], budget=fs["budget"], step_idx=fs["step_idx"],
            a_base=fs["a_base"], a_gov=fs["a_gov"],
            i3_runtime_state=fs["i3_runtime"], t_runtime_state=fs["t_runtime"],
            prior_decisions=fs["prior_decisions"], prior_outcomes=fs["prior_outcomes"],
            utility=utility, general_governor=general_governor,
            exec_governor=exec_governor, res_governor=res_governor,
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

    results_path = output_dir / "five_way_forks_v1.jsonl"
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
    u_e_mean = sum(r["u_e"] for r in all_results) / n

    rg_mean = sum(r["resolution_gain"] for r in all_results) / n
    cg_mean = sum(r["context_gain"] for r in all_results) / n
    aog_mean = sum(r["action_override_gain"] for r in all_results) / n
    rvb_mean = sum(r["resolution_vs_base"] for r in all_results) / n
    cvb_mean = sum(r["context_vs_base"] for r in all_results) / n

    base_success = sum(1 for r in all_results if r["base_success"])
    action_success = sum(1 for r in all_results if r["action_success"])
    generic_success = sum(1 for r in all_results if r["generic_success"])
    resolution_success = sum(1 for r in all_results if r["resolution_success"])
    context_success = sum(1 for r in all_results if r["context_success"])

    action_classes = Counter(r["action_class"] for r in all_results)
    generic_classes = Counter(r["generic_class"] for r in all_results)
    resolution_classes = Counter(r["resolution_class"] for r in all_results)
    context_classes = Counter(r["context_class"] for r in all_results)

    summary = {
        "schema": "DAPH_V2B_I3_6D_A_FIVE_WAY_FORKS_V1",
        "resolution_identity_sha256": identity["resolution_identity_sha256"],
        "n_forks": n,
        "n_tasks": n_tasks_processed,
        "fork_modes": {
            "A": "a_B + pi_B (baseline)",
            "B": "a_G + pi_B (action-only)",
            "C": "a_G + generic execution assist",
            "D": "a_G + resolution assist",
            "E": "a_B + resolution assist (context only)",
        },
        "utility": {
            "mean_u_a_base": round(u_a_mean, 4),
            "mean_u_b_action": round(u_b_mean, 4),
            "mean_u_c_generic": round(u_c_mean, 4),
            "mean_u_d_resolution": round(u_d_mean, 4),
            "mean_u_e_context": round(u_e_mean, 4),
        },
        "contrasts": {
            "mean_resolution_gain": round(rg_mean, 4),
            "mean_context_gain": round(cg_mean, 4),
            "mean_action_override_gain": round(aog_mean, 4),
            "mean_resolution_vs_base": round(rvb_mean, 4),
            "mean_context_vs_base": round(cvb_mean, 4),
        },
        "success": {
            "base": f"{base_success}/{n}",
            "action_only": f"{action_success}/{n}",
            "generic": f"{generic_success}/{n}",
            "resolution": f"{resolution_success}/{n}",
            "context": f"{context_success}/{n}",
        },
        "classification": {
            "base_vs_action": dict(action_classes),
            "base_vs_generic": dict(generic_classes),
            "base_vs_resolution": dict(resolution_classes),
            "base_vs_context": dict(context_classes),
        },
    }

    summary_path = output_dir / "five_way_forks_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*78}")
    print("I3.6d-a FIVE-WAY FORKS SUMMARY")
    print(f"{'='*78}")
    print(f"  Fork sets:                   {n}")
    print(f"  Tasks:                       {n_tasks_processed}")
    print(f"\n  Mean utility (with forced-action cost):")
    print(f"    A: a_B + pi_B:              {u_a_mean:+.4f}")
    print(f"    B: a_G + pi_B:              {u_b_mean:+.4f}")
    print(f"    C: a_G + generic:           {u_c_mean:+.4f}")
    print(f"    D: a_G + resolution:        {u_d_mean:+.4f}")
    print(f"    E: a_B + resolution:        {u_e_mean:+.4f}")
    print(f"\n  Key contrasts:")
    print(f"    ResolutionGain (D-C):       {rg_mean:+.4f}")
    print(f"    ContextGain (E-D):          {cg_mean:+.4f}")
    print(f"    ActionOverrideGain (D-E):   {aog_mean:+.4f}")
    print(f"    ResolutionVsBase (D-A):     {rvb_mean:+.4f}")
    print(f"    ContextVsBase (E-A):        {cvb_mean:+.4f}")
    print(f"\n  Success rates:")
    print(f"    BASE:       {base_success}/{n} ({base_success/n:.1%})")
    print(f"    ACTION_ONLY: {action_success}/{n} ({action_success/n:.1%})")
    print(f"    GENERIC:    {generic_success}/{n} ({generic_success/n:.1%})")
    print(f"    RESOLUTION: {resolution_success}/{n} ({resolution_success/n:.1%})")
    print(f"    CONTEXT:    {context_success}/{n} ({context_success/n:.1%})")
    print(f"\n  Classification (BASE vs RESOLUTION):")
    for cls, cnt in resolution_classes.most_common():
        print(f"    {cls}: {cnt}")
    print(f"\n  Classification (BASE vs CONTEXT):")
    for cls, cnt in context_classes.most_common():
        print(f"    {cls}: {cnt}")


if __name__ == "__main__":
    main()
