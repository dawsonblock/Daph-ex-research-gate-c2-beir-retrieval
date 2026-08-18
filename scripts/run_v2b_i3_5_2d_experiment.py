#!/usr/bin/env python3
"""Run the I3.5.2d Policy-Conditional Intervention Value Experiment.

For every intervention state from the I3.5.2c SELECTIVE_FRAME run:
  1. Replay baseline trajectory up to state s
  2. Fork A: execute baseline action a_B, continue with OFF model → terminal
  3. Fork B: execute governor action a_G, continue with OFF model → terminal
  4. Fork C: execute governor action a_G, continue with SELECTIVE model → terminal
  5. Record realized utilities for all three forks

Three intervention-value quantities:
  A*       = Q*(s, a_G) - Q*(s, a_B)           [oracle, from I3.5.2a]
  A^{π_B}  = U(a_G + OFF continuation) - U(a_B + OFF continuation)
  A^{π_G}  = U(a_G + SEL continuation) - U(a_B + OFF continuation)

Rescueability classification using oracle state graph:
  UNRESCUABLE: V*(s) ≤ failure_value (no path to success)
  RESCUABLE_BASE_FAILS: V*(s) > failure but base model fails
  RESCUABLE_GOV_OPENS_PATH: a_G leads to state with higher V*
  RESCUABLE_GOV_CLOSES_PATH: a_G leads to state with lower V*

Chain macro-pattern analysis from I3.5.2c data.

Usage:
    DEEPSEEK_API_KEY=... python scripts/run_v2b_i3_5_2d_experiment.py \\
        --i352c-results experiments/v2b_i3_5_2/development/i352c_55f93130e87c/results.json \\
        --workers 8
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from hrm_adaptive_memory.executive.actions import ActionProposal
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, TaskRuntime, initial_runtime,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.governor.serializer import frame_sha256
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
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    OraclePolicyTable, build_oracle_policy_table,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState, ResourceExhausted
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.selective_governor import (
    SelectiveGovernorGate, RuleBasedInterventionPredictor,
)
from hrm_adaptive_memory.executive.i3_5_2.modes import GovernorMode

# Fallback penalties (same as shadow dataset)
FALLBACK_PENALTIES: dict[DecisionAction, float] = {
    DecisionAction.ANSWER: -125.11,
    DecisionAction.DEFER: -30.11,
    DecisionAction.STOP: -30.11,
}
DEFAULT_FALLBACK = -125.0
VALID_ACTIONS = tuple(
    a for a in DecisionAction
    if a.value in ("ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP")
)


def get_q_with_source(table: OraclePolicyTable, state_id: str, action: DecisionAction) -> tuple[float, str]:
    q = table.q_values.get((state_id, action))
    if q is not None:
        return q, "oracle_q_values"
    pq = table.proposal_q_values.get((state_id, action))
    if pq is not None:
        return pq, "proposal_q_values"
    return FALLBACK_PENALTIES.get(action, DEFAULT_FALLBACK), "fallback_penalty"


def classify_rescueability(
    table: OraclePolicyTable,
    state_id: str,
    a_base: DecisionAction,
    a_gov: DecisionAction,
    base_success: bool,
) -> dict[str, Any]:
    """Classify rescueability using the oracle state graph."""
    v_star = table.state_values.get(state_id, float("-inf"))
    min_cost = table.minimum_remaining_cost.get(state_id, float("inf"))
    failure_value = min(
        table.state_values.get(s, float("-inf"))
        for s in table.states
    ) if table.states else float("-inf")

    # Check if state is rescuable (V* > failure threshold)
    # A state is rescuable if there exists a path to success
    is_rescuable = v_star > failure_value and min_cost < float("inf")

    # Check what happens after base action
    base_transition = table.transitions.get((state_id, a_base))
    gov_transition = table.transitions.get((state_id, a_gov))

    base_next_v = None
    gov_next_v = None
    if base_transition and base_transition.next_state_id:
        base_next_v = table.state_values.get(base_transition.next_state_id, None)
    if gov_transition and gov_transition.next_state_id:
        gov_next_v = table.state_values.get(gov_transition.next_state_id, None)

    base_terminal_success = base_transition.task_success if base_transition and base_transition.terminal else None
    gov_terminal_success = gov_transition.task_success if gov_transition and gov_transition.terminal else None

    # Classify
    if not is_rescuable:
        category = "UNRESCUABLE"
    elif base_success:
        category = "RESCUABLE_BASE_SUCCEEDS"
    elif gov_next_v is not None and base_next_v is not None and gov_next_v > base_next_v:
        category = "RESCUABLE_GOV_OPENS_PATH"
    elif gov_next_v is not None and base_next_v is not None and gov_next_v < base_next_v:
        category = "RESCUABLE_GOV_CLOSES_PATH"
    else:
        category = "RESCUABLE_AMBIGUOUS"

    return {
        "category": category,
        "v_star": round(v_star, 4),
        "min_remaining_cost": round(min_cost, 4) if min_cost != float("inf") else None,
        "base_next_v_star": round(base_next_v, 4) if base_next_v is not None else None,
        "gov_next_v_star": round(gov_next_v, 4) if gov_next_v is not None else None,
        "base_terminal_success": base_terminal_success,
        "gov_terminal_success": gov_terminal_success,
        "is_rescuable": is_rescuable,
    }


def continue_trajectory_with_model(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    t_runtime: TaskRuntime,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    governor_mode: GovernorMode,
    backend: DeepSeekBackend,
    governor: GeneralGovernor,
    gate: SelectiveGovernorGate,
    utility: MetareasoningUtility,
    max_steps: int = 24,
    start_step: int = 0,
) -> dict[str, Any]:
    """Continue a trajectory from the given state using the specified model mode.

    Returns realized utility, success, steps, and model calls.
    """
    executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
    resources = t_runtime.resources
    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"

    for step_id in range(start_step, max_steps):
        observation = build_observation(
            t_runtime, task, cond,
            tuple(prior_decisions), tuple(prior_outcomes))

        prior_action_strs = tuple(
            d.selected_action if isinstance(d.selected_action, str)
            else d.selected_action.value for d in prior_decisions)

        # Route by mode
        if governor_mode == GovernorMode.OFF:
            packet = build_base_packet(observation)
        elif governor_mode == GovernorMode.SELECTIVE_FRAME:
            gate_decision = gate.assess(
                observation=observation,
                remaining_steps=max_steps - step_id,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )
            if gate_decision.intervene:
                frame = governor.assess(
                    observation=observation,
                    remaining_steps=max_steps - step_id,
                    prior_actions=prior_action_strs,
                    prior_outcomes=tuple(prior_outcomes),
                )
                packet = build_governor_packet(observation, frame)
            else:
                packet = build_base_packet(observation)
        else:
            packet = build_base_packet(observation)

        assert_no_evaluator_leakage(packet)
        user_prompt = packet_json(packet)

        backend.task_id = task.task_id
        backend.condition = f"i352d_{governor_mode.value}"
        backend.pair_id = f"i352d:{task.task_id}:step{start_step}:{governor_mode.value}"

        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
        except Exception:
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
            f"{task.task_id}:cont:{step_id}", action.value,
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
        "terminal_result": terminal_result,
    }


def process_one_intervention(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    table: OraclePolicyTable,
    step_idx: int,
    a_base: DecisionAction,
    a_gov: DecisionAction,
    base_steps_data: list[dict],
    i3_runtime_state: Any,
    t_runtime_state: Any,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    """Process one intervention state: fork into base/gov continuations."""

    state_id = table.state_id_for(i3_runtime_state)

    # Oracle quantities
    q_base, q_base_source = get_q_with_source(table, state_id, a_base)
    q_gov, q_gov_source = get_q_with_source(table, state_id, a_gov)
    a_star = round(q_gov - q_base, 4)

    # Rescueability
    rescue = classify_rescueability(table, state_id, a_base, a_gov, False)

    # Execute the fork action deterministically (no model call needed for the fork point)
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Fork A: execute a_base, then continue with OFF model
    fork_a_i3 = copy.deepcopy(i3_runtime_state)
    fork_a_t = copy.deepcopy(t_runtime_state)
    fork_a_pd = list(prior_decisions)
    fork_a_po = list(prior_outcomes)

    exec_a = oracle_executor.execute(fork_a_i3, a_base)
    exec_a_t = task_executor.execute(fork_a_t, a_base)
    fork_a_pd.append(DecisionSummary(
        f"{task.task_id}:fork_a:{step_idx}", a_base.value,
        "FORK", exec_a.outcome_code))
    fork_a_po.append(exec_a.outcome_code)

    backend = DeepSeekBackend()
    governor = GeneralGovernor()
    gate = SelectiveGovernorGate(predictor=RuleBasedInterventionPredictor())

    if exec_a.terminal:
        cont_a = {
            "realized_utility": utility.terminal_reward(a_base, bool(exec_a.task_success)),
            "success": bool(exec_a.task_success),
            "steps": 0,
            "model_calls": 0,
            "terminal_result": exec_a.outcome_code,
        }
    else:
        cont_a = continue_trajectory_with_model(
            task=task, budget=budget, t_runtime=exec_a_t.runtime,
            prior_decisions=fork_a_pd, prior_outcomes=fork_a_po,
            governor_mode=GovernorMode.OFF,
            backend=backend, governor=governor, gate=gate, utility=utility,
            start_step=step_idx + 1,
        )

    # Fork B: execute a_gov, then continue with OFF model
    fork_b_i3 = copy.deepcopy(i3_runtime_state)
    fork_b_t = copy.deepcopy(t_runtime_state)
    fork_b_pd = list(prior_decisions)
    fork_b_po = list(prior_outcomes)

    exec_b = oracle_executor.execute(fork_b_i3, a_gov)
    exec_b_t = task_executor.execute(fork_b_t, a_gov)
    fork_b_pd.append(DecisionSummary(
        f"{task.task_id}:fork_b:{step_idx}", a_gov.value,
        "FORK", exec_b.outcome_code))
    fork_b_po.append(exec_b.outcome_code)

    if exec_b.terminal:
        cont_b = {
            "realized_utility": utility.terminal_reward(a_gov, bool(exec_b.task_success)),
            "success": bool(exec_b.task_success),
            "steps": 0,
            "model_calls": 0,
            "terminal_result": exec_b.outcome_code,
        }
    else:
        cont_b = continue_trajectory_with_model(
            task=task, budget=budget, t_runtime=exec_b_t.runtime,
            prior_decisions=fork_b_pd, prior_outcomes=fork_b_po,
            governor_mode=GovernorMode.OFF,
            backend=backend, governor=governor, gate=gate, utility=utility,
            start_step=step_idx + 1,
        )

    # Fork C: execute a_gov, then continue with SELECTIVE_FRAME model
    fork_c_i3 = copy.deepcopy(i3_runtime_state)
    fork_c_t = copy.deepcopy(t_runtime_state)
    fork_c_pd = list(prior_decisions)
    fork_c_po = list(prior_outcomes)

    # Re-execute a_gov for fork C (same deterministic result)
    exec_c = oracle_executor.execute(fork_c_i3, a_gov)
    exec_c_t = task_executor.execute(fork_c_t, a_gov)
    fork_c_pd.append(DecisionSummary(
        f"{task.task_id}:fork_c:{step_idx}", a_gov.value,
        "FORK", exec_c.outcome_code))
    fork_c_po.append(exec_c.outcome_code)

    if exec_c.terminal:
        cont_c = {
            "realized_utility": utility.terminal_reward(a_gov, bool(exec_c.task_success)),
            "success": bool(exec_c.task_success),
            "steps": 0,
            "model_calls": 0,
            "terminal_result": exec_c.outcome_code,
        }
    else:
        cont_c = continue_trajectory_with_model(
            task=task, budget=budget, t_runtime=exec_c_t.runtime,
            prior_decisions=fork_c_pd, prior_outcomes=fork_c_po,
            governor_mode=GovernorMode.SELECTIVE_FRAME,
            backend=backend, governor=governor, gate=gate, utility=utility,
            start_step=step_idx + 1,
        )

    # Compute intervention values
    a_pi_base = round(cont_b["realized_utility"] - cont_a["realized_utility"], 4)
    a_pi_gov = round(cont_c["realized_utility"] - cont_a["realized_utility"], 4)

    return {
        "task_id": task.task_id,
        "step_id": step_idx,
        "state_id": state_id,
        "base_action": a_base.value,
        "gov_action": a_gov.value,
        "same_action": a_base.value == a_gov.value,
        # Oracle quantities
        "q_star_base": round(q_base, 4),
        "q_star_gov": round(q_gov, 4),
        "a_star": a_star,
        "q_base_source": q_base_source,
        "q_gov_source": q_gov_source,
        # Policy-conditional quantities
        "u_base_continuation": cont_a["realized_utility"],
        "u_gov_off_continuation": cont_b["realized_utility"],
        "u_gov_sel_continuation": cont_c["realized_utility"],
        "a_pi_base": a_pi_base,
        "a_pi_gov": a_pi_gov,
        # Fork outcomes
        "base_success": cont_a["success"],
        "gov_off_success": cont_b["success"],
        "gov_sel_success": cont_c["success"],
        "base_steps": cont_a["steps"],
        "gov_off_steps": cont_b["steps"],
        "gov_sel_steps": cont_c["steps"],
        "base_model_calls": cont_a["model_calls"],
        "gov_off_model_calls": cont_b["model_calls"],
        "gov_sel_model_calls": cont_c["model_calls"],
        # Rescueability
        "rescueability": rescue,
    }


def analyze_chain_patterns(i352c_results_path: str) -> dict[str, Any]:
    """Analyze intervention chain macro-patterns from I3.5.2c data."""
    data = json.loads(Path(i352c_results_path).read_text())
    blocks = data["results"]

    chain_records = []
    action_sequences = Counter()
    chain_utilities = defaultdict(list)

    for block in blocks:
        trajs = block["trajectories"]
        sel = trajs["SELECTIVE_FRAME"]
        steps = sel["steps"]
        interventions = sel["interventions"]

        if not interventions:
            continue

        # Build intervention chain
        chain_actions = []
        chain_steps = []
        for iv in interventions:
            chain_actions.append(iv["model_action"])
            chain_steps.append(iv["step_id"])

        # Record the action sequence pattern
        pattern = " → ".join(chain_actions)
        action_sequences[pattern] += 1

        # Compute chain-level utility
        off_util = trajs["OFF"]["realized_utility"]
        sel_util = sel["realized_utility"]
        chain_delta = sel_util - off_util
        chain_utilities[len(chain_actions)].append(chain_delta)

        chain_records.append({
            "task_id": block["task_id"],
            "chain_length": len(chain_actions),
            "chain_actions": chain_actions,
            "chain_steps": chain_steps,
            "off_utility": off_util,
            "sel_utility": sel_util,
            "delta_utility": round(chain_delta, 4),
            "off_success": trajs["OFF"]["task_success"],
            "sel_success": sel["task_success"],
        })

    # Chain length statistics
    chain_length_stats = {}
    for length, utils in chain_utilities.items():
        chain_length_stats[length] = {
            "count": len(utils),
            "mean_delta_u": round(sum(utils) / len(utils), 4),
            "min_delta_u": round(min(utils), 4),
            "max_delta_u": round(max(utils), 4),
        }

    return {
        "total_chains": len(chain_records),
        "action_sequence_distribution": dict(action_sequences.most_common(20)),
        "chain_length_stats": chain_length_stats,
        "chain_records": chain_records[:50],  # Sample for inspection
    }


def main():
    parser = argparse.ArgumentParser(description="Run I3.5.2d Policy-Conditional Intervention Value")
    parser.add_argument("--split", default="structure_dev_v2")
    parser.add_argument(
        "--i352c-results",
        default="experiments/v2b_i3_5_2/development/i352c_55f93130e87c/results.json",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-interventions", type=int, default=None,
                        help="Limit total interventions (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Loading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    task_map = {t.task_id: t for t in split_bm.tasks}

    print(f"Loading I3.5.2c results from {args.i352c_results}...")
    i352c_data = json.loads(Path(args.i352c_results).read_text())
    i352c_blocks = i352c_data["results"]
    print(f"Loaded {len(i352c_blocks)} task blocks")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)
    policy = load_frozen_policy(args.policy)
    governor = GeneralGovernor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    # First: analyze chain patterns (no API calls needed)
    print("\nAnalyzing intervention chain macro-patterns from I3.5.2c...")
    chain_analysis = analyze_chain_patterns(args.i352c_results)
    print(f"  Total chains: {chain_analysis['total_chains']}")
    print(f"  Top action sequences:")
    for seq, cnt in list(chain_analysis["action_sequence_distribution"].items())[:5]:
        print(f"    {seq}: {cnt}")
    print(f"  Chain length stats:")
    for length, stats in chain_analysis["chain_length_stats"].items():
        print(f"    Length {length}: n={stats['count']}, mean ΔU={stats['mean_delta_u']}")

    # Collect intervention states from I3.5.2c
    print("\nCollecting intervention states from I3.5.2c SELECTIVE_FRAME...")
    intervention_states = []
    for block in i352c_blocks:
        task_id = block["task_id"]
        sel = block["trajectories"]["SELECTIVE_FRAME"]
        interventions = sel["interventions"]
        off_steps = block["trajectories"]["OFF"]["steps"]

        for iv in interventions:
            intervention_states.append({
                "task_id": task_id,
                "step_id": iv["step_id"],
                "gov_action": iv["governor_top_action"],
                "model_action": iv["model_action"],
                "gate_reason": iv["gate_reason"],
                # We need the baseline action at this step
                # The OFF trajectory's step at this step_id gives us a_base
            })

    # We need to replay baseline trajectories to get the baseline action at each intervention step
    # and the runtime state at that point
    print(f"\nTotal intervention states: {len(intervention_states)}")

    if args.max_interventions:
        intervention_states = intervention_states[:args.max_interventions]
        print(f"Limited to {len(intervention_states)} for testing")

    # Replay baseline trajectories to get state at each intervention point
    print("\nReplaying baseline trajectories to extract intervention states...")

    # Build a map: task_id -> list of (step_id, i3_runtime, t_runtime, prior_decisions, prior_outcomes, a_base)
    task_intervention_data: dict[str, list[dict]] = defaultdict(list)

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    for block in i352c_blocks:
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        # Only process tasks that have interventions
        sel = block["trajectories"]["SELECTIVE_FRAME"]
        if not sel["interventions"]:
            continue

        off_traj = block["trajectories"]["OFF"]
        off_steps = off_traj["steps"]

        # Build oracle table for this task
        table = build_oracle_policy_table(task=task, policy=policy, utility=utility, budget=budget)

        # Replay OFF trajectory
        resources = ResourceState(budget)
        i3_runtime = initial_i3_runtime(task, resources)
        adapter = _I3TaskAdapter(task)
        t_runtime = initial_runtime(adapter, ResourceState(budget))

        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        intervention_steps = {iv["step_id"] for iv in sel["interventions"]}
        intervention_map = {iv["step_id"]: iv for iv in sel["interventions"]}

        for step_idx, step_data in enumerate(off_steps):
            if step_idx in intervention_steps:
                iv = intervention_map[step_idx]
                state_id = table.state_id_for(i3_runtime)
                a_base = DecisionAction(step_data["executed_action"])
                a_gov = DecisionAction(iv["governor_top_action"])

                task_intervention_data[task_id].append({
                    "step_id": step_idx,
                    "state_id": state_id,
                    "a_base": a_base,
                    "a_gov": a_gov,
                    "i3_runtime": copy.deepcopy(i3_runtime),
                    "t_runtime": copy.deepcopy(t_runtime),
                    "prior_decisions": list(prior_decisions),
                    "prior_outcomes": list(prior_outcomes),
                    "table": table,
                    "task": task,
                    "budget": budget,
                    "gate_reason": iv["gate_reason"],
                })

            # Step the baseline
            a_step = DecisionAction(step_data["executed_action"])
            exec_res = oracle_executor.execute(i3_runtime, a_step)
            i3_runtime = exec_res.runtime
            t_runtime = task_executor.execute(t_runtime, a_step).runtime

            prior_decisions.append(DecisionSummary(
                f"{task_id}:step:{step_idx}", a_step.value,
                step_data["reason_code"], exec_res.outcome_code))
            prior_outcomes.append(exec_res.outcome_code)

            if exec_res.terminal:
                break

    # Flatten intervention data
    all_interventions = []
    for task_id, ivs in task_intervention_data.items():
        all_interventions.extend(ivs)

    print(f"Replayed {len(all_interventions)} intervention states across {len(task_intervention_data)} tasks")

    # Run the three forks for each intervention
    print(f"\nRunning policy-conditional continuations ({args.workers} workers)...")
    print(f"Each intervention: 2-3 model continuations (fork A=OFF, fork B=OFF, fork C=SEL)")

    def process_intervention(item):
        idx, iv_data = item
        result = process_one_intervention(
            task=iv_data["task"],
            budget=iv_data["budget"],
            table=iv_data["table"],
            step_idx=iv_data["step_id"],
            a_base=iv_data["a_base"],
            a_gov=iv_data["a_gov"],
            base_steps_data=[],
            i3_runtime_state=iv_data["i3_runtime"],
            t_runtime_state=iv_data["t_runtime"],
            prior_decisions=iv_data["prior_decisions"],
            prior_outcomes=iv_data["prior_outcomes"],
            utility=utility,
            api_key=api_key,
        )
        result["gate_reason"] = iv_data["gate_reason"]
        return idx, result

    results = []
    completed = 0
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_intervention, (i, iv)): i
            for i, iv in enumerate(all_interventions)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results.append((idx, result))
            completed += 1
            if completed % args.progress_every == 0:
                elapsed = time.monotonic() - t_start
                rate = completed / elapsed
                eta = (len(all_interventions) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(all_interventions)}] "
                      f"A*={result['a_star']:+.1f} "
                      f"A_πB={result['a_pi_base']:+.1f} "
                      f"A_πG={result['a_pi_gov']:+.1f} "
                      f"rescue={result['rescueability']['category']} "
                      f"eta={eta:.0f}s")

    results.sort(key=lambda x: x[0])
    intervention_results = [r[1] for r in results]

    elapsed = time.monotonic() - t_start
    print(f"\nCompleted {completed} interventions in {elapsed:.0f}s")

    # Save results
    output_dir = Path(args.output_dir) / "i352d"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "intervention_values_v1.jsonl"
    with open(results_path, "w") as f:
        for r in intervention_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 78)
    print("V2B-I3.5.2d POLICY-CONDITIONAL INTERVENTION VALUE ANALYSIS")
    print("=" * 78)

    n = len(intervention_results)

    # Filter out same-action interventions (a_base == a_gov)
    diff_results = [r for r in intervention_results if not r["same_action"]]
    n_diff = len(diff_results)
    print(f"\nTotal interventions: {n}")
    print(f"Different-action interventions (a_base ≠ a_gov): {n_diff}")

    # A* statistics
    a_stars = [r["a_star"] for r in diff_results]
    a_pi_bs = [r["a_pi_base"] for r in diff_results]
    a_pi_gs = [r["a_pi_gov"] for r in diff_results]

    mean_a_star = sum(a_stars) / n_diff if n_diff else 0
    mean_a_pi_b = sum(a_pi_bs) / n_diff if n_diff else 0
    mean_a_pi_g = sum(a_pi_gs) / n_diff if n_diff else 0

    print(f"\n--- Intervention Value Decomposition (N={n_diff}) ---")
    print(f"  A*       (oracle):       mean={mean_a_star:+.2f}  (Q* advantage)")
    print(f"  A^{{π_B}}  (base cont.):   mean={mean_a_pi_b:+.2f}  (gov action + OFF continuation)")
    print(f"  A^{{π_G}}  (gov cont.):    mean={mean_a_pi_g:+.2f}  (gov action + SEL continuation)")

    # Correlation between A* and A^{π_B}
    if n_diff > 5:
        from statistics import correlation
        try:
            corr_star_pib = correlation(a_stars, a_pi_bs)
        except Exception:
            corr_star_pib = 0.0
        try:
            corr_star_pig = correlation(a_stars, a_pi_gs)
        except Exception:
            corr_star_pig = 0.0
        print(f"\n  Corr(A*, A^{{π_B}}) = {corr_star_pib:.4f}")
        print(f"  Corr(A*, A^{{π_G}}) = {corr_star_pig:.4f}")

    # Distribution of A^{π_B}
    pib_positive = sum(1 for a in a_pi_bs if a > 0)
    pib_negative = sum(1 for a in a_pi_bs if a < 0)
    pib_zero = sum(1 for a in a_pi_bs if a == 0)
    print(f"\n  A^{{π_B}} distribution: positive={pib_positive}, zero={pib_zero}, negative={pib_negative}")

    pig_positive = sum(1 for a in a_pi_gs if a > 0)
    pig_negative = sum(1 for a in a_pi_gs if a < 0)
    pig_zero = sum(1 for a in a_pi_gs if a == 0)
    print(f"  A^{{π_G}} distribution: positive={pig_positive}, zero={pig_zero}, negative={pig_negative}")

    # Where does value disappear?
    star_positive_pib_zero = sum(1 for r in diff_results if r["a_star"] > 5 and abs(r["a_pi_base"]) <= 1)
    star_positive_pib_positive = sum(1 for r in diff_results if r["a_star"] > 5 and r["a_pi_base"] > 1)
    star_positive_pib_negative = sum(1 for r in diff_results if r["a_star"] > 5 and r["a_pi_base"] < -1)
    print(f"\n  Where A* > 5 (oracle says HELP):")
    print(f"    A^{{π_B}} ≈ 0 (model can't continue):  {star_positive_pib_zero}")
    print(f"    A^{{π_B}} > 0 (model benefits):        {star_positive_pib_positive}")
    print(f"    A^{{π_B}} < 0 (model harmed):          {star_positive_pib_negative}")

    star_positive_pig_positive = sum(1 for r in diff_results if r["a_star"] > 5 and r["a_pi_gov"] > 1)
    star_positive_pig_zero = sum(1 for r in diff_results if r["a_star"] > 5 and abs(r["a_pi_gov"]) <= 1)
    print(f"    A^{{π_G}} > 0 (gov continuation helps): {star_positive_pig_positive}")
    print(f"    A^{{π_G}} ≈ 0 (gov continuation neutral): {star_positive_pig_zero}")

    # Rescueability classification
    rescue_counter = Counter()
    for r in intervention_results:
        rescue_counter[r["rescueability"]["category"]] += 1

    print(f"\n--- Rescueability Classification (N={n}) ---")
    for cat, cnt in rescue_counter.most_common():
        print(f"  {cat}: {cnt} ({cnt/n:.1%})")

    # Cross-tab: rescueability × A^{π_B}
    print(f"\n--- Rescueability × A^{{π_B}} (different-action only, N={n_diff}) ---")
    rescue_a_pib = defaultdict(list)
    for r in diff_results:
        rescue_a_pib[r["rescueability"]["category"]].append(r["a_pi_base"])

    for cat, vals in rescue_a_pib.items():
        if vals:
            print(f"  {cat}: n={len(vals)}, mean A^{{π_B}}={sum(vals)/len(vals):+.2f}")

    # Success conversion
    base_successes = sum(1 for r in diff_results if r["base_success"])
    gov_off_successes = sum(1 for r in diff_results if r["gov_off_success"])
    gov_sel_successes = sum(1 for r in diff_results if r["gov_sel_success"])
    print(f"\n--- Success Conversion (different-action, N={n_diff}) ---")
    print(f"  Base continuation success:   {base_successes}/{n_diff}")
    print(f"  Gov+OFF continuation success: {gov_off_successes}/{n_diff}")
    print(f"  Gov+SEL continuation success: {gov_sel_successes}/{n_diff}")

    # Chain pattern analysis
    print(f"\n--- Chain Macro-Patterns (from I3.5.2c) ---")
    print(f"  Total chains: {chain_analysis['total_chains']}")
    for length, stats in chain_analysis["chain_length_stats"].items():
        print(f"  Length {length}: n={stats['count']}, mean ΔU={stats['mean_delta_u']}")
    print(f"  Top action sequences:")
    for seq, cnt in list(chain_analysis["action_sequence_distribution"].items())[:10]:
        print(f"    {seq}: {cnt}")

    # Save summary
    summary = {
        "schema": "DAPH_V2B_I3_5_2D_ANALYSIS_V1",
        "n_interventions": n,
        "n_different_action": n_diff,
        "intervention_value_decomposition": {
            "A_star_mean": round(mean_a_star, 4),
            "A_pi_base_mean": round(mean_a_pi_b, 4),
            "A_pi_gov_mean": round(mean_a_pi_g, 4),
            "A_star_positive_pi_base_zero": star_positive_pib_zero,
            "A_star_positive_pi_base_positive": star_positive_pib_positive,
            "A_star_positive_pi_base_negative": star_positive_pib_negative,
            "A_star_positive_pi_gov_positive": star_positive_pig_positive,
            "A_star_positive_pi_gov_zero": star_positive_pig_zero,
        },
        "rescueability_distribution": dict(rescue_counter.most_common()),
        "rescueability_x_a_pi_base": {
            cat: {"n": len(vals), "mean_a_pi_base": round(sum(vals)/len(vals), 4)}
            for cat, vals in rescue_a_pib.items()
        },
        "success_conversion": {
            "base": base_successes,
            "gov_off": gov_off_successes,
            "gov_sel": gov_sel_successes,
            "total": n_diff,
        },
        "chain_analysis": {
            "total_chains": chain_analysis["total_chains"],
            "chain_length_stats": chain_analysis["chain_length_stats"],
            "top_action_sequences": dict(list(chain_analysis["action_sequence_distribution"].items())[:10]),
        },
    }

    summary_path = output_dir / "analysis_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Scientific interpretation
    print(f"\n{'='*78}")
    print("SCIENTIFIC INTERPRETATION")
    print(f"{'='*78}")

    if mean_a_star > 5 and abs(mean_a_pi_b) < 2:
        print("  RESULT: A* > 0 but A^{π_B} ≈ 0")
        print("  The governor identifies genuinely valuable paths, but the base")
        print("  model cannot continue them. Persistent governor control may be necessary.")
    elif mean_a_star > 5 and mean_a_pi_b > 2:
        print("  RESULT: A* > 0 and A^{π_B} > 0")
        print("  The governor's oracle advantage IS behaviorally realizable by the")
        print("  base model. The I3.5.2c failure must have another explanation.")
    elif mean_a_star > 5 and mean_a_pi_b < -2:
        print("  RESULT: A* > 0 but A^{π_B} < 0")
        print("  The governor's oracle advantage is actively harmful under base model")
        print("  continuation. The model makes worse decisions after the intervention.")
    else:
        print(f"  RESULT: A*={mean_a_star:+.2f}, A^{{π_B}}={mean_a_pi_b:+.2f}, A^{{π_G}}={mean_a_pi_g:+.2f}")

    if mean_a_pi_g > mean_a_pi_b + 2:
        print(f"\n  A^{{π_G}} > A^{{π_B}} by {mean_a_pi_g - mean_a_pi_b:+.2f}")
        print("  Governor continuation is better than base continuation.")
        print("  Persistent governor control may be necessary after intervention.")


if __name__ == "__main__":
    main()
