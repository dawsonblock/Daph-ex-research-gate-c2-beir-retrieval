#!/usr/bin/env python3
"""Build expanded fork dataset for I3.5.3-r1.

For EVERY OFF trajectory state where the governor disagrees with the baseline
action (a_G != a_B), fork into two continuations:

  Fork A: execute a_B, continue with OFF model → terminal → U_A
  Fork B: execute a_G, continue with OFF model → terminal → U_B

Target: ΔQ_π = U_B - U_A

This is broader than I3.5.2d, which only forked at Q*-gate-selected states.
If this expanded dataset still contains zero positive ΔQ_π, we can make the
stronger statement:

  "Across ALL observed governor-baseline disagreements on the development
   distribution, no governor intervention improved return under baseline-model
   continuation."

Usage:
    DEEPSEEK_API_KEY=... python scripts/build_i3_5_3r1_expanded_fork_dataset.py \\
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
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    build_oracle_policy_table,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState, ResourceExhausted
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.selective_governor.features import (
    InterventionFeatures, extract_features,
)


def continue_with_off_model(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    t_runtime: TaskRuntime,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    backend: DeepSeekBackend,
    utility: MetareasoningUtility,
    start_step: int = 0,
    max_steps: int = 24,
) -> dict[str, Any]:
    """Continue a trajectory from the given state using OFF model (base packet only)."""
    executor = DeterministicActionExecutor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
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

        packet = build_base_packet(observation)
        assert_no_evaluator_leakage(packet)
        user_prompt = packet_json(packet)

        backend.task_id = task.task_id
        backend.condition = f"i353r1_fork_OFF"
        backend.pair_id = f"i353r1:{task.task_id}:step{start_step}:off_cont"

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


def process_one_fork(
    task: I3BenchmarkTask,
    budget: ResourceBudget,
    table: Any,
    step_idx: int,
    a_base: DecisionAction,
    a_gov: DecisionAction,
    i3_runtime_state: Any,
    t_runtime_state: Any,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    """Process one fork pair: a_B + OFF continuation vs a_G + OFF continuation."""
    state_id = table.state_id_for(i3_runtime_state)

    # Oracle Q values for reference
    q_base = table.q_values.get((state_id, a_base))
    q_gov = table.q_values.get((state_id, a_gov))
    a_star = round(q_gov - q_base, 4) if q_base is not None and q_gov is not None else None

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Fork A: execute a_base, continue with OFF model
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
            "steps": 0,
            "model_calls": 0,
            "terminal_result": exec_a.outcome_code,
        }
    else:
        cont_a = continue_with_off_model(
            task=task, budget=budget, t_runtime=exec_a_t.runtime,
            prior_decisions=fork_a_pd, prior_outcomes=fork_a_po,
            backend=backend, utility=utility,
            start_step=step_idx + 1)

    # Fork B: execute a_gov, continue with OFF model
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
            "steps": 0,
            "model_calls": 0,
            "terminal_result": exec_b.outcome_code,
        }
    else:
        cont_b = continue_with_off_model(
            task=task, budget=budget, t_runtime=exec_b_t.runtime,
            prior_decisions=fork_b_pd, prior_outcomes=fork_b_po,
            backend=backend, utility=utility,
            start_step=step_idx + 1)

    delta_q_pi = round(cont_b["realized_utility"] - cont_a["realized_utility"], 4)

    return {
        "task_id": task.task_id,
        "step_id": step_idx,
        "state_id": state_id,
        "base_action": a_base.value,
        "gov_action": a_gov.value,
        "a_star": a_star,
        # Policy-conditional target
        "u_base_continuation": cont_a["realized_utility"],
        "u_gov_continuation": cont_b["realized_utility"],
        "delta_q_pi": delta_q_pi,
        # Outcomes
        "base_success": cont_a["success"],
        "gov_success": cont_b["success"],
        "base_steps": cont_a["steps"],
        "gov_steps": cont_b["steps"],
        "base_model_calls": cont_a["model_calls"],
        "gov_model_calls": cont_b["model_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description="Build expanded fork dataset for I3.5.3-r1")
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
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development/i353r1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-forks", type=int, default=None,
                        help="Limit total forks (for testing)")
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

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    # Replay all OFF trajectories, find every state where a_G != a_B
    print("\nReplaying OFF trajectories to find all governor-baseline disagreements...")
    fork_states: list[dict] = []

    for i, block in enumerate(i352c_blocks):
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        off_traj = block["trajectories"]["OFF"]
        off_steps = off_traj["steps"]

        # Build oracle table
        table = build_oracle_policy_table(task=task, policy=policy, utility=utility, budget=budget)

        # Replay OFF trajectory
        resources = ResourceState(budget)
        i3_runtime = initial_i3_runtime(task, resources)
        adapter = _I3TaskAdapter(task)
        t_runtime = initial_runtime(adapter, ResourceState(budget))

        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []

        for step_idx, step_data in enumerate(off_steps):
            a_base_str = step_data["executed_action"]
            a_base = DecisionAction(a_base_str)

            # Compute governor recommendation at this state
            obs = build_observation(t_runtime, task, cond,
                                    tuple(prior_decisions), tuple(prior_outcomes))
            p_actions = tuple(d.selected_action if isinstance(d.selected_action, str)
                              else d.selected_action.value for d in prior_decisions)
            frame = governor.assess(
                observation=obs,
                remaining_steps=24 - step_idx,
                prior_actions=p_actions,
                prior_outcomes=tuple(prior_outcomes),
            )
            a_gov_str = frame.governor_top_action or a_base_str
            a_gov = DecisionAction(a_gov_str)

            # Extract features
            features = extract_features(
                obs, remaining_steps=24 - step_idx,
                prior_actions=p_actions, prior_outcomes=tuple(prior_outcomes),
            )

            # If governor disagrees, record fork state
            if a_base_str != a_gov_str:
                fork_states.append({
                    "task_id": task_id,
                    "step_id": step_idx,
                    "state_id": table.state_id_for(i3_runtime),
                    "a_base": a_base,
                    "a_gov": a_gov,
                    "a_base_str": a_base_str,
                    "a_gov_str": a_gov_str,
                    "i3_runtime": copy.deepcopy(i3_runtime),
                    "t_runtime": copy.deepcopy(t_runtime),
                    "prior_decisions": list(prior_decisions),
                    "prior_outcomes": list(prior_outcomes),
                    "features": features,
                    "table": table,
                    "task": task,
                    "budget": budget,
                })

            # Step forward
            exec_res = oracle_executor.execute(i3_runtime, a_base)
            i3_runtime = exec_res.runtime
            t_runtime = task_executor.execute(t_runtime, a_base).runtime
            prior_decisions.append(DecisionSummary(
                f"{task_id}:step:{step_idx}", a_base.value,
                step_data["reason_code"], exec_res.outcome_code))
            prior_outcomes.append(exec_res.outcome_code)

            if exec_res.terminal:
                break

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(i352c_blocks)}] tasks, "
                  f"{len(fork_states)} disagreement states found")

    print(f"\nTotal governor-baseline disagreement states: {len(fork_states)}")

    # Action pair distribution
    pair_counter = Counter()
    for fs in fork_states:
        pair_counter[(fs["a_base_str"], fs["a_gov_str"])] += 1
    print("\nAction pair distribution:")
    for (b, g), cnt in pair_counter.most_common():
        print(f"  {b} → {g}: {cnt}")

    if args.max_forks:
        fork_states = fork_states[:args.max_forks]
        print(f"\nLimited to {len(fork_states)} forks for testing")

    # Run fork pairs
    print(f"\nRunning {len(fork_states)} fork pairs ({args.workers} workers)...")
    print(f"Each fork: 2 OFF model continuations (fork A: a_B + OFF, fork B: a_G + OFF)")

    def process_fork(item):
        idx, fs = item
        result = process_one_fork(
            task=fs["task"],
            budget=fs["budget"],
            table=fs["table"],
            step_idx=fs["step_id"],
            a_base=fs["a_base"],
            a_gov=fs["a_gov"],
            i3_runtime_state=fs["i3_runtime"],
            t_runtime_state=fs["t_runtime"],
            prior_decisions=fs["prior_decisions"],
            prior_outcomes=fs["prior_outcomes"],
            utility=utility,
            api_key=api_key,
        )
        # Add features for training
        result["features"] = fs["features"].as_dict()
        return idx, result

    results = []
    completed = 0
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_fork, (i, fs)): i
            for i, fs in enumerate(fork_states)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results.append((idx, result))
            completed += 1
            if completed % args.progress_every == 0:
                elapsed = time.monotonic() - t_start
                rate = completed / elapsed
                eta = (len(fork_states) - completed) / rate if rate > 0 else 0
                a_star_str = f"A*={result['a_star']:+.1f}" if result['a_star'] is not None else "A*=N/A"
                print(f"  [{completed}/{len(fork_states)}] "
                      f"ΔQ_π={result['delta_q_pi']:+.1f} "
                      f"{a_star_str} "
                      f"base_s={result['base_success']} "
                      f"gov_s={result['gov_success']} "
                      f"eta={eta:.0f}s")

    results.sort(key=lambda x: x[0])
    fork_results = [r[1] for r in results]

    elapsed = time.monotonic() - t_start
    print(f"\nCompleted {completed} fork pairs in {elapsed:.0f}s")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "expanded_fork_dataset_v1.jsonl"
    with open(results_path, "w") as f:
        for r in fork_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Analysis
    print("\n" + "=" * 78)
    print("EXPANDED FORK DATASET ANALYSIS")
    print("=" * 78)

    n = len(fork_results)
    delta_q_pis = [r["delta_q_pi"] for r in fork_results]
    a_stars = [r["a_star"] for r in fork_results if r["a_star"] is not None]

    mean_dqp = sum(delta_q_pis) / n if n else 0
    mean_astar = sum(a_stars) / len(a_stars) if a_stars else 0

    positive_dqp = sum(1 for d in delta_q_pis if d > 1)
    zero_dqp = sum(1 for d in delta_q_pis if abs(d) <= 1)
    negative_dqp = sum(1 for d in delta_q_pis if d < -1)

    print(f"\nN = {n} fork pairs")
    print(f"Mean ΔQ_π = {mean_dqp:+.4f}")
    print(f"Mean A* = {mean_astar:+.4f}")
    print(f"ΔQ_π > +1 (positive): {positive_dqp} ({positive_dqp/n:.1%})")
    print(f"|ΔQ_π| <= 1 (neutral): {zero_dqp} ({zero_dqp/n:.1%})")
    print(f"ΔQ_π < -1 (negative): {negative_dqp} ({negative_dqp/n:.1%})")

    # Success conversion
    base_successes = sum(1 for r in fork_results if r["base_success"])
    gov_successes = sum(1 for r in fork_results if r["gov_success"])
    print(f"\nBase continuation success: {base_successes}/{n}")
    print(f"Gov continuation success:  {gov_successes}/{n}")

    # Where A* > 0 but ΔQ_π <= 0
    if a_stars:
        star_pos_dqp_nonpos = sum(
            1 for r in fork_results
            if r["a_star"] is not None and r["a_star"] > 5 and r["delta_q_pi"] <= 1
        )
        star_pos = sum(1 for r in fork_results if r["a_star"] is not None and r["a_star"] > 5)
        print(f"\nWhere A* > 5: {star_pos}")
        print(f"  ΔQ_π <= 1 (not realizable): {star_pos_dqp_nonpos}")

    # Action pair × ΔQ_π
    print(f"\n--- Action Pair × ΔQ_π ---")
    pair_dqp = defaultdict(list)
    for r in fork_results:
        pair_dqp[(r["base_action"], r["gov_action"])].append(r["delta_q_pi"])

    for (b, g), vals in sorted(pair_dqp.items(), key=lambda x: -len(x[1])):
        pos = sum(1 for v in vals if v > 1)
        print(f"  {b} → {g}: n={len(vals)}, mean ΔQ_π={sum(vals)/len(vals):+.2f}, positive={pos}")

    # Save summary
    summary = {
        "schema": "DAPH_V2B_I3_5_3R1_EXPANDED_FORK_V1",
        "n_forks": n,
        "mean_delta_q_pi": round(mean_dqp, 4),
        "mean_a_star": round(mean_astar, 4),
        "positive_dqp": positive_dqp,
        "zero_dqp": zero_dqp,
        "negative_dqp": negative_dqp,
        "base_successes": base_successes,
        "gov_successes": gov_successes,
        "action_pair_distribution": {
            f"{b}->{g}": {"count": len(vals), "mean_dqp": round(sum(vals)/len(vals), 4)}
            for (b, g), vals in pair_dqp.items()
        },
    }
    summary_path = output_dir / "expanded_fork_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
