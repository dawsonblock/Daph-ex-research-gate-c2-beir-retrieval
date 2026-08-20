#!/usr/bin/env python3
"""I3.11f: Repeated-Measures R1 vs M3 with interleaved call order.

Everything frozen: T2, R1, M3, A1, prompts, utility, model/config.

For each task:
  A1 × K repeats
  M3 × K repeats
  R1 × K repeats

Calls are interleaved in counterbalanced order to reduce confounding
from provider drift or changing backend instances.

Estimand: per-task mean utility and success probability, then
bootstrap tasks (not individual calls) to get CIs.

Instability metrics:
  - deterministic request rate: fraction of unique request hashes
    producing one action only
  - unstable request rate: fraction producing >1 distinct action
  - STOP <-> DEFER flip rate

Fingerprint micro-experiment:
  Take the exact request from tasks 0009 and 0043 at the divergence
  step and make 20-30 calls each, interleaved. Record system_fingerprint.

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_11f_repeated_measures.py \\
        --workers 4 --k-repeats 5
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
i3_7e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_7e)

_spec_c = importlib.util.spec_from_file_location(
    "i3_11c", ROOT / "scripts" / "run_i3_11c_r1_router.py")
i3_11c = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(i3_11c)
run_r1_trajectory = i3_11c.run_r1_trajectory

_spec_d = importlib.util.spec_from_file_location(
    "i3_11d", ROOT / "scripts" / "run_i3_11d_r1_confirm.py")
i3_11d = importlib.util.module_from_spec(_spec_d)
_spec_d.loader.exec_module(i3_11d)

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
    EvidenceExecutor, EvidenceBenchmark, save_evidence_benchmark,
    initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.evidence_benchmark.serializer import (
    evidence_packet_json,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceSnapshot,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output


# ---------------------------------------------------------------------------
# Trajectory runners that also collect call receipts
# ---------------------------------------------------------------------------

def run_trajectory_with_receipts(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    mode: str,
    api_key: str,
    fork_label: str,
    repeat_idx: int,
) -> dict[str, Any]:
    """Run a trajectory (A1 or M3) and collect call receipts."""
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(task, resources)

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    terminal_action = None
    backend_errors = 0

    continuation_actions: list[str] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps

    call_receipts: list[dict] = []

    for step_id in range(max_steps):
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        if mode == "BASELINE_WITH_AFFORDANCES":
            packet = i3_7e.build_baseline_with_affordances_packet(evidence_snapshot)
            system_prompt = i3_7e.BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT
        elif mode == "MDSG_STATE_WITH_AFFORDANCES":
            packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
            system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown mode: {mode}")

        user_prompt = evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_11f_{mode}"
        backend.pair_id = f"i3_11f:{task.task_id}:{fork_label}:r{repeat_idx}:step{step_id}"

        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
        except Exception:
            backend_errors += 1
            proposal = i3_7e.BACKEND_ERROR_PROPOSAL
        else:
            outcome = decode_output(call_result.raw_output, strict=True)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = i3_7e.FAIL_CLOSED_PROPOSAL

        # Collect receipt from this call
        if backend.call_receipts:
            receipt = backend.call_receipts[-1]
            call_receipts.append(receipt.as_dict())

        action = proposal.action
        target_id = getattr(proposal, "target_id", None)

        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        resources_after = exec_res.runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost
        total_action_cost += step_cost
        step_costs.append(round(step_cost, 4))

        action_str = action.value if hasattr(action, "value") else str(action)
        continuation_actions.append(action_str)

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            terminal_reward = tr
            success = bool(exec_res.task_success)
            terminal = True
            terminal_result = exec_res.outcome_code
            terminal_action = action_str

        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime
        steps_taken += 1

        if exec_res.terminal:
            break

    return {
        "realized_utility": round(realized, 4),
        "success": success,
        "steps": steps_taken,
        "model_calls": model_calls,
        "backend_errors": backend_errors,
        "terminal_action": terminal_action,
        "terminal_result": terminal_result,
        "continuation_actions": continuation_actions,
        "call_receipts": call_receipts,
    }


def run_r1_with_receipts(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
    fork_label: str,
    repeat_idx: int,
) -> dict[str, Any]:
    """Run R1 hybrid trajectory and collect call receipts."""
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(task, resources)

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    terminal_action = None
    backend_errors = 0

    continuation_actions: list[str] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    r1_triggered = False
    r1_trigger_step: int | None = None

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps
    n_hypotheses = len(task.hypotheses)

    call_receipts: list[dict] = []

    for step_id in range(max_steps):
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        internal_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        internal_summary = internal_m3_packet.get("decision_state_summary", {})
        eliminated = internal_summary.get("eliminated_hypotheses", [])

        t2_fires = len(eliminated) == n_hypotheses and n_hypotheses > 0

        if not r1_triggered and t2_fires:
            r1_triggered = True
            r1_trigger_step = step_id

        if r1_triggered:
            packet = internal_m3_packet
            system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
        else:
            packet = i3_7e.build_baseline_with_affordances_packet(evidence_snapshot)
            system_prompt = i3_7e.BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT

        user_prompt = evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_11f_R1"
        backend.pair_id = f"i3_11f:R1:{task.task_id}:{fork_label}:r{repeat_idx}:step{step_id}"

        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
        except Exception:
            backend_errors += 1
            proposal = i3_7e.BACKEND_ERROR_PROPOSAL
        else:
            outcome = decode_output(call_result.raw_output, strict=True)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = i3_7e.FAIL_CLOSED_PROPOSAL

        if backend.call_receipts:
            receipt = backend.call_receipts[-1]
            call_receipts.append(receipt.as_dict())

        action = proposal.action
        target_id = getattr(proposal, "target_id", None)

        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        resources_after = exec_res.runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost
        total_action_cost += step_cost
        step_costs.append(round(step_cost, 4))

        action_str = action.value if hasattr(action, "value") else str(action)
        continuation_actions.append(action_str)

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            terminal_reward = tr
            success = bool(exec_res.task_success)
            terminal = True
            terminal_result = exec_res.outcome_code
            terminal_action = action_str

        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime
        steps_taken += 1

        if exec_res.terminal:
            break

    return {
        "realized_utility": round(realized, 4),
        "success": success,
        "steps": steps_taken,
        "model_calls": model_calls,
        "backend_errors": backend_errors,
        "terminal_action": terminal_action,
        "terminal_result": terminal_result,
        "continuation_actions": continuation_actions,
        "r1_triggered": r1_triggered,
        "r1_trigger_step": r1_trigger_step,
        "call_receipts": call_receipts,
    }


# ---------------------------------------------------------------------------
# Interleaved call ordering
# ---------------------------------------------------------------------------

def build_interleaved_order(task_id: str, k_repeats: int) -> list[tuple[str, int]]:
    """Build interleaved call order for A1, M3, R1 × K repeats.

    Returns a list of (arm, repeat_idx) tuples in counterbalanced order.
    """
    arms = ["A1", "M3", "R1"]
    all_calls = []
    for arm in arms:
        for r in range(k_repeats):
            all_calls.append((arm, r))

    # Seed from task_id for reproducibility
    h = hashlib.sha256(task_id.encode()).hexdigest()
    rng = random.Random(int(h[:8], 16))
    rng.shuffle(all_calls)
    return all_calls


# ---------------------------------------------------------------------------
# Per-task processing with K repeats
# ---------------------------------------------------------------------------

def process_task_repeated(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
    k_repeats: int,
) -> dict[str, Any]:
    """Process one task with K repeats per arm, interleaved."""
    call_order = build_interleaved_order(task.task_id, k_repeats)

    arm_modes = {
        "A1": "BASELINE_WITH_AFFORDANCES",
        "M3": "MDSG_STATE_WITH_AFFORDANCES",
    }

    # We can't truly interleave across threads easily, but we can
    # submit in interleaved order to the pool
    results_by_arm: dict[str, list[dict]] = {"A1": [], "M3": [], "R1": []}

    for arm, repeat_idx in call_order:
        if arm == "R1":
            result = run_r1_with_receipts(
                task, budget, utility, api_key,
                fork_label=f"arm{arm}", repeat_idx=repeat_idx,
            )
        else:
            result = run_trajectory_with_receipts(
                task, budget, utility,
                mode=arm_modes[arm], api_key=api_key,
                fork_label=f"arm{arm}", repeat_idx=repeat_idx,
            )
        result["repeat_idx"] = repeat_idx
        result["arm"] = arm
        results_by_arm[arm].append(result)

    # Sort by repeat_idx
    for arm in results_by_arm:
        results_by_arm[arm].sort(key=lambda r: r["repeat_idx"])

    # Calculate per-task means
    def arm_stats(results):
        n = len(results)
        utilities = [r["realized_utility"] for r in results]
        successes = [r["success"] for r in results]
        steps = [r["steps"] for r in results]
        return {
            "mean_utility": round(sum(utilities) / n, 4),
            "success_probability": round(sum(successes) / n, 4),
            "n_success": sum(successes),
            "n_repeats": n,
            "mean_steps": round(sum(steps) / n, 2),
            "utilities": utilities,
            "successes": successes,
            "steps_list": steps,
            "terminal_actions": [r["terminal_action"] for r in results],
        }

    a1_stats = arm_stats(results_by_arm["A1"])
    m3_stats = arm_stats(results_by_arm["M3"])
    r1_stats = arm_stats(results_by_arm["R1"])

    # Per-task delta
    delta_r1_m3 = round(r1_stats["mean_utility"] - m3_stats["mean_utility"], 4)
    delta_r1_a1 = round(r1_stats["mean_utility"] - a1_stats["mean_utility"], 4)

    # Collect all call receipts for instability analysis
    all_receipts = []
    for arm in ["A1", "M3", "R1"]:
        for r in results_by_arm[arm]:
            for receipt in r.get("call_receipts", []):
                receipt["arm"] = arm
                receipt["repeat_idx"] = r["repeat_idx"]
                receipt["task_id"] = task.task_id
                all_receipts.append(receipt)

    return {
        "task_id": task.task_id,
        "category": task.category,
        "n_hypotheses": len(task.hypotheses),
        "k_repeats": k_repeats,
        "a1": a1_stats,
        "m3": m3_stats,
        "r1": r1_stats,
        "delta_r1_m3": delta_r1_m3,
        "delta_r1_a1": delta_r1_a1,
        "call_receipts": all_receipts,
    }


# ---------------------------------------------------------------------------
# Instability metrics
# ---------------------------------------------------------------------------

def compute_instability_metrics(all_receipts: list[dict]) -> dict[str, Any]:
    """Compute instability metrics from call receipts."""
    # Group by request_sha256
    by_request: dict[str, list[dict]] = defaultdict(list)
    for r in all_receipts:
        if r.get("result_class") == "success":
            req_hash = r.get("request_sha256", "")
            by_request[req_hash].append(r)

    total_requests = len(by_request)
    if total_requests == 0:
        return {"error": "no successful receipts"}

    # For each unique request, count distinct decoded actions
    # We need to decode the raw_output to get the action
    deterministic_count = 0
    unstable_count = 0
    stop_defer_flip_count = 0
    request_action_dist: dict[str, Counter] = {}

    for req_hash, receipts in by_request.items():
        actions = []
        for r in receipts:
            raw = r.get("raw_output", "")
            try:
                parsed = json.loads(raw)
                action = parsed.get("action", "UNKNOWN")
            except Exception:
                action = "PARSE_ERROR"
            actions.append(action)

        action_counter = Counter(actions)
        request_action_dist[req_hash] = action_counter

        n_distinct = len(action_counter)
        if n_distinct == 1:
            deterministic_count += 1
        else:
            unstable_count += 1
            if "STOP" in action_counter and "DEFER" in action_counter:
                stop_defer_flip_count += 1

    # Fingerprint analysis
    by_fingerprint: dict[str, list[dict]] = defaultdict(list)
    for r in all_receipts:
        if r.get("result_class") == "success":
            fp = r.get("system_fingerprint") or "NONE"
            by_fingerprint[fp].append(r)

    fingerprint_distribution = {fp: len(receipts) for fp, receipts in by_fingerprint.items()}

    return {
        "total_unique_requests": total_requests,
        "deterministic_request_rate": round(deterministic_count / total_requests, 4),
        "unstable_request_rate": round(unstable_count / total_requests, 4),
        "stop_defer_flip_rate": round(stop_defer_flip_count / total_requests, 4),
        "n_deterministic": deterministic_count,
        "n_unstable": unstable_count,
        "n_stop_defer_flip": stop_defer_flip_count,
        "fingerprint_distribution": fingerprint_distribution,
        "n_unique_fingerprints": len(fingerprint_distribution),
    }


# ---------------------------------------------------------------------------
# Fingerprint micro-experiment
# ---------------------------------------------------------------------------

def fingerprint_micro_experiment(
    task_map: dict[str, EvidenceTask],
    budget: ResourceBudget,
    api_key: str,
    n_calls: int = 25,
) -> dict[str, Any]:
    """Micro-experiment: take exact request from 0009 and 0043 at divergence
    step, make N interleaved calls each, record system_fingerprint."""
    from hrm_adaptive_memory.executive.evidence_benchmark.serializer import (
        evidence_packet_json,
    )

    target_tasks = ["r1_confirm_v1_0009", "r1_confirm_v1_0043"]
    results = {}

    # First, reconstruct the exact request for each task at the divergence step
    # We need to replay to the divergence step and freeze the packet
    executor = EvidenceExecutor()

    for task_id in target_tasks:
        task = task_map.get(task_id)
        if task is None:
            results[task_id] = {"error": "task not found"}
            continue

        # Replay to divergence step (step 3 for 0009, step 4 for 0043)
        div_step = 3 if "0009" in task_id else 4

        runtime = initial_evidence_runtime(task, ResourceState(budget))
        prior_actions = []
        prior_outcomes = []

        # Replay steps 0..div_step-1 with fixed actions (VERIFY × N, SEARCH_MORE)
        actions_to_replay = ["VERIFY"] * (div_step - 1) + ["SEARCH_MORE"]
        for action_str in actions_to_replay[:div_step]:
            action = DecisionAction(action_str)
            exec_res = executor.execute(runtime, action)
            prior_actions.append(action_str)
            prior_outcomes.append(exec_res.outcome_code)
            runtime = exec_res.runtime

        # Build the frozen snapshot and packet at divergence step
        snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )
        packet = i3_7e.build_mdsg_state_with_affordances_packet(snapshot)
        system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
        user_prompt = evidence_packet_json(packet)

        # Make N interleaved calls
        call_results = []
        for i in range(n_calls):
            backend = DeepSeekBackend()
            backend.task_id = task_id
            backend.condition = f"i3_11f_fingerprint_{i}"
            backend.pair_id = f"i3_11f:fp:{task_id}:call{i}"

            try:
                call_result = backend.generate(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    temperature=0.0, max_tokens=2048)
                outcome = decode_output(call_result.raw_output, strict=True)
                if outcome.valid and outcome.proposal:
                    action = outcome.proposal.action
                    action_str = action.value if hasattr(action, "value") else str(action)
                else:
                    action_str = "FAIL_CLOSED"
            except Exception as e:
                action_str = "BACKEND_ERROR"

            receipt = backend.call_receipts[-1] if backend.call_receipts else None
            call_results.append({
                "call_idx": i,
                "action": action_str,
                "system_fingerprint": receipt.system_fingerprint if receipt else None,
                "reported_model": receipt.reported_model if receipt else None,
                "latency_ms": receipt.latency_ms if receipt else None,
                "finish_reason": receipt.finish_reason if receipt else None,
            })

        # Analyze by fingerprint
        by_fp: dict[str, list[dict]] = defaultdict(list)
        for cr in call_results:
            fp = cr["system_fingerprint"] or "NONE"
            by_fp[fp].append(cr)

        fp_table = {}
        for fp, calls in by_fp.items():
            actions = Counter(c["action"] for c in calls)
            fp_table[fp] = {
                "n": len(calls),
                "actions": dict(actions),
            }

        action_dist = dict(Counter(c["action"] for c in call_results))

        results[task_id] = {
            "n_calls": n_calls,
            "divergence_step": div_step,
            "action_distribution": action_dist,
            "fingerprint_table": fp_table,
            "n_unique_fingerprints": len(by_fp),
            "all_calls": call_results,
        }

        print(f"  {task_id}: {action_dist} ({len(by_fp)} fingerprints)")

    return results


# ---------------------------------------------------------------------------
# Bootstrap and analysis
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(deltas, n_iterations=10000, seed=42):
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    return boot_means[int(0.025 * n_iterations)], boot_means[int(0.975 * n_iterations)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k-repeats", type=int, default=5)
    parser.add_argument("--n-tasks", type=int, default=100,
        help="Number of tasks to run (subset of I3.11d corpus for cost)")
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--output-dir",
        default="experiments/v2b_i3_11/development/i3_11f_repeated_measures")
    parser.add_argument("--fingerprint-calls", type=int, default=25)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"I3.11f: Repeated-Measures R1 vs M3")
    print(f"  K repeats per arm: {args.k_repeats}")
    print(f"  N tasks: {args.n_tasks}")
    print(f"  Total model calls: ~{args.n_tasks * 3 * args.k_repeats * 6} (approx)")
    print(f"  Everything frozen: T2, R1, M3, A1, prompts, utility, model/config")
    print()

    # Use I3.11d corpus but subsample for cost
    full_corpus = i3_11d.generate_i3_11d_corpus(split="r1_confirm_v1")
    task_map = {t.task_id: t for t in full_corpus}

    # Subsample: include all break-category tasks + random sample of others
    break_cats = {"bilateral_conflict_h0", "triple_all_eliminated", "noise_before_conflict"}
    break_tasks = [t for t in full_corpus if t.category in break_cats]
    other_tasks = [t for t in full_corpus if t.category not in break_cats]

    rng = random.Random(42)
    n_other = max(0, args.n_tasks - len(break_tasks))
    sampled_other = rng.sample(other_tasks, min(n_other, len(other_tasks)))
    tasks = break_tasks + sampled_other

    print(f"  Selected {len(tasks)} tasks ({len(break_tasks)} break-category + {len(sampled_other)} other)")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution:")
    for cat in sorted(cats.keys()):
        print(f"    {cat:<40} {cats[cat]}")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks × {args.k_repeats} repeats × 3 arms...")
    all_results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_task_repeated, task, budget, utility, api_key, args.k_repeats): task
                   for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 5 == 0:
                    print(f"  Completed {completed}/{len(tasks)} tasks...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_results)} tasks")

    # Save raw results
    results_path = output_dir / "repeated_measures_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # === Analysis ===
    n = len(all_results)

    # Per-task means
    u_a1_per_task = [r["a1"]["mean_utility"] for r in all_results]
    u_m3_per_task = [r["m3"]["mean_utility"] for r in all_results]
    u_r1_per_task = [r["r1"]["mean_utility"] for r in all_results]

    p_a1_per_task = [r["a1"]["success_probability"] for r in all_results]
    p_m3_per_task = [r["m3"]["success_probability"] for r in all_results]
    p_r1_per_task = [r["r1"]["success_probability"] for r in all_results]

    # Overall means (task-level)
    mean_u_a1 = sum(u_a1_per_task) / n
    mean_u_m3 = sum(u_m3_per_task) / n
    mean_u_r1 = sum(u_r1_per_task) / n

    mean_p_a1 = sum(p_a1_per_task) / n
    mean_p_m3 = sum(p_m3_per_task) / n
    mean_p_r1 = sum(p_r1_per_task) / n

    # Deltas (task-level)
    deltas_r1_a1 = [r["delta_r1_a1"] for r in all_results]
    deltas_r1_m3 = [r["delta_r1_m3"] for r in all_results]
    deltas_m3_a1 = [r["m3"]["mean_utility"] - r["a1"]["mean_utility"] for r in all_results]

    # Success probability deltas
    delta_p_r1_m3 = [p_r1 - p_m3 for p_r1, p_m3 in zip(p_r1_per_task, p_m3_per_task)]
    delta_p_r1_a1 = [p_r1 - p_a1 for p_r1, p_a1 in zip(p_r1_per_task, p_a1_per_task)]

    # Bootstrap CIs (task-level)
    ci_r1_a1 = paired_bootstrap_ci(deltas_r1_a1)
    ci_r1_m3 = paired_bootstrap_ci(deltas_r1_m3)
    ci_m3_a1 = paired_bootstrap_ci(deltas_m3_a1)
    ci_p_r1_m3 = paired_bootstrap_ci(delta_p_r1_m3)
    ci_p_r1_a1 = paired_bootstrap_ci(delta_p_r1_a1)

    # Steps
    mean_steps_a1 = sum(r["a1"]["mean_steps"] for r in all_results) / n
    mean_steps_m3 = sum(r["m3"]["mean_steps"] for r in all_results) / n
    mean_steps_r1 = sum(r["r1"]["mean_steps"] for r in all_results) / n

    # Instability metrics from all call receipts
    all_receipts = []
    for r in all_results:
        all_receipts.extend(r.get("call_receipts", []))

    instability = compute_instability_metrics(all_receipts)

    # Per-task instability: for each task, what fraction of request hashes are unstable?
    per_task_instability = []
    for r in all_results:
        task_receipts = r.get("call_receipts", [])
        task_inst = compute_instability_metrics(task_receipts)
        per_task_instability.append({
            "task_id": r["task_id"],
            "category": r["category"],
            "unstable_request_rate": task_inst.get("unstable_request_rate", 0),
            "n_unstable": task_inst.get("n_unstable", 0),
            "n_unique_requests": task_inst.get("total_unique_requests", 0),
        })

    # Subgroup analysis
    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        subgroups[cat] = {
            "n": cn,
            "mean_u_a1": round(sum(r["a1"]["mean_utility"] for r in cr) / cn, 4),
            "mean_u_m3": round(sum(r["m3"]["mean_utility"] for r in cr) / cn, 4),
            "mean_u_r1": round(sum(r["r1"]["mean_utility"] for r in cr) / cn, 4),
            "mean_p_a1": round(sum(r["a1"]["success_probability"] for r in cr) / cn, 4),
            "mean_p_m3": round(sum(r["m3"]["success_probability"] for r in cr) / cn, 4),
            "mean_p_r1": round(sum(r["r1"]["success_probability"] for r in cr) / cn, 4),
            "mean_steps_r1": round(sum(r["r1"]["mean_steps"] for r in cr) / cn, 2),
            "mean_steps_m3": round(sum(r["m3"]["mean_steps"] for r in cr) / cn, 2),
        }

    # Frozen claims
    frozen_claims = {
        "C1_r1_a1_utility_ci_positive": ci_r1_a1[0] > 0,
        "C2_r1_m3_utility_ci_positive": ci_r1_m3[0] > 0,
        "C3_r1_success_prob_noninferior": mean_p_r1 >= mean_p_m3 - 0.01,
        "C4_r1_steps_lt_m3": mean_steps_r1 < mean_steps_m3,
        "C5_r1_a1_success_ci_positive": ci_p_r1_a1[0] > 0,
    }

    summary = {
        "schema": "DAPH_V2B_I3_11F_REPEATED_MEASURES_V1",
        "n_tasks": n,
        "k_repeats": args.k_repeats,
        "estimand": "per-task mean utility and success probability, bootstrapped over tasks",
        "overall": {
            "mean_u": {"A1": round(mean_u_a1, 4), "M3": round(mean_u_m3, 4), "R1": round(mean_u_r1, 4)},
            "mean_success_prob": {
                "A1": round(mean_p_a1, 4), "M3": round(mean_p_m3, 4), "R1": round(mean_p_r1, 4),
            },
            "bootstrap_ci_r1_a1_utility": [round(ci_r1_a1[0], 4), round(ci_r1_a1[1], 4)],
            "bootstrap_ci_r1_m3_utility": [round(ci_r1_m3[0], 4), round(ci_r1_m3[1], 4)],
            "bootstrap_ci_m3_a1_utility": [round(ci_m3_a1[0], 4), round(ci_m3_a1[1], 4)],
            "bootstrap_ci_r1_m3_success_prob": [round(ci_p_r1_m3[0], 4), round(ci_p_r1_m3[1], 4)],
            "bootstrap_ci_r1_a1_success_prob": [round(ci_p_r1_a1[0], 4), round(ci_p_r1_a1[1], 4)],
            "mean_steps": {
                "A1": round(mean_steps_a1, 2), "M3": round(mean_steps_m3, 2),
                "R1": round(mean_steps_r1, 2),
            },
        },
        "instability_metrics": instability,
        "subgroups": subgroups,
        "frozen_claims": frozen_claims,
    }

    summary_path = output_dir / "repeated_measures_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print results
    print(f"\n{'='*82}")
    print(f"I3.11f REPEATED-MEASURES: A1 vs M3 vs R1 (K={args.k_repeats})")
    print(f"{'='*82}")
    print(f"  Tasks: {n}, Repeats per arm: {args.k_repeats}")
    print(f"\n  Mean utility (task-level):")
    print(f"    A1={mean_u_a1:+.4f}  M3={mean_u_m3:+.4f}  R1={mean_u_r1:+.4f}")
    print(f"\n  Mean success probability (task-level):")
    print(f"    A1={mean_p_a1:.4f}  M3={mean_p_m3:.4f}  R1={mean_p_r1:.4f}")
    print(f"\n  Bootstrap 95% CI (task-level):")
    print(f"    R1-A1 utility:     [{ci_r1_a1[0]:+.4f}, {ci_r1_a1[1]:+.4f}]  <-- CO-PRIMARY")
    print(f"    R1-M3 utility:     [{ci_r1_m3[0]:+.4f}, {ci_r1_m3[1]:+.4f}]  <-- CO-PRIMARY")
    print(f"    M3-A1 utility:     [{ci_m3_a1[0]:+.4f}, {ci_m3_a1[1]:+.4f}]")
    print(f"    R1-M3 success prob:[{ci_p_r1_m3[0]:+.4f}, {ci_p_r1_m3[1]:+.4f}]")
    print(f"    R1-A1 success prob:[{ci_p_r1_a1[0]:+.4f}, {ci_p_r1_a1[1]:+.4f}]")
    print(f"\n  Steps:  A1={mean_steps_a1:.2f}  M3={mean_steps_m3:.2f}  R1={mean_steps_r1:.2f}")

    print(f"\n  INSTABILITY METRICS:")
    print(f"    Total unique requests: {instability['total_unique_requests']}")
    print(f"    Deterministic request rate: {instability['deterministic_request_rate']:.4f}")
    print(f"    Unstable request rate: {instability['unstable_request_rate']:.4f}")
    print(f"    STOP<->DEFER flip rate: {instability['stop_defer_flip_rate']:.4f}")
    print(f"    Unique fingerprints: {instability['n_unique_fingerprints']}")
    if instability.get("fingerprint_distribution"):
        print(f"    Fingerprint distribution: {instability['fingerprint_distribution']}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in frozen_claims.items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")

    print(f"\n{'='*82}")
    print("SUBGROUP ANALYSIS")
    print(f"{'='*82}")
    print(f"  {'Category':<40} {'n':>3} {'A1_U':>8} {'M3_U':>8} {'R1_U':>8} {'A1_P':>6} {'M3_P':>6} {'R1_P':>6}")
    for cat in sorted(subgroups.keys()):
        sg = subgroups[cat]
        print(f"  {cat:<40} {sg['n']:>3} {sg['mean_u_a1']:>+8.2f} {sg['mean_u_m3']:>+8.2f} {sg['mean_u_r1']:>+8.2f} "
              f"{sg['mean_p_a1']:>6.2f} {sg['mean_p_m3']:>6.2f} {sg['mean_p_r1']:>6.2f}")

    # === Fingerprint micro-experiment ===
    print(f"\n{'='*82}")
    print(f"FINGERPRINT MICRO-EXPERIMENT (tasks 0009 + 0043, {args.fingerprint_calls} calls each)")
    print(f"{'='*82}")

    fp_results = fingerprint_micro_experiment(
        task_map, budget, api_key, n_calls=args.fingerprint_calls,
    )

    fp_path = output_dir / "fingerprint_micro_v1.json"
    fp_path.write_text(json.dumps(fp_results, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved: {fp_path}")

    print(f"\n  Fingerprint tables:")
    for task_id, res in fp_results.items():
        if "error" in res:
            print(f"    {task_id}: {res['error']}")
            continue
        print(f"    {task_id}: {res['action_distribution']}")
        for fp, info in res["fingerprint_table"].items():
            print(f"      fp={fp[:20]}...  n={info['n']:>3}  actions={info['actions']}")

    # Add fingerprint results to summary
    summary["fingerprint_micro_experiment"] = {
        tid: {k: v for k, v in res.items() if k != "all_calls"}
        for tid, res in fp_results.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
