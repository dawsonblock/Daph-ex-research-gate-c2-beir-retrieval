#!/usr/bin/env python3
"""I3.12j: 300-task A1/M3/R1 factorial experiment (GOLD x INFERRED).

6 arms per task:
  A1_GOLD, M3_GOLD, R1_GOLD  (oracle supports/contradicts)
  A1_INFERRED, M3_INFERRED, R1_INFERRED  (extractor-inferred relations)

Primary criterion (frozen):
  LCB_95(U_R1_INFERRED - U_A1_INFERRED) > 0

Secondary:
  U_R1_INFERRED - U_M3_INFERRED (not mandatory)
  SemanticGap_R = U_R_GOLD - U_R_INFERRED for R in {A1, M3, R1}

This is a pipeline-transfer test, NOT a semantic robustness test.
The extractor is currently 100% accurate on controlled S1 corpus.

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_12j_factorial.py \\
        --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util

# Import frozen I3.7e trajectory infrastructure
_spec = importlib.util.spec_from_file_location(
    "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
i3_7e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_7e)

# Import frozen I3.11c R1 router
_spec_c = importlib.util.spec_from_file_location(
    "i3_11c", ROOT / "scripts" / "run_i3_11c_r1_router.py")
i3_11c = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(i3_11c)

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
    EvidenceExecutor, EvidenceBenchmark, save_evidence_benchmark,
    initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output

# I3.12 semantic relation infrastructure
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    generate_i3_12_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.integration import (
    build_evidence_snapshot_with_inferred_relations,
    infer_relations_for_runtime,
)
from hrm_adaptive_memory.executive.semantic_relations.serializer import (
    relation_graph_to_supports_contradicts,
)


# ---------------------------------------------------------------------------
# Snapshot builder abstraction
# ---------------------------------------------------------------------------

def make_gold_snapshot_builder():
    """Return a function that builds snapshots with oracle relations."""
    def builder(runtime, *, prior_actions=(), prior_outcomes=()):
        return build_evidence_snapshot(
            runtime,
            prior_actions=prior_actions,
            prior_outcomes=prior_outcomes,
        )
    return builder


def make_inferred_snapshot_builder(extractor):
    """Return a function that builds snapshots with inferred relations.

    The relation graph is computed once per runtime state and cached
    on the first call for each runtime object identity.
    """
    def builder(runtime, *, prior_actions=(), prior_outcomes=()):
        snap, _graph = build_evidence_snapshot_with_inferred_relations(
            runtime, extractor,
            prior_actions=prior_actions,
            prior_outcomes=prior_outcomes,
        )
        return snap
    return builder


# ---------------------------------------------------------------------------
# Modified trajectory runners (adapted from I3.7e and I3.11c)
# The ONLY change: snapshot builder is parameterized
# ---------------------------------------------------------------------------

def run_trajectory_i3_12(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    mode: str,
    api_key: str,
    fork_label: str,
    snapshot_builder: Callable,
) -> dict[str, Any]:
    """Run a full trajectory with parameterized snapshot builder.

    Identical to i3_7e.run_trajectory except snapshot construction
    is delegated to snapshot_builder.
    """
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
    continuation_outcomes: list[str] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0
    decision_state_log: list[dict[str, Any]] = []

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps

    for step_id in range(max_steps):
        evidence_snapshot = snapshot_builder(
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

        if mode == "MDSG_STATE_WITH_AFFORDANCES":
            ds = packet.get("decision_state_summary", {})
            decision_state_log.append({
                "step": step_id,
                "decision_state": ds.get("decision_state"),
                "live_hypotheses": ds.get("live_hypotheses", []),
                "eliminated_hypotheses": ds.get("eliminated_hypotheses", []),
            })

        user_prompt = i3_7e.evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = fork_label
        backend.pair_id = f"i3_12j:{task.task_id}:{fork_label}:step{step_id}"

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
        continuation_outcomes.append(exec_res.outcome_code)

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
        "terminal_result": terminal_result,
        "terminal_action": terminal_action,
        "terminal_reward": round(terminal_reward, 4),
        "total_action_cost": round(total_action_cost, 4),
        "continuation_actions": continuation_actions,
        "continuation_outcomes": continuation_outcomes,
        "step_costs": step_costs,
        "decision_state_log": decision_state_log,
    }


def run_r1_trajectory_i3_12(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
    fork_label: str,
    snapshot_builder: Callable,
) -> dict[str, Any]:
    """Run R1 hybrid trajectory with parameterized snapshot builder.

    Identical to i3_11c.run_r1_trajectory except snapshot construction
    is delegated to snapshot_builder.
    """
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
    continuation_outcomes: list[str] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    r1_triggered = False
    r1_trigger_step: int | None = None
    r1_trigger_decision_state: str | None = None
    r1_pre_trigger_steps = 0
    r1_post_trigger_steps = 0
    r1_pre_trigger_utility = 0.0
    r1_post_trigger_utility = 0.0

    routing_log: list[dict[str, Any]] = []
    decision_state_log: list[dict[str, Any]] = []

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps
    n_hypotheses = len(task.hypotheses)

    for step_id in range(max_steps):
        evidence_snapshot = snapshot_builder(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        internal_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        internal_summary = internal_m3_packet.get("decision_state_summary", {})
        internal_state = internal_summary.get("decision_state")
        eliminated = internal_summary.get("eliminated_hypotheses", [])

        t2_fires = (
            len(eliminated) == n_hypotheses
            and n_hypotheses > 0
        )

        if not r1_triggered and t2_fires:
            r1_triggered = True
            r1_trigger_step = step_id
            r1_trigger_decision_state = internal_state

        if r1_triggered:
            packet = internal_m3_packet
            system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
            current_rep = "M3"
        else:
            packet = i3_7e.build_baseline_with_affordances_packet(evidence_snapshot)
            system_prompt = i3_7e.BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT
            current_rep = "A1"

        routing_log.append({
            "step": step_id,
            "representation": current_rep,
            "triggered": r1_triggered,
            "t2_fires": t2_fires,
            "decision_state": internal_state,
            "eliminated_hypotheses": eliminated,
        })

        decision_state_log.append({
            "step": step_id,
            "decision_state": internal_state,
            "live_hypotheses": internal_summary.get("live_hypotheses", []),
            "eliminated_hypotheses": eliminated,
            "representation": current_rep,
        })

        user_prompt = i3_7e.evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = fork_label
        backend.pair_id = f"i3_12j:{task.task_id}:{fork_label}:step{step_id}"

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
        continuation_outcomes.append(exec_res.outcome_code)

        if r1_triggered and step_id == r1_trigger_step:
            r1_pre_trigger_steps = step_id
            r1_pre_trigger_utility = realized
        elif not r1_triggered:
            r1_pre_trigger_steps = step_id + 1
            r1_pre_trigger_utility = realized

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

        if r1_triggered:
            r1_post_trigger_steps = steps_taken - r1_trigger_step
        r1_post_trigger_utility = realized - r1_pre_trigger_utility

        if exec_res.terminal:
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
        "step_costs": step_costs,
        "decision_state_log": decision_state_log,
        "r1_triggered": r1_triggered,
        "r1_trigger_step": r1_trigger_step,
        "r1_trigger_decision_state": r1_trigger_decision_state,
        "r1_pre_trigger_steps": r1_pre_trigger_steps,
        "r1_post_trigger_steps": r1_post_trigger_steps,
        "r1_pre_trigger_utility": round(r1_pre_trigger_utility, 4),
        "r1_post_trigger_utility": round(r1_post_trigger_utility, 4),
        "routing_log": routing_log,
    }


# ---------------------------------------------------------------------------
# 6-arm factorial processing
# ---------------------------------------------------------------------------

ARMS = ["A1_GOLD", "M3_GOLD", "R1_GOLD",
        "A1_INFERRED", "M3_INFERRED", "R1_INFERRED"]


def counterbalance_6arm(task_id: str) -> list[str]:
    """Counterbalance arm order using task_id hash."""
    h = hashlib.sha256(task_id.encode()).hexdigest()
    # 6! = 720 permutations
    perms = list(itertools.permutations(ARMS))
    return list(perms[int(h[:8], 16) % len(perms)])


def process_one_task(
    semantic_task,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
    extractor: DeterministicRelationExtractor,
) -> dict[str, Any]:
    """Run all 6 arms for one task."""
    task = semantic_task.evidence_task

    gold_builder = make_gold_snapshot_builder()
    inferred_builder = make_inferred_snapshot_builder(extractor)

    results: dict[str, dict] = {}

    # Run arms in counterbalanced order to reduce positional bias
    arm_order = counterbalance_6arm(task.task_id)

    for arm_id in arm_order:
        rep, condition = arm_id.rsplit("_", 1)

        if condition == "GOLD":
            sb = gold_builder
        else:
            sb = inferred_builder

        if rep == "R1":
            result = run_r1_trajectory_i3_12(
                task=task, budget=budget, utility=utility,
                api_key=api_key, fork_label=arm_id,
                snapshot_builder=sb,
            )
        else:
            mode = "BASELINE_WITH_AFFORDANCES" if rep == "A1" else "MDSG_STATE_WITH_AFFORDANCES"
            result = run_trajectory_i3_12(
                task=task, budget=budget, utility=utility,
                mode=mode, api_key=api_key, fork_label=arm_id,
                snapshot_builder=sb,
            )

        results[arm_id] = result

    # Compute relation graph for provenance
    runtime = initial_evidence_runtime(task, ResourceState(budget))
    _, graph = infer_relations_for_runtime(runtime, extractor)

    # Build summary
    summary = {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": sum(1 for e in task.evidence_items if not e.retrieved),
        "relation_graph_sha256": graph.relation_graph_sha256,
        "arm_order": arm_order,
    }

    for arm_id in ARMS:
        r = results[arm_id]
        rep = arm_id.rsplit("_", 1)[0]
        summary[f"u_{arm_id.lower()}"] = r["realized_utility"]
        summary[f"{arm_id.lower()}_success"] = r["success"]
        summary[f"{arm_id.lower()}_steps"] = r["steps"]
        summary[f"{arm_id.lower()}_backend_errors"] = r["backend_errors"]
        summary[f"{arm_id.lower()}_terminal_action"] = r.get("terminal_action")
        if rep == "R1":
            summary[f"{arm_id.lower()}_triggered"] = r.get("r1_triggered", False)
            summary[f"{arm_id.lower()}_trigger_step"] = r.get("r1_trigger_step")

    # Semantic gaps
    for rep in ["a1", "m3", "r1"]:
        u_gold = summary[f"u_{rep}_gold"]
        u_inf = summary[f"u_{rep}_inferred"]
        summary[f"semantic_gap_{rep}"] = round(u_gold - u_inf, 4)

    # Rescues and breaks (INFERRED vs GOLD within each representation)
    for rep in ["a1", "m3", "r1"]:
        gold_success = summary[f"{rep}_gold_success"]
        inf_success = summary[f"{rep}_inferred_success"]
        summary[f"{rep}_inferred_rescue_vs_gold"] = (not gold_success) and inf_success
        summary[f"{rep}_inferred_break_vs_gold"] = gold_success and (not inf_success)

    # R1 vs A1 within each condition
    for cond in ["gold", "inferred"]:
        summary[f"r1_delta_vs_a1_{cond}"] = round(
            summary[f"u_r1_{cond}"] - summary[f"u_a1_{cond}"], 4)
        summary[f"r1_delta_vs_m3_{cond}"] = round(
            summary[f"u_r1_{cond}"] - summary[f"u_m3_{cond}"], 4)
        summary[f"r1_rescues_vs_a1_{cond}"] = (
            not summary[f"a1_{cond}_success"]) and summary[f"r1_{cond}_success"]
        summary[f"r1_breaks_vs_a1_{cond}"] = (
            summary[f"a1_{cond}_success"]) and (not summary[f"r1_{cond}_success"])

    # Task-level GOLD/INFERRED classification for each representation
    for rep in ["a1", "m3", "r1"]:
        g = summary[f"{rep}_gold_success"]
        i = summary[f"{rep}_inferred_success"]
        if g and i:
            summary[f"{rep}_gold_inferred_class"] = "BOTH_SUCCESS"
        elif not g and not i:
            summary[f"{rep}_gold_inferred_class"] = "BOTH_FAIL"
        elif g and not i:
            summary[f"{rep}_gold_inferred_class"] = "INFERRED_BREAK"
        else:
            summary[f"{rep}_gold_inferred_class"] = "INFERRED_RESCUE"

    # Attach full forks for provenance
    summary["forks"] = {arm: results[arm] for arm in ARMS}

    return summary


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(deltas, n_iterations=10000, seed=42, alpha=0.05):
    """Paired bootstrap CI for the mean of paired differences."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int((alpha / 2) * n_iterations)]
    hi = boot_means[int((1 - alpha / 2) * n_iterations)]
    return round(lo, 4), round(hi, 4)


def one_sided_lcb(deltas, n_iterations=10000, seed=42, alpha=0.05):
    """One-sided lower confidence bound (1-alpha)."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    return round(boot_means[int(alpha * n_iterations)], 4)


def mcnemar(a_success, b_success):
    b = sum(1 for a, m in zip(a_success, b_success) if a and not m)
    c = sum(1 for a, m in zip(a_success, b_success) if not a and m)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    larger = max(b, c)
    tail = sum(comb(n, k) * 0.5**k * 0.5**(n-k) for k in range(larger, n + 1))
    p = min(2 * tail, 1.0)
    return {"b": b, "c": c, "p": round(p, 8)}


def analyze_results(results: list[dict]) -> dict:
    """Compute all primary, secondary, and semantic gap statistics."""
    n = len(results)

    # Extract per-arm arrays
    arms = ARMS
    utils = {arm: [r[f"u_{arm.lower()}"] for r in results] for arm in arms}
    successes = {arm: [r[f"{arm.lower()}_success"] for r in results] for arm in arms}
    steps = {arm: [r[f"{arm.lower()}_steps"] for r in results] for arm in arms}
    errors = {arm: [r[f"{arm.lower()}_backend_errors"] for r in results] for arm in arms}

    report = {
        "n_tasks": n,
        "arms": arms,
        "per_arm": {},
    }

    for arm in arms:
        report["per_arm"][arm] = {
            "mean_utility": round(sum(utils[arm]) / n, 4),
            "success_rate": round(sum(successes[arm]) / n, 4),
            "mean_steps": round(sum(steps[arm]) / n, 2),
            "total_backend_errors": sum(errors[arm]),
        }

    # Primary criterion: LCB_95(U_R1_INFERRED - U_A1_INFERRED) > 0
    r1_inf_minus_a1_inf = [r["u_r1_inferred"] - r["u_a1_inferred"] for r in results]
    lcb_primary, ci_hi_primary = paired_bootstrap_ci(r1_inf_minus_a1_inf)
    lcb_one_sided = one_sided_lcb(r1_inf_minus_a1_inf)

    report["primary_criterion"] = {
        "name": "LCB_95(U_R1_INFERRED - U_A1_INFERRED) > 0",
        "mean_delta": round(sum(r1_inf_minus_a1_inf) / n, 4),
        "two_sided_ci_95": [lcb_primary, ci_hi_primary],
        "one_sided_lcb_95": lcb_one_sided,
        "passes": lcb_one_sided > 0,
    }

    # Secondary: U_R1_INFERRED - U_M3_INFERRED
    r1_inf_minus_m3_inf = [r["u_r1_inferred"] - r["u_m3_inferred"] for r in results]
    lcb_sec, hi_sec = paired_bootstrap_ci(r1_inf_minus_m3_inf)
    report["secondary_r1_vs_m3_inferred"] = {
        "mean_delta": round(sum(r1_inf_minus_m3_inf) / n, 4),
        "two_sided_ci_95": [lcb_sec, hi_sec],
        "one_sided_lcb_95": one_sided_lcb(r1_inf_minus_m3_inf),
    }

    # Same for GOLD condition
    r1_gold_minus_a1_gold = [r["u_r1_gold"] - r["u_a1_gold"] for r in results]
    lcb_g, hi_g = paired_bootstrap_ci(r1_gold_minus_a1_gold)
    report["r1_vs_a1_gold"] = {
        "mean_delta": round(sum(r1_gold_minus_a1_gold) / n, 4),
        "two_sided_ci_95": [lcb_g, hi_g],
        "one_sided_lcb_95": one_sided_lcb(r1_gold_minus_a1_gold),
    }

    r1_gold_minus_m3_gold = [r["u_r1_gold"] - r["u_m3_gold"] for r in results]
    lcb_gm, hi_gm = paired_bootstrap_ci(r1_gold_minus_m3_gold)
    report["r1_vs_m3_gold"] = {
        "mean_delta": round(sum(r1_gold_minus_m3_gold) / n, 4),
        "two_sided_ci_95": [lcb_gm, hi_gm],
        "one_sided_lcb_95": one_sided_lcb(r1_gold_minus_m3_gold),
    }

    # Semantic gaps
    report["semantic_gaps"] = {}
    for rep in ["a1", "m3", "r1"]:
        gaps = [r[f"semantic_gap_{rep}"] for r in results]
        lo, hi = paired_bootstrap_ci(gaps)
        report["semantic_gaps"][rep] = {
            "mean_gap": round(sum(gaps) / n, 4),
            "ci_95": [lo, hi],
        }

    # Task-level GOLD/INFERRED classification
    report["gold_inferred_classification"] = {}
    for rep in ["a1", "m3", "r1"]:
        classes = [r[f"{rep}_gold_inferred_class"] for r in results]
        counts = Counter(classes)
        report["gold_inferred_classification"][rep] = {
            "BOTH_SUCCESS": counts.get("BOTH_SUCCESS", 0),
            "BOTH_FAIL": counts.get("BOTH_FAIL", 0),
            "INFERRED_BREAK": counts.get("INFERRED_BREAK", 0),
            "INFERRED_RESCUE": counts.get("INFERRED_RESCUE", 0),
        }

    # T2 trigger rates
    for cond in ["gold", "inferred"]:
        triggered = [r.get(f"r1_{cond}_triggered", False) for r in results]
        trigger_steps = [r.get(f"r1_{cond}_trigger_step") for r in results
                         if r.get(f"r1_{cond}_triggered")]
        report[f"t2_trigger_{cond}"] = {
            "rate": round(sum(triggered) / n, 4),
            "mean_trigger_step": round(sum(trigger_steps) / len(trigger_steps), 2) if trigger_steps else None,
        }

    # Rescues and breaks
    for cond in ["gold", "inferred"]:
        report[f"rescues_breaks_{cond}"] = {
            "r1_rescues_vs_a1": sum(r[f"r1_rescues_vs_a1_{cond}"] for r in results),
            "r1_breaks_vs_a1": sum(r[f"r1_breaks_vs_a1_{cond}"] for r in results),
        }

    # McNemar tests
    report["mcnemar"] = {
        "r1_inf_vs_a1_inf": mcnemar(
            successes["R1_INFERRED"], successes["A1_INFERRED"]),
        "r1_inf_vs_m3_inf": mcnemar(
            successes["R1_INFERRED"], successes["M3_INFERRED"]),
        "r1_gold_vs_a1_gold": mcnemar(
            successes["R1_GOLD"], successes["A1_GOLD"]),
    }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--n-per-category", type=int, default=22)
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir",
        default="experiments/v2b_i3_12/development/i3_12j_factorial")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.12j: 300-Task A1/M3/R1 Factorial Experiment (GOLD x INFERRED)")
    print("  6 arms: A1_GOLD, M3_GOLD, R1_GOLD, A1_INFERRED, M3_INFERRED, R1_INFERRED")
    print("  Primary: LCB_95(U_R1_INFERRED - U_A1_INFERRED) > 0")
    print("  This is a pipeline-transfer test, NOT a semantic robustness test.")
    print()

    # Generate corpus
    all_tasks = generate_i3_12_corpus(
        n_per_category=args.n_per_category, seed=args.seed)
    tasks = all_tasks[:args.n_tasks]
    print(f"  Generated {len(all_tasks)} tasks, using first {len(tasks)}")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution:")
    for cat in sorted(cats.keys()):
        print(f"    {cat:<40} {cats[cat]}")

    # Extractor
    extractor = DeterministicRelationExtractor()
    print(f"\n  Extractor: v{extractor.identity.extractor_version}")
    print(f"  Extractor SHA256: {extractor.identity.sha256}")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Oracle validation
    executor = EvidenceExecutor()
    all_pass = True
    for st in tasks:
        runtime = initial_evidence_runtime(st.evidence_task, ResourceState(budget))
        current = runtime
        final = None
        for step in st.evidence_task.oracle_resolution_path:
            parts = step.split(":")
            action = DecisionAction(parts[0])
            target = parts[1] if len(parts) > 1 else None
            final = executor.execute(current, action, target_evidence_id=target)
            current = final.runtime
            if final.terminal:
                break
        if not final.task_success:
            all_pass = False
            print(f"  ORACLE FAIL: {st.task_id} ({st.category})")
    print(f"\n  All oracle paths succeed: {all_pass}")
    if not all_pass:
        sys.exit(1)

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    # Save manifest
    manifest = {
        "experiment_id": "i3_12j_factorial_v1",
        "n_tasks": len(tasks),
        "n_arms": 6,
        "arms": ARMS,
        "seed": args.seed,
        "n_per_category": args.n_per_category,
        "extractor_version": extractor.identity.extractor_version,
        "extractor_sha256": extractor.identity.sha256,
        "task_ids": [t.task_id for t in tasks],
        "categories": [t.category for t in tasks],
    }
    manifest_path = output_dir / "manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nProcessing {len(tasks)} tasks x 6 arms = {len(tasks) * 6} trajectories")
    print(f"  with {args.workers} workers...")

    all_results = []
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_task, st, budget, utility, api_key, extractor): st
            for st in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate
                    print(f"  Completed {completed}/{len(tasks)} tasks "
                          f"({rate:.1f}/s, ETA {eta:.0f}s)")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    elapsed = time.time() - t0
    print(f"\nCompleted {len(all_results)} tasks in {elapsed:.1f}s")

    # Sort results by task_id
    all_results.sort(key=lambda r: r["task_id"])

    # Save raw results
    results_path = output_dir / "results_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Raw results: {results_path}")

    # Analyze
    report = analyze_results(all_results)
    report["elapsed_seconds"] = round(elapsed, 1)
    report["extractor_version"] = extractor.identity.extractor_version
    report["extractor_sha256"] = extractor.identity.sha256

    report_path = output_dir / "analysis_v1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"I3.12j Results Summary ({n} tasks)")
    print(f"{'='*70}")

    print(f"\nPer-arm results:")
    print(f"  {'Arm':<20} {'Mean U':>10} {'Success':>10} {'Steps':>8} {'Errors':>8}")
    for arm in ARMS:
        pa = report["per_arm"][arm]
        print(f"  {arm:<20} {pa['mean_utility']:>10.2f} "
              f"{pa['success_rate']:>10.4f} {pa['mean_steps']:>8.2f} "
              f"{pa['total_backend_errors']:>8}")

    print(f"\nPrimary criterion:")
    pc = report["primary_criterion"]
    print(f"  LCB_95(U_R1_INF - U_A1_INF) = {pc['one_sided_lcb_95']}")
    print(f"  Mean delta = {pc['mean_delta']}")
    print(f"  Two-sided CI = [{pc['two_sided_ci_95'][0]}, {pc['two_sided_ci_95'][1]}]")
    print(f"  PASSES: {pc['passes']}")

    print(f"\nSecondary (R1 vs M3, INFERRED):")
    sc = report["secondary_r1_vs_m3_inferred"]
    print(f"  Mean delta = {sc['mean_delta']}")
    print(f"  CI = [{sc['two_sided_ci_95'][0]}, {sc['two_sided_ci_95'][1]}]")

    print(f"\nSemantic gaps (GOLD - INFERRED):")
    for rep in ["a1", "m3", "r1"]:
        sg = report["semantic_gaps"][rep]
        print(f"  {rep.upper()}: mean={sg['mean_gap']:.4f} CI=[{sg['ci_95'][0]}, {sg['ci_95'][1]}]")

    print(f"\nGOLD/INFERRED classification:")
    for rep in ["a1", "m3", "r1"]:
        cl = report["gold_inferred_classification"][rep]
        print(f"  {rep.upper()}: BOTH_SUCCESS={cl['BOTH_SUCCESS']} "
              f"BOTH_FAIL={cl['BOTH_FAIL']} "
              f"INFERRED_BREAK={cl['INFERRED_BREAK']} "
              f"INFERRED_RESCUE={cl['INFERRED_RESCUE']}")

    print(f"\nT2 trigger rates:")
    for cond in ["gold", "inferred"]:
        t = report[f"t2_trigger_{cond}"]
        print(f"  {cond}: rate={t['rate']:.4f} mean_step={t['mean_trigger_step']}")

    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
