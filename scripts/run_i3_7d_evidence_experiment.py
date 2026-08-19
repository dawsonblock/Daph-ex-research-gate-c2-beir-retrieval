#!/usr/bin/env python3
"""I3.7d — Semantic-Evidence Development Experiment.

Three arms, no action override, on the new evidence-bearing benchmark:

  A = baseline model + semantic evidence (no resolution context)
  B = baseline model + current resolution context
  C = baseline model + emphasized resolution context

Primary scientific question:
  P(success | ResolutionContext) > P(success | Baseline)

Secondary mechanism chain:
  context -> correct discriminator -> correct evidence operation
  -> evidence acquired -> hypothesis update -> answer condition -> success

For every task, log the information conversion pipeline:
  discriminator_selected_correctly
  target_evidence_retrieved
  target_evidence_verified
  hypothesis_eliminated
  answer_condition_satisfied
  task_success

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_7d_evidence_experiment.py \\
        --n-tasks 50 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)

from hrm_adaptive_memory.executive.evidence_benchmark import (
    load_evidence_benchmark, EvidenceBenchmark, EvidenceTask,
    EvidenceExecutor, initial_evidence_runtime,
    build_evidence_snapshot, serialize_evidence_snapshot,
    assert_no_evidence_leakage,
)
from hrm_adaptive_memory.executive.evidence_benchmark.serializer import (
    evidence_packet_json,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)

# System prompts
BASELINE_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items. Use this when you need more evidence.
  SEARCH_MORE: Search for additional evidence from other sources. Use this when retrieved evidence is insufficient.
  VERIFY: Verify the most recently retrieved unverified evidence item. This changes its verification_state from UNVERIFIED to SUFFICIENT, FALSIFIED, or STALE.
  REASON_MORE: Complete reasoning about current evidence. Use this when reasoning_required.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: Give up due to insufficient evidence. Use when evidence cannot be obtained or verified.
  STOP: Stop without answering.

EVIDENCE STATES:
  UNVERIFIED: Evidence has not been checked yet. Use VERIFY to check it.
  SUFFICIENT: Evidence is verified and supports its hypothesis.
  FALSIFIED: Evidence is verified and found to be false.
  STALE: Evidence is outdated. Use SEARCH_MORE to find current evidence.
  MISSING: Evidence is insufficient and cannot be verified.

DECISION PROCESS:
  1. If you have SUFFICIENT, CURRENT evidence supporting a hypothesis with no contradictions, ANSWER.
  2. If evidence is UNVERIFIED, use VERIFY to check it.
  3. If you need more evidence, use RETRIEVE or SEARCH_MORE.
  4. If evidence is STALE, use SEARCH_MORE to find current evidence.
  5. If you cannot obtain sufficient evidence, DEFER.
  6. Do NOT immediately DEFER if there are hidden evidence items or unverified evidence. Investigate first.

OUTPUT FORMAT:
You must respond with a JSON object containing exactly these three fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "A_SHORT_UPPERCASE_REASON_CODE",
  "target_id": null
}

The reason_code must be uppercase with underscores only (e.g. NEED_MORE_EVIDENCE, VERIFYING_EVIDENCE, SUFFICIENT_EVIDENCE, INSUFFICIENT_EVIDENCE).

The word json appears in this prompt to satisfy the API requirement."""

RESOLUTION_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items. Use this when you need more evidence.
  SEARCH_MORE: Search for additional evidence from other sources. Use this when retrieved evidence is insufficient.
  VERIFY: Verify the most recently retrieved unverified evidence item. This changes its verification_state from UNVERIFIED to SUFFICIENT, FALSIFIED, or STALE.
  REASON_MORE: Complete reasoning about current evidence. Use this when reasoning_required.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: Give up due to insufficient evidence. Use when evidence cannot be obtained or verified.
  STOP: Stop without answering.

EVIDENCE STATES:
  UNVERIFIED: Evidence has not been checked yet. Use VERIFY to check it.
  SUFFICIENT: Evidence is verified and supports its hypothesis.
  FALSIFIED: Evidence is verified and found to be false.
  STALE: Evidence is outdated. Use SEARCH_MORE to find current evidence.
  MISSING: Evidence is insufficient and cannot be verified.

You are given a resolution assistance frame with:
  - hypotheses: competing explanations with answer actions
  - visible_evidence: proposition-level evidence with hypothesis relationships
  - hidden_evidence_count: how many evidence items remain hidden
  - terminal_decision_rule: explicit conditions for each terminal action
  - execution_plan: bounded steps with decision consequences

DECISION PROCESS:
  1. Check the terminal_decision_rule. If any condition is met, take the corresponding terminal action NOW.
  2. If evidence is UNVERIFIED, use VERIFY to check it.
  3. If you need more evidence and hidden_evidence_count > 0, use RETRIEVE.
  4. If retrieved evidence is insufficient, use SEARCH_MORE.
  5. If evidence is STALE, use SEARCH_MORE to find current evidence.
  6. If you cannot obtain sufficient evidence after investigating, DEFER.
  7. Do NOT immediately DEFER if there are hidden evidence items or unverified evidence. Investigate first.

CRITICAL: After each action, check the terminal_decision_rule. If any condition is met, you MUST take the corresponding terminal action immediately. Do not continue searching if an answer condition is already satisfied.

OUTPUT FORMAT:
You must respond with a JSON object containing exactly these three fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "A_SHORT_UPPERCASE_REASON_CODE",
  "target_id": null
}

The reason_code must be uppercase with underscores only (e.g. NEED_MORE_EVIDENCE, VERIFYING_EVIDENCE, SUFFICIENT_EVIDENCE, INSUFFICIENT_EVIDENCE).

The word json appears in this prompt to satisfy the API requirement."""


def counterbalance_order(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    perm_idx = int(h[:8], 16) % 6  # 3! = 6
    perms = list(itertools.permutations(["A", "B", "C"]))
    return list(perms[perm_idx])


def build_resolution_packet(
    snapshot: Any,
    evidence_snapshot: Any,
    prior_actions: tuple[str, ...],
    prior_outcomes: tuple[str, ...],
) -> dict:
    """Build a resolution packet from the evidence snapshot.

    This constructs a resolution-style packet with:
      - hypotheses from the evidence task
      - visible evidence with proposition-level relationships
      - terminal_decision_rule with explicit answer conditions
      - execution plan with answer-condition-check step
    """
    # Build answer conditions from hypotheses
    rules = []
    for h in evidence_snapshot.hypotheses:
        # Condition: this hypothesis has sufficient verified current support
        # and no verified contradiction
        has_support = any(
            e.verification_state == VerificationState.SUFFICIENT
            and e.temporal_status != TemporalStatus.STALE
            and h.hypothesis_id in e.supports
            for e in evidence_snapshot.visible_evidence
        )
        has_contradiction = any(
            e.verification_state == VerificationState.SUFFICIENT
            and e.temporal_status != TemporalStatus.STALE
            and h.hypothesis_id in e.contradicts
            for e in evidence_snapshot.visible_evidence
        )

        if has_support and not has_contradiction:
            rules.append({
                "if": f"{h.hypothesis_id} has sufficient verified current support and no verified contradiction",
                "then": h.answer_action.value,
                "use": h.answer_payload,
                "hypothesis": h.hypothesis_id,
            })

    # Build evidence map
    visible_evidence = []
    for e in evidence_snapshot.visible_evidence:
        ev_dict = e.as_dict()
        ev_dict.pop("verify_result", None)  # never expose
        visible_evidence.append(ev_dict)

    packet = {
        "schema": "DAPH_V2B_I3_7_RESOLUTION_PACKET_V1",
        "task_id": evidence_snapshot.task_id,
        "task_summary": evidence_snapshot.task_summary,
        "hypotheses": [h.as_dict() for h in evidence_snapshot.hypotheses],
        "visible_evidence": visible_evidence,
        "hidden_evidence_count": evidence_snapshot.hidden_evidence_count,
        "verified_count": evidence_snapshot.verified_count,
        "supporting_count": evidence_snapshot.supporting_count,
        "contradicting_count": evidence_snapshot.contradicting_count,
        "searched": evidence_snapshot.searched,
        "reasoning_complete": evidence_snapshot.reasoning_complete,
        "resource_state": dict(evidence_snapshot.resource_state),
        "prior_actions": list(evidence_snapshot.prior_actions),
        "prior_outcomes": list(evidence_snapshot.prior_outcomes),
        "terminal_decision_rule": {
            "instruction": "AFTER each action, check these conditions. If ANY condition is met, take the corresponding terminal action immediately.",
            "rules": rules,
            "defer_if_none_met": "If no answer condition can be satisfied after exhausting evidence operations, DEFER.",
            "critical": "Do not continue searching if an answer condition is already satisfied. Check first.",
        },
        "execution_plan": [
            {
                "operation": "check_answer_conditions",
                "target": "all rules in terminal_decision_rule",
                "purpose": "determine if any hypothesis has sufficient verified support to terminate",
                "decision_consequence": "if any condition met: take terminal action NOW; else: proceed",
            },
            {
                "operation": "retrieve_hidden_evidence",
                "target": f"{evidence_snapshot.hidden_evidence_count} hidden evidence items",
                "purpose": "expose additional evidence that may discriminate between hypotheses",
                "decision_consequence": "new evidence: update hypothesis assessment; none: consider DEFER",
            },
            {
                "operation": "verify_evidence",
                "target": "unverified retrieved evidence items",
                "purpose": "establish verification state of evidence",
                "decision_consequence": "SUFFICIENT: check answer conditions; FALSIFIED: eliminate supported hypothesis; STALE: search for current evidence",
            },
        ],
    }

    assert_no_evidence_leakage(packet)
    return packet


def run_trajectory(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    mode: str,  # "BASELINE", "RESOLUTION", "RESOLUTION_EMPHASIZED"
    api_key: str,
    fork_label: str,
) -> dict[str, Any]:
    """Run a full trajectory on an evidence-bearing task."""
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
    evidence_exposed_log: list[tuple[str, ...]] = []
    evidence_verified_log: list[tuple[str, ...]] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    # Track information conversion pipeline
    target_evidence_retrieved = False
    target_evidence_verified = False
    hypothesis_eliminated = False
    answer_condition_satisfied = False

    # Determine target evidence from oracle path (for logging, not shown to model)
    oracle_evidence_ids = set()
    for step in task.oracle_resolution_path:
        parts = step.split(":")
        if len(parts) > 1:
            oracle_evidence_ids.add(parts[1])

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []

    max_steps = budget.max_executive_steps

    for step_id in range(max_steps):
        # Build snapshot
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Build packet based on mode
        if mode == "BASELINE":
            packet = serialize_evidence_snapshot(evidence_snapshot)
            system_prompt = BASELINE_SYSTEM_PROMPT
        else:
            packet = build_resolution_packet(
                None, evidence_snapshot,
                tuple(prior_actions), tuple(prior_outcomes),
            )
            system_prompt = RESOLUTION_SYSTEM_PROMPT

        user_prompt = evidence_packet_json(packet)

        # Call model
        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_7d_{mode}"
        backend.pair_id = f"i3_7d:{task.task_id}:{fork_label}:step{step_id}"

        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
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
        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action)
        resources_after = exec_res.runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost
        total_action_cost += step_cost
        step_costs.append(round(step_cost, 4))

        action_str = action.value if hasattr(action, "value") else str(action)
        continuation_actions.append(action_str)
        continuation_outcomes.append(exec_res.outcome_code)
        evidence_exposed_log.append(exec_res.evidence_exposed)
        evidence_verified_log.append(exec_res.evidence_verified)

        # Track information conversion
        for eid in exec_res.evidence_exposed:
            if eid in oracle_evidence_ids:
                target_evidence_retrieved = True
        for eid in exec_res.evidence_verified:
            if eid in oracle_evidence_ids:
                target_evidence_verified = True

        # Check if any hypothesis was eliminated (contradicted)
        for e in exec_res.runtime.evidence:
            if (e.verification_state == VerificationState.FALSIFIED
                    and e.retrieved and e.contradicts):
                hypothesis_eliminated = True

        # Check if answer condition could be satisfied
        for e in exec_res.runtime.evidence:
            if (e.verification_state == VerificationState.SUFFICIENT
                    and e.temporal_status != TemporalStatus.STALE
                    and e.retrieved and e.supports):
                answer_condition_satisfied = True

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
        "evidence_exposed_log": [list(e) for e in evidence_exposed_log],
        "evidence_verified_log": [list(e) for e in evidence_verified_log],
        "step_costs": step_costs,
        # Information conversion pipeline
        "target_evidence_retrieved": target_evidence_retrieved,
        "target_evidence_verified": target_evidence_verified,
        "hypothesis_eliminated": hypothesis_eliminated,
        "answer_condition_satisfied": answer_condition_satisfied,
    }


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    """Process one task across three arms: A, B, C."""
    fork_order = counterbalance_order(task.task_id)

    arm_configs = {
        "A": {"mode": "BASELINE"},
        "B": {"mode": "RESOLUTION"},
        "C": {"mode": "RESOLUTION_EMPHASIZED"},
    }

    results: dict[str, dict] = {}
    for arm_id in fork_order:
        cfg = arm_configs[arm_id]
        results[arm_id] = run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=cfg["mode"], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    u_a = results["A"]["realized_utility"]
    u_b = results["B"]["realized_utility"]
    u_c = results["C"]["realized_utility"]

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "oracle_path": list(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a": u_a, "u_b": u_b, "u_c": u_c,
        "b_gain": round(u_b - u_a, 4),
        "c_gain": round(u_c - u_a, 4),
        "c_vs_b": round(u_c - u_b, 4),
        "a_success": results["A"]["success"],
        "b_success": results["B"]["success"],
        "c_success": results["C"]["success"],
        # Information conversion pipeline for each arm
        "a_pipeline": {
            "target_evidence_retrieved": results["A"]["target_evidence_retrieved"],
            "target_evidence_verified": results["A"]["target_evidence_verified"],
            "hypothesis_eliminated": results["A"]["hypothesis_eliminated"],
            "answer_condition_satisfied": results["A"]["answer_condition_satisfied"],
        },
        "b_pipeline": {
            "target_evidence_retrieved": results["B"]["target_evidence_retrieved"],
            "target_evidence_verified": results["B"]["target_evidence_verified"],
            "hypothesis_eliminated": results["B"]["hypothesis_eliminated"],
            "answer_condition_satisfied": results["B"]["answer_condition_satisfied"],
        },
        "c_pipeline": {
            "target_evidence_retrieved": results["C"]["target_evidence_retrieved"],
            "target_evidence_verified": results["C"]["target_evidence_verified"],
            "hypothesis_eliminated": results["C"]["hypothesis_eliminated"],
            "answer_condition_satisfied": results["C"]["answer_condition_satisfied"],
        },
        "fork_a": results["A"],
        "fork_b": results["B"],
        "fork_c": results["C"],
    }


def main():
    parser = argparse.ArgumentParser(description="I3.7d evidence experiment")
    parser.add_argument(
        "--benchmark",
        default="experiments/v2b_i3_7/manifests/i3_7_evidence_benchmark_v1.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_7/development/i3_7d",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evidence benchmark from {args.benchmark}...")
    benchmark = load_evidence_benchmark(args.benchmark)
    tasks = benchmark.tasks[:args.n_tasks]
    budget = benchmark.budget_profiles["STANDARD"]
    print(f"  Loaded {len(tasks)} tasks")

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks with {args.workers} workers...")
    print(f"  A=baseline+evidence, B=resolution context, C=emphasized resolution")

    all_results: list[dict[str, Any]] = []
    completed = 0

    def task_wrapper(task):
        return process_one_task(task, budget, utility, api_key)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task_wrapper, task): task for task in tasks}
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

    results_path = output_dir / "evidence_experiment_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # Compute summary
    n = len(all_results)
    if n == 0:
        print("No tasks completed!")
        return

    u_a_mean = sum(r["u_a"] for r in all_results) / n
    u_b_mean = sum(r["u_b"] for r in all_results) / n
    u_c_mean = sum(r["u_c"] for r in all_results) / n

    a_success = sum(1 for r in all_results if r["a_success"])
    b_success = sum(1 for r in all_results if r["b_success"])
    c_success = sum(1 for r in all_results if r["c_success"])

    # Rescues and breaks
    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    b_classes = Counter(classify(r["a_success"], r["b_success"]) for r in all_results)
    c_classes = Counter(classify(r["a_success"], r["c_success"]) for r in all_results)

    b_rescues = b_classes.get("RESCUE", 0)
    b_breaks = b_classes.get("BREAK", 0)
    c_rescues = c_classes.get("RESCUE", 0)
    c_breaks = c_classes.get("BREAK", 0)

    # Information conversion pipeline
    a_pipe = {
        "target_retrieved": sum(1 for r in all_results if r["a_pipeline"]["target_evidence_retrieved"]),
        "target_verified": sum(1 for r in all_results if r["a_pipeline"]["target_evidence_verified"]),
        "hypothesis_eliminated": sum(1 for r in all_results if r["a_pipeline"]["hypothesis_eliminated"]),
        "answer_condition_satisfied": sum(1 for r in all_results if r["a_pipeline"]["answer_condition_satisfied"]),
    }
    b_pipe = {
        "target_retrieved": sum(1 for r in all_results if r["b_pipeline"]["target_evidence_retrieved"]),
        "target_verified": sum(1 for r in all_results if r["b_pipeline"]["target_evidence_verified"]),
        "hypothesis_eliminated": sum(1 for r in all_results if r["b_pipeline"]["hypothesis_eliminated"]),
        "answer_condition_satisfied": sum(1 for r in all_results if r["b_pipeline"]["answer_condition_satisfied"]),
    }
    c_pipe = {
        "target_retrieved": sum(1 for r in all_results if r["c_pipeline"]["target_evidence_retrieved"]),
        "target_verified": sum(1 for r in all_results if r["c_pipeline"]["target_evidence_verified"]),
        "hypothesis_eliminated": sum(1 for r in all_results if r["c_pipeline"]["hypothesis_eliminated"]),
        "answer_condition_satisfied": sum(1 for r in all_results if r["c_pipeline"]["answer_condition_satisfied"]),
    }

    # Gates
    gates = {
        "G1_BREAK_C_zero": c_breaks == 0,
        "G2_RESCUE_C_ge_1": c_rescues >= 1,
        "G3_RESCUE_gt_BREAK": c_rescues > c_breaks,
        "G4_C_success_gt_A": c_success > a_success,
        "G5_C_success_gt_B": c_success > b_success,
    }

    summary = {
        "schema": "DAPH_V2B_I3_7D_EVIDENCE_EXPERIMENT_V1",
        "n_tasks": n,
        "arms": {
            "A": "baseline model + semantic evidence",
            "B": "baseline model + resolution context",
            "C": "baseline model + emphasized resolution context",
        },
        "utility": {
            "mean_u_a": round(u_a_mean, 4),
            "mean_u_b": round(u_b_mean, 4),
            "mean_u_c": round(u_c_mean, 4),
        },
        "success": {
            "A": f"{a_success}/{n}",
            "B": f"{b_success}/{n}",
            "C": f"{c_success}/{n}",
        },
        "classification": {
            "A_vs_B": dict(b_classes),
            "A_vs_C": dict(c_classes),
        },
        "rescues_and_breaks": {
            "B_rescues": b_rescues, "B_breaks": b_breaks,
            "C_rescues": c_rescues, "C_breaks": c_breaks,
        },
        "information_conversion_pipeline": {
            "A": a_pipe, "B": b_pipe, "C": c_pipe,
        },
        "gates": gates,
    }

    summary_path = output_dir / "evidence_experiment_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*78}")
    print("I3.7d SEMANTIC-EVIDENCE EXPERIMENT SUMMARY")
    print(f"{'='*78}")
    print(f"  Tasks:                   {n}")
    print(f"\n  Mean utility:")
    print(f"    A (baseline):          {u_a_mean:+.4f}")
    print(f"    B (resolution):        {u_b_mean:+.4f}")
    print(f"    C (emphasized):        {u_c_mean:+.4f}")
    print(f"\n  Success rates:")
    print(f"    A: {a_success}/{n} ({a_success/n:.1%})")
    print(f"    B: {b_success}/{n} ({b_success/n:.1%})")
    print(f"    C: {c_success}/{n} ({c_success/n:.1%})")
    print(f"\n  Classification (A vs B):")
    for cls, cnt in b_classes.most_common():
        print(f"    {cls}: {cnt}")
    print(f"\n  Classification (A vs C):")
    for cls, cnt in c_classes.most_common():
        print(f"    {cls}: {cnt}")
    print(f"\n  Rescues and breaks:")
    print(f"    B: rescues={b_rescues}, breaks={b_breaks}")
    print(f"    C: rescues={c_rescues}, breaks={c_breaks}")
    print(f"\n  Information conversion pipeline:")
    print(f"    A: target_ret={a_pipe['target_retrieved']}, target_ver={a_pipe['target_verified']}, hyp_elim={a_pipe['hypothesis_eliminated']}, ans_cond={a_pipe['answer_condition_satisfied']}")
    print(f"    B: target_ret={b_pipe['target_retrieved']}, target_ver={b_pipe['target_verified']}, hyp_elim={b_pipe['hypothesis_eliminated']}, ans_cond={b_pipe['answer_condition_satisfied']}")
    print(f"    C: target_ret={c_pipe['target_retrieved']}, target_ver={c_pipe['target_verified']}, hyp_elim={c_pipe['hypothesis_eliminated']}, ans_cond={c_pipe['answer_condition_satisfied']}")
    print(f"\n  GATES:")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate}: {status}")
    print(f"\n  Total gates passed: {sum(gates.values())}/{len(gates)}")

    # Show rescue details
    c_rescue_details = [r for r in all_results if classify(r["a_success"], r["c_success"]) == "RESCUE"]
    if c_rescue_details:
        print(f"\n  C RESCUE DETAILS:")
        for r in c_rescue_details:
            print(f"    {r['task_id']}: category={r['category']}")
            print(f"      A: U={r['u_a']:+.2f}, actions={r['fork_a']['continuation_actions']}")
            print(f"      C: U={r['u_c']:+.2f}, actions={r['fork_c']['continuation_actions']}")
            print(f"      C pipeline: {r['c_pipeline']}")


if __name__ == "__main__":
    main()
