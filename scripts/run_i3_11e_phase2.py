#!/usr/bin/env python3
"""I3.11e phase 2: Extended boundary discontinuity diagnosis.

Phase 1 found that R1 and M3 have IDENTICAL state at the trigger step,
and all B0-B5 conditions produce the same action (SEARCH_MORE/VERIFY,
never STOP). The STOP divergence occurs LATER in the trajectory.

This phase 2:
  1. Replays the FULL post-trigger trajectory for R1 and M3 on break tasks
  2. Identifies the exact step where R1 and M3 diverge
  3. Runs the B0-B5 factorial at the divergence step
  4. Tests API nondeterminism by calling the same state 5 times

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_11e_phase2.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
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


CLARIFIED_TERMINAL_M3_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items.
  SEARCH_MORE: Search for additional evidence from other sources.
  VERIFY: Verify the most recently retrieved unverified evidence item.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: A terminal epistemic decision that available evidence is insufficient to resolve the task.
  STOP: Administrative termination without making an epistemic answer-or-defer decision.

You are given the evidence state plus a compact decision-state summary:
  - live_hypotheses: hypotheses that are still viable
  - eliminated_hypotheses: hypotheses that have been ruled out
  - verified_support: evidence IDs that provide verified support
  - verified_contradictions: evidence IDs that provide verified contradiction
  - decision_state: READY_TO_ANSWER, SUPPORTED_BUT_UNRESOLVED, NEEDS_DISCRIMINATION, NEEDS_EVIDENCE, or INSUFFICIENT
  - evidence_status: describes what evidence situation remains unresolved
  - action_affordances: which operations are currently callable
    - can_retrieve: true if RETRIEVE can be called (retrieval budget remains)
    - can_search: true if SEARCH_MORE can be called (search budget remains)
    - can_verify: true if VERIFY can be called (verification budget remains and unverified visible evidence exists)

DECISION STATE MEANINGS (epistemic conditions only, no action recommendations):
  READY_TO_ANSWER: One hypothesis has verified current support, no verified contradiction, no unresolved visible evidence, and no hidden evidence remains.
  SUPPORTED_BUT_UNRESOLVED: One hypothesis currently has verified support, but unresolved evidence remains (unverified visible, hidden, or stale).
  NEEDS_DISCRIMINATION: Multiple hypotheses are viable, or unverified visible evidence could discriminate between them.
  NEEDS_EVIDENCE: No hypothesis has verified support, but evidence-gathering operations are possible.
  INSUFFICIENT: No hypothesis can be resolved with available evidence.

Choose the next action yourself based on the epistemic state and which operations are callable.

OUTPUT FORMAT:
You must respond with a JSON object containing exactly these three fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "A_SHORT_UPPERCASE_REASON_CODE",
  "target_id": null
}

The reason_code must be uppercase with underscores only.

The word json appears in this prompt to satisfy the API requirement."""


BREAK_TASK_IDS = [
    "r1_confirm_v1_0001",
    "r1_confirm_v1_0009",
    "r1_confirm_v1_0043",
    "r1_confirm_v1_0051",
    "r1_confirm_v1_0114",
    "r1_confirm_v1_0129",
]


def replay_full_trajectory(
    task: EvidenceTask,
    budget: ResourceBudget,
    mode: str,
    max_steps: int,
    api_key: str,
) -> dict[str, Any]:
    """Replay a full trajectory (A1 or M3) and record state at every step."""
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(task, resources)

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    step_records: list[dict] = []

    for step_id in range(max_steps):
        snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        if mode == "A1":
            packet = i3_7e.build_baseline_with_affordances_packet(snapshot)
            system_prompt = i3_7e.BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT
        elif mode == "M3":
            packet = i3_7e.build_mdsg_state_with_affordances_packet(snapshot)
            system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown mode: {mode}")

        user_prompt = evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_11e_p2_{mode}"
        backend.pair_id = f"i3_11e_p2:{task.task_id}:{mode}:step{step_id}"

        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
            outcome = decode_output(call_result.raw_output, strict=True)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = i3_7e.FAIL_CLOSED_PROPOSAL
        except Exception:
            proposal = i3_7e.BACKEND_ERROR_PROPOSAL

        action = proposal.action
        target_id = getattr(proposal, "target_id", None)
        action_str = action.value if hasattr(action, "value") else str(action)

        # Record state BEFORE execution
        step_records.append({
            "step": step_id,
            "mode": mode,
            "action_chosen": action_str,
            "snapshot": snapshot,
            "packet": packet,
            "system_prompt": system_prompt,
        })

        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime

        if exec_res.terminal:
            return {
                "mode": mode,
                "steps": step_records,
                "terminal_action": action_str,
                "terminal_outcome": exec_res.outcome_code,
                "task_success": bool(exec_res.task_success),
                "total_steps": step_id + 1,
            }

    return {
        "mode": mode,
        "steps": step_records,
        "terminal_action": None,
        "terminal_outcome": "STEP_LIMIT",
        "task_success": False,
        "total_steps": max_steps,
    }


def replay_r1_full(
    task: EvidenceTask,
    budget: ResourceBudget,
    max_steps: int,
    api_key: str,
) -> dict[str, Any]:
    """Replay R1 hybrid: A1 until T2, then M3 (latched). Record all steps."""
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(task, resources)

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    step_records: list[dict] = []
    r1_triggered = False
    trigger_step = None
    n_hyps = len(task.hypotheses)

    for step_id in range(max_steps):
        snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Internal M3 state for T2 check
        internal_m3 = i3_7e.build_mdsg_state_with_affordances_packet(snapshot)
        internal_summary = internal_m3.get("decision_state_summary", {})
        eliminated = internal_summary.get("eliminated_hypotheses", [])

        t2 = len(eliminated) == n_hyps and n_hyps > 0

        if not r1_triggered and t2:
            r1_triggered = True
            trigger_step = step_id

        if r1_triggered:
            packet = internal_m3
            system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
            rep = "M3"
        else:
            packet = i3_7e.build_baseline_with_affordances_packet(snapshot)
            system_prompt = i3_7e.BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT
            rep = "A1"

        user_prompt = evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_11e_p2_R1"
        backend.pair_id = f"i3_11e_p2:R1:{task.task_id}:step{step_id}"

        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
            outcome = decode_output(call_result.raw_output, strict=True)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = i3_7e.FAIL_CLOSED_PROPOSAL
        except Exception:
            proposal = i3_7e.BACKEND_ERROR_PROPOSAL

        action = proposal.action
        target_id = getattr(proposal, "target_id", None)
        action_str = action.value if hasattr(action, "value") else str(action)

        step_records.append({
            "step": step_id,
            "representation": rep,
            "action_chosen": action_str,
            "snapshot": snapshot,
            "packet": packet,
            "system_prompt": system_prompt,
            "t2_fires": t2,
            "internal_decision_state": internal_summary.get("decision_state"),
            "internal_eliminated": eliminated,
        })

        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime

        if exec_res.terminal:
            return {
                "mode": "R1",
                "steps": step_records,
                "trigger_step": trigger_step,
                "terminal_action": action_str,
                "terminal_outcome": exec_res.outcome_code,
                "task_success": bool(exec_res.task_success),
                "total_steps": step_id + 1,
            }

    return {
        "mode": "R1",
        "steps": step_records,
        "trigger_step": trigger_step,
        "terminal_action": None,
        "terminal_outcome": "STEP_LIMIT",
        "task_success": False,
        "total_steps": max_steps,
    }


def make_single_call(
    task: EvidenceTask,
    snapshot: EvidenceSnapshot,
    system_prompt: str,
    packet: dict,
    condition_label: str,
    api_key: str,
) -> dict[str, Any]:
    """Make a single model call and return the action."""
    user_prompt = evidence_packet_json(packet)

    backend = DeepSeekBackend()
    backend.task_id = task.task_id
    backend.condition = condition_label
    backend.pair_id = f"i3_11e_p2:{task.task_id}:{condition_label}"

    try:
        call_result = backend.generate(
            system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=0.0, max_tokens=2048)
        outcome = decode_output(call_result.raw_output, strict=True)
        if outcome.valid and outcome.proposal:
            action = outcome.proposal.action
            action_str = action.value if hasattr(action, "value") else str(action)
            reason_code = getattr(outcome.proposal, "reason_code", None)
            return {"action": action_str, "reason_code": reason_code, "error": None}
        else:
            return {"action": "FAIL_CLOSED", "reason_code": None, "error": "decode_failed"}
    except Exception as e:
        return {"action": "BACKEND_ERROR", "reason_code": None, "error": str(e)}


def find_divergence_step(r1_traj: dict, m3_traj: dict) -> int | None:
    """Find the first step where R1 and M3 choose different actions."""
    r1_steps = r1_traj["steps"]
    m3_steps = m3_traj["steps"]
    min_len = min(len(r1_steps), len(m3_steps))

    for i in range(min_len):
        if r1_steps[i]["action_chosen"] != m3_steps[i]["action_chosen"]:
            return i
    return None


def states_equal(s1: EvidenceSnapshot, s2: EvidenceSnapshot) -> bool:
    """Check if two snapshots have the same controller-visible state."""
    ev1 = {e.evidence_id: (e.verification_state, e.retrieved) for e in s1.visible_evidence}
    ev2 = {e.evidence_id: (e.verification_state, e.retrieved) for e in s2.visible_evidence}
    return (
        ev1 == ev2
        and s1.hidden_evidence_count == s2.hidden_evidence_count
        and s1.resource_state == s2.resource_state
        and s1.prior_actions == s2.prior_actions
        and s1.prior_outcomes == s2.prior_outcomes
        and s1.can_retrieve == s2.can_retrieve
        and s1.can_search == s2.can_search
        and s1.can_verify == s2.can_verify
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
        default="experiments/v2b_i3_11/development/i3_11e_boundary_diagnosis")
    parser.add_argument("--n-repeats", type=int, default=5,
        help="Number of repeated calls for nondeterminism test")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    corpus = i3_11d.generate_i3_11d_corpus(split="r1_confirm_v1")
    task_map = {t.task_id: t for t in corpus}

    results_path = ROOT / "experiments/v2b_i3_11/development/i3_11d_r1_confirm/r1_confirm_v1.jsonl"
    with open(results_path) as f:
        all_results = [json.loads(line) for line in f]

    print(f"I3.11e Phase 2: Extended Boundary Discontinuity Diagnosis")
    print(f"  Break tasks: {len(BREAK_TASK_IDS)}")
    print(f"  Nondeterminism repeats: {args.n_repeats}")
    print()

    all_diagnosis = []

    for task_id in BREAK_TASK_IDS:
        task = task_map[task_id]
        r_result = next(r for r in all_results if r["task_id"] == task_id)

        print(f"\n{'='*70}")
        print(f"Task {task_id} ({r_result['category']})")
        print(f"  Original R1: {r_result['fork_r1']['continuation_actions']}")
        print(f"  Original M3: {r_result['fork_m3']['continuation_actions']}")

        # Replay full R1 and M3 trajectories
        print(f"  Replaying R1...", end=" ")
        r1_traj = replay_r1_full(task, budget, 24, api_key)
        print(f"→ {r1_traj['terminal_action']} ({r1_traj['total_steps']} steps, success={r1_traj['task_success']})")

        print(f"  Replaying M3...", end=" ")
        m3_traj = replay_full_trajectory(task, budget, "M3", 24, api_key)
        print(f"→ {m3_traj['terminal_action']} ({m3_traj['total_steps']} steps, success={m3_traj['task_success']})")

        # Find divergence step
        div_step = find_divergence_step(r1_traj, m3_traj)
        print(f"  Divergence step: {div_step}")

        if div_step is None:
            print(f"  No divergence found — trajectories identical")
            all_diagnosis.append({
                "task_id": task_id,
                "category": r_result["category"],
                "r1_replay_terminal": r1_traj["terminal_action"],
                "m3_replay_terminal": m3_traj["terminal_action"],
                "r1_replay_success": r1_traj["task_success"],
                "m3_replay_success": m3_traj["task_success"],
                "divergence_step": None,
                "finding": "No divergence on replay — original STOP may be API nondeterminism",
            })
            continue

        # Compare states at divergence step
        r1_step = r1_traj["steps"][div_step]
        m3_step = m3_traj["steps"][div_step]

        r1_snap = r1_step["snapshot"]
        m3_snap = m3_step["snapshot"]

        states_match = states_equal(r1_snap, m3_snap)
        print(f"  States at divergence step match: {states_match}")
        print(f"  R1 action: {r1_step['action_chosen']}, M3 action: {m3_step['action_chosen']}")
        print(f"  R1 prior_actions: {r1_snap.prior_actions}")
        print(f"  M3 prior_actions: {m3_snap.prior_actions}")
        print(f"  R1 representation: {r1_step.get('representation', 'M3')}")
        print(f"  M3 representation: M3")

        # If states match, this is likely nondeterminism
        # Run nondeterminism test: call the same state N times
        if states_match:
            print(f"\n  States MATCH at divergence — testing nondeterminism ({args.n_repeats} repeats)...")
            repeats = []
            for i in range(args.n_repeats):
                result = make_single_call(
                    task, r1_snap,
                    i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
                    r1_step["packet"],
                    condition_label=f"nondet_repeat_{i}",
                    api_key=api_key,
                )
                repeats.append(result["action"])
                print(f"    Repeat {i}: {result['action']}")

            action_dist = dict(Counter(repeats))
            print(f"  Action distribution: {action_dist}")

            # Also test clarified terminal at this step
            print(f"  Testing clarified terminal at divergence step...", end=" ")
            clarified = make_single_call(
                task, r1_snap,
                CLARIFIED_TERMINAL_M3_PROMPT,
                r1_step["packet"],
                condition_label="clarified_at_divergence",
                api_key=api_key,
            )
            print(f"→ {clarified['action']}")

            all_diagnosis.append({
                "task_id": task_id,
                "category": r_result["category"],
                "r1_replay_terminal": r1_traj["terminal_action"],
                "m3_replay_terminal": m3_traj["terminal_action"],
                "r1_replay_success": r1_traj["task_success"],
                "m3_replay_success": m3_traj["task_success"],
                "divergence_step": div_step,
                "states_match_at_divergence": True,
                "r1_action_at_divergence": r1_step["action_chosen"],
                "m3_action_at_divergence": m3_step["action_chosen"],
                "nondeterminism_test": action_dist,
                "clarified_terminal_at_divergence": clarified["action"],
                "finding": "States identical at divergence step. STOP vs DEFER is likely API nondeterminism at temperature=0.",
            })
        else:
            # States differ — find what differs
            ev1_ids = set(e.evidence_id for e in r1_snap.visible_evidence)
            ev2_ids = set(e.evidence_id for e in m3_snap.visible_evidence)
            print(f"  States DIFFER at divergence:")
            print(f"    R1 evidence: {ev1_ids}")
            print(f"    M3 evidence: {ev2_ids}")
            print(f"    R1 prior_actions: {r1_snap.prior_actions}")
            print(f"    M3 prior_actions: {m3_snap.prior_actions}")

            # Run factorial at divergence step
            print(f"\n  Running factorial at divergence step...")

            # B0: R1 state + R1 history + M3 prompt
            b0 = make_single_call(task, r1_snap,
                i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
                r1_step["packet"], "B0_div", api_key)
            # B5: M3 state + M3 history + M3 prompt
            b5 = make_single_call(task, m3_snap,
                i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
                m3_step["packet"], "B5_div", api_key)
            # B3: R1 state + clarified terminal
            b3 = make_single_call(task, r1_snap,
                CLARIFIED_TERMINAL_M3_PROMPT,
                r1_step["packet"], "B3_div", api_key)

            print(f"    B0 (R1 state + M3 prompt): {b0['action']}")
            print(f"    B3 (R1 state + clarified): {b3['action']}")
            print(f"    B5 (M3 state + M3 prompt): {b5['action']}")

            all_diagnosis.append({
                "task_id": task_id,
                "category": r_result["category"],
                "r1_replay_terminal": r1_traj["terminal_action"],
                "m3_replay_terminal": m3_traj["terminal_action"],
                "r1_replay_success": r1_traj["task_success"],
                "m3_replay_success": m3_traj["task_success"],
                "divergence_step": div_step,
                "states_match_at_divergence": False,
                "r1_action_at_divergence": r1_step["action_chosen"],
                "m3_action_at_divergence": m3_step["action_chosen"],
                "B0_at_divergence": b0["action"],
                "B3_at_divergence": b3["action"],
                "B5_at_divergence": b5["action"],
                "finding": "States differ at divergence step. A1 prefix changed the trajectory.",
            })

    # Save results
    results_path = output_dir / "diagnosis_v2.jsonl"
    with open(results_path, "w") as f:
        for r in all_diagnosis:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nSaved: {results_path}")

    # Summary
    print(f"\n{'='*82}")
    print("I3.11e PHASE 2: EXTENDED DIAGNOSIS SUMMARY")
    print(f"{'='*82}")

    n_match = sum(1 for r in all_diagnosis if r.get("states_match_at_divergence"))
    n_differ = sum(1 for r in all_diagnosis if r.get("states_match_at_divergence") is False)
    n_no_div = sum(1 for r in all_diagnosis if r.get("divergence_step") is None)

    print(f"\n  Tasks with no divergence on replay: {n_no_div}")
    print(f"  Tasks with states MATCHING at divergence: {n_match}")
    print(f"  Tasks with states DIFFERING at divergence: {n_differ}")

    print(f"\n  Per-task results:")
    for r in all_diagnosis:
        div = r.get("divergence_step")
        match = r.get("states_match_at_divergence")
        finding = r.get("finding", "")[:80]
        print(f"    {r['task_id']}: div_step={div}, states_match={match}")
        print(f"      {finding}")
        if "nondeterminism_test" in r:
            print(f"      Nondeterminism: {r['nondeterminism_test']}")
        if "clarified_terminal_at_divergence" in r:
            print(f"      Clarified terminal: {r['clarified_terminal_at_divergence']}")

    # Overall conclusion
    print(f"\n{'='*82}")
    print("OVERALL DIAGNOSTIC CONCLUSION")
    print(f"{'='*82}")

    if n_no_div == len(all_diagnosis):
        print(f"\n  ALL tasks: No divergence on replay.")
        print(f"  The original STOP-vs-DEFER breaks did NOT reproduce.")
        print(f"  This strongly suggests API NONDETERMINISM at temperature=0.")
        print(f"  The STOP behavior is not a systematic hybrid-trajectory effect.")
    elif n_match > 0:
        print(f"\n  {n_match} tasks: States IDENTICAL at divergence step.")
        print(f"  STOP vs DEFER occurs with identical controller-visible input.")
        print(f"  This is API nondeterminism, not a systematic boundary effect.")
        if n_differ > 0:
            print(f"\n  {n_differ} tasks: States DIFFER at divergence step.")
            print(f"  These may have a systematic component.")

    summary = {
        "schema": "DAPH_V2B_I3_11E_DIAGNOSIS_V2_SUMMARY",
        "n_tasks": len(all_diagnosis),
        "n_no_divergence": n_no_div,
        "n_states_matching": n_match,
        "n_states_differing": n_differ,
        "per_task": all_diagnosis,
    }

    summary_path = output_dir / "diagnosis_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
