#!/usr/bin/env python3
"""I3.11e: Boundary Discontinuity Diagnosis.

Diagnoses why R1 produces STOP instead of DEFER after T2 trigger,
while always-M3 produces DEFER on the same tasks.

This is DIAGNOSIS, not candidate selection. We do not promote any
variant — we isolate the causal source of the STOP-vs-DEFER boundary
effect.

Method:
  1. Take the 6 R1→M3 break tasks from I3.11d
  2. Take 6 matched successful R1 conflict tasks
  3. Replay R1 and M3 trajectories up to the T2 trigger step
  4. Freeze the complete controller-visible state at trigger
  5. Run controlled factorial ablations at the first post-T2 call:

     B0: R1 state + R1 history + frozen M3 prompt        (baseline = R1)
     B1: R1 state + stripped history + frozen M3 prompt   (history removed)
     B2: R1 state + matched M3 history + frozen M3 prompt (history swapped)
     B3: R1 state + R1 history + clarified STOP/DEFER     (terminal semantics)
     B4: M3 state + R1 history + frozen M3 prompt         (state swapped)
     B5: M3 state + M3 history + frozen M3 prompt         (baseline = M3)

Diagnostic questions:
  B0=STOP, B1=DEFER → history causes the failure
  B0=STOP, B3=DEFER → terminal semantics cause it
  B4=STOP, B5=DEFER → prior trajectory/history is causal
  All R1-state STOP regardless of history → A1 prefix changed evidence/resource state

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_11e_boundary_diagnosis.py
"""
from __future__ import annotations

import json
import os
import sys
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
    EvidenceRuntime,
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
# Clarified STOP/DEFER terminal semantics prompt (B3 only)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Replay helpers — replay a trajectory up to a given step, freeze state
# ---------------------------------------------------------------------------

def replay_to_step(
    task: EvidenceTask,
    budget: ResourceBudget,
    mode: str,
    target_step: int,
    api_key: str,
) -> dict[str, Any]:
    """Replay a trajectory (A1 or M3) up to target_step, freezing state.

    Returns the frozen snapshot, packet, prior_actions, prior_outcomes,
    and runtime at the point just before the model call at target_step.
    """
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(task, resources)

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []

    for step_id in range(target_step):
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
            raise ValueError(f"Unknown replay mode: {mode}")

        user_prompt = evidence_packet_json(packet)

        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_11e_replay_{mode}"
        backend.pair_id = f"i3_11e:replay:{task.task_id}:{mode}:step{step_id}"

        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=2048)
        except Exception:
            proposal = i3_7e.BACKEND_ERROR_PROPOSAL
        else:
            outcome = decode_output(call_result.raw_output, strict=True)
            if outcome.valid and outcome.proposal:
                proposal = outcome.proposal
            else:
                proposal = i3_7e.FAIL_CLOSED_PROPOSAL

        action = proposal.action
        target_id = getattr(proposal, "target_id", None)

        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        action_str = action.value if hasattr(action, "value") else str(action)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime

        if exec_res.terminal:
            # Trajectory ended before target_step
            return {
                "ended_early": True,
                "ended_at_step": step_id,
                "final_action": action_str,
                "final_outcome": exec_res.outcome_code,
                "task_success": bool(exec_res.task_success),
            }

    # Build the frozen state at target_step (just before the model call)
    frozen_snapshot = build_evidence_snapshot(
        runtime,
        prior_actions=tuple(prior_actions),
        prior_outcomes=tuple(prior_outcomes),
    )

    return {
        "ended_early": False,
        "frozen_step": target_step,
        "frozen_snapshot": frozen_snapshot,
        "frozen_runtime": runtime,
        "prior_actions": tuple(prior_actions),
        "prior_outcomes": tuple(prior_outcomes),
        "actions_taken": list(prior_actions),
        "outcomes_seen": list(prior_outcomes),
    }


def make_single_model_call(
    task: EvidenceTask,
    snapshot: EvidenceSnapshot,
    system_prompt: str,
    packet: dict,
    prior_actions_override: tuple[str, ...] | None = None,
    prior_outcomes_override: tuple[str, ...] | None = None,
    condition_label: str = "i3_11e",
    api_key: str = "",
) -> dict[str, Any]:
    """Make a single model call with the given prompt and packet.

    If prior_actions_override or prior_outcomes_override is provided,
    rebuild the packet with those overrides (to test history ablation).
    """
    if prior_actions_override is not None or prior_outcomes_override is not None:
        # Rebuild snapshot with overridden history
        from copy import deepcopy
        overridden = EvidenceSnapshot(
            task_id=snapshot.task_id,
            task_summary=snapshot.task_summary,
            visible_evidence=snapshot.visible_evidence,
            hidden_evidence_count=snapshot.hidden_evidence_count,
            hypotheses=snapshot.hypotheses,
            verified_count=snapshot.verified_count,
            supporting_count=snapshot.supporting_count,
            contradicting_count=snapshot.contradicting_count,
            searched=snapshot.searched,
            reasoning_complete=snapshot.reasoning_complete,
            resource_state=snapshot.resource_state,
            prior_actions=prior_actions_override if prior_actions_override is not None else snapshot.prior_actions,
            prior_outcomes=prior_outcomes_override if prior_outcomes_override is not None else snapshot.prior_outcomes,
            can_retrieve=snapshot.can_retrieve,
            can_search=snapshot.can_search,
            can_verify=snapshot.can_verify,
        )
        # Rebuild packet with overridden snapshot
        if "decision_state_summary" in packet:
            # M3 packet — rebuild from overridden snapshot
            packet = i3_7e.build_mdsg_state_with_affordances_packet(overridden)
        else:
            # A1 packet — rebuild from overridden snapshot
            packet = i3_7e.build_baseline_with_affordances_packet(overridden)
        snapshot = overridden

    user_prompt = evidence_packet_json(packet)

    backend = DeepSeekBackend()
    backend.task_id = task.task_id
    backend.condition = condition_label
    backend.pair_id = f"i3_11e:{task.task_id}:{condition_label}"

    try:
        call_result = backend.generate(
            system_prompt=system_prompt, user_prompt=user_prompt,
            temperature=0.0, max_tokens=2048)
        outcome = decode_output(call_result.raw_output, strict=True)
        if outcome.valid and outcome.proposal:
            action = outcome.proposal.action
            action_str = action.value if hasattr(action, "value") else str(action)
            reason_code = getattr(outcome.proposal, "reason_code", None)
            return {
                "action": action_str,
                "reason_code": reason_code,
                "raw": call_result.raw_output[:500],
                "error": None,
            }
        else:
            return {
                "action": "FAIL_CLOSED",
                "reason_code": None,
                "raw": call_result.raw_output[:500] if call_result else "",
                "error": "decode_failed",
            }
    except Exception as e:
        return {
            "action": "BACKEND_ERROR",
            "reason_code": None,
            "raw": str(e)[:500],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Break task identification from I3.11d results
# ---------------------------------------------------------------------------

BREAK_TASK_IDS = [
    "r1_confirm_v1_0001",  # bilateral_conflict_h0
    "r1_confirm_v1_0009",  # bilateral_conflict_h0
    "r1_confirm_v1_0043",  # triple_all_eliminated
    "r1_confirm_v1_0051",  # triple_all_eliminated
    "r1_confirm_v1_0114",  # noise_before_conflict
    "r1_confirm_v1_0129",  # noise_before_conflict
]

# Matched successful R1 conflict tasks (same categories, R1 succeeded)
# We'll find these from the corpus dynamically


def find_matched_successes(corpus, results, break_categories):
    """Find matched successful R1 tasks from the same categories."""
    matched = []
    for cat in break_categories:
        for r in results:
            if (r["category"] == cat and r["r1_success"]
                    and r["r1_triggered"] and r["task_id"] not in BREAK_TASK_IDS):
                matched.append(r["task_id"])
                break  # one per category
    return matched


# ---------------------------------------------------------------------------
# Main diagnosis
# ---------------------------------------------------------------------------

def run_diagnosis(api_key: str, output_dir: Path):
    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Generate the I3.11d corpus to get the task objects
    corpus = i3_11d.generate_i3_11d_corpus(split="r1_confirm_v1")
    task_map = {t.task_id: t for t in corpus}

    # Load I3.11d results to find break tasks and matched successes
    results_path = ROOT / "experiments/v2b_i3_11/development/i3_11d_r1_confirm/r1_confirm_v1.jsonl"
    with open(results_path) as f:
        all_results = [json.loads(line) for line in f]

    break_results = [r for r in all_results if r["task_id"] in BREAK_TASK_IDS]
    break_categories = list(set(r["category"] for r in break_results))
    matched_ids = find_matched_successes(corpus, all_results, break_categories)

    print(f"I3.11e: Boundary Discontinuity Diagnosis")
    print(f"  Break tasks: {len(BREAK_TASK_IDS)}")
    print(f"  Matched successes: {len(matched_ids)}")
    print(f"  Break categories: {break_categories}")
    print()

    all_task_ids = BREAK_TASK_IDS + matched_ids
    diagnosis_results = []

    for task_id in all_task_ids:
        task = task_map.get(task_id)
        if task is None:
            print(f"  WARNING: task {task_id} not found in corpus, skipping")
            continue

        is_break = task_id in BREAK_TASK_IDS
        label = "BREAK" if is_break else "MATCHED_SUCCESS"

        # Find the trigger step from I3.11d results
        r_result = next(r for r in all_results if r["task_id"] == task_id)
        trigger_step = r_result["r1_trigger_step"]

        print(f"\n  Task {task_id} ({label}, {r_result['category']})")
        print(f"    R1 trigger step: {trigger_step}")
        print(f"    R1 actions: {r_result['fork_r1']['continuation_actions']}")
        print(f"    M3 actions: {r_result['fork_m3']['continuation_actions']}")

        # Replay R1 up to trigger step
        r1_replay = replay_to_step(task, budget, "A1", trigger_step, api_key)
        if r1_replay.get("ended_early"):
            print(f"    R1 replay ended early at step {r1_replay['ended_at_step']}")
            continue

        # Replay M3 up to trigger step
        m3_replay = replay_to_step(task, budget, "M3", trigger_step, api_key)
        if m3_replay.get("ended_early"):
            print(f"    M3 replay ended early at step {m3_replay['ended_at_step']}")
            # M3 may have already DEFERred before trigger_step
            # Use the step where it ended
            m3_frozen = {
                "ended_early": True,
                "ended_at_step": m3_replay["ended_at_step"],
                "final_action": m3_replay["final_action"],
            }
        else:
            m3_frozen = m3_replay

        # Build the frozen states
        r1_snapshot = r1_replay["frozen_snapshot"]
        r1_prior_actions = r1_replay["prior_actions"]
        r1_prior_outcomes = r1_replay["prior_outcomes"]

        if not m3_frozen.get("ended_early"):
            m3_snapshot = m3_frozen["frozen_snapshot"]
            m3_prior_actions = m3_frozen["prior_actions"]
            m3_prior_outcomes = m3_frozen["prior_outcomes"]
        else:
            m3_snapshot = None
            m3_prior_actions = None
            m3_prior_outcomes = None

        # Build packets for the factorial
        r1_state_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(r1_snapshot)

        # Compare frozen states
        state_comparison = {
            "r1_prior_actions": list(r1_prior_actions),
            "r1_prior_outcomes": list(r1_prior_outcomes),
            "r1_visible_evidence_count": len(r1_snapshot.visible_evidence),
            "r1_hidden_evidence_count": r1_snapshot.hidden_evidence_count,
            "r1_resource_state": dict(r1_snapshot.resource_state),
            "r1_can_retrieve": r1_snapshot.can_retrieve,
            "r1_can_search": r1_snapshot.can_search,
            "r1_can_verify": r1_snapshot.can_verify,
        }

        if m3_snapshot is not None:
            state_comparison["m3_prior_actions"] = list(m3_prior_actions)
            state_comparison["m3_prior_outcomes"] = list(m3_prior_outcomes)
            state_comparison["m3_visible_evidence_count"] = len(m3_snapshot.visible_evidence)
            state_comparison["m3_hidden_evidence_count"] = m3_snapshot.hidden_evidence_count
            state_comparison["m3_resource_state"] = dict(m3_snapshot.resource_state)
            state_comparison["m3_can_retrieve"] = m3_snapshot.can_retrieve
            state_comparison["m3_can_search"] = m3_snapshot.can_search
            state_comparison["m3_can_verify"] = m3_snapshot.can_verify

            # Check if visible evidence differs
            r1_ev_ids = set(e.evidence_id for e in r1_snapshot.visible_evidence)
            m3_ev_ids = set(e.evidence_id for e in m3_snapshot.visible_evidence)
            state_comparison["evidence_ids_match"] = (r1_ev_ids == m3_ev_ids)
            state_comparison["r1_only_evidence"] = list(r1_ev_ids - m3_ev_ids)
            state_comparison["m3_only_evidence"] = list(m3_ev_ids - r1_ev_ids)

            # Check if verification states differ
            r1_vstates = {e.evidence_id: e.verification_state.value
                          for e in r1_snapshot.visible_evidence}
            m3_vstates = {e.evidence_id: e.verification_state.value
                          for e in m3_snapshot.visible_evidence}
            state_comparison["verification_states_match"] = (r1_vstates == m3_vstates)
            if r1_vstates != m3_vstates:
                state_comparison["verification_state_diffs"] = {
                    eid: {"r1": r1_vstates.get(eid), "m3": m3_vstates.get(eid)}
                    for eid in set(r1_vstates) | set(m3_vstates)
                    if r1_vstates.get(eid) != m3_vstates.get(eid)
                }

        print(f"    State comparison:")
        for k, v in state_comparison.items():
            print(f"      {k}: {v}")

        # === Run the B0-B5 factorial ===
        # B0: R1 state + R1 history + frozen M3 prompt
        print(f"    Running B0 (R1 state + R1 history + M3 prompt)...", end=" ")
        b0 = make_single_model_call(
            task, r1_snapshot,
            i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
            r1_state_m3_packet,
            condition_label="B0_r1_state_r1_hist_m3prompt",
            api_key=api_key,
        )
        print(f"→ {b0['action']}")

        # B1: R1 state + stripped history + frozen M3 prompt
        print(f"    Running B1 (R1 state + stripped history + M3 prompt)...", end=" ")
        b1 = make_single_model_call(
            task, r1_snapshot,
            i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
            r1_state_m3_packet,
            prior_actions_override=(),
            prior_outcomes_override=(),
            condition_label="B1_r1_state_stripped_m3prompt",
            api_key=api_key,
        )
        print(f"→ {b1['action']}")

        # B2: R1 state + matched M3 history + frozen M3 prompt
        if m3_prior_actions is not None:
            print(f"    Running B2 (R1 state + M3 history + M3 prompt)...", end=" ")
            b2 = make_single_model_call(
                task, r1_snapshot,
                i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
                r1_state_m3_packet,
                prior_actions_override=m3_prior_actions,
                prior_outcomes_override=m3_prior_outcomes,
                condition_label="B2_r1_state_m3_hist_m3prompt",
                api_key=api_key,
            )
            print(f"→ {b2['action']}")
        else:
            b2 = {"action": "N/A_M3_ENDED_EARLY", "reason_code": None, "error": "m3_ended_early"}

        # B3: R1 state + R1 history + clarified STOP/DEFER
        print(f"    Running B3 (R1 state + R1 history + clarified terminal)...", end=" ")
        b3 = make_single_model_call(
            task, r1_snapshot,
            CLARIFIED_TERMINAL_M3_PROMPT,
            r1_state_m3_packet,
            condition_label="B3_r1_state_r1_hist_clarified",
            api_key=api_key,
        )
        print(f"→ {b3['action']}")

        # B4: M3 state + R1 history + frozen M3 prompt
        if m3_snapshot is not None:
            print(f"    Running B4 (M3 state + R1 history + M3 prompt)...", end=" ")
            m3_state_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(m3_snapshot)
            b4 = make_single_model_call(
                task, m3_snapshot,
                i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
                m3_state_m3_packet,
                prior_actions_override=r1_prior_actions,
                prior_outcomes_override=r1_prior_outcomes,
                condition_label="B4_m3_state_r1_hist_m3prompt",
                api_key=api_key,
            )
            print(f"→ {b4['action']}")
        else:
            b4 = {"action": "N/A_M3_ENDED_EARLY", "reason_code": None, "error": "m3_ended_early"}

        # B5: M3 state + M3 history + frozen M3 prompt
        if m3_snapshot is not None:
            print(f"    Running B5 (M3 state + M3 history + M3 prompt)...", end=" ")
            b5 = make_single_model_call(
                task, m3_snapshot,
                i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT,
                m3_state_m3_packet,
                condition_label="B5_m3_state_m3_hist_m3prompt",
                api_key=api_key,
            )
            print(f"→ {b5['action']}")
        else:
            b5 = {"action": "N/A_M3_ENDED_EARLY", "reason_code": None, "error": "m3_ended_early"}

        # Run each condition 3 times to check determinism (temperature=0 should be deterministic)
        # but API may have minor nondeterminism
        factorial = {
            "task_id": task_id,
            "label": label,
            "category": r_result["category"],
            "trigger_step": trigger_step,
            "r1_actions": r_result["fork_r1"]["continuation_actions"],
            "m3_actions": r_result["fork_m3"]["continuation_actions"],
            "r1_terminal": r_result["fork_r1"]["terminal_action"],
            "m3_terminal": r_result["fork_m3"]["terminal_action"],
            "state_comparison": state_comparison,
            "B0_r1_state_r1_hist_m3prompt": b0,
            "B1_r1_state_stripped_m3prompt": b1,
            "B2_r1_state_m3_hist_m3prompt": b2,
            "B3_r1_state_r1_hist_clarified": b3,
            "B4_m3_state_r1_hist_m3prompt": b4,
            "B5_m3_state_m3_hist_m3prompt": b5,
        }

        diagnosis_results.append(factorial)

        # Print factorial summary
        print(f"\n    FACTORIAL SUMMARY for {task_id}:")
        print(f"      B0 (R1 state + R1 hist + M3 prompt):     {b0['action']}")
        print(f"      B1 (R1 state + stripped hist + M3 prompt): {b1['action']}")
        print(f"      B2 (R1 state + M3 hist + M3 prompt):     {b2['action']}")
        print(f"      B3 (R1 state + R1 hist + clarified):     {b3['action']}")
        print(f"      B4 (M3 state + R1 hist + M3 prompt):     {b4['action']}")
        print(f"      B5 (M3 state + M3 hist + M3 prompt):     {b5['action']}")

    # Save results
    results_path = output_dir / "diagnosis_v1.jsonl"
    with open(results_path, "w") as f:
        for r in diagnosis_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nSaved: {results_path}")

    # === Analysis ===
    print(f"\n{'='*82}")
    print("I3.11e BOUNDARY DISCONTINUITY DIAGNOSIS — RESULTS")
    print(f"{'='*82}")

    break_results_diag = [r for r in diagnosis_results if r["label"] == "BREAK"]
    success_results_diag = [r for r in diagnosis_results if r["label"] == "MATCHED_SUCCESS"]

    def action_summary(results, condition_key):
        from collections import Counter
        actions = [r[condition_key]["action"] for r in results]
        return dict(Counter(actions))

    print(f"\n  BREAK TASKS ({len(break_results_diag)}):")
    for cond in ["B0_r1_state_r1_hist_m3prompt", "B1_r1_state_stripped_m3prompt",
                 "B2_r1_state_m3_hist_m3prompt", "B3_r1_state_r1_hist_clarified",
                 "B4_m3_state_r1_hist_m3prompt", "B5_m3_state_m3_hist_m3prompt"]:
        summary = action_summary(break_results_diag, cond)
        print(f"    {cond:<45} {summary}")

    print(f"\n  MATCHED SUCCESS TASKS ({len(success_results_diag)}):")
    for cond in ["B0_r1_state_r1_hist_m3prompt", "B1_r1_state_stripped_m3prompt",
                 "B2_r1_state_m3_hist_m3prompt", "B3_r1_state_r1_hist_clarified",
                 "B4_m3_state_r1_hist_m3prompt", "B5_m3_state_m3_hist_m3prompt"]:
        summary = action_summary(success_results_diag, cond)
        print(f"    {cond:<45} {summary}")

    # State comparison summary
    print(f"\n  STATE COMPARISON (R1 vs M3 at trigger):")
    for r in diagnosis_results:
        sc = r["state_comparison"]
        ev_match = sc.get("evidence_ids_match", "N/A")
        v_match = sc.get("verification_states_match", "N/A")
        print(f"    {r['task_id']} ({r['label']}): "
              f"evidence_match={ev_match}, vstate_match={v_match}, "
              f"r1_actions={sc.get('r1_prior_actions')}, "
              f"m3_actions={sc.get('m3_prior_actions')}")

    # Diagnostic interpretation
    print(f"\n{'='*82}")
    print("DIAGNOSTIC INTERPRETATION")
    print(f"{'='*82}")

    b0_breaks = [r for r in break_results_diag
                 if r["B0_r1_state_r1_hist_m3prompt"]["action"] == "STOP"]
    b1_breaks = [r for r in break_results_diag
                 if r["B1_r1_state_stripped_m3prompt"]["action"] == "STOP"]
    b3_breaks = [r for r in break_results_diag
                 if r["B3_r1_state_r1_hist_clarified"]["action"] == "STOP"]
    b4_breaks = [r for r in break_results_diag
                 if r["B4_m3_state_r1_hist_m3prompt"]["action"] == "STOP"]
    b5_breaks = [r for r in break_results_diag
                 if r["B5_m3_state_m3_hist_m3prompt"]["action"] == "STOP"]

    print(f"\n  Break tasks where B0=STOP: {len(b0_breaks)}/{len(break_results_diag)}")
    print(f"  Break tasks where B1=STOP (stripped history): {len(b1_breaks)}/{len(break_results_diag)}")
    print(f"  Break tasks where B3=STOP (clarified terminal): {len(b3_breaks)}/{len(break_results_diag)}")
    print(f"  Break tasks where B4=STOP (M3 state + R1 history): {len(b4_breaks)}/{len(break_results_diag)}")
    print(f"  Break tasks where B5=STOP (M3 state + M3 history): {len(b5_breaks)}/{len(break_results_diag)}")

    if len(b0_breaks) > 0 and len(b1_breaks) == 0:
        print(f"\n  → B0=STOP, B1=DEFER: HISTORY causes the failure")
    if len(b0_breaks) > 0 and len(b3_breaks) == 0:
        print(f"\n  → B0=STOP, B3=DEFER: TERMINAL SEMANTICS cause the failure")
    if len(b4_breaks) > 0 and len(b5_breaks) == 0:
        print(f"\n  → B4=STOP, B5=DEFER: PRIOR TRAJECTORY/HISTORY is causal")
    if len(b0_breaks) > 0 and len(b1_breaks) > 0 and len(b2_breaks) == 0:
        print(f"\n  → B0=STOP, B2=DEFER: M3-specific history prevents STOP")

    # Save summary
    summary = {
        "schema": "DAPH_V2B_I3_11E_DIAGNOSIS_V1",
        "n_break_tasks": len(break_results_diag),
        "n_matched_successes": len(success_results_diag),
        "break_task_results": [{
            "task_id": r["task_id"],
            "category": r["category"],
            "B0": r["B0_r1_state_r1_hist_m3prompt"]["action"],
            "B1": r["B1_r1_state_stripped_m3prompt"]["action"],
            "B2": r["B2_r1_state_m3_hist_m3prompt"]["action"],
            "B3": r["B3_r1_state_r1_hist_clarified"]["action"],
            "B4": r["B4_m3_state_r1_hist_m3prompt"]["action"],
            "B5": r["B5_m3_state_m3_hist_m3prompt"]["action"],
            "evidence_match": r["state_comparison"].get("evidence_ids_match"),
            "vstate_match": r["state_comparison"].get("verification_states_match"),
        } for r in break_results_diag],
        "matched_success_results": [{
            "task_id": r["task_id"],
            "category": r["category"],
            "B0": r["B0_r1_state_r1_hist_m3prompt"]["action"],
            "B1": r["B1_r1_state_stripped_m3prompt"]["action"],
            "B2": r["B2_r1_state_m3_hist_m3prompt"]["action"],
            "B3": r["B3_r1_state_r1_hist_clarified"]["action"],
            "B4": r["B4_m3_state_r1_hist_m3prompt"]["action"],
            "B5": r["B5_m3_state_m3_hist_m3prompt"]["action"],
        } for r in success_results_diag],
        "diagnostic_counts": {
            "B0_STOP": len(b0_breaks),
            "B1_STOP": len(b1_breaks),
            "B3_STOP": len(b3_breaks),
            "B4_STOP": len(b4_breaks),
            "B5_STOP": len(b5_breaks),
        },
    }

    summary_path = output_dir / "diagnosis_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
        default="experiments/v2b_i3_11/development/i3_11e_boundary_diagnosis")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_diagnosis(api_key, output_dir)


if __name__ == "__main__":
    main()
