#!/usr/bin/env python3
"""I3.7e — Compact State Governor Experiment.

Four arms on the same frozen 50-task evidence benchmark:

  A  = semantic evidence baseline (unchanged from I3.7d)
  H  = hypotheses only
  HD = hypotheses + discriminator
  M  = minimal decision state

No action override.  The governor's job is to reduce cognitive search
entropy, not replace the model's policy.

Repaired metrics (vs I3.7d):

  AnswerConditionSatisfied(s)
    = exists h: h is uniquely viable
              AND required evidence for h is verified
              AND no live verified contradiction remains

  Recorded separately:
    answer_condition_satisfied_before_terminal
    terminal_action_matches_condition
    condition_led_to_success

  RedundantActionRate:
    An action is redundant if, immediately before it, the observable
    evidence state already supports the eventual correct terminal
    decision and the action does not change the relevant hypothesis set.

Acceptance gate (stricter than I3.7d):
  success >= baseline
  breaks <= 1
  rescues >= 1
  mean utility >= baseline
  rescues > breaks

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_7e_compact_governor.py \\
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
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceRuntime, EvidenceSnapshot, EvidenceItem, EvidenceHypothesis,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)

# ---------------------------------------------------------------------------
# Decision-state analysis
#
# Two parallel implementations:
#   - _classify_from_snapshot: used by the M-arm packet builder.
#     Operates ONLY on EvidenceSnapshot (controller-visible).
#     Must never receive EvidenceRuntime or EvidenceTask.
#   - classify_hypothesis_viability / answer_condition_satisfied:
#     used by metrics and redundancy checks (evaluation-side).
#     May use EvidenceRuntime.
#
# This separation enforces the observation boundary: the packet builder
# cannot accidentally inspect hidden evidence.
# ---------------------------------------------------------------------------

def _classify_from_snapshot(
    snapshot: EvidenceSnapshot,
) -> dict[str, dict[str, Any]]:
    """Classify hypothesis viability from controller-visible snapshot only.

    This function must never receive EvidenceRuntime.  It derives all
    information from snapshot.hypotheses and snapshot.visible_evidence.
    """
    h_ids = [h.hypothesis_id for h in snapshot.hypotheses]
    result: dict[str, dict[str, Any]] = {}

    for h_id in h_ids:
        has_support = False
        has_contradiction = False
        supporting_evidence: list[str] = []
        contradicting_evidence: list[str] = []
        falsified_support: list[str] = []
        falsified_contradiction: list[str] = []

        for ev in snapshot.visible_evidence:
            # Only visible (retrieved) evidence is considered
            if ev.verification_state == VerificationState.SUFFICIENT:
                if ev.temporal_status == TemporalStatus.STALE:
                    continue
                if h_id in ev.supports:
                    has_support = True
                    supporting_evidence.append(ev.evidence_id)
                if h_id in ev.contradicts:
                    has_contradiction = True
                    contradicting_evidence.append(ev.evidence_id)
            elif ev.verification_state == VerificationState.FALSIFIED:
                if h_id in ev.supports:
                    falsified_support.append(ev.evidence_id)
                if h_id in ev.contradicts:
                    falsified_contradiction.append(ev.evidence_id)

        if has_contradiction:
            status = "ELIMINATED"
        elif has_support:
            status = "VIABLE"
        elif falsified_support and not has_support:
            status = "WEAKENED"
        else:
            status = "UNTESTED"

        result[h_id] = {
            "status": status,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "falsified_support": falsified_support,
            "falsified_contradiction": falsified_contradiction,
        }

    return result


def _answer_condition_from_snapshot(
    snapshot: EvidenceSnapshot,
) -> tuple[bool, str | None]:
    """Check answer condition from snapshot only (controller-visible)."""
    viability = _classify_from_snapshot(snapshot)
    viable_hyps = [h_id for h_id, info in viability.items()
                   if info["status"] == "VIABLE"]
    if len(viable_hyps) == 1:
        return True, viable_hyps[0]
    return False, None


def classify_hypothesis_viability(
    runtime: EvidenceRuntime,
) -> dict[str, dict[str, Any]]:
    """Evaluation-side viability classification (may use runtime).

    Used by metrics and redundancy checks, NOT by packet builders.
    """
    task = runtime.task
    h_ids = [h.hypothesis_id for h in task.hypotheses]
    result: dict[str, dict[str, Any]] = {}

    for h_id in h_ids:
        has_support = False
        has_contradiction = False
        supporting_evidence: list[str] = []
        contradicting_evidence: list[str] = []

        for ev in runtime.evidence:
            if not ev.retrieved:
                continue
            if ev.verification_state != VerificationState.SUFFICIENT:
                continue
            if ev.temporal_status == TemporalStatus.STALE:
                continue
            if h_id in ev.supports:
                has_support = True
                supporting_evidence.append(ev.evidence_id)
            if h_id in ev.contradicts:
                has_contradiction = True
                contradicting_evidence.append(ev.evidence_id)

        falsified_support = []
        for ev in runtime.evidence:
            if not ev.retrieved:
                continue
            if ev.verification_state == VerificationState.FALSIFIED:
                if h_id in ev.supports:
                    falsified_support.append(ev.evidence_id)

        falsified_contradiction = []
        for ev in runtime.evidence:
            if not ev.retrieved:
                continue
            if ev.verification_state == VerificationState.FALSIFIED:
                if h_id in ev.contradicts:
                    falsified_contradiction.append(ev.evidence_id)

        if has_contradiction:
            status = "ELIMINATED"
        elif has_support:
            status = "VIABLE"
        elif falsified_support and not has_support:
            status = "WEAKENED"
        else:
            status = "UNTESTED"

        result[h_id] = {
            "status": status,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "falsified_support": falsified_support,
            "falsified_contradiction": falsified_contradiction,
        }

    return result


def answer_condition_satisfied(runtime: EvidenceRuntime) -> tuple[bool, str | None]:
    """Evaluation-side answer condition check (may use runtime)."""
    viability = classify_hypothesis_viability(runtime)
    viable_hyps = [h_id for h_id, info in viability.items()
                   if info["status"] == "VIABLE"]
    if len(viable_hyps) == 1:
        return True, viable_hyps[0]
    return False, None


def is_redundant_action(
    runtime_before: EvidenceRuntime,
    action: DecisionAction,
    runtime_after: EvidenceRuntime,
    correct_hypothesis_id: str,
) -> bool:
    """Check if an action was redundant.

    An action is redundant if, immediately before it, the observable
    evidence state already supports the eventual correct terminal
    decision and the action does not change the relevant hypothesis set.
    """
    # Check if answer condition was already satisfied before the action
    satisfied_before, h_before = answer_condition_satisfied(runtime_before)

    if not satisfied_before:
        return False

    # The answer condition was already met. Now check if the action
    # changed the relevant hypothesis set.
    if action in (DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP):
        return False  # Terminal actions are never redundant

    # Check if the action changed viability of any hypothesis
    viability_before = classify_hypothesis_viability(runtime_before)
    viability_after = classify_hypothesis_viability(runtime_after)

    for h_id in viability_before:
        if viability_before[h_id]["status"] != viability_after[h_id]["status"]:
            return False  # The action changed hypothesis viability

    # The action didn't change any hypothesis's viability → redundant
    return True


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

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

HYPOTHESES_ONLY_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items. Use this when you need more evidence.
  SEARCH_MORE: Search for additional evidence from other sources.
  VERIFY: Verify the most recently retrieved unverified evidence item.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: Give up due to insufficient evidence.
  STOP: Stop without answering.

EVIDENCE STATES:
  UNVERIFIED: Evidence has not been checked yet. Use VERIFY to check it.
  SUFFICIENT: Evidence is verified and supports its hypothesis.
  FALSIFIED: Evidence is verified and found to be false.
  STALE: Evidence is outdated. Use SEARCH_MORE to find current evidence.

You are given the evidence state plus a list of candidate hypotheses with their answer actions.
Use the hypotheses to organize your assessment of the evidence.

DECISION PROCESS:
  1. If exactly one hypothesis has sufficient verified current support and no verified contradiction, ANSWER.
  2. If evidence is UNVERIFIED, use VERIFY to check it.
  3. If you need more evidence, use RETRIEVE or SEARCH_MORE.
  4. If you cannot obtain sufficient evidence, DEFER.
  5. Do NOT immediately DEFER if there are hidden evidence items or unverified evidence. Investigate first.

OUTPUT FORMAT:
You must respond with a JSON object containing exactly these three fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "A_SHORT_UPPERCASE_REASON_CODE",
  "target_id": null
}

The reason_code must be uppercase with underscores only.

The word json appears in this prompt to satisfy the API requirement."""

HYPOTHESES_DISCRIMINATOR_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items. Use this when you need more evidence.
  SEARCH_MORE: Search for additional evidence from other sources.
  VERIFY: Verify the most recently retrieved unverified evidence item.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: Give up due to insufficient evidence.
  STOP: Stop without answering.

EVIDENCE STATES:
  UNVERIFIED: Evidence has not been checked yet. Use VERIFY to check it.
  SUFFICIENT: Evidence is verified and supports its hypothesis.
  FALSIFIED: Evidence is verified and found to be false.
  STALE: Evidence is outdated. Use SEARCH_MORE to find current evidence.

You are given the evidence state plus:
  - candidate hypotheses with their answer actions
  - a discriminator: which evidence item, if verified, would distinguish between hypotheses

Use the discriminator to focus your evidence operations on the most decision-relevant item.

DECISION PROCESS:
  1. If exactly one hypothesis has sufficient verified current support and no verified contradiction, ANSWER.
  2. If the discriminator evidence is unverified, VERIFY it.
  3. If the discriminator evidence is hidden, RETRIEVE it.
  4. If you need more evidence, use SEARCH_MORE.
  5. If you cannot obtain sufficient evidence, DEFER.
  6. Do NOT immediately DEFER if there are hidden evidence items or unverified evidence. Investigate first.

OUTPUT FORMAT:
You must respond with a JSON object containing exactly these three fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "A_SHORT_UPPERCASE_REASON_CODE",
  "target_id": null
}

The reason_code must be uppercase with underscores only.

The word json appears in this prompt to satisfy the API requirement."""

MINIMAL_DECISION_STATE_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items. Use this when you need more evidence.
  SEARCH_MORE: Search for additional evidence from other sources.
  VERIFY: Verify the most recently retrieved unverified evidence item.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: Give up due to insufficient evidence.
  STOP: Stop without answering.

You are given the evidence state plus a compact decision-state summary:
  - live_hypotheses: hypotheses that are still viable
  - eliminated_hypotheses: hypotheses that have been ruled out
  - verified_support: evidence IDs that provide verified support
  - verified_contradictions: evidence IDs that provide verified contradiction
  - decision_state: READY_TO_ANSWER, NEEDS_DISCRIMINATION, NEEDS_EVIDENCE, or INSUFFICIENT
  - remaining_blocker: the single most important unresolved item, or null

This summary tells you the current decision-relevant state. Use it to avoid unnecessary actions.

CRITICAL RULES:
  - If decision_state is READY_TO_ANSWER, ANSWER immediately. Do not search or verify further.
  - If decision_state is NEEDS_DISCRIMINATION, focus on the remaining_blocker.
  - If decision_state is NEEDS_EVIDENCE, RETRIEVE or SEARCH_MORE.
  - If decision_state is INSUFFICIENT, DEFER.
  - Do NOT take actions that don't address the remaining_blocker.

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
# Packet builders for each arm
# ---------------------------------------------------------------------------

def build_baseline_packet(snapshot: EvidenceSnapshot) -> dict:
    """A arm: just the evidence snapshot."""
    packet = serialize_evidence_snapshot(snapshot)
    assert_no_evidence_leakage(packet)
    return packet


def build_hypotheses_only_packet(snapshot: EvidenceSnapshot) -> dict:
    """H arm: evidence + hypotheses only (no execution plan, no terminal rules)."""
    visible_evidence = []
    for e in snapshot.visible_evidence:
        ev_dict = e.as_dict()
        ev_dict.pop("verify_result", None)
        visible_evidence.append(ev_dict)

    packet = {
        "schema": "DAPH_V2B_I3_7_HYPOTHESES_ONLY_PACKET_V1",
        "task_id": snapshot.task_id,
        "task_summary": snapshot.task_summary,
        "hypotheses": [h.as_dict() for h in snapshot.hypotheses],
        "visible_evidence": visible_evidence,
        "hidden_evidence_count": snapshot.hidden_evidence_count,
        "verified_count": snapshot.verified_count,
        "supporting_count": snapshot.supporting_count,
        "contradicting_count": snapshot.contradicting_count,
        "searched": snapshot.searched,
        "reasoning_complete": snapshot.reasoning_complete,
        "resource_state": dict(snapshot.resource_state),
        "prior_actions": list(snapshot.prior_actions),
        "prior_outcomes": list(snapshot.prior_outcomes),
    }
    assert_no_evidence_leakage(packet)
    return packet


def build_hypotheses_discriminator_packet(
    snapshot: EvidenceSnapshot,
    runtime: EvidenceRuntime,
) -> dict:
    """HD arm: evidence + hypotheses + discriminator.

    The discriminator is the single evidence item that, if verified,
    would most reduce hypothesis uncertainty. It is derived from the
    current evidence state, not from the oracle path.
    """
    packet = build_hypotheses_only_packet(snapshot)
    packet["schema"] = "DAPH_V2B_I3_7_HYPOTHESES_DISCRIMINATOR_PACKET_V1"

    # Find the discriminator: the unverified evidence item that
    # supports or contradicts the most hypotheses
    discriminator = None
    best_score = -1

    for ev in runtime.evidence:
        if not ev.retrieved:
            continue
        if ev.verification_state != VerificationState.UNVERIFIED:
            continue
        # Score: how many hypotheses does this evidence touch?
        score = len(set(ev.supports + ev.contradicts))
        if score > best_score:
            best_score = score
            discriminator = {
                "evidence_id": ev.evidence_id,
                "proposition": ev.proposition,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "rationale": f"Verifying {ev.evidence_id} would resolve support/contradiction for {score} hypothesis(es).",
            }

    # If no unverified retrieved evidence, check hidden evidence
    if discriminator is None:
        hidden = [e for e in runtime.evidence if not e.retrieved]
        if hidden:
            # Pick the hidden evidence that touches the most hypotheses
            for ev in hidden:
                score = len(set(ev.supports + ev.contradicts))
                if score > best_score:
                    best_score = score
                    discriminator = {
                        "evidence_id": ev.evidence_id,
                        "proposition": "(hidden - retrieve first)",
                        "supports": list(ev.supports),
                        "contradicts": list(ev.contradicts),
                        "rationale": f"Retrieving {ev.evidence_id} would expose evidence relevant to {score} hypothesis(es).",
                        "is_hidden": True,
                    }

    packet["discriminator"] = discriminator
    assert_no_evidence_leakage(packet)
    return packet


def build_minimal_decision_state_packet(
    snapshot: EvidenceSnapshot,
) -> dict:
    """M arm: evidence + minimal decision state.

    STRICTLY a function of the controller-visible snapshot.
    Must never receive EvidenceRuntime or EvidenceTask.

    The minimal decision state compresses the current evidence into:
      - live_hypotheses
      - eliminated_hypotheses
      - verified_support
      - verified_contradictions
      - decision_state
      - remaining_blocker
    """
    packet = build_hypotheses_only_packet(snapshot)
    packet["schema"] = "DAPH_V2B_I3_7_MINIMAL_DECISION_STATE_PACKET_V1"

    viability = _classify_from_snapshot(snapshot)

    live_hyps = [h_id for h_id, info in viability.items()
                 if info["status"] == "VIABLE"]
    eliminated_hyps = [h_id for h_id, info in viability.items()
                       if info["status"] == "ELIMINATED"]
    weakened_hyps = [h_id for h_id, info in viability.items()
                     if info["status"] == "WEAKENED"]
    untested_hyps = [h_id for h_id, info in viability.items()
                     if info["status"] == "UNTESTED"]

    verified_support: list[str] = []
    verified_contradictions: list[str] = []
    for h_id, info in viability.items():
        verified_support.extend(info["supporting_evidence"])
        verified_contradictions.extend(info["contradicting_evidence"])
    verified_support = sorted(set(verified_support))
    verified_contradictions = sorted(set(verified_contradictions))

    # Determine decision state from snapshot only
    satisfied, satisfied_h = _answer_condition_from_snapshot(snapshot)

    # Unverified visible evidence (from snapshot only)
    unverified_visible = [
        ev.evidence_id for ev in snapshot.visible_evidence
        if ev.verification_state == VerificationState.UNVERIFIED
    ]

    if satisfied:
        decision_state = "READY_TO_ANSWER"
        remaining_blocker = None
    elif len(eliminated_hyps) == len(viability) - 1 and len(live_hyps) == 0:
        # All but one eliminated, remaining one has no verified support
        remaining_h = [h for h in viability if h not in eliminated_hyps]
        if remaining_h:
            # Check visible unverified evidence that supports remaining hypothesis
            unverified_support = [
                ev.evidence_id for ev in snapshot.visible_evidence
                if ev.verification_state == VerificationState.UNVERIFIED
                and remaining_h[0] in ev.supports
            ]
            if unverified_support:
                decision_state = "NEEDS_DISCRIMINATION"
                remaining_blocker = {
                    "evidence_id": unverified_support[0],
                    "operation": "VERIFY",
                    "rationale": f"Verify {unverified_support[0]} to confirm {remaining_h[0]}.",
                }
            elif snapshot.hidden_evidence_count > 0:
                # Hidden evidence exists but we cannot name it
                decision_state = "NEEDS_EVIDENCE"
                remaining_blocker = {
                    "evidence_id": None,
                    "operation": "RETRIEVE",
                    "rationale": "Additional hidden evidence remains available.",
                }
            else:
                decision_state = "INSUFFICIENT"
                remaining_blocker = None
        else:
            decision_state = "INSUFFICIENT"
            remaining_blocker = None
    elif len(live_hyps) > 1:
        # Multiple viable hypotheses — need discrimination
        # Find visible unverified evidence that touches live hypotheses
        best_blocker = None
        best_score = -1
        for ev in snapshot.visible_evidence:
            if ev.verification_state != VerificationState.UNVERIFIED:
                continue
            touched = set(ev.supports + ev.contradicts) & set(live_hyps)
            if len(touched) > best_score:
                best_score = len(touched)
                best_blocker = {
                    "evidence_id": ev.evidence_id,
                    "operation": "VERIFY",
                    "rationale": f"Verify {ev.evidence_id} to discriminate between {len(touched)} live hypotheses.",
                }
        if best_blocker:
            decision_state = "NEEDS_DISCRIMINATION"
            remaining_blocker = best_blocker
        elif snapshot.hidden_evidence_count > 0:
            # Hidden evidence may help discriminate, but we cannot name it
            decision_state = "NEEDS_EVIDENCE"
            remaining_blocker = {
                "evidence_id": None,
                "operation": "RETRIEVE",
                "rationale": "Additional hidden evidence remains available.",
            }
        else:
            decision_state = "INSUFFICIENT"
            remaining_blocker = None
    elif len(live_hyps) == 0 and len(untested_hyps) > 0:
        # No verified support for anything yet
        if unverified_visible:
            decision_state = "NEEDS_DISCRIMINATION"
            remaining_blocker = {
                "evidence_id": unverified_visible[0],
                "operation": "VERIFY",
                "rationale": f"Verify {unverified_visible[0]} to establish hypothesis support.",
            }
        elif snapshot.hidden_evidence_count > 0:
            decision_state = "NEEDS_EVIDENCE"
            remaining_blocker = {
                "evidence_id": None,
                "operation": "RETRIEVE",
                "rationale": "Additional hidden evidence remains available.",
            }
        else:
            decision_state = "INSUFFICIENT"
            remaining_blocker = None
    else:
        # Weakened or mixed state
        if unverified_visible:
            decision_state = "NEEDS_DISCRIMINATION"
            remaining_blocker = {
                "evidence_id": unverified_visible[0],
                "operation": "VERIFY",
                "rationale": f"Verify {unverified_visible[0]} to resolve hypothesis status.",
            }
        elif snapshot.hidden_evidence_count > 0:
            decision_state = "NEEDS_EVIDENCE"
            remaining_blocker = {
                "evidence_id": None,
                "operation": "RETRIEVE",
                "rationale": "Additional hidden evidence remains available.",
            }
        else:
            decision_state = "INSUFFICIENT"
            remaining_blocker = None

    packet["decision_state_summary"] = {
        "live_hypotheses": live_hyps,
        "eliminated_hypotheses": eliminated_hyps,
        "weakened_hypotheses": weakened_hyps,
        "untested_hypotheses": untested_hyps,
        "verified_support": verified_support,
        "verified_contradictions": verified_contradictions,
        "unverified_relevant_evidence": unverified_visible,
        "decision_state": decision_state,
        "remaining_blocker": remaining_blocker,
    }

    assert_no_evidence_leakage(packet)
    return packet


# ---------------------------------------------------------------------------
# M1: MDSG-StateOnly — state estimation without action prescription
#
# Differences from M0 (MINIMAL_DECISION_STATE):
#   1. No operation recommendations in remaining_blocker
#   2. Conservative READY: requires no unresolved visible evidence AND
#      no hidden evidence remaining (PROVISIONALLY_READY if hidden exists)
#   3. Distinguishes SUPPORTED from DECISION-SUFFICIENT
#   4. Reports evidence_status instead of operation
# ---------------------------------------------------------------------------

MDSG_STATE_ONLY_SYSTEM_PROMPT = """You are a metareasoning controller for a retrieval-verification task.

You must choose one bounded action from the frozen seven-action vocabulary:
  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP

ACTION SEMANTICS:
  RETRIEVE: Expose hidden evidence items. Use this when you need more evidence.
  SEARCH_MORE: Search for additional evidence from other sources.
  VERIFY: Verify the most recently retrieved unverified evidence item.
  ANSWER: Provide the final answer. Only use when you have sufficient verified current evidence.
  DEFER: Give up due to insufficient evidence.
  STOP: Stop without answering.

You are given the evidence state plus a compact decision-state summary:
  - live_hypotheses: hypotheses that are still viable
  - eliminated_hypotheses: hypotheses that have been ruled out
  - verified_support: evidence IDs that provide verified support
  - verified_contradictions: evidence IDs that provide verified contradiction
  - decision_state: READY_TO_ANSWER, PROVISIONALLY_READY, NEEDS_DISCRIMINATION, NEEDS_EVIDENCE, or INSUFFICIENT
  - evidence_status: describes what evidence situation remains unresolved

This summary tells you the current decision-relevant state. Use it to avoid unnecessary actions.

CRITICAL RULES:
  - If decision_state is READY_TO_ANSWER, you have sufficient verified evidence. ANSWER immediately.
  - If decision_state is PROVISIONALLY_READY, one hypothesis is supported but hidden evidence remains. You may ANSWER or gather more evidence. Consider whether the visible evidence is reliable enough.
  - If decision_state is NEEDS_DISCRIMINATION, multiple hypotheses are viable. VERIFY unverified evidence to discriminate.
  - If decision_state is NEEDS_EVIDENCE, no hypothesis has verified support. RETRIEVE or SEARCH_MORE.
  - If decision_state is INSUFFICIENT, DEFER.
  - You retain full authority over which action to take. The decision state is advisory.

OUTPUT FORMAT:
You must respond with a JSON object containing exactly these three fields:
{
  "action": "one of ANSWER RETRIEVE VERIFY SEARCH_MORE REASON_MORE DEFER STOP",
  "reason_code": "A_SHORT_UPPERCASE_REASON_CODE",
  "target_id": null
}

The reason_code must be uppercase with underscores only.

The word json appears in this prompt to satisfy the API requirement."""


def build_mdsg_state_only_packet(
    snapshot: EvidenceSnapshot,
) -> dict:
    """M1 arm: MDSG-StateOnly.

    Strictly a function of the controller-visible snapshot.
    Must never receive EvidenceRuntime or EvidenceTask.

    Key differences from M0:
      1. No operation recommendations (no "operation" field in remaining_blocker)
      2. Conservative READY: only when no hidden evidence remains
      3. PROVISIONALLY_READY when one hypothesis is viable but hidden evidence exists
      4. evidence_status describes the situation without prescribing an action
    """
    packet = build_hypotheses_only_packet(snapshot)
    packet["schema"] = "DAPH_V2B_I3_9_MDSG_STATE_ONLY_PACKET_V1"

    viability = _classify_from_snapshot(snapshot)

    live_hyps = [h_id for h_id, info in viability.items()
                 if info["status"] == "VIABLE"]
    eliminated_hyps = [h_id for h_id, info in viability.items()
                       if info["status"] == "ELIMINATED"]
    weakened_hyps = [h_id for h_id, info in viability.items()
                     if info["status"] == "WEAKENED"]
    untested_hyps = [h_id for h_id, info in viability.items()
                     if info["status"] == "UNTESTED"]

    verified_support: list[str] = []
    verified_contradictions: list[str] = []
    for h_id, info in viability.items():
        verified_support.extend(info["supporting_evidence"])
        verified_contradictions.extend(info["contradicting_evidence"])
    verified_support = sorted(set(verified_support))
    verified_contradictions = sorted(set(verified_contradictions))

    # Check answer condition from snapshot
    satisfied, satisfied_h = _answer_condition_from_snapshot(snapshot)

    # Unverified visible evidence
    unverified_visible = [
        ev.evidence_id for ev in snapshot.visible_evidence
        if ev.verification_state == VerificationState.UNVERIFIED
    ]

    # Check for stale verified evidence
    has_stale_verified = any(
        ev.verification_state == VerificationState.SUFFICIENT
        and ev.temporal_status == TemporalStatus.STALE
        for ev in snapshot.visible_evidence
    )

    # Check for multiple viable hypotheses
    multiple_viable = len(live_hyps) > 1

    # Determine decision state — conservative version
    if satisfied and not multiple_viable:
        # One uniquely viable hypothesis with verified support
        if snapshot.hidden_evidence_count > 0:
            # Hidden evidence remains — provisional, not certain
            decision_state = "PROVISIONALLY_READY"
            evidence_status = "ONE_HYPOTHESIS_SUPPORTED_HIDDEN_EVIDENCE_REMAINS"
            remaining_blocker = None  # No action prescription
        elif has_stale_verified:
            # Some verified evidence is stale — provisional
            decision_state = "PROVISIONALLY_READY"
            evidence_status = "SUPPORTED_WITH_STALE_EVIDENCE"
            remaining_blocker = None
        else:
            # Fully ready: one viable hypothesis, no hidden evidence, no stale
            decision_state = "READY_TO_ANSWER"
            evidence_status = "DECISION_SUFFICIENT"
            remaining_blocker = None
    elif multiple_viable:
        decision_state = "NEEDS_DISCRIMINATION"
        evidence_status = "MULTIPLE_VIABLE_HYPOTHESES"
        remaining_blocker = None
    elif len(eliminated_hyps) == len(viability) - 1 and len(live_hyps) == 0:
        # All but one eliminated, remaining has no verified support
        if unverified_visible:
            decision_state = "NEEDS_DISCRIMINATION"
            evidence_status = "UNVERIFIED_VISIBLE_EVIDENCE_REMAINS"
            remaining_blocker = None
        elif snapshot.hidden_evidence_count > 0:
            decision_state = "NEEDS_EVIDENCE"
            evidence_status = "NO_VERIFIED_SUPPORT_HIDDEN_EVIDENCE_REMAINS"
            remaining_blocker = None
        else:
            decision_state = "INSUFFICIENT"
            evidence_status = "NO_VERIFIED_SUPPORT_NO_HIDDEN_EVIDENCE"
            remaining_blocker = None
    elif len(live_hyps) == 0 and len(untested_hyps) > 0:
        if unverified_visible:
            decision_state = "NEEDS_DISCRIMINATION"
            evidence_status = "NO_VERIFIED_SUPPORT_UNVERIFIED_VISIBLE"
            remaining_blocker = None
        elif snapshot.hidden_evidence_count > 0:
            decision_state = "NEEDS_EVIDENCE"
            evidence_status = "NO_VERIFIED_SUPPORT_HIDDEN_EVIDENCE_REMAINS"
            remaining_blocker = None
        else:
            decision_state = "INSUFFICIENT"
            evidence_status = "NO_EVIDENCE_AVAILABLE"
            remaining_blocker = None
    else:
        # Weakened or mixed state
        if unverified_visible:
            decision_state = "NEEDS_DISCRIMINATION"
            evidence_status = "WEAKENED_HYPOTHESES_UNVERIFIED_EVIDENCE"
            remaining_blocker = None
        elif snapshot.hidden_evidence_count > 0:
            decision_state = "NEEDS_EVIDENCE"
            evidence_status = "WEAKENED_HYPOTHESES_HIDDEN_EVIDENCE_REMAINS"
            remaining_blocker = None
        else:
            decision_state = "INSUFFICIENT"
            evidence_status = "WEAKENED_HYPOTHESES_NO_EVIDENCE"
            remaining_blocker = None

    packet["decision_state_summary"] = {
        "live_hypotheses": live_hyps,
        "eliminated_hypotheses": eliminated_hyps,
        "weakened_hypotheses": weakened_hyps,
        "untested_hypotheses": untested_hyps,
        "verified_support": verified_support,
        "verified_contradictions": verified_contradictions,
        "unverified_relevant_evidence": unverified_visible,
        "decision_state": decision_state,
        "evidence_status": evidence_status,
        "remaining_blocker": remaining_blocker,
        "hidden_evidence_count": snapshot.hidden_evidence_count,
    }

    assert_no_evidence_leakage(packet)
    return packet


# ---------------------------------------------------------------------------
# Trajectory runner
# ---------------------------------------------------------------------------

def counterbalance_order(task_id: str, n_arms: int = 4) -> list[str]:
    """Counterbalance arm order using task_id hash."""
    arms = ["A", "H", "HD", "M"][:n_arms]
    h = hashlib.sha256(task_id.encode()).hexdigest()
    perm_idx = int(h[:8], 16) % len(list(itertools.permutations(arms)))
    perms = list(itertools.permutations(arms))
    return list(perms[perm_idx])


def run_trajectory(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    mode: str,  # "BASELINE", "HYPOTHESES_ONLY", "HYPOTHESES_DISCRIMINATOR", "MINIMAL_DECISION_STATE"
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

    # Repaired metrics
    target_evidence_retrieved = False
    target_evidence_verified = False
    hypothesis_eliminated = False

    # answer_condition_satisfied_before_terminal
    acs_before_terminal = False
    acs_hypothesis: str | None = None

    # Redundant action tracking
    redundant_actions: list[dict[str, Any]] = []
    redundant_action_count = 0

    # Per-step decision state log (for M arms only)
    decision_state_log: list[dict[str, Any]] = []

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
            packet = build_baseline_packet(evidence_snapshot)
            system_prompt = BASELINE_SYSTEM_PROMPT
        elif mode == "HYPOTHESES_ONLY":
            packet = build_hypotheses_only_packet(evidence_snapshot)
            system_prompt = HYPOTHESES_ONLY_SYSTEM_PROMPT
        elif mode == "HYPOTHESES_DISCRIMINATOR":
            packet = build_hypotheses_discriminator_packet(evidence_snapshot, runtime)
            system_prompt = HYPOTHESES_DISCRIMINATOR_SYSTEM_PROMPT
        elif mode == "MINIMAL_DECISION_STATE":
            packet = build_minimal_decision_state_packet(evidence_snapshot)
            system_prompt = MINIMAL_DECISION_STATE_SYSTEM_PROMPT
        elif mode == "MDSG_STATE_ONLY":
            packet = build_mdsg_state_only_packet(evidence_snapshot)
            system_prompt = MDSG_STATE_ONLY_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Log per-step decision state for M arms
        if mode in ("MINIMAL_DECISION_STATE", "MDSG_STATE_ONLY"):
            ds_info = packet.get("decision_state_summary", {})
            decision_state_log.append({
                "step": step_id,
                "decision_state": ds_info.get("decision_state"),
                "live_hypotheses": ds_info.get("live_hypotheses", []),
                "eliminated_hypotheses": ds_info.get("eliminated_hypotheses", []),
                "remaining_blocker": ds_info.get("remaining_blocker"),
            })

        user_prompt = evidence_packet_json(packet)

        # Check answer condition BEFORE the action (for acs_before_terminal)
        acs_now, acs_h = answer_condition_satisfied(runtime)
        if acs_now:
            acs_before_terminal = True
            acs_hypothesis = acs_h

        # Call model
        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_7e_{mode}"
        backend.pair_id = f"i3_7e:{task.task_id}:{fork_label}:step{step_id}"

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
        target_id = getattr(proposal, "target_id", None)

        # Check if this action is redundant
        resources_before = runtime.resources
        runtime_before = runtime
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        resources_after = exec_res.runtime.resources

        is_redundant = is_redundant_action(
            runtime_before, action, exec_res.runtime,
            task.correct_hypothesis_id,
        )
        if is_redundant:
            redundant_action_count += 1
            redundant_actions.append({
                "step": step_id,
                "action": action.value if hasattr(action, "value") else str(action),
                "acs_hypothesis": acs_hypothesis,
            })

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

        # Track hypothesis elimination
        for e in exec_res.runtime.evidence:
            if (e.verification_state == VerificationState.SUFFICIENT
                    and e.retrieved and e.contradicts):
                hypothesis_eliminated = True

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

    # Compute derived metrics
    # terminal_action_matches_condition: did the terminal action match what the condition suggested?
    terminal_matches_condition = False
    condition_led_to_success = False
    if acs_before_terminal and acs_hypothesis is not None:
        # Find the answer action for the satisfied hypothesis
        for h in task.hypotheses:
            if h.hypothesis_id == acs_hypothesis:
                if terminal_action == h.answer_action.value:
                    terminal_matches_condition = True
                if terminal_matches_condition and success:
                    condition_led_to_success = True
                break

    redundant_action_rate = redundant_action_count / max(steps_taken, 1)

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
        # Repaired information conversion pipeline
        "target_evidence_retrieved": target_evidence_retrieved,
        "target_evidence_verified": target_evidence_verified,
        "hypothesis_eliminated": hypothesis_eliminated,
        "answer_condition_satisfied_before_terminal": acs_before_terminal,
        "acs_hypothesis": acs_hypothesis,
        "terminal_action_matches_condition": terminal_matches_condition,
        "condition_led_to_success": condition_led_to_success,
        # Redundant action metric
        "redundant_action_count": redundant_action_count,
        "redundant_action_rate": round(redundant_action_rate, 4),
        "redundant_actions": redundant_actions,
        # Per-step decision state log (M arms only, empty for baseline)
        "decision_state_log": decision_state_log,
    }


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    """Process one task across four arms: A, H, HD, M."""
    fork_order = counterbalance_order(task.task_id, n_arms=4)

    arm_configs = {
        "A": {"mode": "BASELINE"},
        "H": {"mode": "HYPOTHESES_ONLY"},
        "HD": {"mode": "HYPOTHESES_DISCRIMINATOR"},
        "M": {"mode": "MINIMAL_DECISION_STATE"},
    }

    results: dict[str, dict] = {}
    for arm_id in fork_order:
        cfg = arm_configs[arm_id]
        results[arm_id] = run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=cfg["mode"], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    # Compute pairwise comparisons against A
    u_a = results["A"]["realized_utility"]
    comparisons = {}
    for arm_id in ["H", "HD", "M"]:
        u_arm = results[arm_id]["realized_utility"]
        comparisons[arm_id] = {
            "u": u_arm,
            "u_gain": round(u_arm - u_a, 4),
            "success": results[arm_id]["success"],
        }

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "oracle_path": list(task.oracle_resolution_path),
        "fork_order": fork_order,
        "u_a": u_a,
        "a_success": results["A"]["success"],
        # Per-arm results
        "fork_a": results["A"],
        "fork_h": results["H"],
        "fork_hd": results["HD"],
        "fork_m": results["M"],
        # Comparisons
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# First-divergence analysis
# ---------------------------------------------------------------------------

def first_divergence(
    actions_a: list[str],
    actions_b: list[str],
) -> tuple[int, str, str] | None:
    """Find the first step where two action sequences diverge.

    Returns (step_index, action_a, action_b) or None if identical.
    """
    for i in range(min(len(actions_a), len(actions_b))):
        if actions_a[i] != actions_b[i]:
            return (i, actions_a[i], actions_b[i])
    if len(actions_a) != len(actions_b):
        return (min(len(actions_a), len(actions_b)),
                actions_a[min(len(actions_a), len(actions_b))] if len(actions_a) > len(actions_b) else "END",
                actions_b[min(len(actions_a), len(actions_b))] if len(actions_b) > len(actions_a) else "END")
    return None


def analyze_divergences(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze first-divergence patterns for breaks and rescues."""
    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    analysis: dict[str, list] = {"rescues": [], "breaks": []}

    for r in results:
        for arm_id in ["H", "HD", "M"]:
            arm_key = f"fork_{arm_id.lower()}"
            cls = classify(r["a_success"], r[arm_key]["success"])
            div = first_divergence(
                r["fork_a"]["continuation_actions"],
                r[arm_key]["continuation_actions"],
            )

            if div is None:
                continue

            step_idx, a_action, arm_action = div

            # Get evidence state at divergence point
            # (from arm A's evidence logs up to that point)
            a_exposed = r["fork_a"]["evidence_exposed_log"][:step_idx]
            a_verified = r["fork_a"]["evidence_verified_log"][:step_idx]
            arm_exposed = r[arm_key]["evidence_exposed_log"][:step_idx]
            arm_verified = r[arm_key]["evidence_verified_log"][:step_idx]

            entry = {
                "task_id": r["task_id"],
                "category": r["category"],
                "arm": arm_id,
                "classification": cls,
                "divergence_step": step_idx,
                "a_action": a_action,
                "arm_action": arm_action,
                "a_actions_full": r["fork_a"]["continuation_actions"],
                "arm_actions_full": r[arm_key]["continuation_actions"],
                "a_u": r["fork_a"]["realized_utility"],
                "arm_u": r[arm_key]["realized_utility"],
                "u_delta": round(r[arm_key]["realized_utility"] - r["fork_a"]["realized_utility"], 4),
                "a_exposed_before_div": [list(e) for e in a_exposed],
                "a_verified_before_div": [list(e) for e in a_verified],
                "arm_exposed_before_div": [list(e) for e in arm_exposed],
                "arm_verified_before_div": [list(e) for e in arm_verified],
                "a_redundant_count": r["fork_a"]["redundant_action_count"],
                "arm_redundant_count": r[arm_key]["redundant_action_count"],
                "a_acs_before_terminal": r["fork_a"]["answer_condition_satisfied_before_terminal"],
                "arm_acs_before_terminal": r[arm_key]["answer_condition_satisfied_before_terminal"],
            }

            if cls == "RESCUE":
                analysis["rescues"].append(entry)
            elif cls == "BREAK":
                analysis["breaks"].append(entry)

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="I3.7e compact governor experiment")
    parser.add_argument(
        "--benchmark",
        default="experiments/v2b_i3_7/manifests/i3_7_evidence_benchmark_v1.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_7/development/i3_7e",
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
    print(f"  A=baseline, H=hypotheses, HD=hypotheses+discriminator, M=minimal decision state")

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

    results_path = output_dir / "compact_governor_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # First-divergence analysis
    print("\nComputing first-divergence analysis...")
    divergence = analyze_divergences(all_results)
    divergence_path = output_dir / "divergence_analysis_v1.json"
    divergence_path.write_text(json.dumps(divergence, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {divergence_path}")

    # Summary
    n = len(all_results)
    if n == 0:
        print("No tasks completed!")
        return

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    def arm_summary(arm_key: str, arm_label: str):
        u_vals = [r[arm_key]["realized_utility"] for r in all_results]
        successes = [r for r in all_results if r[arm_key]["success"]]
        u_mean = sum(u_vals) / n

        # Classification vs A
        classes = Counter(classify(r["a_success"], r[arm_key]["success"]) for r in all_results)
        rescues = classes.get("RESCUE", 0)
        breaks = classes.get("BREAK", 0)

        # Repaired metrics
        acs_before = sum(1 for r in all_results if r[arm_key]["answer_condition_satisfied_before_terminal"])
        terminal_matches = sum(1 for r in all_results if r[arm_key]["terminal_action_matches_condition"])
        condition_led = sum(1 for r in all_results if r[arm_key]["condition_led_to_success"])

        # Redundant actions
        redundant_total = sum(r[arm_key]["redundant_action_count"] for r in all_results)
        steps_total = sum(r[arm_key]["steps"] for r in all_results)
        redundant_rate = redundant_total / max(steps_total, 1)

        # Target evidence
        target_ret = sum(1 for r in all_results if r[arm_key]["target_evidence_retrieved"])
        target_ver = sum(1 for r in all_results if r[arm_key]["target_evidence_verified"])
        hyp_elim = sum(1 for r in all_results if r[arm_key]["hypothesis_eliminated"])

        return {
            "arm": arm_label,
            "mean_u": round(u_mean, 4),
            "success": f"{len(successes)}/{n}",
            "success_rate": round(len(successes) / n, 4),
            "rescues": rescues,
            "breaks": breaks,
            "acs_before_terminal": acs_before,
            "terminal_matches_condition": terminal_matches,
            "condition_led_to_success": condition_led,
            "redundant_action_count": redundant_total,
            "redundant_action_rate": round(redundant_rate, 4),
            "target_evidence_retrieved": target_ret,
            "target_evidence_verified": target_ver,
            "hypothesis_eliminated": hyp_elim,
            "mean_steps": round(sum(r[arm_key]["steps"] for r in all_results) / n, 2),
            "mean_model_calls": round(sum(r[arm_key]["model_calls"] for r in all_results) / n, 2),
        }

    arm_summaries = {
        "A": arm_summary("fork_a", "A"),
        "H": arm_summary("fork_h", "H"),
        "HD": arm_summary("fork_hd", "HD"),
        "M": arm_summary("fork_m", "M"),
    }

    # Gates (stricter)
    gates = {}
    for arm_id in ["H", "HD", "M"]:
        arm_key = f"fork_{arm_id.lower()}"
        s = arm_summaries[arm_id]
        a_success = arm_summaries["A"]["success_rate"]
        a_u = arm_summaries["A"]["mean_u"]
        gates[arm_id] = {
            "success_ge_baseline": s["success_rate"] >= a_success,
            "breaks_le_1": s["breaks"] <= 1,
            "rescues_ge_1": s["rescues"] >= 1,
            "mean_u_ge_baseline": s["mean_u"] >= a_u,
            "rescues_gt_breaks": s["rescues"] > s["breaks"],
        }

    summary = {
        "schema": "DAPH_V2B_I3_7E_COMPACT_GOVERNOR_V1",
        "n_tasks": n,
        "arms": {
            "A": "semantic evidence baseline",
            "H": "hypotheses only",
            "HD": "hypotheses + discriminator",
            "M": "minimal decision state",
        },
        "arm_summaries": arm_summaries,
        "gates": gates,
        "divergence_summary": {
            "total_rescues": len(divergence["rescues"]),
            "total_breaks": len(divergence["breaks"]),
        },
    }

    summary_path = output_dir / "compact_governor_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Summary saved: {summary_path}")

    # Print summary
    print(f"\n{'='*90}")
    print("I3.7e COMPACT GOVERNOR EXPERIMENT SUMMARY")
    print(f"{'='*90}")
    print(f"  Tasks: {n}")
    print(f"\n  {'Arm':<6} {'Success':<12} {'Mean U':<12} {'Resc':<6} {'Brk':<6} {'Redund':<8} {'ACS':<6} {'Match':<6} {'Led':<6} {'Steps':<8}")
    print(f"  {'-'*80}")
    for arm_id in ["A", "H", "HD", "M"]:
        s = arm_summaries[arm_id]
        print(f"  {arm_id:<6} {s['success']:<12} {s['mean_u']:+.4f}    {s['rescues']:<6} {s['breaks']:<6} {s['redundant_action_rate']:<8.4f} {s['acs_before_terminal']:<6} {s['terminal_matches_condition']:<6} {s['condition_led_to_success']:<6} {s['mean_steps']:<8.2f}")

    print(f"\n  GATES (per arm vs A):")
    for arm_id in ["H", "HD", "M"]:
        g = gates[arm_id]
        all_pass = all(g.values())
        print(f"    {arm_id}: {'ALL PASS' if all_pass else 'FAIL'}", end="")
        for gate_name, passed in g.items():
            if not passed:
                print(f"  {gate_name}=FAIL", end="")
        print()

    # Print divergence analysis
    print(f"\n  DIVERGENCE ANALYSIS:")
    print(f"    Total rescues: {len(divergence['rescues'])}")
    print(f"    Total breaks: {len(divergence['breaks'])}")

    if divergence["rescues"]:
        print(f"\n  RESCUE DETAILS:")
        for entry in divergence["rescues"]:
            print(f"    {entry['task_id']} arm={entry['arm']} cat={entry['category']}")
            print(f"      div step={entry['divergence_step']}: A={entry['a_action']} -> {entry['arm']}_action={entry['arm_action']}")
            print(f"      A actions: {entry['a_actions_full']}")
            print(f"      {entry['arm']} actions: {entry['arm_actions_full']}")
            print(f"      U: A={entry['a_u']:+.2f}, {entry['arm']}={entry['arm_u']:+.2f}, delta={entry['u_delta']:+.2f}")
            print(f"      A redundant={entry['a_redundant_count']}, {entry['arm']} redundant={entry['arm_redundant_count']}")
            print(f"      A ACS_before={entry['a_acs_before_terminal']}, {entry['arm']} ACS_before={entry['arm_acs_before_terminal']}")

    if divergence["breaks"]:
        print(f"\n  BREAK DETAILS:")
        for entry in divergence["breaks"]:
            print(f"    {entry['task_id']} arm={entry['arm']} cat={entry['category']}")
            print(f"      div step={entry['divergence_step']}: A={entry['a_action']} -> {entry['arm']}_action={entry['arm_action']}")
            print(f"      A actions: {entry['a_actions_full']}")
            print(f"      {entry['arm']} actions: {entry['arm_actions_full']}")
            print(f"      U: A={entry['a_u']:+.2f}, {entry['arm']}={entry['arm_u']:+.2f}, delta={entry['u_delta']:+.2f}")
            print(f"      A redundant={entry['a_redundant_count']}, {entry['arm']} redundant={entry['arm_redundant_count']}")
            print(f"      A ACS_before={entry['a_acs_before_terminal']}, {entry['arm']} ACS_before={entry['arm_acs_before_terminal']}")


if __name__ == "__main__":
    main()
