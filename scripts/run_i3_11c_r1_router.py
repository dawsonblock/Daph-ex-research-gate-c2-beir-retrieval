#!/usr/bin/env python3
"""I3.11c: R1 Epistemic-Conflict Router — A1 → T2 → M3 on fresh decorrelated corpus.

R1 is a dynamic phase-switching router:
  - Start in A1 (acquisition representation)
  - Before each model call, compute M3's state estimator internally
    (without exposing M3 context to the LLM)
  - When T2 fires (all hypotheses eliminated by visible verified evidence),
    permanently latch to M3 (resolution representation)
  - M3 context is exposed to the LLM only after T2 fires

T2 is an observed-state phase transition detector, NOT a predictor.

Arms:
  A1 = always baseline + affordances
  M3 = always MDSG state + affordances
  R1 = A1 until T2, then M3 (latched)

Co-primary criteria:
  C1: LCB_95(U_R1 - U_A1) > 0
  C2: LCB_95(U_R1 - U_M3) > 0

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_11c_r1_router.py \\
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
from collections import Counter
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

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
    EvidenceExecutor, EvidenceBenchmark, save_evidence_benchmark,
    initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.evidence_benchmark.structural_ood_generator import (
    STRUCTURAL_TEMPLATES, _seeded_rng,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output


# ---------------------------------------------------------------------------
# Decorrelated corpus under current epistemic algebra
# ---------------------------------------------------------------------------

def _make_h1_h2(template: dict):
    h1 = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition=template["h1_proposition"],
        answer_action=template["h1_answer"],
        answer_payload=template["h1_payload"],
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=template["h2_proposition"],
        answer_action=template["h2_answer"],
        answer_payload=template["h2_payload"],
    )
    return h1, h2


def _noise_evidence(eid: str, subject: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        proposition=f"A tangential reference mentions {subject} in passing without substantive analysis.",
        source_class="search",
        supports=(),
        contradicts=(),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=False,
        verify_result="MISSING",
    )


# --- T2-should-activate categories ---

def gen_conflict_unresolved_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """Genuine unresolvable conflict. Both sides SUFFICIENT. T2 should fire.
    hidden_count noise items added. T2 should still fire (tests T2 > T1)."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A definitively confirms {subject}.",
            source_class="primary",
            supports=("H1",), contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B definitively refutes {subject}.",
            source_class="primary",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
    ]
    search_exposes = ()
    for i in range(hidden_count):
        eid = f"E{3+i}"
        evidence.append(_noise_evidence(eid, subject))
        search_exposes = search_exposes + (eid,)

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"conflict_unresolved_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


# --- T2-should-NOT-activate categories ---

def gen_single_verify_ready_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """1 visible SUFFICIENT evidence. After VERIFY, ready to ANSWER.
    T2 should never fire (only one hypothesis eliminated)."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"The primary documentation confirms that {subject}.",
            source_class="primary",
            supports=("H1",), contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
    ]
    search_exposes = ()
    if hidden_count >= 1:
        evidence.append(EvidenceItem(
            evidence_id="E2",
            proposition=f"A secondary source does not address {subject}.",
            source_class="search",
            supports=("H2",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False, verify_result="MISSING",
        ))
        search_exposes = ("E2",)
    if hidden_count >= 2:
        evidence.append(_noise_evidence("E3", subject))
        search_exposes = search_exposes + ("E3",)

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"single_verify_ready_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5, budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_triple_verify_ready_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """3 visible, verify all 3, answer. H2 evidence FALSIFIED.
    T2 should never fire (H1 remains viable)."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A claims {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B also claims {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"Source C contradicts, claiming not-{subject}.",
            source_class="initial",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="FALSIFIED",
        ),
    ]
    search_exposes = ()
    for i in range(hidden_count):
        eid = f"E{4+i}"
        evidence.append(_noise_evidence(eid, subject))
        search_exposes = search_exposes + (eid,)

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"triple_verify_ready_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_varying_visible_split_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """3 visible + hidden. H2 evidence FALSIFIED. T2 should never fire."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A confirms {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B also confirms {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"Source C contradicts, claiming not-{subject}.",
            source_class="initial",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="FALSIFIED",
        ),
    ]
    search_exposes = ()
    if hidden_count >= 1:
        evidence.append(EvidenceItem(
            evidence_id="E4",
            proposition=f"A hidden source provides additional confirmation of {subject}.",
            source_class="search",
            supports=("H1",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False, verify_result="SUFFICIENT",
        ))
        search_exposes = ("E4",)
    if hidden_count >= 2:
        evidence.append(_noise_evidence("E5", subject))
        search_exposes = search_exposes + ("E5",)

    oracle = ("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER")
    if hidden_count >= 1:
        oracle = ("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "SEARCH_MORE:E4", "VERIFY:E4", "ANSWER")

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"varying_visible_split_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5, budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=search_exposes,
        oracle_resolution_path=oracle,
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_early_false_ready_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """Apparent conflict later resolves. E2 FALSIFIED. T2 should never fire.
    hidden_count >= 1 (E3 hidden, must retrieve)."""
    subject = template["subject"]
    h1_answer = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition=template["h1_proposition"],
        answer_action=DecisionAction.ANSWER,
        answer_payload=template["h1_payload"],
    )
    h2_answer = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=f"the documentation refutes the claim about {subject}, so the answer should be ANSWER with refutation",
        answer_action=DecisionAction.ANSWER,
        answer_payload=f"refuted: {template['h1_payload']}",
    )

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"An initial source claims {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Another source contradicts, claiming not-{subject}.",
            source_class="initial",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="FALSIFIED",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"A definitive source refutes the claim about {subject}.",
            source_class="primary",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False, verify_result="SUFFICIENT",
        ),
    ]
    retrieve_exposes = ("E3",)
    search_exposes = ()
    if hidden_count >= 2:
        evidence.append(_noise_evidence("E4", subject))
        search_exposes = ("E4",)

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"early_false_ready_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=(h1_answer, h2_answer),
        evidence_items=tuple(evidence),
        retrieve_exposes=retrieve_exposes, search_exposes=search_exposes,
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H2",
    )


def gen_stale_support_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """Visible support is STALE. Must search for current evidence.
    T2 should never fire (H1 becomes viable after current evidence found)."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"An older source claims {subject}, but may be outdated.",
            source_class="initial",
            supports=("H1",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.STALE,
            retrieved=True, verify_result="STALE",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"A source is silent on {subject}.",
            source_class="initial",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="FALSIFIED",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"A current source confirms {subject}.",
            source_class="search",
            supports=("H1",), contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False, verify_result="SUFFICIENT",
        ),
    ]
    search_exposes = ("E3",)
    if hidden_count >= 1:
        evidence.append(_noise_evidence("E4", subject))
        search_exposes = search_exposes + ("E4",)

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"stale_support_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "SEARCH_MORE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_multi_hypothesis_ambiguity_h(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """3 hypotheses, two look viable. Hidden evidence breaks tie.
    T2 should never fire (multiple viable, not all eliminated)."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)
    h3 = EvidenceHypothesis(
        hypothesis_id="H3",
        proposition=f"the documentation is ambiguous about {subject}, so the answer should be DEFER",
        answer_action=DecisionAction.DEFER,
        answer_payload="documentation is ambiguous",
    )

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A claims {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B claims not-{subject}.",
            source_class="initial",
            supports=("H2",), contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"A definitive source confirms {subject}.",
            source_class="search",
            supports=("H1",), contradicts=("H2", "H3"),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False, verify_result="SUFFICIENT",
        ),
    ]
    search_exposes = ("E3",)
    if hidden_count >= 1:
        evidence.append(_noise_evidence("E4", subject))
        search_exposes = search_exposes + ("E4",)

    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category=f"multi_hypothesis_ambiguity_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=(h1, h2, h3),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=search_exposes,
        oracle_resolution_path=(
            "VERIFY:E1", "VERIFY:E2", "SEARCH_MORE:E3", "VERIFY:E3", "ANSWER",
        ),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_bilateral_one_side_falsified(
    task_id: str, template: dict, rng: random.Random,
) -> EvidenceTask:
    """Bilateral visible relations but only one side verifies SUFFICIENT.
    The other side FALSIFIED. T2 should never fire.
    Tests: bilateral visible support does not mean T2 fires."""
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A confirms {subject}.",
            source_class="initial",
            supports=("H1",), contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B claims not-{subject}.",
            source_class="initial",
            supports=("H2",), contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True, verify_result="FALSIFIED",
        ),
    ]
    return EvidenceTask(
        task_id=task_id, split="r1_stress_v1",
        category="bilateral_one_falsified",
        task_summary=f"Determine {subject}.",
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def generate_r1_corpus(split: str = "r1_stress_v1") -> list[EvidenceTask]:
    """Generate I3.11c corpus under current epistemic algebra.

    Distribution (300 tasks):
      T2-should-activate:
        conflict_unresolved × {h0, h1, h2}:  30 + 30 + 30 = 90
          (all-eliminated with noise — tests T2 > T1)

      T2-should-NOT-activate:
        single_verify_ready × {h0, h1, h2}:  25 + 25 + 25 = 75
        triple_verify_ready × {h0, h1, h2}:  10 + 10 + 10 = 30
        varying_visible_split × {h0, h1, h2}: 15 + 15 + 15 = 45
        early_false_ready × {h1, h2}:         15 + 15     = 30
        stale_support × {h0, h1}:             10 + 10     = 20
        multi_hypothesis_ambiguity × {h0, h1}: 5 + 5      = 10
        bilateral_one_falsified:                           10

    Total: 90 + 75 + 30 + 45 + 30 + 20 + 10 + 10 = 310 → trim to 300
    """
    target = [
        ("conflict_unresolved", 0, 30),
        ("conflict_unresolved", 1, 30),
        ("conflict_unresolved", 2, 30),
        ("single_verify_ready", 0, 25),
        ("single_verify_ready", 1, 25),
        ("single_verify_ready", 2, 25),
        ("triple_verify_ready", 0, 10),
        ("triple_verify_ready", 1, 10),
        ("triple_verify_ready", 2, 10),
        ("varying_visible_split", 0, 15),
        ("varying_visible_split", 1, 15),
        ("varying_visible_split", 2, 15),
        ("early_false_ready", 1, 15),
        ("early_false_ready", 2, 15),
        ("stale_support", 0, 10),
        ("stale_support", 1, 10),
        ("multi_hypothesis_ambiguity", 0, 5),
        ("multi_hypothesis_ambiguity", 1, 5),
        ("bilateral_one_falsified", 0, 10),
    ]

    generators = {
        "conflict_unresolved": gen_conflict_unresolved_h,
        "single_verify_ready": gen_single_verify_ready_h,
        "triple_verify_ready": gen_triple_verify_ready_h,
        "varying_visible_split": gen_varying_visible_split_h,
        "early_false_ready": gen_early_false_ready_h,
        "stale_support": gen_stale_support_h,
        "multi_hypothesis_ambiguity": gen_multi_hypothesis_ambiguity_h,
    }

    tasks: list[EvidenceTask] = []
    task_idx = 0
    for category, hidden_count, count in target:
        for i in range(count):
            task_id = f"{split}_{task_idx:04d}"
            template = STRUCTURAL_TEMPLATES[task_idx % len(STRUCTURAL_TEMPLATES)]
            rng = _seeded_rng(task_id)
            if category == "bilateral_one_falsified":
                task = gen_bilateral_one_falsified(task_id, template, rng)
            else:
                gen = generators[category]
                task = gen(task_id, template, rng, hidden_count)
            tasks.append(task)
            task_idx += 1

    return tasks


# ---------------------------------------------------------------------------
# R1 trajectory runner — hybrid A1→T2→M3
# ---------------------------------------------------------------------------

def run_r1_trajectory(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
    fork_label: str,
) -> dict[str, Any]:
    """Run R1 hybrid trajectory: A1 until T2, then M3 (latched).

    Before each model call:
      1. Build snapshot
      2. Compute M3 state internally (build_mdsg_state_with_affordances_packet)
      3. Check T2: all hypotheses eliminated
      4. If T2 fires and not yet triggered, latch to M3
      5. If triggered, use M3 packet + prompt; else A1 packet + prompt

    No extra model call. M3 state is deterministic.
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
    evidence_exposed_log: list[tuple[str, ...]] = []
    evidence_verified_log: list[tuple[str, ...]] = []
    step_costs: list[float] = []
    total_action_cost = 0.0
    terminal_reward = 0.0

    # R1 routing telemetry
    r1_triggered = False
    r1_trigger_step: int | None = None
    r1_trigger_decision_state: str | None = None
    r1_pre_trigger_steps = 0
    r1_post_trigger_steps = 0
    r1_pre_trigger_utility = 0.0
    r1_post_trigger_utility = 0.0

    # Per-step routing log
    routing_log: list[dict[str, Any]] = []

    # Per-step decision state log (from internal M3 computation)
    decision_state_log: list[dict[str, Any]] = []

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps

    n_hypotheses = len(task.hypotheses)

    for step_id in range(max_steps):
        # Build snapshot
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Compute M3 state internally (always, for trigger check)
        internal_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        internal_summary = internal_m3_packet.get("decision_state_summary", {})
        internal_state = internal_summary.get("decision_state")
        eliminated = internal_summary.get("eliminated_hypotheses", [])

        # T2 trigger check: all hypotheses eliminated
        t2_fires = (
            len(eliminated) == n_hypotheses
            and n_hypotheses > 0
        )

        # Latch: first T2 activation permanently switches to M3
        if not r1_triggered and t2_fires:
            r1_triggered = True
            r1_trigger_step = step_id
            r1_trigger_decision_state = internal_state

        # Select representation based on trigger state
        if r1_triggered:
            packet = internal_m3_packet
            system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
            current_rep = "M3"
        else:
            packet = i3_7e.build_baseline_with_affordances_packet(evidence_snapshot)
            system_prompt = i3_7e.BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT
            current_rep = "A1"

        # Log routing decision
        routing_log.append({
            "step": step_id,
            "representation": current_rep,
            "triggered": r1_triggered,
            "t2_fires": t2_fires,
            "decision_state": internal_state,
            "eliminated_hypotheses": eliminated,
        })

        # Log decision state (always computed internally)
        decision_state_log.append({
            "step": step_id,
            "decision_state": internal_state,
            "live_hypotheses": internal_summary.get("live_hypotheses", []),
            "eliminated_hypotheses": eliminated,
            "representation": current_rep,
        })

        user_prompt = i3_7e.evidence_packet_json(packet)

        # Call model
        backend = DeepSeekBackend()
        backend.task_id = task.task_id
        backend.condition = f"i3_11c_R1"
        backend.pair_id = f"i3_11c:{task.task_id}:{fork_label}:step{step_id}"

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

        # Execute
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
        evidence_exposed_log.append(exec_res.evidence_exposed)
        evidence_verified_log.append(exec_res.evidence_verified)

        # Track routing telemetry
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

    redundant_action_rate = 0.0  # computed below if needed

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
        "decision_state_log": decision_state_log,
        # R1 routing telemetry
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
# Experiment runner
# ---------------------------------------------------------------------------

def counterbalance_3arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    arms = ["A1", "M3", "R1"]
    perms = list(itertools.permutations(arms))
    return list(perms[int(h[:8], 16) % len(perms)])


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    arm_modes = {
        "A1": "BASELINE_WITH_AFFORDANCES",
        "M3": "MDSG_STATE_WITH_AFFORDANCES",
    }

    results: dict[str, dict] = {}
    for arm_id in ["A1", "M3"]:
        results[arm_id] = i3_7e.run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=arm_modes[arm_id], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )

    # R1 hybrid arm
    results["R1"] = run_r1_trajectory(
        task=task, budget=budget, utility=utility,
        api_key=api_key, fork_label="armR1",
    )

    r1 = results["R1"]
    a1 = results["A1"]
    m3 = results["M3"]

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": sum(1 for e in task.evidence_items if not e.retrieved),
        "oracle_steps": len(task.oracle_resolution_path),
        "u_a1": a1["realized_utility"],
        "u_m3": m3["realized_utility"],
        "u_r1": r1["realized_utility"],
        "r1_delta_vs_a1": round(r1["realized_utility"] - a1["realized_utility"], 4),
        "r1_delta_vs_m3": round(r1["realized_utility"] - m3["realized_utility"], 4),
        "a1_success": a1["success"],
        "m3_success": m3["success"],
        "r1_success": r1["success"],
        "a1_steps": a1["steps"],
        "m3_steps": m3["steps"],
        "r1_steps": r1["steps"],
        "r1_triggered": r1["r1_triggered"],
        "r1_trigger_step": r1["r1_trigger_step"],
        "r1_pre_trigger_steps": r1["r1_pre_trigger_steps"],
        "r1_post_trigger_steps": r1["r1_post_trigger_steps"],
        "m3_rescues_vs_a1": (not a1["success"]) and m3["success"],
        "m3_breaks_vs_a1": a1["success"] and (not m3["success"]),
        "r1_rescues_vs_a1": (not a1["success"]) and r1["success"],
        "r1_breaks_vs_a1": a1["success"] and (not r1["success"]),
        "r1_rescues_vs_m3": (not m3["success"]) and r1["success"],
        "r1_breaks_vs_m3": m3["success"] and (not r1["success"]),
        "fork_a1": a1,
        "fork_m3": m3,
        "fork_r1": r1,
    }


def paired_bootstrap_ci(deltas, n_iterations=10000, seed=42):
    rng = random.Random(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    return boot_means[int(0.025 * n_iterations)], boot_means[int(0.975 * n_iterations)]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_11/development/i3_11c_r1_router",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.11c: R1 Epistemic-Conflict Router — A1 → T2 → M3")
    print("  R1: A1 until all hypotheses eliminated (T2), then M3 (latched)")
    print("  T2 is an observed-state phase transition detector, not a predictor")
    print()

    tasks = generate_r1_corpus(split="r1_stress_v1")
    print(f"  Generated {len(tasks)} tasks")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution:")
    for cat in sorted(cats.keys()):
        print(f"    {cat:<40} {cats[cat]}")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Save corpus manifest
    benchmark = EvidenceBenchmark(
        benchmark_id="i3_11c_r1_stress_v1",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_11/manifests/r1_stress_v1.json")

    # Oracle validation
    executor = EvidenceExecutor()
    all_pass = True
    for task in tasks:
        runtime = initial_evidence_runtime(task, ResourceState(budget))
        current = runtime
        final = None
        for step in task.oracle_resolution_path:
            parts = step.split(":")
            action = DecisionAction(parts[0])
            target = parts[1] if len(parts) > 1 else None
            final = executor.execute(current, action, target_evidence_id=target)
            current = final.runtime
            if final.terminal:
                break
        if not final.task_success:
            all_pass = False
            print(f"  ORACLE FAIL: {task.task_id} ({task.category})")
    print(f"\n  All oracle paths succeed: {all_pass}")
    if not all_pass:
        sys.exit(1)

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks with {args.workers} workers...")
    all_results: list[dict[str, Any]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_task, task, budget, utility, api_key): task
                   for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Completed {completed}/{len(tasks)} tasks...")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    print(f"\nCompleted {len(all_results)} tasks")

    results_path = output_dir / "r1_stress_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # === Analysis ===
    n = len(all_results)
    a1_s = sum(1 for r in all_results if r["a1_success"])
    m3_s = sum(1 for r in all_results if r["m3_success"])
    r1_s = sum(1 for r in all_results if r["r1_success"])

    u_a1 = sum(r["u_a1"] for r in all_results) / n
    u_m3 = sum(r["u_m3"] for r in all_results) / n
    u_r1 = sum(r["u_r1"] for r in all_results) / n

    r1_a1_deltas = [r["r1_delta_vs_a1"] for r in all_results]
    r1_m3_deltas = [r["r1_delta_vs_m3"] for r in all_results]
    m3_a1_deltas = [r["u_m3"] - r["u_a1"] for r in all_results]

    r1_a1_ci = paired_bootstrap_ci(r1_a1_deltas)
    r1_m3_ci = paired_bootstrap_ci(r1_m3_deltas)
    m3_a1_ci = paired_bootstrap_ci(m3_a1_deltas)

    mc_r1_a1 = mcnemar([r["a1_success"] for r in all_results], [r["r1_success"] for r in all_results])
    mc_r1_m3 = mcnemar([r["m3_success"] for r in all_results], [r["r1_success"] for r in all_results])

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    r1_a1_cl = Counter(classify(r["a1_success"], r["r1_success"]) for r in all_results)
    r1_m3_cl = Counter(classify(r["m3_success"], r["r1_success"]) for r in all_results)
    m3_a1_cl = Counter(classify(r["a1_success"], r["m3_success"]) for r in all_results)

    a1_steps = sum(r["a1_steps"] for r in all_results)
    m3_steps = sum(r["m3_steps"] for r in all_results)
    r1_steps = sum(r["r1_steps"] for r in all_results)

    # R1 routing telemetry
    r1_trigger_count = sum(1 for r in all_results if r["r1_triggered"])
    r1_trigger_steps = [r["r1_trigger_step"] for r in all_results if r["r1_triggered"]]
    r1_trigger_mean = sum(r1_trigger_steps) / len(r1_trigger_steps) if r1_trigger_steps else None
    r1_trigger_median = sorted(r1_trigger_steps)[len(r1_trigger_steps) // 2] if r1_trigger_steps else None
    r1_pre_trigger_mean = sum(r["r1_pre_trigger_steps"] for r in all_results if r["r1_triggered"]) / max(r1_trigger_count, 1)
    r1_post_trigger_mean = sum(r["r1_post_trigger_steps"] for r in all_results if r["r1_triggered"]) / max(r1_trigger_count, 1)

    # Useful activation precision: of tasks where R1 triggered, how many were M3 rescues?
    r1_triggered_rescues = sum(1 for r in all_results if r["r1_triggered"] and r["m3_rescues_vs_a1"])
    useful_activation_precision = r1_triggered_rescues / max(r1_trigger_count, 1)

    # Coverage: of M3 rescue opportunities, how many did R1 trigger on?
    m3_rescues = sum(1 for r in all_results if r["m3_rescues_vs_a1"])
    r1_coverage = sum(1 for r in all_results if r["m3_rescues_vs_a1"] and r["r1_triggered"]) / max(m3_rescues, 1)

    # False activation: of A1-success tasks, how many did R1 trigger on?
    a1_success_count = sum(1 for r in all_results if r["a1_success"])
    r1_false_activation = sum(1 for r in all_results if r["a1_success"] and r["r1_triggered"]) / max(a1_success_count, 1)

    # Subgroup analysis
    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        ca1 = sum(1 for r in cr if r["a1_success"])
        cm3 = sum(1 for r in cr if r["m3_success"])
        cr1 = sum(1 for r in cr if r["r1_success"])
        cr1_trig = sum(1 for r in cr if r["r1_triggered"])
        subgroups[cat] = {
            "n": cn,
            "a1_success": f"{ca1}/{cn} ({ca1/cn*100:.1f}%)",
            "m3_success": f"{cm3}/{cn} ({cm3/cn*100:.1f}%)",
            "r1_success": f"{cr1}/{cn} ({cr1/cn*100:.1f}%)",
            "r1_triggered": f"{cr1_trig}/{cn}",
            "mean_u_a1": round(sum(r["u_a1"] for r in cr) / cn, 4),
            "mean_u_m3": round(sum(r["u_m3"] for r in cr) / cn, 4),
            "mean_u_r1": round(sum(r["u_r1"] for r in cr) / cn, 4),
            "m3_rescues_vs_a1": sum(1 for r in cr if r["m3_rescues_vs_a1"]),
            "r1_rescues_vs_a1": sum(1 for r in cr if r["r1_rescues_vs_a1"]),
            "r1_breaks_vs_a1": sum(1 for r in cr if r["r1_breaks_vs_a1"]),
            "r1_rescues_vs_m3": sum(1 for r in cr if r["r1_rescues_vs_m3"]),
            "r1_breaks_vs_m3": sum(1 for r in cr if r["r1_breaks_vs_m3"]),
            "mean_steps_a1": round(sum(r["a1_steps"] for r in cr) / cn, 2),
            "mean_steps_m3": round(sum(r["m3_steps"] for r in cr) / cn, 2),
            "mean_steps_r1": round(sum(r["r1_steps"] for r in cr) / cn, 2),
        }

    # Frozen claims
    frozen_claims = {
        "C1_r1_a1_ci_positive": r1_a1_ci[0] > 0,
        "C2_r1_m3_ci_positive": r1_m3_ci[0] > 0,
        "C3_r1_success_ge_m3_minus_1pp": r1_s >= m3_s - 3,
        "C4_r1_steps_lt_m3": r1_steps < m3_steps,
        "C5_r1_rescues_gt_breaks_vs_a1": r1_a1_cl.get("RESCUE", 0) > r1_a1_cl.get("BREAK", 0),
        "C6_false_activation_lt_5pct": r1_false_activation < 0.05,
        "C7_no_catastrophic_subgroup": not any(
            (sum(1 for r in all_results if r["category"] == cat and r["r1_success"]) /
             max(len([r for r in all_results if r["category"] == cat]), 1)) <
            max(
                sum(1 for r in all_results if r["category"] == cat and r["a1_success"]) /
                max(len([r for r in all_results if r["category"] == cat]), 1),
                sum(1 for r in all_results if r["category"] == cat and r["m3_success"]) /
                max(len([r for r in all_results if r["category"] == cat]), 1),
            ) - 0.10
            for cat in categories
        ),
    }

    summary = {
        "schema": "DAPH_V2B_I3_11C_R1_ROUTER_V1",
        "n_tasks": n,
        "arms": {
            "A1": "baseline + public affordances",
            "M3": "frozen MDSG-StateWithAffordances",
            "R1": "A1 until T2 (all_hypotheses_eliminated), then M3 (latched)",
        },
        "overall": {
            "mean_u": {"A1": round(u_a1, 4), "M3": round(u_m3, 4), "R1": round(u_r1, 4)},
            "success": {"A1": f"{a1_s}/{n}", "M3": f"{m3_s}/{n}", "R1": f"{r1_s}/{n}"},
            "bootstrap_ci_r1_a1": [round(r1_a1_ci[0], 4), round(r1_a1_ci[1], 4)],
            "bootstrap_ci_r1_m3": [round(r1_m3_ci[0], 4), round(r1_m3_ci[1], 4)],
            "bootstrap_ci_m3_a1": [round(m3_a1_ci[0], 4), round(m3_a1_ci[1], 4)],
            "mcnemar_r1_a1": mc_r1_a1,
            "mcnemar_r1_m3": mc_r1_m3,
            "r1_a1_classification": dict(r1_a1_cl),
            "r1_m3_classification": dict(r1_m3_cl),
            "m3_a1_classification": dict(m3_a1_cl),
            "mean_steps": {
                "A1": round(a1_steps / n, 2), "M3": round(m3_steps / n, 2),
                "R1": round(r1_steps / n, 2),
            },
        },
        "r1_routing_telemetry": {
            "trigger_rate": round(r1_trigger_count / n, 4),
            "trigger_count": r1_trigger_count,
            "trigger_step_mean": round(r1_trigger_mean, 2) if r1_trigger_mean is not None else None,
            "trigger_step_median": r1_trigger_median,
            "useful_activation_precision": round(useful_activation_precision, 4),
            "coverage_of_m3_rescues": round(r1_coverage, 4),
            "false_activation_on_a1_success": round(r1_false_activation, 4),
            "a1_prefix_length_mean": round(r1_pre_trigger_mean, 2),
            "m3_post_trigger_length_mean": round(r1_post_trigger_mean, 2),
        },
        "subgroups": subgroups,
        "frozen_claims": frozen_claims,
    }

    summary_path = output_dir / "r1_stress_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print results
    print(f"\n{'='*82}")
    print("I3.11c R1 EPISTEMIC-CONFLICT ROUTER: A1 vs M3 vs R1")
    print(f"{'='*82}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A1={u_a1:+.4f}  M3={u_m3:+.4f}  R1={u_r1:+.4f}")
    print(f"  Success:       A1={a1_s}/{n}  M3={m3_s}/{n}  R1={r1_s}/{n}")
    print(f"\n  Bootstrap 95% CI:")
    print(f"    R1-A1: [{r1_a1_ci[0]:+.4f}, {r1_a1_ci[1]:+.4f}]  <-- CO-PRIMARY (R1 must beat A1)")
    print(f"    R1-M3: [{r1_m3_ci[0]:+.4f}, {r1_m3_ci[1]:+.4f}]  <-- CO-PRIMARY (R1 must beat M3)")
    print(f"    M3-A1: [{m3_a1_ci[0]:+.4f}, {m3_a1_ci[1]:+.4f}]")
    print(f"\n  McNemar:")
    print(f"    R1-A1: b={mc_r1_a1['b']}, c={mc_r1_a1['c']}, p={mc_r1_a1['p']}")
    print(f"    R1-M3: b={mc_r1_m3['b']}, c={mc_r1_m3['c']}, p={mc_r1_m3['p']}")
    print(f"\n  R1 vs A1: rescues={r1_a1_cl.get('RESCUE',0)}, breaks={r1_a1_cl.get('BREAK',0)}")
    print(f"  R1 vs M3: rescues={r1_m3_cl.get('RESCUE',0)}, breaks={r1_m3_cl.get('BREAK',0)}")
    print(f"\n  Steps:  A1={a1_steps/n:.2f}  M3={m3_steps/n:.2f}  R1={r1_steps/n:.2f}")

    print(f"\n  R1 ROUTING TELEMETRY:")
    print(f"    Trigger rate: {r1_trigger_count}/{n} ({r1_trigger_count/n*100:.1f}%)")
    print(f"    Trigger step: mean={r1_trigger_mean:.2f}, median={r1_trigger_median}" if r1_trigger_mean else "    Trigger step: N/A")
    print(f"    Useful activation precision: {useful_activation_precision:.4f}")
    print(f"    Coverage of M3 rescues: {r1_coverage:.4f}")
    print(f"    False activation on A1-success: {r1_false_activation:.4f}")
    print(f"    A1 prefix length (mean): {r1_pre_trigger_mean:.2f}")
    print(f"    M3 post-trigger length (mean): {r1_post_trigger_mean:.2f}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in frozen_claims.items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")

    print(f"\n{'='*82}")
    print("SUBGROUP ANALYSIS")
    print(f"{'='*82}")
    print(f"  {'Category':<40} {'n':>3} {'A1%':>6} {'M3%':>6} {'R1%':>6} {'R1_trig':>7} {'A1_U':>8} {'M3_U':>8} {'R1_U':>8}")
    for cat in sorted(subgroups.keys()):
        sg = subgroups[cat]
        a1p = sg["a1_success"].split("(")[1].rstrip(")")
        m3p = sg["m3_success"].split("(")[1].rstrip(")")
        r1p = sg["r1_success"].split("(")[1].rstrip(")")
        print(f"  {cat:<40} {sg['n']:>3} {a1p:>6} {m3p:>6} {r1p:>6} {sg['r1_triggered']:>7} "
              f"{sg['mean_u_a1']:>+8.2f} {sg['mean_u_m3']:>+8.2f} {sg['mean_u_r1']:>+8.2f}")


if __name__ == "__main__":
    main()
