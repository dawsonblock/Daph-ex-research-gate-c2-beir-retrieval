#!/usr/bin/env python3
"""I3.11d: R1 confirmation under expanded hypothesis topology.

Uses the exact repaired R1 implementation from I3.11c byte-for-byte.
Tests whether T2 (all hypotheses eliminated) is a robust representation-
phase boundary across:
  - 2, 3, 4, 5 hypotheses
  - Only subset eliminated (T2 should NOT fire)
  - All eliminated after different verification step counts
  - Conflict appearing late (not always step 2)
  - Irrelevant evidence before and after conflict
  - Stale/falsified apparent contradictions
  - Asymmetric conflict graphs
  - Tasks where T2 never occurs

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_11d_r1_confirm.py \\
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

# Import the exact R1 trajectory runner from I3.11c
_spec_c = importlib.util.spec_from_file_location(
    "i3_11c", ROOT / "scripts" / "run_i3_11c_r1_router.py")
i3_11c = importlib.util.module_from_spec(_spec_c)
_spec_c.loader.exec_module(i3_11c)
run_r1_trajectory = i3_11c.run_r1_trajectory

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


# ---------------------------------------------------------------------------
# Hypothesis builders for variable topology
# ---------------------------------------------------------------------------

def _make_hyps(n: int, template: dict) -> tuple[EvidenceHypothesis, ...]:
    """Create n hypotheses. H1=ANSWER, H2=DEFER, H3+=DEFER variants."""
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
    hyps = [h1, h2]
    for i in range(3, n + 1):
        hyps.append(EvidenceHypothesis(
            hypothesis_id=f"H{i}",
            proposition=f"the documentation is ambiguous about {template['subject']}, so the answer should be DEFER",
            answer_action=DecisionAction.DEFER,
            answer_payload=f"insufficient evidence (hypothesis {i})",
        ))
    return tuple(hyps)


def _noise(eid: str, subject: str, retrieved: bool = False) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        proposition=f"A tangential reference mentions {subject} in passing.",
        source_class="search",
        supports=(), contradicts=(),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved, verify_result="MISSING",
    )


def _suff(eid: str, prop: str, supports: tuple, contradicts: tuple,
          retrieved: bool = True) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid, proposition=prop,
        source_class="primary" if retrieved else "search",
        supports=supports, contradicts=contradicts,
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved, verify_result="SUFFICIENT",
    )


def _falsified(eid: str, prop: str, supports: tuple, contradicts: tuple,
               retrieved: bool = True) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid, proposition=prop,
        source_class="initial",
        supports=supports, contradicts=contradicts,
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved, verify_result="FALSIFIED",
    )


def _stale(eid: str, prop: str, supports: tuple, retrieved: bool = True) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid, proposition=prop,
        source_class="initial",
        supports=supports, contradicts=(),
        verification_state=VerificationState.UNVERIFIED,
        temporal_status=TemporalStatus.STALE,
        retrieved=retrieved, verify_result="STALE",
    )


# ---------------------------------------------------------------------------
# T2-should-activate generators (all hypotheses eliminated → DEFER)
# ---------------------------------------------------------------------------

def gen_bilateral_conflict_h0(task_id, template, rng):
    """2 hyps, bilateral SUFFICIENT conflict, no hidden. T2 at step 2."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B refutes {s}.", ("H2",), ("H1",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="bilateral_conflict_h0", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_bilateral_conflict_h1_noise(task_id, template, rng):
    """2 hyps, bilateral conflict + 1 hidden noise. T2 at step 2 (noise irrelevant)."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B refutes {s}.", ("H2",), ("H1",)),
        _noise("E3", s, retrieved=False),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="bilateral_conflict_h1_noise", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_triple_all_eliminated(task_id, template, rng):
    """3 hyps, all eliminated by verified contradictions. T2 should fire."""
    s = template["subject"]
    h = _make_hyps(3, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B refutes {s}.", ("H2",), ("H1",)),
        _suff("E3", f"Source C contradicts H3.", ("H3",), ("H3",)),
    )
    # E3 supports and contradicts H3 — but SUFFICIENT contradiction eliminates H3
    # Actually need evidence that contradicts H3
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B refutes {s}.", ("H2",), ("H1",)),
        _suff("E3", f"Source C contradicts H3's claim.", (), ("H3",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="triple_all_eliminated", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_triple_all_eliminated_with_noise(task_id, template, rng):
    """3 hyps, all eliminated + hidden noise. T2 should fire despite noise."""
    s = template["subject"]
    h = _make_hyps(3, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B refutes {s}.", ("H2",), ("H1",)),
        _suff("E3", f"Source C contradicts H3's claim.", (), ("H3",)),
        _noise("E4", s, retrieved=False),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="triple_all_eliminated_noise", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=("E4",),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_quad_all_eliminated(task_id, template, rng):
    """4 hyps, all eliminated. T2 should fire with 4 hypotheses."""
    s = template["subject"]
    h = _make_hyps(4, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B refutes {s}.", ("H2",), ("H1",)),
        _suff("E3", f"Source C contradicts H3.", (), ("H3",)),
        _suff("E4", f"Source D contradicts H4.", (), ("H4",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="quad_all_eliminated", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "VERIFY:E4", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_late_conflict(task_id, template, rng):
    """Conflict appears late. First verify non-conflicting evidence, then conflict.
    T2 should fire later (step 4+), not step 2."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"Source A mentions {s} in a neutral context.", ("H1",), ()),
        _falsified("E2", f"Source B is silent on {s}.", ("H2",), ("H1",)),
        _suff("E3", f"Source C definitively confirms {s}.", ("H1",), ("H2",)),
        _suff("E4", f"Source D definitively refutes {s}.", ("H2",), ("H1",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="late_conflict", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "VERIFY:E4", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_noise_before_conflict(task_id, template, rng):
    """Irrelevant visible evidence before conflict evidence. T2 fires after noise + conflict."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _noise("E1", s, retrieved=True),
        _noise("E2", s, retrieved=True),
        _suff("E3", f"Source C confirms {s}.", ("H1",), ("H2",)),
        _suff("E4", f"Source D refutes {s}.", ("H2",), ("H1",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="noise_before_conflict", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E3", "VERIFY:E4", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_asymmetric_conflict(task_id, template, rng):
    """Asymmetric: H1 eliminated by 2 contradictions, H2 eliminated by 1.
    T2 should fire when all eliminated."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B also confirms {s}.", ("H1",), ("H2",)),
        _suff("E3", f"Source C refutes {s}.", ("H2",), ("H1",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="asymmetric_conflict", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


# ---------------------------------------------------------------------------
# T2-should-NOT-activate generators
# ---------------------------------------------------------------------------

def gen_subset_eliminated(task_id, template, rng):
    """3 hyps, only H2 and H3 eliminated. H1 remains viable. T2 should NOT fire.
    Correct answer is ANSWER (H1)."""
    s = template["subject"]
    h = _make_hyps(3, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B contradicts H3.", (), ("H3",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="subset_eliminated", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_one_viable_among_eliminated(task_id, template, rng):
    """4 hyps, H2/H3/H4 eliminated, H1 viable. T2 should NOT fire."""
    s = template["subject"]
    h = _make_hyps(4, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _suff("E2", f"Source B contradicts H3.", (), ("H3",)),
        _suff("E3", f"Source C contradicts H4.", (), ("H4",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="one_viable_among_eliminated", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_falsified_contradiction(task_id, template, rng):
    """Apparent contradiction that FALSIFIES. T2 should NOT fire.
    H2's evidence is FALSIFIED, so H2 is not eliminated by SUFFICIENT contradiction."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
        _falsified("E2", f"Source B claims not-{s}.", ("H2",), ("H1",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="falsified_contradiction", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_stale_support_then_current(task_id, template, rng):
    """Stale support, then current support found. T2 should NOT fire."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _stale("E1", f"An older source claims {s}.", ("H1",)),
        _falsified("E2", f"A source is silent on {s}.", ("H2",), ("H1",)),
        _suff("E3", f"A current source confirms {s}.", ("H1",), ("H2",), retrieved=False),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="stale_then_current", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=("E3",),
        oracle_resolution_path=("VERIFY:E1", "SEARCH_MORE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_multi_viable(task_id, template, rng):
    """3 hyps, H1 and H2 both viable. T2 should NOT fire (multiple viable)."""
    s = template["subject"]
    h = _make_hyps(3, template)
    ev = (
        _suff("E1", f"Source A claims {s}.", ("H1",), ()),
        _suff("E2", f"Source B claims not-{s}.", ("H2",), ()),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="multi_viable", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER, correct_hypothesis_id="H2")


def gen_simple_answer(task_id, template, rng):
    """Simple 1-verify answer. T2 should NOT fire."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"The documentation confirms {s}.", ("H1",), ("H2",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="simple_answer", task_summary=f"Determine {s}.",
        high_stakes=rng.random() > 0.5, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_triple_verify_answer(task_id, template, rng):
    """3 visible, verify all, answer. T2 should NOT fire."""
    s = template["subject"]
    h = _make_hyps(2, template)
    ev = (
        _suff("E1", f"Source A claims {s}.", ("H1",), ()),
        _suff("E2", f"Source B also claims {s}.", ("H1",), ("H2",)),
        _falsified("E3", f"Source C contradicts.", ("H2",), ("H1",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="triple_verify_answer", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_penta_simple_answer(task_id, template, rng):
    """5 hypotheses, but only H1 has support. T2 should NOT fire."""
    s = template["subject"]
    h = _make_hyps(5, template)
    ev = (
        _suff("E1", f"Source A confirms {s}.", ("H1",), ("H2",)),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="penta_simple_answer", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=h,
        evidence_items=ev, retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H1")


def gen_early_false_ready(task_id, template, rng):
    """Apparent conflict, hidden evidence resolves. T2 should NOT fire."""
    s = template["subject"]
    h1 = EvidenceHypothesis(
        hypothesis_id="H1", proposition=template["h1_proposition"],
        answer_action=DecisionAction.ANSWER, answer_payload=template["h1_payload"],
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=f"the documentation refutes {s}, so ANSWER with refutation",
        answer_action=DecisionAction.ANSWER,
        answer_payload=f"refuted: {template['h1_payload']}",
    )
    ev = (
        _suff("E1", f"An initial source claims {s}.", ("H1",), ()),
        _falsified("E2", f"Another source contradicts.", ("H2",), ("H1",)),
        _suff("E3", f"A definitive source refutes {s}.", ("H2",), ("H1",), retrieved=False),
    )
    return EvidenceTask(task_id=task_id, split="r1_confirm_v1",
        category="early_false_ready", task_summary=f"Determine {s}.",
        high_stakes=True, budget_profile="STANDARD", hypotheses=(h1, h2),
        evidence_items=ev, retrieve_exposes=("E3",), search_exposes=(),
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER, correct_hypothesis_id="H2")


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def generate_i3_11d_corpus(split: str = "r1_confirm_v1") -> list[EvidenceTask]:
    """Generate I3.11d corpus with expanded hypothesis topology.

    Target: 330 tasks

    T2-should-activate (150 tasks):
      bilateral_conflict_h0:           20  (2 hyps, step 2)
      bilateral_conflict_h1_noise:     20  (2 hyps + noise, step 2)
      triple_all_eliminated:           20  (3 hyps, step 3)
      triple_all_eliminated_noise:     15  (3 hyps + noise, step 3)
      quad_all_eliminated:             15  (4 hyps, step 4)
      late_conflict:                   20  (2 hyps, step 4+)
      noise_before_conflict:           20  (2 hyps, noise then conflict)
      asymmetric_conflict:             20  (2 hyps, asymmetric elimination)

    T2-should-NOT-activate (180 tasks):
      subset_eliminated:               20  (3 hyps, only 2 eliminated)
      one_viable_among_eliminated:     15  (4 hyps, 3 eliminated, 1 viable)
      falsified_contradiction:         20  (apparent conflict FALSIFIES)
      stale_then_current:              15  (stale support, then current)
      multi_viable:                    20  (3 hyps, 2 viable)
      simple_answer:                   25  (1-verify answer)
      triple_verify_answer:            20  (3-verify answer)
      penta_simple_answer:             15  (5 hyps, simple answer)
      early_false_ready:               30  (apparent conflict, hidden resolves)
    """
    target = [
        ("bilateral_conflict_h0", 20),
        ("bilateral_conflict_h1_noise", 20),
        ("triple_all_eliminated", 20),
        ("triple_all_eliminated_noise", 15),
        ("quad_all_eliminated", 15),
        ("late_conflict", 20),
        ("noise_before_conflict", 20),
        ("asymmetric_conflict", 20),
        ("subset_eliminated", 20),
        ("one_viable_among_eliminated", 15),
        ("falsified_contradiction", 20),
        ("stale_then_current", 15),
        ("multi_viable", 20),
        ("simple_answer", 25),
        ("triple_verify_answer", 20),
        ("penta_simple_answer", 15),
        ("early_false_ready", 30),
    ]

    generators = {
        "bilateral_conflict_h0": gen_bilateral_conflict_h0,
        "bilateral_conflict_h1_noise": gen_bilateral_conflict_h1_noise,
        "triple_all_eliminated": gen_triple_all_eliminated,
        "triple_all_eliminated_noise": gen_triple_all_eliminated_with_noise,
        "quad_all_eliminated": gen_quad_all_eliminated,
        "late_conflict": gen_late_conflict,
        "noise_before_conflict": gen_noise_before_conflict,
        "asymmetric_conflict": gen_asymmetric_conflict,
        "subset_eliminated": gen_subset_eliminated,
        "one_viable_among_eliminated": gen_one_viable_among_eliminated,
        "falsified_contradiction": gen_falsified_contradiction,
        "stale_then_current": gen_stale_support_then_current,
        "multi_viable": gen_multi_viable,
        "simple_answer": gen_simple_answer,
        "triple_verify_answer": gen_triple_verify_answer,
        "penta_simple_answer": gen_penta_simple_answer,
        "early_false_ready": gen_early_false_ready,
    }

    tasks: list[EvidenceTask] = []
    task_idx = 0
    for category, count in target:
        gen = generators[category]
        for i in range(count):
            task_id = f"{split}_{task_idx:04d}"
            template = STRUCTURAL_TEMPLATES[task_idx % len(STRUCTURAL_TEMPLATES)]
            rng = _seeded_rng(task_id)
            task = gen(task_id, template, rng)
            tasks.append(task)
            task_idx += 1

    return tasks


# ---------------------------------------------------------------------------
# Experiment runner (reuses I3.11c process_one_task pattern)
# ---------------------------------------------------------------------------

def counterbalance_3arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    arms = ["A1", "M3", "R1"]
    perms = list(itertools.permutations(arms))
    return list(perms[int(h[:8], 16) % len(perms)])


def process_one_task(task, budget, utility, api_key):
    arm_modes = {
        "A1": "BASELINE_WITH_AFFORDANCES",
        "M3": "MDSG_STATE_WITH_AFFORDANCES",
    }
    results = {}
    for arm_id in ["A1", "M3"]:
        results[arm_id] = i3_7e.run_trajectory(
            task=task, budget=budget, utility=utility,
            mode=arm_modes[arm_id], api_key=api_key,
            fork_label=f"arm{arm_id}",
        )
    results["R1"] = run_r1_trajectory(
        task=task, budget=budget, utility=utility,
        api_key=api_key, fork_label="armR1",
    )

    a1 = results["A1"]
    m3 = results["M3"]
    r1 = results["R1"]

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": sum(1 for e in task.evidence_items if not e.retrieved),
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
        "r1_rescues_vs_a1": (not a1["success"]) and r1["success"],
        "r1_breaks_vs_a1": a1["success"] and (not r1["success"]),
        "r1_rescues_vs_m3": (not m3["success"]) and r1["success"],
        "r1_breaks_vs_m3": m3["success"] and (not r1["success"]),
        "fork_a1": a1, "fork_m3": m3, "fork_r1": r1,
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
    parser.add_argument("--output-dir",
        default="experiments/v2b_i3_11/development/i3_11d_r1_confirm")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.11d: R1 Confirmation Under Expanded Hypothesis Topology")
    print("  R1: A1 until T2 (all_hypotheses_eliminated), then M3 (latched)")
    print("  Testing T2 robustness across 2-5 hypotheses, asymmetric graphs, late conflict")
    print()

    tasks = generate_i3_11d_corpus(split="r1_confirm_v1")
    print(f"  Generated {len(tasks)} tasks")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution:")
    for cat in sorted(cats.keys()):
        print(f"    {cat:<40} {cats[cat]}")

    # Report hypothesis topology
    hyp_counts = Counter(len(t.hypotheses) for t in tasks)
    print(f"\n  Hypothesis count distribution:")
    for n_h in sorted(hyp_counts.keys()):
        print(f"    {n_h} hypotheses: {hyp_counts[n_h]} tasks")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Save corpus manifest
    benchmark = EvidenceBenchmark(
        benchmark_id="i3_11d_r1_confirm_v1",
        tasks=tasks, budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_11/manifests/r1_confirm_v1.json")

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
    all_results = []
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

    results_path = output_dir / "r1_confirm_v1.jsonl"
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

    r1_a1_ci = paired_bootstrap_ci([r["r1_delta_vs_a1"] for r in all_results])
    r1_m3_ci = paired_bootstrap_ci([r["r1_delta_vs_m3"] for r in all_results])
    m3_a1_ci = paired_bootstrap_ci([r["u_m3"] - r["u_a1"] for r in all_results])

    mc_r1_a1 = mcnemar([r["a1_success"] for r in all_results], [r["r1_success"] for r in all_results])
    mc_r1_m3 = mcnemar([r["m3_success"] for r in all_results], [r["r1_success"] for r in all_results])

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    r1_a1_cl = Counter(classify(r["a1_success"], r["r1_success"]) for r in all_results)
    r1_m3_cl = Counter(classify(r["m3_success"], r["r1_success"]) for r in all_results)

    a1_steps = sum(r["a1_steps"] for r in all_results)
    m3_steps = sum(r["m3_steps"] for r in all_results)
    r1_steps = sum(r["r1_steps"] for r in all_results)

    # Routing telemetry
    r1_trigger_count = sum(1 for r in all_results if r["r1_triggered"])
    r1_trigger_steps = [r["r1_trigger_step"] for r in all_results if r["r1_triggered"]]
    r1_trigger_mean = sum(r1_trigger_steps) / len(r1_trigger_steps) if r1_trigger_steps else None
    r1_trigger_median = sorted(r1_trigger_steps)[len(r1_trigger_steps) // 2] if r1_trigger_steps else None

    m3_rescues = sum(1 for r in all_results if r["m3_rescues_vs_a1"])
    r1_coverage = sum(1 for r in all_results if r["m3_rescues_vs_a1"] and r["r1_triggered"]) / max(m3_rescues, 1)
    a1_success_count = sum(1 for r in all_results if r["a1_success"])
    r1_false_activation = sum(1 for r in all_results if r["a1_success"] and r["r1_triggered"]) / max(a1_success_count, 1)

    # TriggerPrecision = P(R1 useful | T2 fires)
    # "R1 useful" = R1 succeeds AND A1 fails (rescue), OR R1 succeeds where M3 also succeeds (no harm)
    # More precisely: TriggerPrecision = P(rescue_or_no_harm | triggered)
    # But the user asked for P(R1 useful | T2) — useful means R1 helped vs A1
    r1_triggered_rescues = sum(1 for r in all_results if r["r1_triggered"] and r["r1_rescues_vs_a1"])
    r1_triggered_both_fail = sum(1 for r in all_results if r["r1_triggered"] and not r["r1_success"] and not r["a1_success"])
    trigger_precision = r1_triggered_rescues / max(r1_trigger_count, 1)

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
            "n_hypotheses": cr[0]["n_hypotheses"],
            "a1_success": f"{ca1}/{cn} ({ca1/cn*100:.1f}%)",
            "m3_success": f"{cm3}/{cn} ({cm3/cn*100:.1f}%)",
            "r1_success": f"{cr1}/{cn} ({cr1/cn*100:.1f}%)",
            "r1_triggered": f"{cr1_trig}/{cn}",
            "mean_u_a1": round(sum(r["u_a1"] for r in cr) / cn, 4),
            "mean_u_m3": round(sum(r["u_m3"] for r in cr) / cn, 4),
            "mean_u_r1": round(sum(r["u_r1"] for r in cr) / cn, 4),
            "r1_rescues_vs_a1": sum(1 for r in cr if r["r1_rescues_vs_a1"]),
            "r1_breaks_vs_a1": sum(1 for r in cr if r["r1_breaks_vs_a1"]),
            "mean_steps_r1": round(sum(r["r1_steps"] for r in cr) / cn, 2),
            "mean_steps_m3": round(sum(r["m3_steps"] for r in cr) / cn, 2),
        }

    # Hypothesis-count subgroup
    hyp_subgroups = {}
    for n_h in sorted(set(r["n_hypotheses"] for r in all_results)):
        hr = [r for r in all_results if r["n_hypotheses"] == n_h]
        hn = len(hr)
        hyp_subgroups[f"{n_h}_hypotheses"] = {
            "n": hn,
            "a1_success": f"{sum(1 for r in hr if r['a1_success'])}/{hn}",
            "m3_success": f"{sum(1 for r in hr if r['m3_success'])}/{hn}",
            "r1_success": f"{sum(1 for r in hr if r['r1_success'])}/{hn}",
            "r1_triggered": sum(1 for r in hr if r["r1_triggered"]),
            "mean_u_a1": round(sum(r["u_a1"] for r in hr) / hn, 4),
            "mean_u_m3": round(sum(r["u_m3"] for r in hr) / hn, 4),
            "mean_u_r1": round(sum(r["u_r1"] for r in hr) / hn, 4),
        }

    frozen_claims = {
        "C1_r1_a1_ci_positive": r1_a1_ci[0] > 0,
        "C2_r1_m3_ci_positive": r1_m3_ci[0] > 0,
        "C3_r1_success_ge_m3_minus_1pp": r1_s >= m3_s - max(3, n // 100),
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
        "schema": "DAPH_V2B_I3_11D_R1_CONFIRM_V1",
        "n_tasks": n,
        "arms": {
            "A1": "baseline + public affordances",
            "M3": "frozen MDSG-StateWithAffordances",
            "R1": "A1 until T2 (all_hypotheses_eliminated), then M3 (latched) — exact I3.11c executable",
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
            "mean_steps": {
                "A1": round(a1_steps / n, 2), "M3": round(m3_steps / n, 2),
                "R1": round(r1_steps / n, 2),
            },
        },
        "r1_routing_telemetry": {
            "trigger_rate": round(r1_trigger_count / n, 4),
            "trigger_count": r1_trigger_count,
            "trigger_step_mean": round(r1_trigger_mean, 2) if r1_trigger_mean else None,
            "trigger_step_median": r1_trigger_median,
            "trigger_precision": round(trigger_precision, 4),
            "trigger_precision_explanation": "P(R1 rescues vs A1 | T2 fires). Independent from rescue coverage.",
            "coverage_of_m3_rescues": round(r1_coverage, 4),
            "false_activation_on_a1_success": round(r1_false_activation, 4),
            "triggered_both_fail": r1_triggered_both_fail,
        },
        "subgroups": subgroups,
        "hypothesis_count_subgroups": hyp_subgroups,
        "frozen_claims": frozen_claims,
    }

    summary_path = output_dir / "r1_confirm_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Print results
    print(f"\n{'='*82}")
    print("I3.11d R1 CONFIRMATION: A1 vs M3 vs R1 (expanded topology)")
    print(f"{'='*82}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A1={u_a1:+.4f}  M3={u_m3:+.4f}  R1={u_r1:+.4f}")
    print(f"  Success:       A1={a1_s}/{n}  M3={m3_s}/{n}  R1={r1_s}/{n}")
    print(f"\n  Bootstrap 95% CI:")
    print(f"    R1-A1: [{r1_a1_ci[0]:+.4f}, {r1_a1_ci[1]:+.4f}]  <-- CO-PRIMARY")
    print(f"    R1-M3: [{r1_m3_ci[0]:+.4f}, {r1_m3_ci[1]:+.4f}]  <-- CO-PRIMARY")
    print(f"    M3-A1: [{m3_a1_ci[0]:+.4f}, {m3_a1_ci[1]:+.4f}]")
    print(f"\n  McNemar:")
    print(f"    R1-A1: b={mc_r1_a1['b']}, c={mc_r1_a1['c']}, p={mc_r1_a1['p']}")
    print(f"    R1-M3: b={mc_r1_m3['b']}, c={mc_r1_m3['c']}, p={mc_r1_m3['p']}")
    print(f"\n  R1 vs A1: rescues={r1_a1_cl.get('RESCUE',0)}, breaks={r1_a1_cl.get('BREAK',0)}")
    print(f"  R1 vs M3: rescues={r1_m3_cl.get('RESCUE',0)}, breaks={r1_m3_cl.get('BREAK',0)}")
    print(f"\n  Steps:  A1={a1_steps/n:.2f}  M3={m3_steps/n:.2f}  R1={r1_steps/n:.2f}")

    print(f"\n  R1 ROUTING TELEMETRY:")
    print(f"    Trigger rate: {r1_trigger_count}/{n} ({r1_trigger_count/n*100:.1f}%)")
    if r1_trigger_mean:
        print(f"    Trigger step: mean={r1_trigger_mean:.2f}, median={r1_trigger_median}")
    print(f"    Trigger precision: {trigger_precision:.4f}")
    print(f"    Coverage of M3 rescues: {r1_coverage:.4f}")
    print(f"    False activation on A1-success: {r1_false_activation:.4f}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in frozen_claims.items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")

    print(f"\n{'='*82}")
    print("SUBGROUP ANALYSIS BY CATEGORY")
    print(f"{'='*82}")
    print(f"  {'Category':<40} {'n':>3} {'hyp':>3} {'A1%':>6} {'M3%':>6} {'R1%':>6} {'trig':>5} {'A1_U':>8} {'M3_U':>8} {'R1_U':>8}")
    for cat in sorted(subgroups.keys()):
        sg = subgroups[cat]
        a1p = sg["a1_success"].split("(")[1].rstrip(")")
        m3p = sg["m3_success"].split("(")[1].rstrip(")")
        r1p = sg["r1_success"].split("(")[1].rstrip(")")
        print(f"  {cat:<40} {sg['n']:>3} {sg['n_hypotheses']:>3} {a1p:>6} {m3p:>6} {r1p:>6} {sg['r1_triggered']:>5} "
              f"{sg['mean_u_a1']:>+8.2f} {sg['mean_u_m3']:>+8.2f} {sg['mean_u_r1']:>+8.2f}")

    print(f"\n{'='*82}")
    print("SUBGROUP ANALYSIS BY HYPOTHESIS COUNT")
    print(f"{'='*82}")
    for key in sorted(hyp_subgroups.keys()):
        sg = hyp_subgroups[key]
        print(f"  {key:<20} n={sg['n']:>3}  A1={sg['a1_success']:>8}  M3={sg['m3_success']:>8}  "
              f"R1={sg['r1_success']:>8}  trig={sg['r1_triggered']:>3}  "
              f"U: A1={sg['mean_u_a1']:>+8.2f}  M3={sg['mean_u_m3']:>+8.2f}  R1={sg['mean_u_r1']:>+8.2f}")


if __name__ == "__main__":
    main()
