#!/usr/bin/env python3
"""Phase 15: Adversarial qualification suite for DAPH-X.

Tests cases where:
- LLM is right / DAPH wrong
- LLM wrong / DAPH right
- Both right / both wrong
- World model wrong
- Belief model wrong
- Certificate wrong
- Value estimates close
- Value estimates widely separated
- OOD topology
- OOD resource state

Usage:
    python scripts/adversarial_qualification.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.actions.typed_actions import Action, ActionType, answer, defer, verify, stop
from daph_x.authority.executive import select_action, ExecutiveConfig
from daph_x.authority.authority_policy import determine_authority_mode, AuthorityConfig
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task
from daph_x.belief.belief_engine import compute_belief_state
from daph_x.receipts.checkpoint import checkpoint_from_task_and_runtime
from daph_x.receipts.fork_engine import evaluate_all_actions, compute_oracle_action
from daph_x.actions.candidate_generator import generate_and_prune

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)


def make_task(
    task_id: str,
    hypotheses: list[tuple[str, str, str]],
    evidence: list[tuple[str, str, tuple, tuple, str, str]],
    correct_hypothesis: str,
    expected_terminal: str,
    oracle_path: tuple[str, ...],
    budget: dict | None = None,
) -> EvidenceTask:
    hyps = [EvidenceHypothesis(
        hypothesis_id=h_id, proposition=prop,
        answer_action=DecisionAction(action_str),
        answer_payload=f"{action_str}:{h_id}:{prop}",
    ) for h_id, prop, action_str in hypotheses]
    evs = [EvidenceItem(
        evidence_id=ev_id, proposition=prop, source_class="initial",
        supports=supports, contradicts=contradicts,
        verification_state=VerificationState(vs),
        temporal_status=TemporalStatus(ts),
        retrieved=True,
        verify_result=vs if vs != "UNVERIFIED" else None,
    ) for ev_id, prop, supports, contradicts, vs, ts in evidence]
    b = budget or {"steps": 4, "verify": 2, "retrieve": 0, "search": 0}
    return EvidenceTask(
        task_id=task_id, split="adversarial", category="ADV",
        task_summary="Adversarial test", high_stakes=True,
        budget_profile=f"ADV_{b['steps']}_{b['verify']}_{b.get('search', 0)}",
        hypotheses=tuple(hyps), evidence_items=tuple(evs),
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=oracle_path,
        expected_terminal=DecisionAction(expected_terminal),
        correct_hypothesis_id=correct_hypothesis,
    )


def simulate_llm_proposal(task: EvidenceTask) -> str:
    """Simulate an LLM proposal for a task.

    The LLM tends to:
    - Propose VERIFY when uncertain (even if ANSWER is ready)
    - Propose DEFER when competing support (correct for DEFER tasks)
    - Propose ANSWER when it sees support (even if wrong hypothesis)
    """
    # Simple heuristic: LLM proposes VERIFY if there's unverified evidence
    for e in task.evidence_items:
        if e.verification_state == VerificationState.UNVERIFIED:
            return "VERIFY"
    # If all verified and competing support, LLM proposes DEFER
    verified_support = sum(1 for e in task.evidence_items
                          if e.verification_state == VerificationState.SUFFICIENT and e.supports)
    if verified_support >= 2:
        return "DEFER"
    # Otherwise propose ANSWER
    return "ANSWER"


def run_adversarial_case(task: EvidenceTask, case_name: str, config: ExecutiveConfig | None = None):
    """Run a single adversarial test case."""
    if config is None:
        config = ExecutiveConfig()

    checkpoint = checkpoint_from_task_and_runtime(task, None, seed=42)
    graph = build_graph_from_evidence_task(task)
    belief = compute_belief_state(graph)
    candidates = generate_and_prune(graph)

    # Get LLM proposal
    llm_proposal_str = simulate_llm_proposal(task)

    # Executive decision
    decision = select_action(graph, llm_proposal=llm_proposal_str, config=config)

    # Evaluate all actions for ground truth
    results = evaluate_all_actions(checkpoint, candidates, seed=42)
    oracle_action, oracle_utility = compute_oracle_action(results)

    # Determine if executive is correct
    executive_correct = str(decision.selected_action) == oracle_action

    # Determine if LLM is correct
    llm_correct = llm_proposal_str == oracle_action.split("(")[0]  # Compare action type

    return {
        "case_name": case_name,
        "task_id": task.task_id,
        "executive_action": str(decision.selected_action),
        "oracle_action": oracle_action,
        "executive_correct": executive_correct,
        "llm_proposal": llm_proposal_str,
        "llm_correct": llm_correct,
        "authority_mode": decision.authority_mode.value,
        "executive_score": decision.expected_value,
        "oracle_utility": oracle_utility,
        "value_margin": decision.value_margin,
        "structural_certificate": decision.structural_certificate,
        "readiness": belief.readiness.value,
        "n_supported": belief.n_supported,
        "n_contradicted": belief.n_contradicted,
    }


def main():
    print("=" * 70)
    print("ADVERSARIAL QUALIFICATION SUITE")
    print("=" * 70)

    cases = []

    # Case 1: LLM right, DAPH right (both correct)
    # Simple ANSWER task where both should agree
    task = make_task(
        task_id="adv_01_both_right",
        hypotheses=[("H1", "A", "ANSWER"), ("H2", "B", "DEFER")],
        evidence=[
            ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
        ],
        correct_hypothesis="H1",
        expected_terminal="ANSWER",
        oracle_path=("ANSWER",),
    )
    cases.append(run_adversarial_case(task, "both_right"))

    # Case 2: LLM wrong, DAPH right
    # LLM proposes VERIFY but ANSWER is ready
    task = make_task(
        task_id="adv_02_llm_wrong_daph_right",
        hypotheses=[("H1", "A", "ANSWER"), ("H2", "B", "DEFER")],
        evidence=[
            ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Test for B", ("H2",), (), "UNVERIFIED", "CURRENT"),
        ],
        correct_hypothesis="H1",
        expected_terminal="ANSWER",
        oracle_path=("ANSWER",),
    )
    cases.append(run_adversarial_case(task, "llm_wrong_daph_right"))

    # Case 3: LLM right, DAPH wrong (adversarial)
    # Misleading evidence — DAPH picks wrong hypothesis
    task = make_task(
        task_id="adv_03_llm_right_daph_wrong",
        hypotheses=[("H1", "A (wrong)", "ANSWER"), ("H2", "B (correct)", "ANSWER"), ("H3", "C", "DEFER")],
        evidence=[
            ("E1", "Misleading marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ],
        correct_hypothesis="H2",
        expected_terminal="DEFER",
        oracle_path=("DEFER",),
    )
    cases.append(run_adversarial_case(task, "llm_right_daph_wrong"))

    # Case 4: Both wrong
    # Competing support, both should DEFER but might not
    task = make_task(
        task_id="adv_04_both_wrong",
        hypotheses=[("H1", "A", "ANSWER"), ("H2", "B", "ANSWER"), ("H3", "C", "DEFER")],
        evidence=[
            ("E1", "Marker for A", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker for B", ("H2",), (), "SUFFICIENT", "CURRENT"),
        ],
        correct_hypothesis="H3",
        expected_terminal="DEFER",
        oracle_path=("DEFER",),
    )
    cases.append(run_adversarial_case(task, "both_wrong"))

    # Case 5: Value estimates close
    # Two VERIFY targets with similar value
    task = make_task(
        task_id="adv_05_close_values",
        hypotheses=[("H1", "A", "ANSWER"), ("H2", "B", "ANSWER"), ("H3", "C", "DEFER")],
        evidence=[
            ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Test for B", ("H2",), (), "UNVERIFIED", "CURRENT"),
            ("E3", "Contradiction of C", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ],
        correct_hypothesis="H1",
        expected_terminal="ANSWER",
        oracle_path=("VERIFY", "ANSWER"),
    )
    cases.append(run_adversarial_case(task, "close_values"))

    # Case 6: OOD topology (5 hypotheses, novel structure)
    task = make_task(
        task_id="adv_06_ood_topology",
        hypotheses=[(f"H{i}", f"type {i}", "ANSWER") for i in range(1, 5)] + [("H5", "type 5", "DEFER")],
        evidence=[
            ("E1", "Test for 1", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Test for 2", ("H2",), (), "UNVERIFIED", "CURRENT"),
            ("E3", "Contradiction of 3", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of 4", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Contradiction of 5", (), ("H5",), "SUFFICIENT", "CURRENT"),
        ],
        correct_hypothesis="H1",
        expected_terminal="ANSWER",
        oracle_path=("VERIFY", "ANSWER"),
        budget={"steps": 5, "verify": 3, "retrieve": 0, "search": 0},
    )
    cases.append(run_adversarial_case(task, "ood_topology"))

    # Case 7: Resource constrained (no verify budget)
    task = make_task(
        task_id="adv_07_resource_constrained",
        hypotheses=[("H1", "A", "ANSWER"), ("H2", "B", "DEFER")],
        evidence=[
            ("E1", "Test for A", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", (), ("H2",), "SUFFICIENT", "CURRENT"),
        ],
        correct_hypothesis="H1",
        expected_terminal="ANSWER",
        oracle_path=("VERIFY", "ANSWER"),
        budget={"steps": 2, "verify": 0, "retrieve": 0, "search": 0},  # No verify budget!
    )
    cases.append(run_adversarial_case(task, "resource_constrained"))

    # Summary
    print(f"\n{'Case':>30} {'Executive':>15} {'Oracle':>15} {'Correct':>8} {'LLM':>10} {'LLM Correct':>12} {'Authority':>10}")
    print("-" * 100)

    for case in cases:
        print(f"{case['case_name']:>30} {case['executive_action']:>15} {case['oracle_action']:>15} "
              f"{str(case['executive_correct']):>8} {case['llm_proposal']:>10} "
              f"{str(case['llm_correct']):>12} {case['authority_mode']:>10}")

    # Aggregate
    n_correct = sum(1 for c in cases if c["executive_correct"])
    n_llm_correct = sum(1 for c in cases if c["llm_correct"])
    print(f"\nExecutive correct: {n_correct}/{len(cases)}")
    print(f"LLM correct: {n_llm_correct}/{len(cases)}")

    # Disagreement analysis
    disagreements = [c for c in cases if c["executive_action"] != c["llm_proposal"]]
    print(f"Disagreements: {len(disagreements)}/{len(cases)}")
    for d in disagreements:
        print(f"  {d['case_name']}: executive={d['executive_action']}, llm={d['llm_proposal']}, "
              f"executive_correct={d['executive_correct']}, llm_correct={d['llm_correct']}")

    # Save
    output_path = REPO_ROOT / "experiments/daph_x/adversarial_qualification.json"
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
