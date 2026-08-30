"""D1 DEFER-ready training stratum for Q_V3R3.

Generates terminal DEFER states where continuation is legal but
causally dominated. These states teach the Q model that DEFER is
high-value when:
- Verification budget is exhausted or very low
- No verified support exists for any hypothesis
- No unverified discriminating evidence remains
- Continuing would only waste resources

This addresses the D1 Q-calibration weakness where V3R2 sometimes
overvalues REASON_MORE when the state is already DEFER-ready.

Key design: the model must learn that DEFER > REASON_MORE when
continuation cannot resolve the state, even if REASON_MORE is legal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget


# Fresh domain templates for D1 DEFER-ready stratum
D1_DEFER_DOMAINS = [
    {
        "summary": "Does the patient have condition A or condition B?",
        "hypotheses": [
            ("H1", "condition A", "ANSWER"),
            ("H2", "condition B", "DEFER"),
        ],
        "evidence": [
            # All evidence is UNVERIFIED or FALSIFIED — no verified support
            ("E1", "Initial lab finding suggesting A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradictory finding for B", "initial", (), ("H2",), "FALSIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the diagnosis X or Y based on available tests?",
        "hypotheses": [
            ("H1", "diagnosis X", "ANSWER"),
            ("H2", "diagnosis Y", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test result supporting X", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Test result contradicting Y", "initial", (), ("H2",), "FALSIFIED", "CURRENT"),
            ("E3", "Ambiguous marker", "initial", ("H1", "H2"), (), "UNVERIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the cause infectious or autoimmune?",
        "hypotheses": [
            ("H1", "infectious cause", "ANSWER"),
            ("H2", "autoimmune cause", "DEFER"),
        ],
        "evidence": [
            ("E1", "Culture result", "initial", ("H1",), (), "FALSIFIED", "CURRENT"),
            ("E2", "Antibody panel", "initial", ("H2",), (), "UNVERIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the tumor benign or malignant?",
        "hypotheses": [
            ("H1", "benign tumor", "ANSWER"),
            ("H2", "malignant tumor", "DEFER"),
        ],
        "evidence": [
            ("E1", "Biopsy result", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Imaging characteristic", "initial", (), ("H2",), "FALSIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the failure acute or chronic?",
        "hypotheses": [
            ("H1", "acute failure", "ANSWER"),
            ("H2", "chronic condition", "DEFER"),
        ],
        "evidence": [
            ("E1", "Acute marker", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Chronic indicator", "initial", ("H2",), (), "FALSIFIED", "CURRENT"),
            ("E3", "Timeline evidence", "initial", (), ("H1",), "UNVERIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the reaction allergic or toxic?",
        "hypotheses": [
            ("H1", "allergic reaction", "ANSWER"),
            ("H2", "toxic reaction", "DEFER"),
        ],
        "evidence": [
            ("E1", "Skin test result", "initial", ("H1",), (), "FALSIFIED", "CURRENT"),
            ("E2", "Exposure history", "initial", ("H2",), (), "UNVERIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the deficit structural or functional?",
        "hypotheses": [
            ("H1", "structural deficit", "ANSWER"),
            ("H2", "functional deficit", "DEFER"),
        ],
        "evidence": [
            ("E1", "Imaging scan", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Functional test", "initial", ("H2",), (), "FALSIFIED", "CURRENT"),
        ],
    },
    {
        "summary": "Is the source endogenous or exogenous?",
        "hypotheses": [
            ("H1", "endogenous source", "ANSWER"),
            ("H2", "exogenous source", "DEFER"),
        ],
        "evidence": [
            ("E1", "Metabolite analysis", "initial", ("H1",), (), "FALSIFIED", "CURRENT"),
            ("E2", "Exposure marker", "initial", ("H2",), (), "UNVERIFIED", "CURRENT"),
            ("E3", "Genetic marker", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
        ],
    },
]


def generate_d1_defer_ready_tasks(
    seed: int = 7777,
    n_per_domain: int = 10,
) -> list[EvidenceTask]:
    """Generate D1 DEFER-ready training tasks.

    These tasks have:
    - No verified support for any hypothesis
    - Exhausted or near-exhausted verification budget
    - DEFER as the correct terminal action
    - Continuation is legal but causally dominated (cannot resolve)

    Args:
        seed: Random seed
        n_per_domain: Tasks per domain template

    Returns:
        List of EvidenceTask objects
    """
    rng = random.Random(seed)
    tasks = []

    for domain_idx, domain in enumerate(D1_DEFER_DOMAINS):
        for i in range(n_per_domain):
            task_id = f"i3_30r3_d1dr_{domain_idx:02d}_{i:03d}"

            # Build hypotheses
            hypotheses = []
            for h_id, prop, action_str in domain["hypotheses"]:
                action = DecisionAction(action_str)
                payload = f"{action_str}:{h_id}:{prop}"
                hypotheses.append(EvidenceHypothesis(
                    hypothesis_id=h_id,
                    proposition=prop,
                    answer_action=action,
                    answer_payload=payload,
                ))

            # Build evidence
            evidence_items = []
            for ev_id, prop, source, supports, contradicts, vstate_str, tstatus_str in domain["evidence"]:
                # Vary verification state slightly across tasks
                # Some tasks have all FALSIFIED, some have UNVERIFIED
                actual_vstate = vstate_str
                if rng.random() < 0.3 and vstate_str == "UNVERIFIED":
                    actual_vstate = "FALSIFIED"

                evidence_items.append(EvidenceItem(
                    evidence_id=ev_id,
                    proposition=prop,
                    source_class=source,
                    supports=supports,
                    contradicts=contradicts,
                    verification_state=VerificationState(actual_vstate),
                    temporal_status=TemporalStatus(tstatus_str),
                    retrieved=True,
                    verify_result=actual_vstate if actual_vstate != "UNVERIFIED" else None,
                ))

            # Budget: verification exhausted or nearly exhausted
            # This makes continuation legal but futile
            verify_budget = rng.choice([0, 0, 0, 1])  # mostly exhausted
            step_budget = rng.choice([1, 2, 3])

            budget = ResourceBudget(
                max_executive_steps=step_budget,
                max_reasoning_tokens=256,
                max_retrieval_calls=0,  # no retrieval available
                max_verification_calls=verify_budget,
                max_search_calls=0,  # no search available
                max_elapsed_ms=10000,
                max_monetary_cost_microusd=0,
            )

            # Correct hypothesis is H2 (DEFER) — no verified support exists
            # The state is DEFER_READY because continuation cannot resolve
            task = EvidenceTask(
                task_id=task_id,
                split="i3_30r3_d1dr",
                category="D1_DEFER_READY",
                task_summary=domain["summary"],
                high_stakes=True,
                budget_profile=f"D1DR_{verify_budget}_{step_budget}",
                hypotheses=tuple(hypotheses),
                evidence_items=tuple(evidence_items),
                retrieve_exposes=(),
                search_exposes=(),
                oracle_resolution_path=("DEFER",),  # DEFER is the correct path
                expected_terminal=DecisionAction.DEFER,
                correct_hypothesis_id="H2",
            )
            tasks.append(task)

    return tasks
