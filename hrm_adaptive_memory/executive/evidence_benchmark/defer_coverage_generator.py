"""P5: DEFER authority causal coverage benchmark.

Generates fresh tasks where DEFER is causally correct but the model
has a realistic tendency to continue reasoning, searching, or answering.

Design principles:
1. DEFER is the correct terminal action (expected_terminal = DEFER)
2. Continuation is legal (resources remain) but causally dominated
3. The state has genuine epistemic ambiguity — not manufactured
4. The model's tendency to continue is natural, not forced by prompt

Task categories:
- EXHAUSTED_AMBIGUITY: All evidence verified, no hypothesis has
  unique verified support, no more evidence to retrieve. DEFER is
  correct because the state is genuinely unresolved.
- COMPETING_SUPPORT: Multiple hypotheses have verified support,
  no discriminator exists or can be obtained. DEFER is correct
  because the state is genuinely ambiguous.
- RESOURCE_CEILING: Verification budget is nearly exhausted, remaining
  evidence is unlikely to discriminate. DEFER is correct because
  the cost of continuation exceeds expected information gain.
- POST_CONTRADICTION: Evidence contradicts the primary hypothesis,
  no alternative has sufficient support. DEFER is correct because
  the state is genuinely uncertain.

These tasks provide legitimate opportunities for DEFER authority
to override a model that naturally wants to keep reasoning or answer.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget


DEFER_COVERAGE_DOMAINS = [
    # EXHAUSTED_AMBIGUITY: all verified, no unique support
    {
        "category": "EXHAUSTED_AMBIGUITY",
        "summary": "Is the condition A, B, or indeterminate?",
        "hypotheses": [
            ("H1", "condition A", "ANSWER"),
            ("H2", "condition B", "ANSWER"),
            ("H3", "indeterminate", "DEFER"),
        ],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker for B", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            # Both A and B have support — no unique answer
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # COMPETING_SUPPORT: multiple verified, no discriminator
    {
        "category": "COMPETING_SUPPORT",
        "summary": "Is the diagnosis X or Y — both have lab support?",
        "hypotheses": [
            ("H1", "diagnosis X", "ANSWER"),
            ("H2", "diagnosis Y", "ANSWER"),
            ("H3", "unresolved — defer for specialist", "DEFER"),
        ],
        "evidence": [
            ("E1", "Lab finding for X", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Lab finding for Y", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            ("E3", "Inconclusive marker", "initial", ("H1", "H2"), (), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 3, "verify": 1, "retrieve": 0, "search": 0},
    },
    # RESOURCE_CEILING: budget nearly exhausted
    {
        "category": "RESOURCE_CEILING",
        "summary": "Is the failure mode A or B with limited testing?",
        "hypotheses": [
            ("H1", "failure mode A", "ANSWER"),
            ("H2", "failure mode B", "ANSWER"),
            ("H3", "cannot determine — defer", "DEFER"),
        ],
        "evidence": [
            ("E1", "Test suggesting A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Test suggesting B", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 1, "verify": 0, "retrieve": 0, "search": 0},
    },
    # POST_CONTRADICTION: primary contradicted, no alternative
    {
        "category": "POST_CONTRADICTION",
        "summary": "Is the cause A, B, or unknown after contradiction?",
        "hypotheses": [
            ("H1", "cause A", "ANSWER"),
            ("H2", "cause B", "ANSWER"),
            ("H3", "unknown — defer for further workup", "DEFER"),
        ],
        "evidence": [
            ("E1", "Evidence for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of A", "initial", (), ("H1",), "SUFFICIENT", "CURRENT"),
            ("E3", "Weak evidence for B", "initial", ("H2",), (), "UNVERIFIED", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 2, "verify": 1, "retrieve": 0, "search": 0},
    },
    # EXHAUSTED_AMBIGUITY with more hypotheses
    {
        "category": "EXHAUSTED_AMBIGUITY",
        "summary": "Is the phenotype type 1, 2, 3, or unspecified?",
        "hypotheses": [
            ("H1", "type 1", "ANSWER"),
            ("H2", "type 2", "ANSWER"),
            ("H3", "type 3", "ANSWER"),
            ("H4", "unspecified — defer", "DEFER"),
        ],
        "evidence": [
            ("E1", "Marker for type 1", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker for type 2", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            ("E3", "Marker for type 3", "initial", ("H3",), (), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H4",
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # COMPETING_SUPPORT with search available but unlikely to help
    {
        "category": "COMPETING_SUPPORT",
        "summary": "Is the lesion benign or malignant — both have evidence?",
        "hypotheses": [
            ("H1", "benign", "ANSWER"),
            ("H2", "malignant", "ANSWER"),
            ("H3", "defer for biopsy", "DEFER"),
        ],
        "evidence": [
            ("E1", "Imaging suggesting benign", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker suggesting malignant", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 3, "verify": 0, "retrieve": 0, "search": 2},
    },
    # RESOURCE_CEILING with reasoning budget
    {
        "category": "RESOURCE_CEILING",
        "summary": "Is the drug reaction allergic or toxic — limited info?",
        "hypotheses": [
            ("H1", "allergic", "ANSWER"),
            ("H2", "toxic", "ANSWER"),
            ("H3", "defer for allergy testing", "DEFER"),
        ],
        "evidence": [
            ("E1", "Symptom consistent with allergic", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Symptom consistent with toxic", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 1, "verify": 0, "retrieve": 0, "search": 0},
    },
    # POST_CONTRADICTION with no verify budget
    {
        "category": "POST_CONTRADICTION",
        "summary": "Is the deficit structural or functional after mixed results?",
        "hypotheses": [
            ("H1", "structural", "ANSWER"),
            ("H2", "functional", "ANSWER"),
            ("H3", "defer for advanced imaging", "DEFER"),
        ],
        "evidence": [
            ("E1", "Scan suggesting structural", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of structural", "initial", (), ("H1",), "SUFFICIENT", "CURRENT"),
            ("E3", "Functional test", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of functional", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
]


def generate_defer_coverage_tasks(
    seed: int = 9999,
    n_per_domain: int = 15,
) -> list[EvidenceTask]:
    """Generate DEFER coverage tasks.

    Args:
        seed: Random seed
        n_per_domain: Tasks per domain template

    Returns:
        List of EvidenceTask objects
    """
    rng = random.Random(seed)
    tasks = []

    for domain_idx, domain in enumerate(DEFER_COVERAGE_DOMAINS):
        for i in range(n_per_domain):
            task_id = f"i3_30r3_dcov_{domain_idx:02d}_{i:03d}"

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
                evidence_items.append(EvidenceItem(
                    evidence_id=ev_id,
                    proposition=prop,
                    source_class=source,
                    supports=supports,
                    contradicts=contradicts,
                    verification_state=VerificationState(vstate_str),
                    temporal_status=TemporalStatus(tstatus_str),
                    retrieved=True,
                    verify_result=vstate_str if vstate_str != "UNVERIFIED" else None,
                ))

            # Budget with slight variation
            b = domain["budget"]
            budget = ResourceBudget(
                max_executive_steps=b["steps"] + rng.choice([0, 0, 1]),
                max_reasoning_tokens=256,
                max_retrieval_calls=b["retrieve"],
                max_verification_calls=b["verify"],
                max_search_calls=b["search"],
                max_elapsed_ms=10000,
                max_monetary_cost_microusd=0,
            )

            task = EvidenceTask(
                task_id=task_id,
                split="i3_30r3_dcov",
                category=f"DEFER_COVERAGE_{domain['category']}",
                task_summary=domain["summary"],
                high_stakes=True,
                budget_profile=f"DCOV_{b['steps']}_{b['verify']}_{b['search']}",
                hypotheses=tuple(hypotheses),
                evidence_items=tuple(evidence_items),
                retrieve_exposes=(),
                search_exposes=(),
                oracle_resolution_path=("DEFER",),
                expected_terminal=DecisionAction.DEFER,
                correct_hypothesis_id=domain["correct_hypothesis"],
            )
            tasks.append(task)

    return tasks


def compute_defer_coverage_hash(tasks: Sequence[EvidenceTask]) -> str:
    """Compute deterministic hash of the DEFER coverage benchmark."""
    data = []
    for t in tasks:
        data.append({
            "task_id": t.task_id,
            "category": t.category,
            "summary": t.task_summary,
            "expected_terminal": t.expected_terminal.value,
            "correct_hypothesis_id": t.correct_hypothesis_id,
            "hypotheses": [
                {"id": h.hypothesis_id, "action": h.answer_action.value}
                for h in t.hypotheses
            ],
            "evidence": [
                {
                    "id": ev.evidence_id,
                    "state": ev.verification_state.value,
                    "supports": list(ev.supports),
                    "contradicts": list(ev.contradicts),
                }
                for ev in t.evidence_items
            ],
            "budget_profile": t.budget_profile,
        })
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
