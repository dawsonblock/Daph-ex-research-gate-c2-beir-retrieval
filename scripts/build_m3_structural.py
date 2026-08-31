#!/usr/bin/env python3
"""Build structural-heldout and topology-OOD datasets for M3.

Creates training and test sets with ZERO structural signature overlap.
The test set contains topology families never seen in training.

Also includes target-specific discrimination cases where two actions
of the same type have different values (VERIFY(e1) != VERIFY(e2)).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph_x.receipts.checkpoint import checkpoint_from_task_and_runtime
from daph_x.receipts.causal_dataset import build_causal_dataset
from daph_x.actions.candidate_generator import generate_and_prune
from daph_x.graph.epistemic_graph import build_graph_from_evidence_task

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceHypothesis, EvidenceItem, EvidenceTask,
)


def compute_structural_signature(task: EvidenceTask) -> str:
    """Compute a canonical structural signature for a task."""
    # Use hypothesis count, evidence count, verification states, and topology
    n_hyp = len(task.hypotheses)
    n_ev = len(task.evidence_items)
    n_verified = sum(1 for e in task.evidence_items if e.verification_state != VerificationState.UNVERIFIED)
    n_unverified = n_ev - n_verified
    n_support = sum(1 for e in task.evidence_items if e.supports)
    n_contradict = sum(1 for e in task.evidence_items if e.contradicts)

    # Oracle path length and type
    oracle_str = "_".join(task.oracle_resolution_path)

    sig = f"h{n_hyp}_e{n_ev}_v{n_verified}_u{n_unverified}_s{n_support}_c{n_contradict}_{oracle_str}_{task.expected_terminal.value}"
    return sig


# Training templates: simple, low-complexity structures
TRAIN_TEMPLATES = [
    # 2 hypotheses, 1 evidence, simple VERIFY→ANSWER
    {
        "category": "TRAIN_2HYP_VERIFY",
        "summary": "Simple 2-hypothesis verify",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "DEFER")],
        "evidence": [
            ("E1", "Test for A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),
        "budget": {"steps": 3, "verify": 1, "retrieve": 0, "search": 0},
    },
    # 3 hypotheses, all verified, unique support
    {
        "category": "TRAIN_3HYP_UNIQUE",
        "summary": "3-hypothesis unique support",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "DEFER")],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of B", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("ANSWER",),
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # 3 hypotheses, competing support → DEFER
    {
        "category": "TRAIN_3HYP_COMPETING_DEFER",
        "summary": "3-hypothesis competing support",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "DEFER")],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker for B", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H3",
        "expected_terminal": "DEFER",
        "oracle_path": ("DEFER",),
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # 4 hypotheses, one unverified support
    {
        "category": "TRAIN_4HYP_ONE_UNVERIFIED",
        "summary": "4-hypothesis one unverified",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "ANSWER"), ("H4", "type D", "DEFER")],
        "evidence": [
            ("E1", "Test for A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Contradiction of B", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),
        "budget": {"steps": 4, "verify": 2, "retrieve": 0, "search": 0},
    },
]

# Test templates: novel structures never seen in training
TEST_TEMPLATES = [
    # 5 hypotheses, multiple unverified supports — VERIFY target matters
    {
        "category": "TEST_5HYP_MULTI_VERIFY",
        "summary": "5-hypothesis multiple unverified",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "ANSWER"), ("H4", "type D", "ANSWER"), ("H5", "type E", "DEFER")],
        "evidence": [
            ("E1", "Test for A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Test for B", "initial", ("H2",), (), "UNVERIFIED", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Contradiction of E", "initial", (), ("H5",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),  # Must verify E1 (H1), not E2 (H2)
        "budget": {"steps": 5, "verify": 3, "retrieve": 0, "search": 0},
    },
    # 6 hypotheses, competing verified + unverified
    {
        "category": "TEST_6HYP_MIXED_COMPETING",
        "summary": "6-hypothesis mixed competing",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "ANSWER"), ("H4", "type D", "ANSWER"), ("H5", "type E", "ANSWER"), ("H6", "type F", "DEFER")],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Marker for B", "initial", ("H2",), (), "SUFFICIENT", "CURRENT"),
            ("E3", "Test for C", "initial", ("H3",), (), "UNVERIFIED", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Contradiction of E", "initial", (), ("H5",), "SUFFICIENT", "CURRENT"),
            ("E6", "Contradiction of F", "initial", (), ("H6",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",  # H1 and H2 both supported → competing
        "expected_terminal": "DEFER",
        "oracle_path": ("DEFER",),
        "budget": {"steps": 4, "verify": 2, "retrieve": 0, "search": 0},
    },
    # 7 hypotheses, many eliminated
    {
        "category": "TEST_7HYP_MANY_ELIM",
        "summary": "7-hypothesis many eliminated",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "ANSWER"), ("H4", "type D", "ANSWER"), ("H5", "type E", "ANSWER"), ("H6", "type F", "ANSWER"), ("H7", "type G", "DEFER")],
        "evidence": [
            ("E1", "Marker for A", "initial", ("H1",), (), "SUFFICIENT", "CURRENT"),
            ("E2", "Contradiction of B", "initial", (), ("H2",), "SUFFICIENT", "CURRENT"),
            ("E3", "Contradiction of C", "initial", (), ("H3",), "SUFFICIENT", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Contradiction of E", "initial", (), ("H5",), "SUFFICIENT", "CURRENT"),
            ("E6", "Contradiction of F", "initial", (), ("H6",), "SUFFICIENT", "CURRENT"),
            ("E7", "Contradiction of G", "initial", (), ("H7",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("ANSWER",),
        "budget": {"steps": 2, "verify": 0, "retrieve": 0, "search": 0},
    },
    # 5 hypotheses, VERIFY discrimination matters
    {
        "category": "TEST_5HYP_VERIFY_DISCRIMINATION",
        "summary": "5-hypothesis verify discrimination",
        "hypotheses": [("H1", "type A", "ANSWER"), ("H2", "type B", "ANSWER"), ("H3", "type C", "ANSWER"), ("H4", "type D", "ANSWER"), ("H5", "type E", "DEFER")],
        "evidence": [
            ("E1", "Test for A", "initial", ("H1",), (), "UNVERIFIED", "CURRENT"),
            ("E2", "Test for B", "initial", ("H2",), (), "UNVERIFIED", "CURRENT"),
            ("E3", "Test for C", "initial", ("H3",), (), "UNVERIFIED", "CURRENT"),
            ("E4", "Contradiction of D", "initial", (), ("H4",), "SUFFICIENT", "CURRENT"),
            ("E5", "Contradiction of E", "initial", (), ("H5",), "SUFFICIENT", "CURRENT"),
        ],
        "correct_hypothesis": "H1",
        "expected_terminal": "ANSWER",
        "oracle_path": ("VERIFY", "ANSWER"),  # Must verify E1
        "budget": {"steps": 5, "verify": 3, "retrieve": 0, "search": 0},
    },
]


def generate_task(template: dict, idx: int) -> EvidenceTask:
    """Generate a task from a template."""
    task_id = f"m3_{template['category'].lower()}_{idx:03d}"

    hypotheses = []
    for h_id, prop, action_str in template["hypotheses"]:
        action = DecisionAction(action_str)
        hypotheses.append(EvidenceHypothesis(
            hypothesis_id=h_id,
            proposition=prop,
            answer_action=action,
            answer_payload=f"{action_str}:{h_id}:{prop}",
        ))

    evidence_items = []
    for ev_id, prop, source, supports, contradicts, vstate_str, tstatus_str in template["evidence"]:
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

    b = template["budget"]
    return EvidenceTask(
        task_id=task_id,
        split="m3",
        category=template["category"],
        task_summary=template["summary"],
        high_stakes=True,
        budget_profile=f"M3_{b['steps']}_{b['verify']}_{b['search']}",
        hypotheses=tuple(hypotheses),
        evidence_items=tuple(evidence_items),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=template["oracle_path"],
        expected_terminal=DecisionAction(template["expected_terminal"]),
        correct_hypothesis_id=template["correct_hypothesis"],
    )


def build_dataset(templates, n_per_template, seed, split_name):
    """Build a causal dataset from templates."""
    all_records = []
    for template in templates:
        for i in range(n_per_template):
            task = generate_task(template, i)
            checkpoint = checkpoint_from_task_and_runtime(task, None, seed=seed)
            graph = build_graph_from_evidence_task(task)
            candidates = generate_and_prune(graph)
            if not candidates:
                continue
            records = build_causal_dataset(checkpoint, candidates, seed=seed)
            all_records.extend(records)

    # Compute structural signatures
    signatures = set()
    for r in all_records:
        # Use checkpoint hash as proxy for structural signature
        signatures.add(r.checkpoint_hash)

    print(f"{split_name}: {len(all_records)} records, {len(signatures)} unique checkpoints")
    return all_records


def main():
    output_dir = REPO_ROOT / "experiments/daph_x/m3_structural"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build training set (simple structures)
    train_records = build_dataset(TRAIN_TEMPLATES, n_per_template=20, seed=42, split_name="TRAIN")

    # Build test set (novel structures)
    test_records = build_dataset(TEST_TEMPLATES, n_per_template=20, seed=42, split_name="TEST")

    # Check structural overlap
    train_hashes = set(r.checkpoint_hash for r in train_records)
    test_hashes = set(r.checkpoint_hash for r in test_records)
    overlap = train_hashes & test_hashes
    print(f"\nStructural overlap: {len(overlap)} (should be 0)")

    # Write
    from daph_x.receipts.causal_dataset import write_causal_dataset
    write_causal_dataset(train_records, output_dir / "m3_train.jsonl")
    write_causal_dataset(test_records, output_dir / "m3_test.jsonl")

    # Save metadata
    metadata = {
        "train_templates": [t["category"] for t in TRAIN_TEMPLATES],
        "test_templates": [t["category"] for t in TEST_TEMPLATES],
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_groups": len(set(r.counterfactual_group_id for r in train_records)),
        "test_groups": len(set(r.counterfactual_group_id for r in test_records)),
        "structural_overlap": len(overlap),
    }
    with open(output_dir / "m3_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
