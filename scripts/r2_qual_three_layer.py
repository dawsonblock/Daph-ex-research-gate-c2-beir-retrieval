#!/usr/bin/env python3
"""
R2-QUAL — Three-Layer Gate Qualification.

Decomposes gate errors into three layers:

    Semantic inference → T2 classification → R2d gate

Layer 1: Semantic inference
    Did the semantic extractor correctly identify contradictions?

Layer 2: T2 classification
    Given correct semantics, did the T2 classifier correctly identify
    the all-eliminated state?

Layer 3: R2d gate
    Given correct T2, did the R2d gate fire correctly?

The gate implementation itself should be perfect:
    FalseGate_{R2d} = 0,  MissedGate_{R2d} = 0

End-to-end may not be zero because the semantic extractor can be wrong.
That is a scientific result, not necessarily a gate-code defect.

Usage:
    PYTHONPATH=scripts:. python3 scripts/r2_qual_three_layer.py \
        --dataset /path/to/balanced_dataset.jsonl \
        --output /path/to/r2_qual_v2/
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from r2_allowed_actions import (
    ACTION_VOCABULARY,
    ActionState,
    C0, D, E, DE,
    ALL_ARMS,
    compute_allowed_actions,
)


@dataclass
class GateAssessment:
    """Assessment of a single task's gate behavior across three layers."""
    task_id: str
    stratum: str
    gold_t2: bool
    gold_should_gate: bool
    semantic_error_class: str | None

    # Layer 1: Semantic inference
    # (assessed from stratum: FALSE_CONTRADICTION / MISSED_CONTRADICTION)
    semantic_correct: bool
    semantic_error_type: str | None  # "false_contradiction", "missed_contradiction", None

    # Layer 2: T2 classification
    # (would be assessed from runtime; here we use gold as perfect inference)
    inferred_t2: bool
    t2_correct: bool

    # Layer 3: R2d gate
    gate_fired: bool
    gate_should_fire: bool
    gate_correct: bool

    # Confusion matrix category
    tp: bool = False  # gate fired and should fire
    fp: bool = False  # gate fired but should not (false gate)
    fn: bool = False  # gate did not fire but should (missed gate)
    tn: bool = False  # gate did not fire and should not


def assess_task(task: dict) -> GateAssessment:
    """Assess a single task's gate behavior across three layers.

    For R2-QUAL (structural qualification), we use gold labels as
    perfect inference. End-to-end assessment with real inference
    happens during R2-DEV-V2.
    """
    gold_t2 = task["gold_t2"]
    gold_should_gate = task["gold_should_gate_verify"]
    semantic_error_class = task.get("semantic_error_class")

    # Layer 1: Semantic inference
    # FALSE_CONTRADICTION: semantic extractor falsely inferred a contradiction
    # MISSED_CONTRADICTION: semantic extractor missed a real contradiction
    if semantic_error_class == "FALSE_CONTRADICTION":
        semantic_correct = False
        semantic_error_type = "false_contradiction"
    elif semantic_error_class == "MISSED_CONTRADICTION":
        semantic_correct = False
        semantic_error_type = "missed_contradiction"
    else:
        semantic_correct = True
        semantic_error_type = None

    # Layer 2: T2 classification
    # For structural qual, use gold T2 as inferred T2 (perfect inference)
    inferred_t2 = gold_t2
    t2_correct = True  # perfect inference in structural qual

    # Layer 3: R2d gate
    state = ActionState(
        t2=inferred_t2,
        executive_steps_remaining=5,
        can_retrieve=task["retrieval_budget_case"] == "available",
        can_search=task["search_budget_case"] == "available",
        can_verify=True,
    )
    d_decision = compute_allowed_actions(state, D)
    gate_fired = d_decision.verify_gate_condition_active
    gate_should_fire = gold_should_gate

    gate_correct = (gate_fired == gate_should_fire)

    tp = gate_fired and gate_should_fire
    fp = gate_fired and not gate_should_fire
    fn = not gate_fired and gate_should_fire
    tn = not gate_fired and not gate_should_fire

    return GateAssessment(
        task_id=task["task_id"],
        stratum=task["stratum"],
        gold_t2=gold_t2,
        gold_should_gate=gold_should_gate,
        semantic_error_class=semantic_error_class,
        semantic_correct=semantic_correct,
        semantic_error_type=semantic_error_type,
        inferred_t2=inferred_t2,
        t2_correct=t2_correct,
        gate_fired=gate_fired,
        gate_should_fire=gate_should_fire,
        gate_correct=gate_correct,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


@dataclass
class ThreeLayerReport:
    """Full three-layer qualification report."""
    # Layer 3: R2d gate (should be perfect)
    r2d_false_gate: int
    r2d_missed_gate: int
    r2d_true_gate: int
    r2d_true_no_gate: int
    r2d_false_gate_rate: float
    r2d_missed_gate_rate: float

    # Layer 2: T2 classification
    t2_errors: int
    t2_error_rate: float

    # Layer 1: Semantic inference
    semantic_errors: int
    semantic_error_types: dict[str, int]

    # End-to-end
    e2e_false_gate: int
    e2e_missed_gate: int
    e2e_false_gate_rate: float
    e2e_missed_gate_rate: float

    # Per-stratum breakdown
    per_stratum: dict[str, dict]

    # All assessments
    assessments: list[dict] = field(default_factory=list)

    @property
    def r2d_gate_perfect(self) -> bool:
        """R2d gate implementation should be perfect."""
        return self.r2d_false_gate == 0 and self.r2d_missed_gate == 0

    def to_dict(self) -> dict:
        return {
            "r2d_gate": {
                "false_gate": self.r2d_false_gate,
                "missed_gate": self.r2d_missed_gate,
                "true_gate": self.r2d_true_gate,
                "true_no_gate": self.r2d_true_no_gate,
                "false_gate_rate": self.r2d_false_gate_rate,
                "missed_gate_rate": self.r2d_missed_gate_rate,
                "perfect": self.r2d_gate_perfect,
            },
            "t2_classification": {
                "errors": self.t2_errors,
                "error_rate": self.t2_error_rate,
            },
            "semantic_inference": {
                "errors": self.semantic_errors,
                "error_types": self.semantic_error_types,
            },
            "end_to_end": {
                "false_gate": self.e2e_false_gate,
                "missed_gate": self.e2e_missed_gate,
                "false_gate_rate": self.e2e_false_gate_rate,
                "missed_gate_rate": self.e2e_missed_gate_rate,
            },
            "per_stratum": self.per_stratum,
            "assessments": self.assessments,
        }


def run_three_layer_qualification(tasks: list[dict]) -> ThreeLayerReport:
    """Run three-layer gate qualification on a dataset."""
    assessments = [assess_task(t) for t in tasks]

    # Layer 3: R2d gate (using gold T2 as inferred T2 = perfect inference)
    r2d_tp = sum(1 for a in assessments if a.tp)
    r2d_fp = sum(1 for a in assessments if a.fp)
    r2d_fn = sum(1 for a in assessments if a.fn)
    r2d_tn = sum(1 for a in assessments if a.tn)

    r2d_fgr = r2d_fp / (r2d_fp + r2d_tn) if (r2d_fp + r2d_tn) > 0 else 0.0
    r2d_mgr = r2d_fn / (r2d_fn + r2d_tp) if (r2d_fn + r2d_tp) > 0 else 0.0

    # Layer 2: T2 classification (perfect in structural qual)
    t2_errors = sum(1 for a in assessments if not a.t2_correct)
    t2_error_rate = t2_errors / len(assessments) if assessments else 0.0

    # Layer 1: Semantic inference
    semantic_errors = sum(1 for a in assessments if not a.semantic_correct)
    semantic_error_types = Counter(
        a.semantic_error_type for a in assessments if a.semantic_error_type
    )

    # End-to-end (same as R2d in structural qual since T2 is perfect)
    e2e_false_gate = r2d_fp
    e2e_missed_gate = r2d_fn
    e2e_fgr = r2d_fgr
    e2e_mgr = r2d_mgr

    # Per-stratum breakdown
    per_stratum: dict[str, dict] = {}
    by_stratum = defaultdict(list)
    for a in assessments:
        by_stratum[a.stratum].append(a)

    for stratum, stratum_assessments in sorted(by_stratum.items()):
        tp = sum(1 for a in stratum_assessments if a.tp)
        fp = sum(1 for a in stratum_assessments if a.fp)
        fn = sum(1 for a in stratum_assessments if a.fn)
        tn = sum(1 for a in stratum_assessments if a.tn)
        per_stratum[stratum] = {
            "n": len(stratum_assessments),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "gate_correct": sum(1 for a in stratum_assessments if a.gate_correct),
            "semantic_errors": sum(1 for a in stratum_assessments if not a.semantic_correct),
        }

    return ThreeLayerReport(
        r2d_false_gate=r2d_fp,
        r2d_missed_gate=r2d_fn,
        r2d_true_gate=r2d_tp,
        r2d_true_no_gate=r2d_tn,
        r2d_false_gate_rate=r2d_fgr,
        r2d_missed_gate_rate=r2d_mgr,
        t2_errors=t2_errors,
        t2_error_rate=t2_error_rate,
        semantic_errors=semantic_errors,
        semantic_error_types=dict(semantic_error_types),
        e2e_false_gate=e2e_false_gate,
        e2e_missed_gate=e2e_missed_gate,
        e2e_false_gate_rate=e2e_fgr,
        e2e_missed_gate_rate=e2e_mgr,
        per_stratum=per_stratum,
        assessments=[
            {
                "task_id": a.task_id,
                "stratum": a.stratum,
                "gold_t2": a.gold_t2,
                "gold_should_gate": a.gold_should_gate,
                "semantic_correct": a.semantic_correct,
                "semantic_error_type": a.semantic_error_type,
                "inferred_t2": a.inferred_t2,
                "t2_correct": a.t2_correct,
                "gate_fired": a.gate_fired,
                "gate_should_fire": a.gate_should_fire,
                "gate_correct": a.gate_correct,
                "tp": a.tp,
                "fp": a.fp,
                "fn": a.fn,
                "tn": a.tn,
            }
            for a in assessments
        ],
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="R2-QUAL Three-Layer Gate Qualification")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to balanced dataset JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for qualification report")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        tasks = [json.loads(line) for line in f]

    print("R2-QUAL Three-Layer Gate Qualification")
    print(f"  Dataset: {args.dataset}")
    print(f"  Tasks: {len(tasks)}")
    print()

    report = run_three_layer_qualification(tasks)

    # Print results
    print("=== Layer 3: R2d Gate (should be perfect) ===")
    print(f"  True gates (TP):     {report.r2d_true_gate}")
    print(f"  False gates (FP):    {report.r2d_false_gate}")
    print(f"  Missed gates (FN):   {report.r2d_missed_gate}")
    print(f"  True no-gates (TN):  {report.r2d_true_no_gate}")
    print(f"  FalseGateRate:       {report.r2d_false_gate_rate:.4f}")
    print(f"  MissedGateRate:      {report.r2d_missed_gate_rate:.4f}")
    print(f"  Gate perfect:        {report.r2d_gate_perfect}")
    print()

    print("=== Layer 2: T2 Classification ===")
    print(f"  Errors:              {report.t2_errors}")
    print(f"  Error rate:          {report.t2_error_rate:.4f}")
    print()

    print("=== Layer 1: Semantic Inference ===")
    print(f"  Errors:              {report.semantic_errors}")
    print(f"  Error types:         {report.semantic_error_types}")
    print()

    print("=== End-to-End ===")
    print(f"  FalseGate:           {report.e2e_false_gate}")
    print(f"  MissedGate:          {report.e2e_missed_gate}")
    print(f"  FalseGateRate:       {report.e2e_false_gate_rate:.4f}")
    print(f"  MissedGateRate:      {report.e2e_missed_gate_rate:.4f}")
    print()

    print("=== Per-Stratum Breakdown ===")
    for stratum, stats in sorted(report.per_stratum.items()):
        print(f"  {stratum:40s} n={stats['n']:2d}  "
              f"TP={stats['tp']} FP={stats['fp']} "
              f"FN={stats['fn']} TN={stats['tn']}  "
              f"sem_err={stats['semantic_errors']}")
    print()

    # Write report
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "three_layer_qualification.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)

    print(f"Report written to: {report_path}")

    if not report.r2d_gate_perfect:
        print("\n*** R2d GATE NOT PERFECT — DO NOT PROCEED TO R2-DEV-V2 ***")
        sys.exit(1)
    else:
        print("\n*** R2d GATE PERFECT — R2-DEV-V2 may proceed ***")


if __name__ == "__main__":
    main()
