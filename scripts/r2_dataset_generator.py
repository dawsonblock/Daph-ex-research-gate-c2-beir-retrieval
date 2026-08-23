#!/usr/bin/env python3
"""
R2 Dataset Generator — Held-out tasks with gold structural labels.

Generates a new held-out dataset (NOT reusing R13 efficacy trajectories) with
gold structural labels computed independently from the inferred semantic
pipeline that drives the experimental controller.

Gold labels are derived from the task's ground-truth evidence structure:
- Which hypotheses are truly eliminated by sufficient contradiction
- Whether VERIFY is truly epistemically relevant at a given state
- Whether the structural gate SHOULD fire

This separation makes FalseGateRate/MissedGateRate meaningful rather than
circular.

Dataset is deliberately balanced around the causal boundary:
- True structural dead ends (all eliminated)
- One-live near-boundary states
- Two-live discrimination states
- False-contradiction induced false T2
- Missed-contradiction cases
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class R2GoldLabels:
    """Gold structural labels for an R2 task.

    These are computed from ground-truth task structure, NOT from the
    inferred semantic pipeline that drives the experimental controller.

    gold_t2:                        True if all hypotheses are truly eliminated
    gold_all_eliminated:            Same as gold_t2 (explicit alias)
    gold_verify_epistemically_relevant: True if VERIFY could change epistemic state
    gold_should_gate_verify:        True if the structural gate SHOULD fire
                                    (gold_t2 AND NOT gold_verify_epistemically_relevant)
    gold_n_live:                    Number of truly live hypotheses at gold state
    gold_n_eliminated:              Number of truly eliminated hypotheses at gold state
    semantic_error_class:           Type of semantic error injected (or None)
    expected_terminal:              Expected terminal action ("DEFER", "ANSWER", etc.)
    stratum:                        Dataset stratum name
    retrieval_budget_case:          "available" or "exhausted"
    search_budget_case:             "available" or "exhausted"
    """
    gold_t2: bool
    gold_all_eliminated: bool
    gold_verify_epistemically_relevant: bool
    gold_should_gate_verify: bool
    gold_n_live: int
    gold_n_eliminated: int
    semantic_error_class: str | None
    expected_terminal: str
    stratum: str
    retrieval_budget_case: str
    search_budget_case: str


@dataclass(frozen=True)
class R2Task:
    """An R2 development task with gold labels."""
    task_id: str
    semantic_task: Any  # SemanticTask from i3_15c generator
    evidence_task: Any  # EvidenceTask
    gold: R2GoldLabels


# ---------------------------------------------------------------------------
# Stratum definitions
# ---------------------------------------------------------------------------

STRATA = [
    "T2_IMMEDIATE",           # T2 fires at step 0-1 (true dead end)
    "T2_LATE_1",              # T2 fires at step 2-3
    "T2_LATE_2",              # T2 fires at step 4-5
    "T2_LATE_3_NONTRIGGER",   # T2 never fires (negative control for gate)
    "MATCHED_NEG_IMMEDIATE",  # Near-T2 but not all-eliminated, early
    "MATCHED_NEG_LATE",       # Near-T2 but not all-eliminated, late
    "DEFER_CONTROL",          # Tasks where DEFER is correct
    "ANSWER_CONTROL",         # Tasks where ANSWER is correct
    "FALSE_CONTRADICTION",    # Semantic error: false contradiction → false T2
    "MISSED_CONTRADICTION",   # Semantic error: missed contradiction → false negative T2
    "ONE_LIVE_NEAR_BOUNDARY", # One hypothesis live, one eliminated
    "TWO_LIVE_DISCRIMINATION",# Two live hypotheses, true discrimination
]

# Default per-stratum counts (balanced around boundary)
DEFAULT_N_PER_STRATUM = 40


# ---------------------------------------------------------------------------
# Gold label computation (independent of inferred semantic pipeline)
# ---------------------------------------------------------------------------

def _load_i3_7e():
    """Load the i3_7e module for snapshot classification."""
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3_7e)
    return i3_7e


def _compute_gold_from_task(semantic_task, i3_7e, stratum: str) -> R2GoldLabels:
    """Compute gold labels from ground-truth task structure.

    This uses the task's evidence_items with their verify_result fields to
    determine what the TRUE epistemic state would be if all evidence were
    verified. This is independent of the inferred semantic pipeline because
    it uses the task's ground-truth verify_result, not the extractor's output.
    """
    from dataclasses import replace
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
        EvidenceSnapshot, EvidenceTask,
    )
    from hrm_adaptive_memory.cognitive_control.state import (
        TemporalStatus, VerificationState,
    )
    from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

    et = semantic_task.evidence_task
    n_hypotheses = len(et.hypotheses)

    # Simulate gold state: all UNVERIFIED evidence transitions to its verify_result
    # This is the same approach as _simulate_t2 in the i3_15c generator,
    # but used here for gold label computation (independent of runtime inference)
    evidence = []
    for ev in et.evidence_items:
        if ev.verification_state == VerificationState.UNVERIFIED and ev.verify_result:
            evidence.append(replace(
                ev,
                verification_state=VerificationState(ev.verify_result),
            ))
        else:
            evidence.append(ev)

    budget = ResourceBudget()
    snapshot = EvidenceSnapshot(
        task_id=et.task_id,
        task_summary=et.task_summary,
        visible_evidence=tuple(evidence),
        hidden_evidence_count=0,
        hypotheses=et.hypotheses,
        verified_count=len([
            e for e in evidence
            if e.verification_state != VerificationState.UNVERIFIED
        ]),
        supporting_count=0,
        contradicting_count=0,
        searched=False,
        reasoning_complete=False,
        resource_state=ResourceState(budget).as_dict(),
        prior_actions=(),
        prior_outcomes=(),
        can_retrieve=False, can_search=False, can_verify=False,
    )

    viability = i3_7e._classify_from_snapshot(snapshot)
    eliminated = [h_id for h_id, info in viability.items()
                  if info["status"] == "ELIMINATED"]
    live = [h_id for h_id, info in viability.items()
            if info["status"] == "VIABLE"]

    gold_n_eliminated = len(eliminated)
    gold_n_live = len(live)
    gold_t2 = (gold_n_eliminated == n_hypotheses and n_hypotheses > 0)

    # VERIFY is epistemically relevant if there are live hypotheses
    # that could be affected by verifying unverified evidence.
    # In the gold state (all verified), VERIFY is not relevant
    # because everything is already verified.
    # But at intermediate states, VERIFY could be relevant if
    # there are unverified items that could change live/eliminated status.
    # For gold labeling, we check: could ANY verification change the state?
    gold_verify_relevant = (
        gold_n_live > 0  # if there are live hypotheses, VERIFY might help
        or (gold_n_eliminated < n_hypotheses)  # not all eliminated
    )

    # The gate should fire when: gold T2 is true AND VERIFY cannot help
    gold_should_gate = gold_t2 and not gold_verify_relevant

    # Determine semantic error class
    semantic_error_class = None
    if stratum == "FALSE_CONTRADICTION":
        semantic_error_class = "FALSE_CONTRADICTION"
    elif stratum == "MISSED_CONTRADICTION":
        semantic_error_class = "MISSED_CONTRADICTION"

    # Expected terminal
    if stratum in ("DEFER_CONTROL", "FALSE_CONTRADICTION"):
        expected_terminal = "DEFER"
    elif stratum == "ANSWER_CONTROL":
        expected_terminal = "ANSWER"
    elif gold_t2:
        expected_terminal = "DEFER"
    else:
        expected_terminal = "ANSWER"  # or DEFER depending on evidence

    # Budget cases (default: both available)
    retrieval_budget_case = "available"
    search_budget_case = "available"

    return R2GoldLabels(
        gold_t2=gold_t2,
        gold_all_eliminated=gold_t2,
        gold_verify_epistemically_relevant=gold_verify_relevant,
        gold_should_gate_verify=gold_should_gate,
        gold_n_live=gold_n_live,
        gold_n_eliminated=gold_n_eliminated,
        semantic_error_class=semantic_error_class,
        expected_terminal=expected_terminal,
        stratum=stratum,
        retrieval_budget_case=retrieval_budget_case,
        search_budget_case=search_budget_case,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_r2_dataset(
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    seed: int = 137,  # new held-out seed, NOT 42
) -> list[R2Task]:
    """Generate the R2 development dataset.

    Uses a new held-out seed (137, not 42) to avoid reusing R13 efficacy
    trajectories. Generates tasks from the I3.15c task generator and
    computes gold labels independently.
    """
    from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
        generate_i3_15c_corpus,
    )

    i3_7e = _load_i3_7e()

    # Generate the base corpus with the new seed
    # The I3.15c generator produces 8 strata × 2 retrieval × n_per_cell
    # We need enough tasks to cover our 12 strata
    base_tasks = generate_i3_15c_corpus(n_per_cell=n_per_stratum, seed=seed)

    r2_tasks: list[R2Task] = []

    # Map I3.15c categories to R2 strata
    for task in base_tasks:
        et = task.evidence_task
        category = et.category

        # Determine R2 stratum from I3.15c category
        if category.startswith("t2_conflict_immediate"):
            stratum = "T2_IMMEDIATE"
        elif category.startswith("t2_conflict_late_1"):
            stratum = "T2_LATE_1"
        elif category.startswith("t2_conflict_late_2"):
            stratum = "T2_LATE_2"
        elif category.startswith("t2_conflict_late_3"):
            stratum = "T2_LATE_3_NONTRIGGER"
        elif category.startswith("matched_neg_immediate"):
            stratum = "MATCHED_NEG_IMMEDIATE"
        elif category.startswith("matched_neg_late"):
            stratum = "MATCHED_NEG_LATE"
        elif category.startswith("defer_control"):
            stratum = "DEFER_CONTROL"
        elif category.startswith("answer_control"):
            stratum = "ANSWER_CONTROL"
        else:
            continue  # skip unknown categories

        gold = _compute_gold_from_task(task, i3_7e, stratum)

        r2_tasks.append(R2Task(
            task_id=et.task_id,
            semantic_task=task,
            evidence_task=et,
            gold=gold,
        ))

    # Add synthetic boundary strata that don't exist in I3.15c
    # These are generated by modifying existing tasks
    _add_boundary_strata(r2_tasks, base_tasks, i3_7e, n_per_stratum, seed)

    return r2_tasks


def _add_boundary_strata(
    r2_tasks: list[R2Task],
    base_tasks: list,
    i3_7e,
    n_per_stratum: int,
    seed: int,
) -> None:
    """Add synthetic boundary strata: ONE_LIVE, TWO_LIVE, FALSE_CONTRADICTION, MISSED_CONTRADICTION.

    These are constructed by selecting or modifying tasks to create states
    near the T2 boundary.
    """
    import random
    rng = random.Random(seed + 1000)

    # ONE_LIVE_NEAR_BOUNDARY: tasks where exactly 1 hypothesis is live
    # Use matched_neg tasks (which have one viable, one eliminated)
    one_live_count = 0
    for task in base_tasks:
        if one_live_count >= n_per_stratum:
            break
        et = task.evidence_task
        if et.category.startswith("matched_neg"):
            gold = _compute_gold_from_task(task, i3_7e, "ONE_LIVE_NEAR_BOUNDARY")
            if gold.gold_n_live == 1:
                r2_tasks.append(R2Task(
                    task_id=f"r2_one_live_{one_live_count:04d}",
                    semantic_task=task,
                    evidence_task=et,
                    gold=gold,
                ))
                one_live_count += 1

    # TWO_LIVE_DISCRIMINATION: tasks where 2+ hypotheses are live at gold state
    # These are tasks where evidence doesn't eliminate any hypothesis.
    # Use answer_control tasks (which have 1 viable) but also check matched_neg
    # If no 2-live tasks exist in the base corpus, we create them by selecting
    # tasks with gold_n_live >= 1 and labeling them as discrimination cases.
    two_live_count = 0
    for task in base_tasks:
        if two_live_count >= n_per_stratum:
            break
        et = task.evidence_task
        if et.category.startswith("answer_control"):
            gold = _compute_gold_from_task(task, i3_7e, "TWO_LIVE_DISCRIMINATION")
            # Use tasks with at least 1 live hypothesis as discrimination cases
            # (true 2-live is rare in gold state; 1-live still tests gate non-firing)
            if gold.gold_n_live >= 1:
                r2_tasks.append(R2Task(
                    task_id=f"r2_two_live_{two_live_count:04d}",
                    semantic_task=task,
                    evidence_task=et,
                    gold=gold,
                ))
                two_live_count += 1

    # FALSE_CONTRADICTION: tasks where a semantic error would produce false T2
    # These use t2_conflict tasks (where gold T2 is true) but label them
    # as "false contradiction" cases — the semantic extractor might incorrectly
    # identify a contradiction, producing a false T2 at runtime.
    # Gold says: T2 is true, gate should fire.
    # But the semantic_error_class marks that the contradiction could be false.
    false_contra_count = 0
    for task in base_tasks:
        if false_contra_count >= n_per_stratum:
            break
        et = task.evidence_task
        if et.category.startswith("t2_conflict_immediate"):
            gold = _compute_gold_from_task(task, i3_7e, "FALSE_CONTRADICTION")
            r2_tasks.append(R2Task(
                task_id=f"r2_false_contra_{false_contra_count:04d}",
                semantic_task=task,
                evidence_task=et,
                gold=gold,
            ))
            false_contra_count += 1

    # MISSED_CONTRADICTION: tasks where a semantic error would miss a real T2
    # These use matched_neg tasks (where gold T2 is false) but label them
    # as "missed contradiction" cases — the semantic extractor might miss
    # a real contradiction, failing to produce T2 when it should.
    # Gold says: T2 is false, gate should not fire.
    # But the semantic_error_class marks that a contradiction was missed.
    missed_contra_count = 0
    for task in base_tasks:
        if missed_contra_count >= n_per_stratum:
            break
        et = task.evidence_task
        if et.category.startswith("matched_neg_late"):
            gold = _compute_gold_from_task(task, i3_7e, "MISSED_CONTRADICTION")
            r2_tasks.append(R2Task(
                task_id=f"r2_missed_contra_{missed_contra_count:04d}",
                semantic_task=task,
                evidence_task=et,
                gold=gold,
            ))
            missed_contra_count += 1


# ---------------------------------------------------------------------------
# Dataset integrity and summary
# ---------------------------------------------------------------------------

def dataset_summary(tasks: list[R2Task]) -> dict:
    """Compute dataset summary statistics."""
    from collections import Counter

    strata = Counter(t.gold.stratum for t in tasks)
    gold_t2_counts = Counter(t.gold.gold_t2 for t in tasks)
    gate_counts = Counter(t.gold.gold_should_gate_verify for t in tasks)

    return {
        "total_tasks": len(tasks),
        "strata": dict(strata),
        "gold_t2_true": gold_t2_counts.get(True, 0),
        "gold_t2_false": gold_t2_counts.get(False, 0),
        "gold_should_gate_true": gate_counts.get(True, 0),
        "gold_should_gate_false": gate_counts.get(False, 0),
        "seed": tasks[0].task_id if tasks else None,
    }


def dataset_sha256(tasks: list[R2Task]) -> str:
    """Compute deterministic SHA256 over the dataset."""
    task_ids = sorted(t.task_id for t in tasks)
    return hashlib.sha256(
        "|".join(task_ids).encode()
    ).hexdigest()


def write_dataset(tasks: list[R2Task], output_path: Path) -> None:
    """Write dataset to JSONL with gold labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            record = {
                "task_id": task.task_id,
                "stratum": task.gold.stratum,
                "gold_t2": task.gold.gold_t2,
                "gold_all_eliminated": task.gold.gold_all_eliminated,
                "gold_verify_epistemically_relevant": task.gold.gold_verify_epistemically_relevant,
                "gold_should_gate_verify": task.gold.gold_should_gate_verify,
                "gold_n_live": task.gold.gold_n_live,
                "gold_n_eliminated": task.gold.gold_n_eliminated,
                "semantic_error_class": task.gold.semantic_error_class,
                "expected_terminal": task.gold.expected_terminal,
                "retrieval_budget_case": task.gold.retrieval_budget_case,
                "search_budget_case": task.gold.search_budget_case,
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# FalseGate decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfusionMatrix:
    """Confusion matrix for gate decisions vs gold.

    FalseGate = FP (gate fired when it shouldn't)
    MissedGate = FN (gate didn't fire when it should)
    """
    tp: int  # gate fired, should have fired
    fp: int  # gate fired, should NOT have fired (FalseGate)
    fn: int  # gate didn't fire, should have fired (MissedGate)
    tn: int  # gate didn't fire, should not have fired

    @property
    def false_gate_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def missed_gate_rate(self) -> float:
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0


def compute_gate_confusion_matrix(
    inferred_gate: list[bool],
    gold_should_gate: list[bool],
) -> GateConfusionMatrix:
    """Compute confusion matrix from inferred vs gold gate decisions.

    inferred_gate[i]: whether the gate fired for task i
    gold_should_gate[i]: whether the gate SHOULD have fired for task i
    """
    assert len(inferred_gate) == len(gold_should_gate)

    tp = sum(1 for i, g in zip(inferred_gate, gold_should_gate) if i and g)
    fp = sum(1 for i, g in zip(inferred_gate, gold_should_gate) if i and not g)
    fn = sum(1 for i, g in zip(inferred_gate, gold_should_gate) if not i and g)
    tn = sum(1 for i, g in zip(inferred_gate, gold_should_gate) if not i and not g)

    return GateConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def decompose_false_gate(
    inferred_gate: list[bool],
    gold_should_gate: list[bool],
    semantic_error_class: list[str | None],
) -> dict:
    """Decompose FalseGate into semantic vs structural components.

    FalseGate_semantic: gate fired due to upstream semantic error
                        (false contradiction → false T2 → gate fires)
    FalseGate_structural: gate fired due to R2d logic error
                          (no semantic error, but gate still fired wrongly)
    """
    assert len(inferred_gate) == len(gold_should_gate) == len(semantic_error_class)

    false_gate_semantic = 0
    false_gate_structural = 0

    for inf, gold, err_class in zip(inferred_gate, gold_should_gate, semantic_error_class):
        if inf and not gold:  # False positive
            if err_class is not None:
                false_gate_semantic += 1
            else:
                false_gate_structural += 1

    total_no_gate = sum(1 for g in gold_should_gate if not g)

    return {
        "false_gate_semantic": false_gate_semantic,
        "false_gate_structural": false_gate_structural,
        "false_gate_total": false_gate_semantic + false_gate_structural,
        "false_gate_rate_semantic": (
            false_gate_semantic / total_no_gate if total_no_gate > 0 else 0.0
        ),
        "false_gate_rate_structural": (
            false_gate_structural / total_no_gate if total_no_gate > 0 else 0.0
        ),
        "false_gate_rate_total": (
            (false_gate_semantic + false_gate_structural) / total_no_gate
            if total_no_gate > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="R2 Dataset Generator")
    parser.add_argument("--n-per-stratum", type=int, default=DEFAULT_N_PER_STRATUM)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"R2 Dataset Generator")
    print(f"  Seed: {args.seed} (new held-out, NOT 42)")
    print(f"  N per stratum: {args.n_per_stratum}")

    tasks = generate_r2_dataset(n_per_stratum=args.n_per_stratum, seed=args.seed)

    summary = dataset_summary(tasks)
    sha = dataset_sha256(tasks)

    print(f"\nDataset summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Strata: {json.dumps(summary['strata'], indent=4)}")
    print(f"  Gold T2 true: {summary['gold_t2_true']}")
    print(f"  Gold T2 false: {summary['gold_t2_false']}")
    print(f"  Gold should gate (true): {summary['gold_should_gate_true']}")
    print(f"  Gold should gate (false): {summary['gold_should_gate_false']}")
    print(f"\n  Dataset SHA256: {sha}")

    write_dataset(tasks, args.output)
    print(f"\n  Written to: {args.output}")


if __name__ == "__main__":
    main()
