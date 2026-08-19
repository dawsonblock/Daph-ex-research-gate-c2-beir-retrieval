#!/usr/bin/env python3
"""I3.11a: Routing Invariance Stress Test — R0 vs A1 vs M3 on decorrelated corpus.

Tests whether the R0 routing rule (hidden_evidence_count == 0 -> M3, else -> A1)
generalizes when hidden_evidence_count is deliberately crossed with task semantics.

Distinguishes:
  H1: zero hidden evidence itself predicts MDSG applicability
  H2: zero hidden evidence merely identifies conflict_unresolved in current generator

Critical decorrelation cases:
  1. conflict_unresolved WITH hidden evidence (hidden=1, hidden=2)
     - If H1: M3 should NOT help (hidden>0)
     - If H2: M3 SHOULD help (still conflict_unresolved)
     - R0 routes to A1 (hidden>0). If M3 would rescue, R0 loses.
  2. ANSWER tasks with hidden=0
     - If H1: M3 should help (hidden=0)
     - If H2: M3 should NOT help (not conflict_unresolved)
     - R0 routes to M3. If M3 wastes steps, R0 loses efficiency.

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_11a_routing_stress.py \\
        --n-tasks 300 --workers 4
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


# ---------------------------------------------------------------------------
# Decorrelated corpus generator
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
    """Create a noise evidence item that supports neither hypothesis."""
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


def gen_conflict_unresolved_crossed(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """conflict_unresolved with variable hidden_evidence_count.

    hidden=0: original (both SUFFICIENT, no hidden, DEFER)
    hidden=1: + 1 hidden noise evidence (doesn't resolve conflict)
    hidden=2: + 2 hidden noise evidence
    """
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A definitively confirms {subject}.",
            source_class="primary",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B definitively refutes {subject}.",
            source_class="primary",
            supports=("H2",),
            contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
    ]

    retrieve_exposes = ()
    search_exposes = ()
    for i in range(hidden_count):
        eid = f"E{3+i}"
        evidence.append(_noise_evidence(eid, subject))
        # Noise evidence is exposed via search but doesn't help
        search_exposes = search_exposes + (eid,)

    return EvidenceTask(
        task_id=task_id, split="routing_stress_v1",
        category=f"conflict_unresolved_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=retrieve_exposes,
        search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "DEFER"),
        expected_terminal=DecisionAction.DEFER,
        correct_hypothesis_id="H2",
    )


def gen_single_verify_ready_crossed(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """single_verify_ready with variable hidden_evidence_count.

    hidden=0: 1 visible evidence, verify, answer. No hidden.
    hidden=1: original (1 visible + 1 hidden via search)
    hidden=2: 1 visible + 1 hidden via search + 1 hidden noise
    """
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"The primary documentation confirms that {subject}.",
            source_class="primary",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
    ]

    search_exposes = ()
    if hidden_count >= 1:
        evidence.append(EvidenceItem(
            evidence_id="E2",
            proposition=f"A secondary source does not address {subject}.",
            source_class="search",
            supports=("H2",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False,
            verify_result="MISSING",
        ))
        search_exposes = ("E2",)

    if hidden_count >= 2:
        evidence.append(_noise_evidence("E3", subject))
        search_exposes = search_exposes + ("E3",)

    return EvidenceTask(
        task_id=task_id, split="routing_stress_v1",
        category=f"single_verify_ready_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(),
        search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_varying_visible_split_crossed(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """varying_visible_split with variable hidden_evidence_count.

    hidden=0: 3 visible, no hidden. Verify all 3, answer.
    hidden=1: original (3 visible + 1 hidden via search)
    hidden=2: 3 visible + 1 hidden via search + 1 hidden noise
    """
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A confirms {subject}.",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B also confirms {subject}.",
            source_class="initial",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"Source C contradicts, claiming not-{subject}.",
            source_class="initial",
            supports=("H2",),
            contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="FALSIFIED",
        ),
    ]

    search_exposes = ()
    if hidden_count >= 1:
        evidence.append(EvidenceItem(
            evidence_id="E4",
            proposition=f"A hidden source provides additional confirmation of {subject}.",
            source_class="search",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False,
            verify_result="SUFFICIENT",
        ))
        search_exposes = ("E4",)

    if hidden_count >= 2:
        evidence.append(_noise_evidence("E5", subject))
        search_exposes = search_exposes + ("E5",)

    oracle = ("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER")
    if hidden_count >= 1:
        oracle = ("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "SEARCH_MORE:E4", "VERIFY:E4", "ANSWER")

    return EvidenceTask(
        task_id=task_id, split="routing_stress_v1",
        category=f"varying_visible_split_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=rng.random() > 0.5,
        budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(),
        search_exposes=search_exposes,
        oracle_resolution_path=oracle,
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_triple_verify_ready_crossed(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """triple_verify_ready with variable hidden_evidence_count.

    hidden=0: original (3 visible, 0 hidden)
    hidden=1: 3 visible + 1 hidden noise
    hidden=2: 3 visible + 2 hidden noise
    """
    subject = template["subject"]
    h1, h2 = _make_h1_h2(template)

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            proposition=f"Source A claims {subject}.",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Source B also claims {subject}.",
            source_class="initial",
            supports=("H1",),
            contradicts=("H2",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"Source C contradicts, claiming not-{subject}.",
            source_class="initial",
            supports=("H2",),
            contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="FALSIFIED",
        ),
    ]

    search_exposes = ()
    for i in range(hidden_count):
        eid = f"E{4+i}"
        evidence.append(_noise_evidence(eid, subject))
        search_exposes = search_exposes + (eid,)

    return EvidenceTask(
        task_id=task_id, split="routing_stress_v1",
        category=f"triple_verify_ready_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=(h1, h2),
        evidence_items=tuple(evidence),
        retrieve_exposes=(),
        search_exposes=search_exposes,
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


def gen_early_false_ready_crossed(
    task_id: str, template: dict, rng: random.Random, hidden_count: int,
) -> EvidenceTask:
    """early_false_ready with variable hidden_evidence_count.

    hidden=1: original (E3 hidden via retrieve, must be found)
    hidden=2: E3 hidden via retrieve + E4 hidden noise via search
    """
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
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition=f"Another source contradicts, claiming not-{subject}.",
            source_class="initial",
            supports=("H2",),
            contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="FALSIFIED",
        ),
        EvidenceItem(
            evidence_id="E3",
            proposition=f"A definitive source refutes the claim about {subject}.",
            source_class="primary",
            supports=("H2",),
            contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=False,
            verify_result="SUFFICIENT",
        ),
    ]

    retrieve_exposes = ("E3",)
    search_exposes = ()
    if hidden_count >= 2:
        evidence.append(_noise_evidence("E4", subject))
        search_exposes = ("E4",)

    return EvidenceTask(
        task_id=task_id, split="routing_stress_v1",
        category=f"early_false_ready_h{hidden_count}",
        task_summary=f"Determine {subject}.",
        high_stakes=True,
        budget_profile="STANDARD",
        hypotheses=(h1_answer, h2_answer),
        evidence_items=tuple(evidence),
        retrieve_exposes=retrieve_exposes,
        search_exposes=search_exposes,
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H2",
    )


def generate_decorrelated_corpus(split: str = "routing_stress_v1") -> list[EvidenceTask]:
    """Generate decorrelated corpus crossing hidden_evidence_count with task semantics.

    Target: 300 tasks
    Distribution:
      conflict_unresolved × {h0, h1, h2}:    30 + 30 + 30 = 90
      single_verify_ready × {h0, h1, h2}:    25 + 25 + 25 = 75
      varying_visible_split × {h0, h1, h2}:  15 + 15 + 15 = 45
      triple_verify_ready × {h0, h1, h2}:    10 + 10 + 10 = 30
      early_false_ready × {h1, h2}:          15 + 15     = 30
      (stale_support requires hidden to resolve — excluded from crossing)
    Total: 90 + 75 + 45 + 30 + 30 = 270 + 30 buffer = 300
    """
    target = {
        ("conflict_unresolved", 0): 30,
        ("conflict_unresolved", 1): 30,
        ("conflict_unresolved", 2): 30,
        ("single_verify_ready", 0): 25,
        ("single_verify_ready", 1): 25,
        ("single_verify_ready", 2): 25,
        ("varying_visible_split", 0): 15,
        ("varying_visible_split", 1): 15,
        ("varying_visible_split", 2): 15,
        ("triple_verify_ready", 0): 10,
        ("triple_verify_ready", 1): 10,
        ("triple_verify_ready", 2): 10,
        ("early_false_ready", 1): 15,
        ("early_false_ready", 2): 15,
    }

    generators = {
        "conflict_unresolved": gen_conflict_unresolved_crossed,
        "single_verify_ready": gen_single_verify_ready_crossed,
        "varying_visible_split": gen_varying_visible_split_crossed,
        "triple_verify_ready": gen_triple_verify_ready_crossed,
        "early_false_ready": gen_early_false_ready_crossed,
    }

    tasks: list[EvidenceTask] = []
    task_idx = 0
    for (category, hidden_count), count in target.items():
        gen = generators[category]
        for i in range(count):
            task_id = f"{split}_{task_idx:04d}"
            template = STRUCTURAL_TEMPLATES[task_idx % len(STRUCTURAL_TEMPLATES)]
            rng = _seeded_rng(task_id)
            task = gen(task_id, template, rng, hidden_count)
            tasks.append(task)
            task_idx += 1

    return tasks


# ---------------------------------------------------------------------------
# R0 Router
# ---------------------------------------------------------------------------

def r0_route(hidden_evidence_count: int) -> str:
    """R0: deterministic hidden-count router.

    Returns 'M3' if hidden_evidence_count == 0, else 'A1'.
    This is the ONLY input to the router — no other field.
    """
    if hidden_evidence_count == 0:
        return "M3"
    return "A1"


def assert_r0_leakage(router_input: dict) -> None:
    """Assert that R0 only receives hidden_evidence_count."""
    allowed = {"hidden_evidence_count"}
    extra = set(router_input.keys()) - allowed
    if extra:
        raise ValueError(f"R0 LEAKAGE: router received forbidden fields: {extra}")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def counterbalance_3arm(task_id: str) -> list[str]:
    h = hashlib.sha256(task_id.encode()).hexdigest()
    arms = ["A1", "M3", "R0"]
    perms = list(itertools.permutations(arms))
    return list(perms[int(h[:8], 16) % len(perms)])


def process_one_task(
    task: EvidenceTask,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
) -> dict[str, Any]:
    # R0 routing decision from step-0 hidden_evidence_count ONLY
    # Count hidden evidence items (retrieved=False)
    hidden_count = sum(1 for e in task.evidence_items if not e.retrieved)
    router_input = {"hidden_evidence_count": hidden_count}
    assert_r0_leakage(router_input)
    r0_decision = r0_route(hidden_count)

    # Run all three arms
    fork_order = counterbalance_3arm(task.task_id)
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

    # R0 uses the routed arm's result
    r0_arm = r0_decision
    results["R0"] = results[r0_arm]

    return {
        "task_id": task.task_id,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": len(task.hypotheses),
        "n_hidden": hidden_count,
        "oracle_steps": len(task.oracle_resolution_path),
        "r0_decision": r0_decision,
        "fork_order": fork_order,
        "u_a1": results["A1"]["realized_utility"],
        "u_m3": results["M3"]["realized_utility"],
        "u_r0": results["R0"]["realized_utility"],
        "r0_delta_vs_a1": round(results["R0"]["realized_utility"] - results["A1"]["realized_utility"], 4),
        "r0_delta_vs_m3": round(results["R0"]["realized_utility"] - results["M3"]["realized_utility"], 4),
        "a1_success": results["A1"]["success"],
        "m3_success": results["M3"]["success"],
        "r0_success": results["R0"]["success"],
        "a1_steps": results["A1"]["steps"],
        "m3_steps": results["M3"]["steps"],
        "r0_steps": results["R0"]["steps"],
        "m3_rescues_vs_a1": (not results["A1"]["success"]) and results["M3"]["success"],
        "m3_breaks_vs_a1": results["A1"]["success"] and (not results["M3"]["success"]),
        "fork_a1": results["A1"],
        "fork_m3": results["M3"],
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
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_11/development/i3_11a_routing_stress",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.11a: Routing Invariance Stress Test")
    print("  R0: hidden_evidence_count == 0 -> M3, else -> A1")
    print("  Testing on decorrelated corpus (hidden_count crossed with task semantics)")
    print()

    tasks = generate_decorrelated_corpus(split="routing_stress_v1")
    print(f"  Generated {len(tasks)} decorrelated tasks")

    cats = Counter(t.category for t in tasks)
    print(f"  Category distribution:")
    for cat in sorted(cats.keys()):
        print(f"    {cat:<35} {cats[cat]}")

    # Verify hidden counts are crossed
    hidden_by_cat = {}
    for t in tasks:
        hc = sum(1 for e in t.evidence_items if not e.retrieved)
        hidden_by_cat.setdefault(t.category, set()).add(hc)
    print(f"\n  Hidden count by category (should show crossing):")
    for cat in sorted(hidden_by_cat.keys()):
        print(f"    {cat:<35} hidden_counts={sorted(hidden_by_cat[cat])}")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Save corpus manifest
    benchmark = EvidenceBenchmark(
        benchmark_id="i3_11a_routing_stress_v1",
        tasks=tasks,
        budget_profiles={"STANDARD": budget},
    )
    save_evidence_benchmark(benchmark, "experiments/v2b_i3_11/manifests/routing_stress_v1.json")

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

    results_path = output_dir / "routing_stress_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"Saved: {results_path}")

    # === Analysis ===
    n = len(all_results)
    a1_s = sum(1 for r in all_results if r["a1_success"])
    m3_s = sum(1 for r in all_results if r["m3_success"])
    r0_s = sum(1 for r in all_results if r["r0_success"])

    u_a1 = sum(r["u_a1"] for r in all_results) / n
    u_m3 = sum(r["u_m3"] for r in all_results) / n
    u_r0 = sum(r["u_r0"] for r in all_results) / n

    r0_a1_deltas = [r["r0_delta_vs_a1"] for r in all_results]
    r0_m3_deltas = [r["r0_delta_vs_m3"] for r in all_results]
    m3_a1_deltas = [r["u_m3"] - r["u_a1"] for r in all_results]

    r0_a1_ci = paired_bootstrap_ci(r0_a1_deltas)
    r0_m3_ci = paired_bootstrap_ci(r0_m3_deltas)
    m3_a1_ci = paired_bootstrap_ci(m3_a1_deltas)

    mc_r0_a1 = mcnemar([r["a1_success"] for r in all_results], [r["r0_success"] for r in all_results])
    mc_r0_m3 = mcnemar([r["m3_success"] for r in all_results], [r["r0_success"] for r in all_results])

    def classify(base_ok, treat_ok):
        if base_ok and treat_ok: return "BOTH_SUCCESS"
        elif not base_ok and not treat_ok: return "BOTH_FAIL"
        elif not base_ok and treat_ok: return "RESCUE"
        else: return "BREAK"

    r0_a1_cl = Counter(classify(r["a1_success"], r["r0_success"]) for r in all_results)
    r0_m3_cl = Counter(classify(r["m3_success"], r["r0_success"]) for r in all_results)
    m3_a1_cl = Counter(classify(r["a1_success"], r["m3_success"]) for r in all_results)

    a1_steps = sum(r["a1_steps"] for r in all_results)
    m3_steps = sum(r["m3_steps"] for r in all_results)
    r0_steps = sum(r["r0_steps"] for r in all_results)

    # Subgroup analysis by crossed category
    categories = sorted(set(r["category"] for r in all_results))
    subgroups = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        ca1 = sum(1 for r in cr if r["a1_success"])
        cm3 = sum(1 for r in cr if r["m3_success"])
        cr0 = sum(1 for r in cr if r["r0_success"])
        subgroups[cat] = {
            "n": cn,
            "hidden_count": cr[0]["n_hidden"],
            "r0_decision": cr[0]["r0_decision"],
            "a1_success": f"{ca1}/{cn} ({ca1/cn*100:.1f}%)",
            "m3_success": f"{cm3}/{cn} ({cm3/cn*100:.1f}%)",
            "r0_success": f"{cr0}/{cn} ({cr0/cn*100:.1f}%)",
            "mean_u_a1": round(sum(r["u_a1"] for r in cr) / cn, 4),
            "mean_u_m3": round(sum(r["u_m3"] for r in cr) / cn, 4),
            "mean_u_r0": round(sum(r["u_r0"] for r in cr) / cn, 4),
            "m3_rescues_vs_a1": sum(1 for r in cr if r["m3_rescues_vs_a1"]),
            "m3_breaks_vs_a1": sum(1 for r in cr if r["m3_breaks_vs_a1"]),
            "m3_steps": round(sum(r["m3_steps"] for r in cr) / cn, 2),
            "r0_steps": round(sum(r["r0_steps"] for r in cr) / cn, 2),
            "a1_steps": round(sum(r["a1_steps"] for r in cr) / cn, 2),
        }

    # Key analysis: hidden_count crossing
    print(f"\n{'='*82}")
    print("HIDDEN_COUNT CROSSING ANALYSIS")
    print(f"{'='*82}")
    print(f"  {'Category':<35} {'n':>3} {'hid':>3} {'R0':>4} {'A1%':>6} {'M3%':>6} {'R0%':>6} {'A1_U':>8} {'M3_U':>8} {'R0_U':>8} {'resc':>5} {'brk':>4}")
    for cat in sorted(subgroups.keys()):
        sg = subgroups[cat]
        a1p = sg["a1_success"].split("(")[1].rstrip(")")
        m3p = sg["m3_success"].split("(")[1].rstrip(")")
        r0p = sg["r0_success"].split("(")[1].rstrip(")")
        print(f"  {cat:<35} {sg['n']:>3} {sg['hidden_count']:>3} {sg['r0_decision']:>4} "
              f"{a1p:>6} {m3p:>6} {r0p:>6} {sg['mean_u_a1']:>+8.2f} {sg['mean_u_m3']:>+8.2f} "
              f"{sg['mean_u_r0']:>+8.2f} {sg['m3_rescues_vs_a1']:>5} {sg['m3_breaks_vs_a1']:>4}")

    # Critical test: conflict_unresolved with hidden>0
    print(f"\n{'='*82}")
    print("CRITICAL TEST 1: conflict_unresolved with hidden>0")
    print("  If H1 (hidden_count matters): M3 should NOT rescue here (hidden>0)")
    print("  If H2 (generator proxy): M3 SHOULD rescue here (still conflict_unresolved)")
    cu_hidden = [r for r in all_results if r["category"].startswith("conflict_unresolved_h") and r["n_hidden"] > 0]
    if cu_hidden:
        cn = len(cu_hidden)
        a1_s_cu = sum(1 for r in cu_hidden if r["a1_success"])
        m3_s_cu = sum(1 for r in cu_hidden if r["m3_success"])
        rescues = sum(1 for r in cu_hidden if r["m3_rescues_vs_a1"])
        breaks = sum(1 for r in cu_hidden if r["m3_breaks_vs_a1"])
        print(f"  n={cn}, A1 success={a1_s_cu}/{cn}, M3 success={m3_s_cu}/{cn}")
        print(f"  M3 rescues={rescues}, M3 breaks={breaks}")
        print(f"  R0 routed these to A1 (hidden>0). R0 success={sum(1 for r in cu_hidden if r['r0_success'])}/{cn}")
        if rescues > 0:
            print(f"  → H2 SUPPORTED: M3 rescues conflict_unresolved even with hidden>0")
            print(f"  → R0 LOSES {rescues} rescues by routing to A1")
        else:
            print(f"  → H1 SUPPORTED: M3 does not rescue when hidden>0 (even conflict_unresolved)")

    # Critical test: ANSWER tasks with hidden=0
    print(f"\n{'='*82}")
    print("CRITICAL TEST 2: ANSWER tasks with hidden=0")
    print("  If H1 (hidden_count matters): M3 should help here (hidden=0)")
    print("  If H2 (generator proxy): M3 should NOT help here (not conflict_unresolved)")
    answer_h0 = [r for r in all_results if r["expected_terminal"] == "ANSWER" and r["n_hidden"] == 0]
    if answer_h0:
        cn = len(answer_h0)
        a1_s_ah = sum(1 for r in answer_h0 if r["a1_success"])
        m3_s_ah = sum(1 for r in answer_h0 if r["m3_success"])
        rescues = sum(1 for r in answer_h0 if r["m3_rescues_vs_a1"])
        breaks = sum(1 for r in answer_h0 if r["m3_breaks_vs_a1"])
        mean_delta = sum(r["u_m3"] - r["u_a1"] for r in answer_h0) / cn
        print(f"  n={cn}, A1 success={a1_s_ah}/{cn}, M3 success={m3_s_ah}/{cn}")
        print(f"  M3 rescues={rescues}, M3 breaks={breaks}")
        print(f"  Mean delta U (M3-A1)={mean_delta:+.4f}")
        print(f"  R0 routed these to M3 (hidden=0). R0 success={sum(1 for r in answer_h0 if r['r0_success'])}/{cn}")
        if mean_delta < 0:
            print(f"  → H2 SUPPORTED: M3 is costly on ANSWER tasks even with hidden=0")
            print(f"  → R0 LOSES efficiency by routing to M3")
        elif rescues > 0:
            print(f"  → H1 SUPPORTED: M3 rescues ANSWER tasks with hidden=0")
        else:
            print(f"  → Neutral: M3 neither helps nor hurts on these ANSWER+hidden=0 tasks")

    # Frozen claims
    frozen_claims = {
        "C1_r0_a1_ci_positive": r0_a1_ci[0] > 0,
        "C2_r0_m3_ci_positive": r0_m3_ci[0] > 0,
        "C3_r0_success_ge_max_a1_m3_minus_1pp": r0_s >= max(a1_s, m3_s) - 3,
        "C4_r0_rescues_gt_breaks_vs_a1": r0_a1_cl.get("RESCUE", 0) > r0_a1_cl.get("BREAK", 0),
        "C5_no_catastrophic_subgroup": not any(
            (sum(1 for r in all_results if r["category"] == cat and r["r0_success"]) /
             max(len([r for r in all_results if r["category"] == cat]), 1)) <
            (sum(1 for r in all_results if r["category"] == cat and r["a1_success"]) /
             max(len([r for r in all_results if r["category"] == cat]), 1)) - 0.10
            for cat in categories
        ),
    }

    summary = {
        "schema": "DAPH_V2B_I3_11A_ROUTING_STRESS_V1",
        "n_tasks": n,
        "arms": {
            "A1": "baseline + public affordances",
            "M3": "frozen MDSG-StateWithAffordances",
            "R0": "deterministic router: hidden_evidence_count==0 -> M3, else -> A1",
        },
        "overall": {
            "mean_u": {"A1": round(u_a1, 4), "M3": round(u_m3, 4), "R0": round(u_r0, 4)},
            "success": {"A1": f"{a1_s}/{n}", "M3": f"{m3_s}/{n}", "R0": f"{r0_s}/{n}"},
            "bootstrap_ci_r0_a1": [round(r0_a1_ci[0], 4), round(r0_a1_ci[1], 4)],
            "bootstrap_ci_r0_m3": [round(r0_m3_ci[0], 4), round(r0_m3_ci[1], 4)],
            "bootstrap_ci_m3_a1": [round(m3_a1_ci[0], 4), round(m3_a1_ci[1], 4)],
            "mcnemar_r0_a1": mc_r0_a1,
            "mcnemar_r0_m3": mc_r0_m3,
            "r0_a1_classification": dict(r0_a1_cl),
            "r0_m3_classification": dict(r0_m3_cl),
            "m3_a1_classification": dict(m3_a1_cl),
            "mean_steps": {
                "A1": round(a1_steps / n, 2), "M3": round(m3_steps / n, 2),
                "R0": round(r0_steps / n, 2),
            },
        },
        "subgroups": subgroups,
        "frozen_claims": frozen_claims,
        "r0_router": {
            "rule": "hidden_evidence_count == 0 -> M3, else -> A1",
            "input_fields": ["hidden_evidence_count"],
            "leakage_assertion": "assert_r0_leakage verifies only hidden_evidence_count is used",
        },
    }

    summary_path = output_dir / "routing_stress_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*82}")
    print("I3.11a ROUTING STRESS TEST: A1 vs M3 vs R0")
    print(f"{'='*82}")
    print(f"  Tasks: {n}")
    print(f"\n  Mean utility:  A1={u_a1:+.4f}  M3={u_m3:+.4f}  R0={u_r0:+.4f}")
    print(f"  Success:       A1={a1_s}/{n}  M3={m3_s}/{n}  R0={r0_s}/{n}")
    print(f"\n  Bootstrap 95% CI:")
    print(f"    R0-A1: [{r0_a1_ci[0]:+.4f}, {r0_a1_ci[1]:+.4f}]  <-- PRIMARY (R0 must beat A1)")
    print(f"    R0-M3: [{r0_m3_ci[0]:+.4f}, {r0_m3_ci[1]:+.4f}]  <-- PRIMARY (R0 must beat M3)")
    print(f"    M3-A1: [{m3_a1_ci[0]:+.4f}, {m3_a1_ci[1]:+.4f}]")
    print(f"\n  McNemar:")
    print(f"    R0-A1: b={mc_r0_a1['b']}, c={mc_r0_a1['c']}, p={mc_r0_a1['p']}")
    print(f"    R0-M3: b={mc_r0_m3['b']}, c={mc_r0_m3['c']}, p={mc_r0_m3['p']}")
    print(f"\n  R0 vs A1: rescues={r0_a1_cl.get('RESCUE',0)}, breaks={r0_a1_cl.get('BREAK',0)}")
    print(f"  R0 vs M3: rescues={r0_m3_cl.get('RESCUE',0)}, breaks={r0_m3_cl.get('BREAK',0)}")
    print(f"\n  Steps:  A1={a1_steps/n:.2f}  M3={m3_steps/n:.2f}  R0={r0_steps/n:.2f}")

    print(f"\n  FROZEN CLAIMS:")
    for claim, passed in frozen_claims.items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
