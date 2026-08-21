#!/usr/bin/env python3
"""I3.15c smoke test: 12-trajectory live test on the T2-eligible benchmark.

6 tasks (2 T2_CONFLICT, 2 DEFER_CONTROL, 2 ANSWER_CONTROL) x 2 arms (A1, R1)
under Q3_RERANKED retrieval = 12 trajectories.

Verifies:
  S1: Zero decoder failures
  S2: Zero fail-closed calls
  S3: T2_CONFLICT tasks reach T2 (at least 1 R1 T2 trigger)
  S4: DEFER_CONTROL tasks do NOT fire T2
  S5: ANSWER_CONTROL tasks do NOT fire T2
  S6: A1 and R1 diverge on at least 1 T2_CONFLICT task
  S7: R1 post-T2 packet differs from A1 packet
  S8: T2_CONFLICT expected terminal = DEFER
  S9: ANSWER_CONTROL expected terminal = ANSWER
  S10: Packet hash diversity across trajectories
  S11: Provenance fields present (schema hash, prompt hash, packet hash)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load i3_12j module
spec_12j = importlib.util.spec_from_file_location(
    "i3_12j", str(REPO_ROOT / "scripts" / "run_i3_12j_factorial.py"))
i3_12j = importlib.util.module_from_spec(spec_12j)
spec_12j.loader.exec_module(i3_12j)
i3_7e = i3_12j.i3_7e

from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
    generate_i3_15c_corpus, validate_t2_eligibility, get_i3_15c_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend
from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import build_retriever

from scripts.run_i3_15_r1_balanced import (
    build_corpus_index, get_required_passage_ids, build_retrieved_evidence_task,
    TOP_K, adapt_local_system_prompt, SHARED_ACTION_SEMANTICS_V1,
)
from hrm_adaptive_memory.memory.chunking import Chunk


def build_corpus_index_15c():
    """Build corpus index from the combined I3.15c corpus."""
    corpus_passages = get_i3_15c_corpus()
    chunks = [
        Chunk(
            chunk_id=p.passage_id,
            source_id=p.source,
            source_type="document",
            title=p.domain,
            section="",
            content=p.text,
            token_count=len(p.text.split()),
            metadata={"domain": p.domain, "source": p.source},
        )
        for p in corpus_passages
    ]
    corpus_by_text = {p.text: p.passage_id for p in corpus_passages}
    corpus_by_id = {p.passage_id: p for p in corpus_passages}
    return chunks, corpus_by_text, corpus_by_id


def select_smoke_tasks(tasks):
    """Select 6 smoke tasks: 2 T2_CONFLICT, 2 DEFER_CONTROL, 2 ANSWER_CONTROL."""
    t2_conflict = [t for t in tasks if t.evidence_task.category.startswith("t2_conflict")]
    defer_control = [t for t in tasks if t.evidence_task.category.startswith("defer_control")]
    answer_control = [t for t in tasks if t.evidence_task.category.startswith("answer_control")]

    # Pick 2 from each stratum, preferring different domains
    selected = []
    for stratum_tasks in [t2_conflict, defer_control, answer_control]:
        # Pick first 2 with different domains
        seen_domains = set()
        for t in stratum_tasks:
            domain = t.evidence_task.task_summary
            if domain not in seen_domains:
                selected.append(t)
                seen_domains.add(domain)
                if len([s for s in selected if s.evidence_task.category == t.evidence_task.category]) >= 2:
                    break

    return selected


def run_smoke():
    print("=" * 80)
    print("I3.15c Smoke Test: 12-trajectory live test on T2-eligible benchmark")
    print("=" * 80)

    # Generate tasks
    all_tasks = generate_i3_15c_corpus(n_per_cell=25, seed=42)
    print(f"\nGenerated {len(all_tasks)} tasks")

    # Structural validation
    validation = validate_t2_eligibility(all_tasks)
    print(f"Structural T2 validation: {'PASSED' if validation['passed'] else 'FAILED'}")
    print(f"  T2 positive reachable: {validation['t2_positive_reachable']}/{validation['t2_positive_expected']}")
    print(f"  T2 negative incorrectly reachable: {validation['t2_negative_incorrectly_reachable']}")
    if not validation["passed"]:
        print("STRUCTURAL VALIDATION FAILED — aborting smoke test.")
        return

    # Select 6 smoke tasks
    smoke_tasks = select_smoke_tasks(all_tasks)
    print(f"\nSelected {len(smoke_tasks)} smoke tasks:")
    for t in smoke_tasks:
        et = t.evidence_task
        print(f"  {et.task_id}  category={et.category}  "
              f"expected={et.expected_terminal.value}  "
              f"evidence={len(et.evidence_items)} items")

    # Build corpus and retriever
    chunks, corpus_by_text, corpus_by_id = build_corpus_index_15c()
    retriever = build_retriever("Q3_RERANKED", chunks)
    print(f"\nCorpus: {len(chunks)} passages")
    print(f"Retriever: Q3_RERANKED")

    # Setup
    extractor = DeterministicRelationExtractor()
    snapshot_builder = i3_12j.make_inferred_snapshot_builder(extractor)
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    def backend_factory():
        return LocalLlamaBackend()

    max_tokens = 2048

    # Run trajectories
    results = []
    for task in smoke_tasks:
        et = task.evidence_task
        required_ids = get_required_passage_ids(task, corpus_by_text)

        # Retrieve
        query = et.task_summary
        retrieved = retriever.search(query, top_k=TOP_K)
        retrieved_passages = [
            corpus_by_id[c.chunk_id] for c, _ in retrieved
            if c.chunk_id in corpus_by_id
        ]
        new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text)

        recall = len({p.passage_id for p in retrieved_passages} & required_ids) / max(len(required_ids), 1)

        for arm in ["A1_INFERRED", "R1_INFERRED"]:
            print(f"\n  Running {et.task_id} {arm}...")
            t0 = time.time()

            if arm == "A1_INFERRED":
                result = i3_12j.run_trajectory_i3_12(
                    new_et, budget, utility,
                    mode="BASELINE_WITH_AFFORDANCES",
                    api_key="", fork_label=f"smoke:{et.task_id}:{arm}",
                    snapshot_builder=snapshot_builder,
                    backend_factory=backend_factory,
                    strict_decode=True,
                    max_tokens=max_tokens,
                    system_prompt_transform=adapt_local_system_prompt,
                )
                result["arm"] = "A1_INFERRED"
            else:
                result = i3_12j.run_r1_trajectory_i3_12(
                    new_et, budget, utility,
                    api_key="", fork_label=f"smoke:{et.task_id}:{arm}",
                    snapshot_builder=snapshot_builder,
                    backend_factory=backend_factory,
                    strict_decode=True,
                    max_tokens=max_tokens,
                    system_prompt_transform=adapt_local_system_prompt,
                )
                result["arm"] = "R1_INFERRED"

            result["task_id"] = et.task_id
            result["category"] = et.category
            result["expected_terminal"] = et.expected_terminal.value
            result["retrieval_recall"] = round(recall, 3)
            result["wall_time_s"] = round(time.time() - t0, 1)

            actions = result.get("continuation_actions", [])
            t2_triggered = result.get("r1_triggered", False)
            print(f"    actions={actions}")
            print(f"    terminal={result.get('terminal_action')} "
                  f"success={result.get('success')} "
                  f"steps={result.get('steps')} "
                  f"t2={t2_triggered} "
                  f"time={result['wall_time_s']}s")

            results.append(result)

    # Save results
    output_dir = REPO_ROOT / "experiments" / "v2b_i3_15" / "development" / "i3_15c_smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke_results_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Evaluate smoke criteria
    print("\n" + "=" * 80)
    print("SMOKE CRITERIA EVALUATION")
    print("=" * 80)

    criteria = {}

    # S1: Zero decoder failures
    decoder_failures = sum(
        1 for r in results
        for call in r.get("model_call_log", [])
        if not call.get("decoder_valid", True)
    )
    criteria["S1_decoder_failures_zero"] = {
        "passed": decoder_failures == 0,
        "value": decoder_failures,
        "threshold": 0,
    }

    # S2: Zero fail-closed calls
    fail_closed = sum(
        1 for r in results
        for call in r.get("model_call_log", [])
        if call.get("fail_closed", False)
    )
    criteria["S2_fail_closed_zero"] = {
        "passed": fail_closed == 0,
        "value": fail_closed,
        "threshold": 0,
    }

    # S3: T2_CONFLICT tasks reach T2 (at least 1 R1 T2 trigger)
    t2_conflict_r1 = [r for r in results
                      if r.get("category", "").startswith("t2_conflict")
                      and r.get("arm") == "R1_INFERRED"]
    t2_triggers = sum(1 for r in t2_conflict_r1 if r.get("r1_triggered", False))
    criteria["S3_t2_conflict_t2_fires"] = {
        "passed": t2_triggers >= 1,
        "value": f"{t2_triggers}/{len(t2_conflict_r1)}",
        "threshold": ">=1",
    }

    # S4: DEFER_CONTROL tasks do NOT fire T2
    defer_control_r1 = [r for r in results
                        if r.get("category", "").startswith("defer_control")
                        and r.get("arm") == "R1_INFERRED"]
    defer_t2_fires = sum(1 for r in defer_control_r1 if r.get("r1_triggered", False))
    criteria["S4_defer_control_no_t2"] = {
        "passed": defer_t2_fires == 0,
        "value": f"{defer_t2_fires}/{len(defer_control_r1)}",
        "threshold": 0,
    }

    # S5: ANSWER_CONTROL tasks do NOT fire T2
    answer_control_r1 = [r for r in results
                         if r.get("category", "").startswith("answer_control")
                         and r.get("arm") == "R1_INFERRED"]
    answer_t2_fires = sum(1 for r in answer_control_r1 if r.get("r1_triggered", False))
    criteria["S5_answer_control_no_t2"] = {
        "passed": answer_t2_fires == 0,
        "value": f"{answer_t2_fires}/{len(answer_control_r1)}",
        "threshold": 0,
    }

    # S6: A1 and R1 diverge on at least 1 T2_CONFLICT task
    t2_conflict_tasks = set(
        r["task_id"] for r in results
        if r.get("category", "").startswith("t2_conflict")
    )
    diverged = 0
    for tid in t2_conflict_tasks:
        a1 = next((r for r in results if r["task_id"] == tid and r["arm"] == "A1_INFERRED"), None)
        r1 = next((r for r in results if r["task_id"] == tid and r["arm"] == "R1_INFERRED"), None)
        if a1 and r1:
            a1_actions = a1.get("continuation_actions", [])
            r1_actions = r1.get("continuation_actions", [])
            if a1_actions != r1_actions:
                diverged += 1
                print(f"  Divergence on {tid}: A1={a1_actions} vs R1={r1_actions}")
    criteria["S6_a1_r1_divergence"] = {
        "passed": diverged >= 1,
        "value": f"{diverged}/{len(t2_conflict_tasks)}",
        "threshold": ">=1",
    }

    # S7: R1 post-T2 packet differs from A1 packet
    # Check if any R1 trajectory has a packet with M3 representation
    # that differs from the A1 packet at the same step
    post_t2_diff = 0
    for tid in t2_conflict_tasks:
        a1 = next((r for r in results if r["task_id"] == tid and r["arm"] == "A1_INFERRED"), None)
        r1 = next((r for r in results if r["task_id"] == tid and r["arm"] == "R1_INFERRED"), None)
        if not (a1 and r1):
            continue
        a1_calls = a1.get("model_call_log", [])
        r1_calls = r1.get("model_call_log", [])
        # Find the step where R1 switches to M3
        for r1_call in r1_calls:
            if r1_call.get("representation") == "M3":
                r1_packet = r1_call.get("packet_sha256")
                step = r1_call.get("step")
                # Compare with A1 packet at the same step
                a1_call = next((c for c in a1_calls if c.get("step") == step), None)
                if a1_call and a1_call.get("packet_sha256") != r1_packet:
                    post_t2_diff += 1
                    print(f"  Post-T2 packet diff on {tid} step={step}: "
                          f"A1={a1_call.get('packet_sha256','')[:12]}... "
                          f"R1/M3={r1_packet[:12]}...")
                break
    criteria["S7_r1_post_t2_packet_differs"] = {
        "passed": post_t2_diff >= 1,
        "value": post_t2_diff,
        "threshold": ">=1",
    }

    # S8: T2_CONFLICT expected terminal = DEFER
    t2_conflict_terminals = set(
        r["expected_terminal"] for r in results
        if r.get("category", "").startswith("t2_conflict")
    )
    criteria["S8_t2_conflict_expected_defer"] = {
        "passed": t2_conflict_terminals == {"DEFER"},
        "value": t2_conflict_terminals,
        "threshold": {"DEFER"},
    }

    # S9: ANSWER_CONTROL expected terminal = ANSWER
    answer_terminals = set(
        r["expected_terminal"] for r in results
        if r.get("category", "").startswith("answer_control")
    )
    criteria["S9_answer_control_expected_answer"] = {
        "passed": answer_terminals == {"ANSWER"},
        "value": answer_terminals,
        "threshold": {"ANSWER"},
    }

    # S10: Packet hash diversity
    all_packets = set()
    for r in results:
        for call in r.get("model_call_log", []):
            ph = call.get("packet_sha256")
            if ph:
                all_packets.add(ph)
    criteria["S10_packet_hash_diversity"] = {
        "passed": len(all_packets) >= 6,
        "value": len(all_packets),
        "threshold": ">=6",
    }

    # S11: Provenance fields present
    required_provenance = [
        "system_prompt_sha256", "packet_sha256",
        "decoder_valid", "decoded_action",
    ]
    missing_provenance = 0
    total_calls = 0
    for r in results:
        for call in r.get("model_call_log", []):
            total_calls += 1
            for field in required_provenance:
                if field not in call:
                    missing_provenance += 1
    criteria["S11_provenance_fields_present"] = {
        "passed": missing_provenance == 0,
        "value": f"{total_calls * len(required_provenance) - missing_provenance}/{total_calls * len(required_provenance)}",
        "threshold": "all present",
    }

    # Print results
    print()
    all_passed = True
    for name, info in criteria.items():
        status = "PASS" if info["passed"] else "FAIL"
        if not info["passed"]:
            all_passed = False
        print(f"  {name}: {status}  value={info['value']}  threshold={info['threshold']}")

    print(f"\n{'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")

    # Print T2 trigger details
    print("\nT2 trigger details:")
    for r in results:
        if r.get("arm") == "R1_INFERRED":
            routing = r.get("routing_log", [])
            t2_steps = [e for e in routing if e.get("t2_fires")]
            print(f"  {r['task_id']} R1: t2_triggered={r.get('r1_triggered')} "
                  f"t2_fires_steps={[e['step'] for e in t2_steps]} "
                  f"trigger_step={r.get('r1_trigger_step')}")

    # Save criteria
    criteria_path = output_dir / "smoke_criteria_v1.json"
    with open(criteria_path, "w") as f:
        json.dump(criteria, f, indent=2, default=str)
    print(f"\nCriteria saved to {criteria_path}")


if __name__ == "__main__":
    run_smoke()
