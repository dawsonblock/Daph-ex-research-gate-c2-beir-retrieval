#!/usr/bin/env python3
"""I3.15: Retrieval-Fair Epistemically-Hard Benchmark.

3x2 factorial: Q0_BM25 / Q3_RERANKED / Q4_ORACLE x A1_INFERRED / R1_INFERRED
= 150 tasks x 3 retrieval x 2 arms = 900 trajectories

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python3 -u scripts/run_i3_15_epistemic_benchmark.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
_env_file = ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Load I3.12j runner for the trajectory infrastructure (same as I3.13)
_spec_12j = importlib.util.spec_from_file_location(
    "i3_12j", str(ROOT / "scripts" / "run_i3_12j_factorial.py"))
i3_12j = importlib.util.module_from_spec(_spec_12j)
_spec_12j.loader.exec_module(i3_12j)

# Load I3.13 module for shared utilities
_spec_13 = importlib.util.spec_from_file_location(
    "run_i3_13_retrieved_evidence", str(ROOT / "scripts" / "run_i3_13_retrieved_evidence.py"))
i3_13 = importlib.util.module_from_spec(_spec_13)
_spec_13.loader.exec_module(i3_13)

from hrm_adaptive_memory.executive.semantic_relations.i3_15_epistemic_corpus import (
    get_corpus as get_i3_15_corpus, corpus_sha256 as i3_15_corpus_sha256,
    I3_15Passage,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_15_task_generator import (
    generate_i3_15_corpus,
)
from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import build_retriever
from hrm_adaptive_memory.memory.chunking import Chunk
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark import EvidenceItem, EvidenceTask
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.resources import ResourceBudget

RETRIEVAL_LEVELS = ["Q0_BM25", "Q3_RERANKED", "Q4_ORACLE"]
ARMS = ["A1_INFERRED", "R1_INFERRED"]
TOP_K = 10


def build_corpus_index(corpus_passages):
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


def get_required_passage_ids(task, corpus_by_text):
    required = set()
    for ev in task.evidence_task.evidence_items:
        if not ev.retrieved:
            continue
        pid = corpus_by_text.get(ev.proposition)
        if pid:
            required.add(pid)
    return required


def build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text):
    """Build an EvidenceTask with evidence from retrieved passages."""
    et = task.evidence_task
    retrieved_texts = {p.text for p in retrieved_passages}

    evidence_items = []
    for ev in et.evidence_items:
        if ev.proposition in retrieved_texts:
            evidence_items.append(ev)

    existing_texts = {ev.proposition for ev in evidence_items}
    for p in retrieved_passages:
        if p.text not in existing_texts:
            evidence_items.append(EvidenceItem(
                evidence_id=f"D{p.passage_id}",
                proposition=p.text,
                source_class="search",
                supports=(),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="MISSING",
            ))

    new_et = EvidenceTask(
        task_id=et.task_id, split=et.split,
        category=et.category,
        task_summary=et.task_summary,
        high_stakes=et.high_stakes, budget_profile=et.budget_profile,
        hypotheses=et.hypotheses,
        evidence_items=tuple(evidence_items),
        retrieve_exposes=et.retrieve_exposes,
        search_exposes=et.search_exposes,
        oracle_resolution_path=et.oracle_resolution_path,
        expected_terminal=et.expected_terminal,
        correct_hypothesis_id=et.correct_hypothesis_id,
    )
    return new_et


def compute_retrieval_recall(retrieved_passages, required_ids):
    if not required_ids:
        return 1.0
    retrieved_ids = {p.passage_id for p in retrieved_passages}
    found = len(retrieved_ids & required_ids)
    return found / len(required_ids)


def run_single(work_item):
    """Run a single trajectory. Must be picklable for ProcessPoolExecutor."""
    (task_id, retrieval_level, arm, api_key, budget_dict) = work_item

    # Reconstruct task
    from hrm_adaptive_memory.executive.semantic_relations.i3_15_task_generator import (
        generate_i3_15_corpus,
    )
    from hrm_adaptive_memory.executive.semantic_relations.i3_15_epistemic_corpus import (
        get_corpus,
    )
    all_tasks = generate_i3_15_corpus(n_per_cell=25, seed=42)
    task = next(t for t in all_tasks if t.evidence_task.task_id == task_id)

    corpus = get_corpus()
    chunks, corpus_by_text, corpus_by_id = build_corpus_index(corpus)

    required_ids = get_required_passage_ids(task, corpus_by_text)

    # Get retrieved passages
    if retrieval_level == "Q4_ORACLE":
        retrieved_passages = [p for p in corpus if p.passage_id in required_ids]
    else:
        retriever = build_retriever(retrieval_level, chunks)
        query = task.evidence_task.task_summary
        results = retriever.search(query, top_k=TOP_K)
        # Convert Chunk objects back to I3_15Passage using corpus_by_id
        retrieved_passages = [corpus_by_id[chunk.chunk_id] for chunk, _ in results
                              if chunk.chunk_id in corpus_by_id]

    recall = compute_retrieval_recall(retrieved_passages, required_ids)
    recall_any = recall > 0 if required_ids else True
    recall_all = recall == 1.0 if required_ids else True

    # Build retrieved evidence task
    new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text)

    if not new_et.evidence_items:
        return {
            "task_id": task_id,
            "category": task.evidence_task.category,
            "retrieval_level": retrieval_level,
            "arm": arm,
            "retrieval_recall": 0.0,
            "recall_any": False,
            "recall_all": False,
            "n_required": len(required_ids),
            "n_retrieved": 0,
            "success": False,
            "utility": -150.0,
            "steps": 0,
            "terminal_action": "NO_EVIDENCE",
            "t2_triggered": False,
            "relation_correct": False,
            "relation_error": None,
            "relation_accuracy": 0.0,
            "failure_attribution": "RETRIEVAL_ERROR",
        }

    # Compute relation accuracy
    extractor = DeterministicRelationExtractor()
    rel_info = i3_13.compute_relation_accuracy(new_et, task, extractor)

    # Run trajectory
    budget = ResourceBudget(**budget_dict)
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")

    use_gold = "GOLD" in arm
    arch = arm.split("_")[0]

    if use_gold:
        sb = i3_12j.make_gold_snapshot_builder()
    else:
        sb = i3_12j.make_inferred_snapshot_builder(extractor)

    try:
        if arch == "R1":
            result = i3_12j.run_r1_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
            )
        else:
            mode = "BASELINE_WITH_AFFORDANCES" if arch == "A1" else "MDSG_STATE_WITH_AFFORDANCES"
            result = i3_12j.run_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                mode=mode, api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
            )
    except Exception as e:
        return {
            "task_id": task_id,
            "category": task.evidence_task.category,
            "retrieval_level": retrieval_level,
            "arm": arm,
            "retrieval_recall": round(recall, 4),
            "recall_any": recall_any,
            "recall_all": recall_all,
            "n_required": len(required_ids),
            "n_retrieved": len(required_ids & {p.passage_id for p in retrieved_passages}),
            "success": False,
            "utility": -150.0,
            "steps": 0,
            "terminal_action": "ERROR",
            "t2_triggered": False,
            "relation_correct": rel_info["correct"],
            "relation_error": rel_info["error_type"],
            "relation_accuracy": round(rel_info["accuracy"], 4),
            "failure_attribution": "POLICY_ERROR",
            "error": str(e)[:200],
        }

    success = result.get("success", False)
    utility_val = result.get("realized_utility", 0.0)
    steps = result.get("steps", 0)
    terminal = result.get("terminal_action", "UNKNOWN")
    t2_triggered = result.get("r1_triggered", False)

    # Failure attribution
    if success:
        failure_attribution = "NO_ERROR"
    elif not recall_all:
        failure_attribution = "RETRIEVAL_ERROR"
    elif not rel_info["correct"]:
        failure_attribution = "SEMANTIC_EXTRACTION"
    else:
        failure_attribution = "POLICY_ERROR"

    return {
        "task_id": task_id,
        "category": task.evidence_task.category,
        "retrieval_level": retrieval_level,
        "arm": arm,
        "retrieval_recall": round(recall, 4),
        "recall_any": recall_any,
        "recall_all": recall_all,
        "n_required": len(required_ids),
        "n_retrieved": len(required_ids & {p.passage_id for p in retrieved_passages}),
        "success": success,
        "utility": utility_val,
        "steps": steps,
        "terminal_action": terminal,
        "t2_triggered": t2_triggered,
        "t2_trigger_step": result.get("r1_trigger_step"),
        "relation_correct": rel_info["correct"],
        "relation_error": rel_info["error_type"],
        "relation_accuracy": round(rel_info["accuracy"], 4),
        "failure_attribution": failure_attribution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    corpus = get_i3_15_corpus()
    tasks = generate_i3_15_corpus(n_per_cell=25, seed=42)

    print(f"I3.15: Retrieval-Fair Epistemically-Hard Benchmark")
    print(f"  Corpus: {len(corpus)} passages, SHA256: {i3_15_corpus_sha256()[:16]}...")
    print(f"  Tasks: {len(tasks)} (seed=42, 25 per cell, 6 cells)")
    print(f"  Retrieval levels: {RETRIEVAL_LEVELS}")
    print(f"  Arms: {ARMS}")
    print(f"  Top-k: {TOP_K}")
    n_traj = len(tasks) * len(RETRIEVAL_LEVELS) * len(ARMS)
    print(f"  Trajectories: {len(tasks)} x {len(RETRIEVAL_LEVELS)} x {len(ARMS)} = {n_traj}")
    print(f"  Workers: {args.workers}")
    print()

    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")
    budget_dict = {"max_executive_steps": 10, "max_retrieval_calls": 3,
                   "max_search_calls": 2, "max_verification_calls": 5}

    # Build work items
    work_items = []
    for task in tasks:
        task_id = task.evidence_task.task_id
        for retrieval_level in RETRIEVAL_LEVELS:
            for arm in ARMS:
                work_items.append((task_id, retrieval_level, arm, api_key, budget_dict))

    print(f"  Total work items: {len(work_items)}")
    print()

    out_dir = ROOT / "experiments/v2b_i3_15/development/i3_15_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v1.jsonl"

    t0 = time.time()
    n_completed = 0

    with open(results_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single, wi): wi for wi in work_items}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush()
                    n_completed += 1
                    if n_completed % 50 == 0 or n_completed == len(work_items):
                        elapsed = time.time() - t0
                        rate = n_completed / elapsed
                        eta = (len(work_items) - n_completed) / rate if rate > 0 else 0
                        print(f"  Completed {n_completed}/{len(work_items)} "
                              f"({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
                except Exception as e:
                    n_completed += 1
                    print(f"  ERROR: {e}", flush=True)
                    f_out.write(json.dumps({"error": str(e)}) + "\n")
                    f_out.flush()

    elapsed = time.time() - t0
    print(f"\nCompleted {n_completed} trajectories in {elapsed:.1f}s")
    print(f"  Results: {results_path}")

    # Analysis
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    results.append(r)

    print(f"\n{'='*80}")
    print(f"I3.15 Results Summary ({len(tasks)} tasks, {len(results)} trajectories)")
    print(f"{'='*80}")

    print(f"\n{'Retrieval':<14} {'Arm':<14} {'N':>4} {'Recall':>8} {'RecAll':>8} {'Success':>8} {'MeanU':>8} {'RelAcc':>8}")
    print("-" * 80)
    for retrieval in RETRIEVAL_LEVELS:
        for arm in ARMS:
            rs = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == arm]
            if not rs:
                continue
            recall = sum(r["retrieval_recall"] for r in rs) / len(rs)
            recall_all = sum(r["recall_all"] for r in rs) / len(rs)
            success = sum(r["success"] for r in rs) / len(rs)
            mean_u = sum(r["utility"] for r in rs) / len(rs)
            rel_acc = sum(r["relation_accuracy"] for r in rs) / len(rs)
            print(f"  {retrieval:<14} {arm:<14} {len(rs):>4} {recall:>8.4f} {recall_all:>8.4f} {success:>8.4f} {mean_u:>8.2f} {rel_acc:>8.4f}")

    # Delta_R(Q)
    print(f"\n{'='*80}")
    print("DELTA_R(Q) = U(R1,Q) - U(A1,Q)")
    print(f"{'='*80}")
    deltas = {}
    for retrieval in RETRIEVAL_LEVELS:
        r1 = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"]
        a1 = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == "A1_INFERRED"]
        if not r1 or not a1:
            continue
        u_r1 = sum(r["utility"] for r in r1) / len(r1)
        u_a1 = sum(r["utility"] for r in a1) / len(a1)
        delta = u_r1 - u_a1
        deltas[retrieval] = delta
        print(f"  {retrieval}: Delta_R = {delta:+.2f}  (R1={u_r1:.2f}, A1={u_a1:.2f})")

    if "Q0_BM25" in deltas and "Q3_RERANKED" in deltas:
        interaction = deltas["Q3_RERANKED"] - deltas["Q0_BM25"]
        print(f"\n  Interaction E_retrieval x E_MDSG = Delta_R(Q3) - Delta_R(Q0) = {interaction:+.2f}")

    # Per-cell breakdown
    print(f"\n{'='*80}")
    print("PER-CELL (2x3 matrix, R1 vs A1)")
    print(f"{'='*80}")
    print(f"{'Cell':<32} {'Retr':>5} {'N':>4} {'R1_S':>8} {'A1_S':>8} {'R1_U':>8} {'A1_U':>8} {'Delta':>8}")
    print("-" * 90)

    cells = sorted(set(r["category"] for r in results))
    for cell in cells:
        for retrieval in RETRIEVAL_LEVELS:
            r1 = [r for r in results if r["category"] == cell and r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"]
            a1 = [r for r in results if r["category"] == cell and r["retrieval_level"] == retrieval and r["arm"] == "A1_INFERRED"]
            if not r1 or not a1:
                continue
            r1_s = sum(r["success"] for r in r1) / len(r1)
            a1_s = sum(r["success"] for r in a1) / len(a1)
            r1_u = sum(r["utility"] for r in r1) / len(r1)
            a1_u = sum(r["utility"] for r in a1) / len(a1)
            delta = r1_u - a1_u
            retr_short = retrieval[:5]
            print(f"  {cell[:30]:<31} {retr_short:>5} {len(r1):>4} {r1_s:>8.4f} {a1_s:>8.4f} {r1_u:>8.2f} {a1_u:>8.2f} {delta:>+8.2f}")

    # Conditional survival
    print(f"\n{'='*80}")
    print("CONDITIONAL SURVIVAL (R1_INFERRED)")
    print(f"{'='*80}")
    for retrieval in RETRIEVAL_LEVELS:
        rs = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"]
        n = len(rs)
        if n == 0:
            continue
        p_succ = sum(r["success"] for r in rs) / n
        all_r = [r for r in rs if r["recall_all"]]
        rel_r = [r for r in rs if r["relation_correct"]]
        all_and_rel = [r for r in all_r if r["relation_correct"]]
        print(f"  {retrieval}:")
        print(f"    P(success) = {p_succ:.4f} (n={n})")
        if all_r:
            print(f"    P(success | RecallAll=1) = {sum(r['success'] for r in all_r)/len(all_r):.4f} (n={len(all_r)})")
        if all_and_rel:
            print(f"    P(success | RecallAll=1, RelCorrect=1) = {sum(r['success'] for r in all_and_rel)/len(all_and_rel):.4f} (n={len(all_and_rel)})")

    # Failure attribution
    print(f"\n{'='*80}")
    print("FAILURE ATTRIBUTION")
    print(f"{'='*80}")
    fa = defaultdict(int)
    for r in results:
        fa[r["failure_attribution"]] += 1
    for k, v in sorted(fa.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Save analysis
    analysis = {
        "experiment": "I3.15",
        "n_tasks": len(tasks),
        "n_trajectories": len(results),
        "corpus_sha256": i3_15_corpus_sha256(),
        "top_k": TOP_K,
        "deltas": deltas,
        "interaction": deltas.get("Q3_RERANKED", 0) - deltas.get("Q0_BM25", 0) if "Q3_RERANKED" in deltas and "Q0_BM25" in deltas else None,
    }
    analysis_path = out_dir / "analysis_v1.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
