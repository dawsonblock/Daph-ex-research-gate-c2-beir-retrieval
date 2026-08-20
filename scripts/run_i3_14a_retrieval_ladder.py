#!/usr/bin/env python3
"""I3.14a: Retrieval Sufficiency Repair.

Tests a retrieval ladder (Q0-Q4) while keeping all downstream components
frozen from I3.13. The scientific question is whether increasing
RequiredEvidenceRecall produces proportional TaskUtility increase.

Retrieval ladder:
  Q0_BM25:     Frozen BM25 baseline (current I3.13 R1_REAL)
  Q1_DENSE:    BGE-small-en-v1.5 dense retrieval
  Q2_HYBRID:   BM25 + dense with reciprocal rank fusion
  Q3_RERANKED: Hybrid + cross-encoder reranking
  Q4_ORACLE:   Oracle retrieval ceiling

Frozen downstream:
  semantic extractor v2.6.0, MDSG, T2, R1, A1, prompts, executor, utility, model

Usage:
    PYTHONPATH=. python3 -u scripts/run_i3_14a_retrieval_ladder.py --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util

# Load I3.12j runner for the trajectory infrastructure
_spec = importlib.util.spec_from_file_location(
    "i3_12j", ROOT / "scripts" / "run_i3_12j_factorial.py")
i3_12j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_12j)

from hrm_adaptive_memory.executive.semantic_relations.i3_13_task_generator import (
    generate_i3_13_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_13_document_corpus import (
    get_corpus, corpus_sha256,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import (
    build_retriever, RETRIEVAL_CONDITIONS, retriever_digest,
)
from hrm_adaptive_memory.memory.chunking import Chunk
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceRuntime,
    initial_evidence_runtime,
)
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility


# ---------------------------------------------------------------------------
# Corpus indexing
# ---------------------------------------------------------------------------

def build_corpus_index(corpus_passages):
    """Build index structures from the corpus."""
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
    """Get the passage IDs for evidence that should be retrieved.

    Only counts evidence with retrieved=True. Evidence with retrieved=False
    (e.g., hidden noise in conflict_with_noise) is not required for retrieval
    — it's meant to be found through SEARCH actions, not initial retrieval.
    """
    required = set()
    for ev in task.evidence_task.evidence_items:
        if not ev.retrieved:
            continue
        pid = corpus_by_text.get(ev.proposition)
        if pid:
            required.add(pid)
    return required


def retrieve_oracle(task, corpus_by_text, corpus_by_id):
    """Q4_ORACLE: return exactly the passages used in the task."""
    passages = []
    for ev in task.evidence_task.evidence_items:
        pid = corpus_by_text.get(ev.proposition)
        if pid and pid in corpus_by_id:
            passages.append(corpus_by_id[pid])
    return passages


def retrieve_with_ladder(task, retriever, corpus_by_id, k=5):
    """Retrieve passages using a ladder retriever."""
    query = task.evidence_task.task_summary
    results = retriever.search(query, top_k=k)
    passages = []
    for chunk, score in results:
        p = corpus_by_id.get(chunk.chunk_id)
        if p:
            passages.append(p)
    return passages


def compute_retrieval_recall(retrieved_passages, required_ids):
    if not required_ids:
        return 1.0
    retrieved_ids = {p.passage_id for p in retrieved_passages}
    found = len(retrieved_ids & required_ids)
    return found / len(required_ids)


# ---------------------------------------------------------------------------
# Build evidence task from retrieved passages (from I3.13)
# ---------------------------------------------------------------------------

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

    if not evidence_items:
        return None

    return EvidenceTask(
        task_id=et.task_id,
        split=et.split,
        category=et.category,
        task_summary=et.task_summary,
        high_stakes=et.high_stakes,
        budget_profile=et.budget_profile,
        hypotheses=et.hypotheses,
        evidence_items=tuple(evidence_items),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=et.oracle_resolution_path,
        expected_terminal=et.expected_terminal,
        correct_hypothesis_id=et.correct_hypothesis_id,
    )


# ---------------------------------------------------------------------------
# Relation accuracy (from I3.13)
# ---------------------------------------------------------------------------

def compute_relation_accuracy(new_et, task, extractor):
    """Compute per-edge relation accuracy for the retrieved evidence task."""
    correct = True
    error_type = None
    n_edges = 0
    n_correct = 0

    for ev in [e for e in new_et.evidence_items if e.retrieved]:
        for hyp in new_et.hypotheses:
            result = extractor.extract(
                evidence_id=ev.evidence_id,
                evidence_proposition=ev.proposition,
                hypothesis_id=hyp.hypothesis_id,
                hypothesis_proposition=hyp.proposition,
            )
            inferred = result.relation.relation.value

            # Find gold relation
            gold_rel = "NEUTRAL"
            if not ev.evidence_id.startswith("D"):
                for gr in task.gold_relations:
                    if gr.evidence_id == ev.evidence_id and gr.hypothesis_id == hyp.hypothesis_id:
                        gold_rel = gr.relation
                        break

            n_edges += 1
            if inferred == gold_rel:
                n_correct += 1
            else:
                correct = False
                if error_type is None:
                    if gold_rel == "SUPPORT" and inferred == "CONTRADICT":
                        error_type = "FALSE_CONTRADICTION"
                    elif gold_rel == "SUPPORT" and inferred == "NEUTRAL":
                        error_type = "MISSED_SUPPORT"
                    elif gold_rel == "CONTRADICT" and inferred == "SUPPORT":
                        error_type = "FALSE_SUPPORT"
                    elif gold_rel == "CONTRADICT" and inferred == "NEUTRAL":
                        error_type = "MISSED_CONTRADICTION"
                    elif gold_rel == "NEUTRAL" and inferred == "SUPPORT":
                        error_type = "FALSE_SUPPORT"
                    elif gold_rel == "NEUTRAL" and inferred == "CONTRADICT":
                        error_type = "FALSE_CONTRADICTION"

    return {
        "correct": correct,
        "error_type": error_type,
        "n_edges": n_edges,
        "n_correct": n_correct,
        "accuracy": n_correct / n_edges if n_edges > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# Single trajectory runner
# ---------------------------------------------------------------------------

def run_single(work_item):
    """Run a single trajectory. Must be picklable for ProcessPoolExecutor."""
    (task_dict, condition, arm, corpus_data, api_key, budget_dict) = work_item

    # Reconstruct task from dict — regenerate the full corpus
    from hrm_adaptive_memory.executive.semantic_relations.i3_13_task_generator import (
        generate_i3_13_corpus,
    )
    all_tasks = generate_i3_13_corpus(n_per_category=25, seed=42)
    task = next(t for t in all_tasks if t.task_id == task_dict["task_id"])

    # Reconstruct corpus
    from hrm_adaptive_memory.executive.semantic_relations.i3_13_document_corpus import (
        DocumentPassage, get_corpus,
    )
    corpus = get_corpus()
    chunks, corpus_by_text, corpus_by_id = build_corpus_index(corpus)

    # Build retriever (Q4_ORACLE is handled separately)
    retriever = None
    if condition != "Q4_ORACLE":
        retriever = build_retriever(condition, chunks)

    # Retrieve passages
    required_ids = get_required_passage_ids(task, corpus_by_text)
    if condition == "Q4_ORACLE":
        passages = retrieve_oracle(task, corpus_by_text, corpus_by_id)
    else:
        passages = retrieve_with_ladder(task, retriever, corpus_by_id, k=5)

    retrieval_recall = compute_retrieval_recall(passages, required_ids)
    required_retrieved = retrieval_recall == 1.0

    # Build evidence task
    new_et = build_retrieved_evidence_task(task, passages, corpus_by_text)
    if new_et is None:
        return {
            "task_id": task.task_id,
            "category": task.category,
            "retrieval_condition": condition,
            "arm": arm,
            "retrieval_recall": 0.0,
            "required_evidence_retrieved": False,
            "recall_any": False,
            "recall_all": False,
            "n_evidence": 0,
            "n_required": len(required_ids),
            "success": False,
            "utility": -150.0,
            "steps": 0,
            "terminal_action": "NO_EVIDENCE",
            "t2_triggered": False,
            "t2_trigger_step": None,
            "relation_correct": False,
            "relation_error": None,
            "relation_accuracy": 0.0,
            "failure_attribution": "RETRIEVAL_ERROR",
        }

    # Compute relation accuracy
    extractor = DeterministicRelationExtractor()
    rel_info = compute_relation_accuracy(new_et, task, extractor)

    # Run trajectory
    budget = ResourceBudget(**budget_dict)
    utility = MetareasoningUtility.from_file(
        ROOT / "configs/v2b_i3_1_utility_v1.json")

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
            mode = "BASELINE_WITH_AFFORDANCES"
            result = i3_12j.run_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                mode=mode, api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
            )
    except Exception as e:
        return {
            "task_id": task.task_id,
            "category": task.category,
            "retrieval_condition": condition,
            "arm": arm,
            "retrieval_recall": round(retrieval_recall, 4),
            "required_evidence_retrieved": required_retrieved,
            "recall_any": retrieval_recall > 0,
            "recall_all": required_retrieved,
            "n_evidence": len(new_et.evidence_items),
            "n_required": len(required_ids),
            "success": False,
            "utility": -150.0,
            "steps": 0,
            "terminal_action": "ERROR",
            "t2_triggered": False,
            "t2_trigger_step": None,
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

    if success:
        failure_attribution = "NO_ERROR"
    elif not required_retrieved:
        failure_attribution = "RETRIEVAL_ERROR"
    elif not rel_info["correct"]:
        failure_attribution = "SEMANTIC_EXTRACTION"
    else:
        failure_attribution = "POLICY_ERROR"

    return {
        "task_id": task.task_id,
        "category": task.category,
        "retrieval_condition": condition,
        "arm": arm,
        "retrieval_recall": round(retrieval_recall, 4),
        "required_evidence_retrieved": required_retrieved,
        "recall_any": retrieval_recall > 0,
        "recall_all": required_retrieved,
        "n_evidence": len(new_et.evidence_items),
        "n_required": len(required_ids),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-per-category", type=int, default=25)
    parser.add_argument("--retrieval-conditions", nargs="+",
                        default=RETRIEVAL_CONDITIONS)
    parser.add_argument("--arms", nargs="+",
                        default=["A1_GOLD", "A1_INFERRED", "R1_GOLD", "R1_INFERRED"])
    parser.add_argument("--output-dir", type=str,
                        default="experiments/v2b_i3_14/development/i3_14a_ladder")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Generate corpus
    tasks = generate_i3_13_corpus(n_per_category=args.n_per_category, seed=42)
    corpus = get_corpus()
    corpus_data = [
        {"passage_id": p.passage_id, "text": p.text, "source": p.source,
         "domain": p.domain, "gold_relations": [list(r) for r in p.gold_relations]}
        for p in corpus
    ]

    extractor = DeterministicRelationExtractor()
    src = (ROOT / "hrm_adaptive_memory/executive/semantic_relations/deterministic_rules.py").read_bytes()
    extractor_sha = hashlib.sha256(src).hexdigest()

    n_tasks = len(tasks)
    n_conditions = len(args.retrieval_conditions)
    n_arms = len(args.arms)
    total = n_tasks * n_conditions * n_arms

    print(f"I3.14a: Retrieval Sufficiency Repair")
    print(f"  {n_tasks} tasks x {n_conditions} conditions x {n_arms} arms = {total} trajectories")
    print(f"  Retrieval conditions: {args.retrieval_conditions}")
    print(f"  Arms: {args.arms}")
    print(f"  Extractor: v2.6.0 (FROZEN), SHA256: {extractor_sha[:16]}...")
    print(f"  Corpus: {len(corpus)} passages, SHA256: {corpus_sha256()[:16]}...")
    print(f"  Primary: RequiredEvidenceRecall increase -> TaskUtility increase")
    print()

    # Build work items
    budget_dict = {"max_executive_steps": 10, "max_retrieval_calls": 3,
                   "max_search_calls": 2, "max_verification_calls": 5}
    task_dicts = [{"task_id": t.task_id, "category": t.category} for t in tasks]

    work_items = []
    for td in task_dicts:
        for cond in args.retrieval_conditions:
            for arm in args.arms:
                work_items.append((td, cond, arm, corpus_data, api_key, budget_dict))

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v1.jsonl"
    log_path = out_dir / "run_log.txt"

    start = time.time()
    results = []
    completed = 0

    with open(log_path, "w") as logf, open(results_path, "w") as rf:
        logf.write(f"I3.14a: Retrieval Sufficiency Repair\n")
        logf.write(f"  {n_tasks} tasks x {n_conditions} conditions x {n_arms} arms = {total} trajectories\n")
        logf.write(f"  Extractor: v2.6.0 (FROZEN), SHA256: {extractor_sha[:16]}...\n")
        logf.write(f"  Corpus: {len(corpus)} passages, SHA256: {corpus_sha256()[:16]}...\n\n")
        logf.write(f"Processing {total} trajectories with {args.workers} workers...\n")
        logf.flush()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_single, wi): wi for wi in work_items}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    rf.write(json.dumps(result) + "\n")
                    rf.flush()
                except Exception as e:
                    logf.write(f"ERROR: {e}\n{traceback.format_exc()}\n")
                    logf.flush()
                completed += 1
                if completed % 50 == 0 or completed == total:
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    msg = f"  Completed {completed}/{total} ({rate:.1f}/s, ETA {eta:.0f}s)"
                    print(msg)
                    logf.write(msg + "\n")
                    logf.flush()

    elapsed = time.time() - start
    print(f"\nCompleted {total} trajectories in {elapsed:.1f}s")
    print(f"  Results: {results_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"I3.14a Results Summary ({n_tasks} tasks, {total} trajectories)")
    print(f"{'='*80}")

    by_ca = defaultdict(list)
    for r in results:
        by_ca[(r["retrieval_condition"], r["arm"])].append(r)

    print(f"\n{'Condition':<16} {'Arm':<16} {'N':>5} {'Recall':>8} {'RecAll':>8} {'Success':>8} {'MeanU':>8} {'RelAcc':>8}")
    for (cond, arm), rs in sorted(by_ca.items()):
        n = len(rs)
        recall = sum(r["retrieval_recall"] for r in rs) / n
        recall_all = sum(r.get("recall_all", False) for r in rs) / n
        success = sum(r["success"] for r in rs) / n
        mean_u = sum(r["utility"] for r in rs) / n
        rel_acc = sum(r.get("relation_accuracy", 1.0) for r in rs) / n
        print(f"  {cond:<16} {arm:<16} {n:>5} {recall:>8.4f} {recall_all:>8.4f} {success:>8.4f} {mean_u:>8.2f} {rel_acc:>8.4f}")

    # Recall decomposition per condition
    print(f"\n{'='*80}")
    print("RETRIEVAL RECALL DECOMPOSITION")
    print(f"{'='*80}")
    for cond in args.retrieval_conditions:
        r1i = [r for r in results if r["retrieval_condition"] == cond and r["arm"] == "R1_INFERRED"]
        if not r1i:
            continue
        recall = sum(r["retrieval_recall"] for r in r1i) / len(r1i)
        recall_any = sum(r.get("recall_any", False) for r in r1i) / len(r1i)
        recall_all = sum(r.get("recall_all", False) for r in r1i) / len(r1i)
        print(f"  {cond}: Recall={recall:.4f} RecallAny={recall_any:.4f} RecallAll={recall_all:.4f}")

    # P(success | RecallAll=1) vs P(success | RecallAll=0) per condition
    print(f"\n{'='*80}")
    print("CONDITIONAL PERFORMANCE: P(success | RecallAll)")
    print(f"{'='*80}")
    for cond in args.retrieval_conditions:
        r1i = [r for r in results if r["retrieval_condition"] == cond and r["arm"] == "R1_INFERRED"]
        if not r1i:
            continue
        all_retr = [r for r in r1i if r.get("recall_all", False)]
        not_all = [r for r in r1i if not r.get("recall_all", False)]
        p_all = sum(r["success"] for r in all_retr) / len(all_retr) if all_retr else 0
        p_not = sum(r["success"] for r in not_all) / len(not_all) if not_all else 0
        print(f"  {cond}: P(succ|RecallAll=1)={p_all:.4f} (n={len(all_retr)})  P(succ|RecallAll=0)={p_not:.4f} (n={len(not_all)})")

    # Gap decomposition
    print(f"\n{'='*80}")
    print("GAP DECOMPOSITION (mean utility)")
    print(f"{'='*80}")
    for cond in args.retrieval_conditions:
        cond_results = [r for r in results if r["retrieval_condition"] == cond]
        if not cond_results:
            continue
        by_arm = defaultdict(list)
        for r in cond_results:
            by_arm[r["arm"]].append(r)
        r1g = sum(r["utility"] for r in by_arm.get("R1_GOLD", [])) / len(by_arm["R1_GOLD"]) if by_arm.get("R1_GOLD") else 0
        r1i = sum(r["utility"] for r in by_arm.get("R1_INFERRED", [])) / len(by_arm["R1_INFERRED"]) if by_arm.get("R1_INFERRED") else 0
        a1i = sum(r["utility"] for r in by_arm.get("A1_INFERRED", [])) / len(by_arm["A1_INFERRED"]) if by_arm.get("A1_INFERRED") else 0
        a1g = sum(r["utility"] for r in by_arm.get("A1_GOLD", [])) / len(by_arm["A1_GOLD"]) if by_arm.get("A1_GOLD") else 0
        sem_gap = r1g - r1i
        route_gap = r1i - a1i
        print(f"  {cond}: R1_GOLD={r1g:.2f} R1_INF={r1i:.2f} A1_INF={a1i:.2f} A1_GOLD={a1g:.2f} SemGap={sem_gap:.2f} RouteGap={route_gap:.2f}")

    # Retrieval gap (Q4 - Qk)
    print(f"\n{'='*80}")
    print("RETRIEVAL GAP (Q4_ORACLE - Qk, R1_INFERRED)")
    print(f"{'='*80}")
    q4_r1i = [r for r in results if r["retrieval_condition"] == "Q4_ORACLE" and r["arm"] == "R1_INFERRED"]
    if q4_r1i:
        q4_u = sum(r["utility"] for r in q4_r1i) / len(q4_r1i)
        q4_recall = sum(r["retrieval_recall"] for r in q4_r1i) / len(q4_r1i)
        for cond in args.retrieval_conditions:
            if cond == "Q4_ORACLE":
                continue
            qk_r1i = [r for r in results if r["retrieval_condition"] == cond and r["arm"] == "R1_INFERRED"]
            if not qk_r1i:
                continue
            qk_u = sum(r["utility"] for r in qk_r1i) / len(qk_r1i)
            qk_recall = sum(r["retrieval_recall"] for r in qk_r1i) / len(qk_r1i)
            retr_gap = q4_u - qk_u
            print(f"  {cond}: U={qk_u:.2f} Recall={qk_recall:.4f} RetrievalGap={retr_gap:.2f}")

    # Failure attribution
    print(f"\nFailure attribution (all results):")
    fa_counts = Counter(r["failure_attribution"] for r in results)
    for fa, count in fa_counts.most_common():
        print(f"  {fa}: {count}")

    # Save analysis
    analysis = {
        "n_tasks": n_tasks,
        "n_trajectories": total,
        "elapsed_s": round(elapsed, 1),
        "extractor_sha256": extractor_sha,
        "corpus_sha256": corpus_sha256(),
        "retrieval_conditions": args.retrieval_conditions,
        "arms": args.arms,
        "failure_attribution": dict(fa_counts),
    }
    analysis_path = out_dir / "analysis_v1.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
