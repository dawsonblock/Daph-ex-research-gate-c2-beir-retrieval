#!/usr/bin/env python3
"""I3.14b: Retrieval Qualification.

Retrieval-only evaluation of the Q0-Q4 ladder on the frozen I3.13 corpus.
No LLM, no A1/R1/MDSG/T2 involvement.

Primary criteria:
  RecallAll_Q3 > RecallAll_Q0 + 0.15
  RecallAll_Q3 >= 0.75

Secondary:
  RequiredEvidenceRecall_Q3 > RequiredEvidenceRecall_Q1

Also measures latency, candidate counts, and memory.

Usage:
    PYTHONPATH=. python3 -u scripts/run_i3_14b_retrieval_qualification.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.semantic_relations.i3_13_task_generator import (
    generate_i3_13_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_13_document_corpus import (
    get_corpus, corpus_sha256,
)
from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import (
    build_retriever, RETRIEVAL_CONDITIONS, retriever_digest,
)
from hrm_adaptive_memory.memory.chunking import Chunk


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
    return chunks, corpus_by_text


def get_required_passage_ids(task, corpus_by_text):
    """Get passage IDs for evidence with retrieved=True only."""
    required = set()
    for ev in task.evidence_task.evidence_items:
        if not ev.retrieved:
            continue
        pid = corpus_by_text.get(ev.proposition)
        if pid:
            required.add(pid)
    return required


def main():
    corpus = get_corpus()
    tasks = generate_i3_13_corpus(n_per_category=25, seed=42)
    chunks, corpus_by_text = build_corpus_index(corpus)

    print(f"I3.14b: Retrieval Qualification")
    print(f"  Corpus: {len(corpus)} passages, SHA256: {corpus_sha256()[:16]}...")
    print(f"  Tasks: {len(tasks)} (seed=42, 25 per category)")
    print(f"  Retrievers: {RETRIEVAL_CONDITIONS}")
    print(f"  No LLM involvement — retrieval metrics only")
    print()

    # Compute required passage IDs for all tasks
    task_required = {}
    for task in tasks:
        req = get_required_passage_ids(task, corpus_by_text)
        task_required[task.task_id] = req

    n_with_required = sum(1 for r in task_required.values() if r)
    print(f"  Tasks with >=1 required passage: {n_with_required}/{len(tasks)}")
    n_multi = sum(1 for r in task_required.values() if len(r) > 1)
    print(f"  Tasks with >1 required passage: {n_multi}/{len(tasks)}")
    print()

    # Evaluate each retriever
    results = {}
    perf_data = {}

    for cond in RETRIEVAL_CONDITIONS:
        print(f"  Evaluating {cond}...", flush=True)

        if cond == "Q4_ORACLE":
            # Oracle: all required passages
            recalls = []
            any_count = 0
            all_count = 0
            per_task = []
            for task in tasks:
                req = task_required[task.task_id]
                if not req:
                    recall = 1.0
                else:
                    recall = 1.0  # Oracle always retrieves all
                recalls.append(recall)
                if recall > 0: any_count += 1
                if recall == 1.0: all_count += 1
                per_task.append({
                    "task_id": task.task_id,
                    "category": task.category,
                    "recall": recall,
                    "recall_any": recall > 0,
                    "recall_all": recall == 1.0,
                    "n_required": len(req),
                    "n_retrieved": len(req),
                    "retrieved_ids": sorted(req),
                })
            results[cond] = {
                "recall_mean": sum(recalls) / len(recalls),
                "recall_any": any_count / len(tasks),
                "recall_all": all_count / len(tasks),
                "per_task": per_task,
            }
            perf_data[cond] = {"latency_ms": 0.0, "candidate_count": 0}
            print(f"    Recall={results[cond]['recall_mean']:.4f} "
                  f"RecallAny={results[cond]['recall_any']:.4f} "
                  f"RecallAll={results[cond]['recall_all']:.4f}")
            continue

        # Build retriever
        t0 = time.time()
        retriever = build_retriever(cond, chunks)
        build_time = time.time() - t0

        # Evaluate
        recalls = []
        any_count = 0
        all_count = 0
        per_task = []
        latencies = []

        for task in tasks:
            req = task_required[task.task_id]
            query = task.evidence_task.task_summary

            t1 = time.time()
            search_results = retriever.search(query, top_k=5)
            latency_ms = (time.time() - t1) * 1000
            latencies.append(latency_ms)

            retrieved_ids = {chunk.chunk_id for chunk, _ in search_results}
            if not req:
                recall = 1.0
            else:
                recall = len(req & retrieved_ids) / len(req)

            recalls.append(recall)
            if recall > 0: any_count += 1
            if recall == 1.0: all_count += 1
            per_task.append({
                "task_id": task.task_id,
                "category": task.category,
                "recall": round(recall, 4),
                "recall_any": recall > 0,
                "recall_all": recall == 1.0,
                "n_required": len(req),
                "n_retrieved": len(req & retrieved_ids),
                "retrieved_ids": sorted(req & retrieved_ids),
                "missing_ids": sorted(req - retrieved_ids),
            })

        mean_recall = sum(recalls) / len(recalls)
        mean_latency = sum(latencies) / len(latencies)
        results[cond] = {
            "recall_mean": mean_recall,
            "recall_any": any_count / len(tasks),
            "recall_all": all_count / len(tasks),
            "per_task": per_task,
        }
        perf_data[cond] = {
            "build_time_s": round(build_time, 3),
            "latency_ms_mean": round(mean_latency, 2),
            "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "candidate_count": 5,
        }
        print(f"    Recall={mean_recall:.4f} "
              f"RecallAny={any_count/len(tasks):.4f} "
              f"RecallAll={all_count/len(tasks):.4f} "
              f"Latency={mean_latency:.1f}ms")

    # Per-category breakdown
    print(f"\n{'='*80}")
    print("PER-CATEGORY RECALL")
    print(f"{'='*80}")
    print(f"{'Category':<25} {'N':>4}", end="")
    for cond in RETRIEVAL_CONDITIONS:
        print(f" {cond:>14}", end="")
    print()
    for cat in sorted(set(t.category for t in tasks)):
        cat_tasks = [t for t in tasks if t.category == cat]
        print(f"{cat:<25} {len(cat_tasks):>4}", end="")
        for cond in RETRIEVAL_CONDITIONS:
            cat_results = [r for r in results[cond]["per_task"] if r["category"] == cat]
            cat_recall = sum(r["recall"] for r in cat_results) / len(cat_results)
            print(f" {cat_recall:>14.4f}", end="")
        print()

    # Primary criteria
    print(f"\n{'='*80}")
    print("PRIMARY CRITERIA")
    print(f"{'='*80}")
    r_q3 = results["Q3_RERANKED"]["recall_all"]
    r_q0 = results["Q0_BM25"]["recall_all"]
    r_q1 = results["Q1_DENSE"]["recall_mean"]
    r_q3_mean = results["Q3_RERANKED"]["recall_mean"]

    margin = r_q3 - r_q0
    crit1_pass = margin >= 0.15
    crit2_pass = r_q3 >= 0.75
    crit3_pass = r_q3_mean > r_q1

    print(f"  RecallAll_Q3 > RecallAll_Q0 + 0.15:")
    print(f"    {r_q3:.4f} > {r_q0:.4f} + 0.15 = {r_q0 + 0.15:.4f}")
    print(f"    Margin = {margin:.4f} ({margin*100:.1f}pp)")
    print(f"    PASS: {crit1_pass}")
    print()
    print(f"  RecallAll_Q3 >= 0.75:")
    print(f"    {r_q3:.4f} >= 0.75")
    print(f"    PASS: {crit2_pass}")
    print()
    print(f"  RequiredEvidenceRecall_Q3 > RequiredEvidenceRecall_Q1 (reranker justifies cost):")
    print(f"    {r_q3_mean:.4f} > {r_q1:.4f}")
    print(f"    PASS: {crit3_pass}")
    print()
    overall_pass = crit1_pass and crit2_pass
    print(f"  OVERALL (primary): {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass and crit3_pass:
        print(f"  RECOMMENDATION: Freeze Q3_RERANKED as production retrieval stack")

    # Performance summary
    print(f"\n{'='*80}")
    print("PERFORMANCE")
    print(f"{'='*80}")
    print(f"{'Condition':<16} {'Build(s)':>10} {'Latency(ms)':>12} {'P95(ms)':>10}")
    for cond in RETRIEVAL_CONDITIONS:
        p = perf_data.get(cond, {})
        print(f"  {cond:<16} {p.get('build_time_s', 0):>10.3f} "
              f"{p.get('latency_ms_mean', 0):>12.1f} {p.get('latency_ms_p95', 0):>10.1f}")

    # Save qualification artifact
    qualification = {
        "experiment": "I3.14b",
        "status": "FROZEN",
        "corpus_sha256": corpus_sha256(),
        "n_tasks": len(tasks),
        "n_passages": len(corpus),
        "retrievers": {cond: retriever_digest(cond) for cond in RETRIEVAL_CONDITIONS},
        "results": {
            cond: {
                "recall_mean": round(r["recall_mean"], 4),
                "recall_any": round(r["recall_any"], 4),
                "recall_all": round(r["recall_all"], 4),
            }
            for cond, r in results.items()
        },
        "performance": perf_data,
        "primary_criteria": {
            "RecallAll_Q3_gt_Q0_plus_15pp": {
                "value": round(margin, 4),
                "pass": crit1_pass,
            },
            "RecallAll_Q3_ge_0_75": {
                "value": round(r_q3, 4),
                "pass": crit2_pass,
            },
            "Recall_Q3_gt_Q1": {
                "value": round(r_q3_mean - r_q1, 4),
                "pass": crit3_pass,
            },
            "overall_primary_pass": overall_pass,
        },
        "freeze_recommendation": "Q3_RERANKED" if overall_pass else "NONE",
    }

    out_path = ROOT / "experiments/v2b_i3_14/development/i3_14b_qualification_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qualification, f, indent=2)
    print(f"\n  Qualification: {out_path}")


if __name__ == "__main__":
    main()
