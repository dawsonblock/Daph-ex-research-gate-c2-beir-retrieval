"""Stage A: Prepare evidence for I3.15c experiments.

This script pre-computes all retrieval results and writes immutable retrieval
receipts. Stage B (run_policy.py) never calls the retriever, making inference
experiments deterministic with respect to retrieval.

Usage:
    PYTHONPATH=. python3 scripts/i3_15c_prepare_evidence.py \
        --n-per-cell 10 \
        --retrieval-levels Q0_BM25,Q3_RERANKED,Q4_ORACLE \
        --output experiments/v2b_i3_15c/confirmation/retrieval_receipts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser(description="Stage A: Prepare evidence")
    parser.add_argument("--n-per-cell", type=int, default=10)
    parser.add_argument("--retrieval-levels", default="Q0_BM25,Q3_RERANKED,Q4_ORACLE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    retrieval_levels = args.retrieval_levels.split(",")

    from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
        generate_i3_15c_corpus, get_i3_15c_corpus,
    )
    from hrm_adaptive_memory.memory.chunking import Chunk
    from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import build_retriever
    from scripts.run_i3_15c_factorial import get_required_passage_ids, TOP_K

    # Generate tasks
    print(f"Generating tasks (n_per_cell={args.n_per_cell}, seed={args.seed})...")
    tasks = generate_i3_15c_corpus(n_per_cell=args.n_per_cell, seed=args.seed)
    print(f"Generated {len(tasks)} tasks")

    # Build corpus
    corpus_passages = get_i3_15c_corpus()
    chunks = [
        Chunk(
            chunk_id=p.passage_id, source_id=p.source, source_type="doc",
            title=p.passage_id, section="", content=p.text,
            token_count=len(p.text.split()),
        )
        for p in corpus_passages
    ]
    corpus_by_id = {p.passage_id: p for p in corpus_passages}
    corpus_by_text = {p.text: p.passage_id for p in corpus_passages}

    corpus_sha = hashlib.sha256(
        json.dumps(
            [{"id": p.passage_id, "text": p.text, "domain": p.domain}
             for p in corpus_passages],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    print(f"Corpus: {len(chunks)} passages, SHA256: {corpus_sha[:16]}...")

    # Pre-retrieve for each task x retrieval_level
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPre-retrieving evidence for {len(tasks)} tasks x {len(retrieval_levels)} levels...")
    t_start = time.time()

    retrievers = {}
    for level in retrieval_levels:
        if level != "Q4_ORACLE":
            print(f"  Building retriever: {level}...")
            retrievers[level] = build_retriever(level, chunks)

    receipts = []
    for i, task in enumerate(tasks):
        et = task.evidence_task
        required_ids = get_required_passage_ids(task, corpus_by_text)

        for level in retrieval_levels:
            if level == "Q4_ORACLE":
                retrieved_passages = [
                    corpus_by_id[pid] for pid in required_ids
                    if pid in corpus_by_id
                ]
                scores = [1.0] * len(retrieved_passages)
            else:
                retriever = retrievers[level]
                retrieved = retriever.search(et.task_summary, top_k=TOP_K)
                retrieved_passages = [
                    corpus_by_id[c.chunk_id] for c, _ in retrieved
                    if c.chunk_id in corpus_by_id
                ]
                scores = [s for c, s in retrieved if c.chunk_id in corpus_by_id]

            retrieved_ids = [p.passage_id for p in retrieved_passages]
            recall = len(set(retrieved_ids) & required_ids) / max(len(required_ids), 1)

            receipt = {
                "task_id": et.task_id,
                "category": et.category,
                "retrieval_condition": level,
                "retrieved_chunk_ids": retrieved_ids,
                "scores": [round(s, 6) for s in scores],
                "recall": round(recall, 4),
                "required_ids": sorted(list(str(pid) for pid in required_ids)),
                "retrieval_sha256": hashlib.sha256(
                    json.dumps(retrieved_ids, sort_keys=True).encode()
                ).hexdigest(),
                "query": et.task_summary,
            }
            receipts.append(receipt)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tasks)} tasks done ({time.time() - t_start:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\nPre-retrieval complete: {len(receipts)} receipts in {elapsed:.1f}s")

    # Write receipts
    with open(output_path, "w") as f:
        for r in receipts:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote {output_path} ({len(receipts)} receipts)")

    # ---- Qualification gates ----
    print("\n" + "=" * 80)
    print("RETRIEVAL QUALIFICATION GATES")
    print("=" * 80)

    gate_failures = []

    # Gate 1: Corpus coverage = 1.0 (all required IDs present in corpus)
    all_required = set()
    for r in receipts:
        all_required.update(r["required_ids"])
    corpus_id_set = {c.chunk_id for c in chunks}
    missing_from_corpus = all_required - corpus_id_set
    corpus_coverage = 1.0 if not missing_from_corpus else (
        1.0 - len(missing_from_corpus) / max(len(all_required), 1)
    )
    print(f"\nGate 1: CorpusCoverage = {corpus_coverage:.4f}")
    if missing_from_corpus:
        print(f"  FAIL: {len(missing_from_corpus)} required IDs missing from corpus")
        for mid in sorted(list(missing_from_corpus))[:10]:
            print(f"    {mid}")
        gate_failures.append("CORPUS_COVERAGE_LT_1")
    else:
        print(f"  PASS: all {len(all_required)} required IDs present in corpus")

    # Gate 2: Receipt count == expected
    expected_receipts = len(tasks) * len(retrieval_levels)
    print(f"\nGate 2: ReceiptCount = {len(receipts)} (expected {expected_receipts})")
    if len(receipts) != expected_receipts:
        print(f"  FAIL: receipt count mismatch")
        gate_failures.append("RECEIPT_COUNT_MISMATCH")
    else:
        print(f"  PASS")

    # Gate 3: No duplicate canonical IDs in corpus
    corpus_ids = [p.passage_id for p in corpus_passages]
    duplicate_ids = set([pid for pid in corpus_ids if corpus_ids.count(pid) > 1])
    print(f"\nGate 3: DuplicateCanonicalIDs = {len(duplicate_ids)}")
    if duplicate_ids:
        print(f"  FAIL: duplicates: {duplicate_ids}")
        gate_failures.append("DUPLICATE_CANONICAL_IDS")
    else:
        print(f"  PASS: no duplicates")

    # Gate 4: Q3 recall > Q0 recall
    q3_receipts = [r for r in receipts if r["retrieval_condition"] == "Q3_RERANKED"]
    q0_receipts = [r for r in receipts if r["retrieval_condition"] == "Q0_BM25"]
    if q3_receipts and q0_receipts:
        q3_mean = sum(r["recall"] for r in q3_receipts) / len(q3_receipts)
        q0_mean = sum(r["recall"] for r in q0_receipts) / len(q0_receipts)
        print(f"\nGate 4: Q3 recall ({q3_mean:.4f}) > Q0 recall ({q0_mean:.4f})")
        if q3_mean <= q0_mean:
            print(f"  WARN: Q3 not better than Q0")
            gate_failures.append("Q3_NOT_BETTER_THAN_Q0")
        else:
            print(f"  PASS")

    # Gate 5: Q3 RecallAll >= 0.75
    if q3_receipts:
        q3_recall_all = sum(1 for r in q3_receipts if r["recall"] == 1.0) / len(q3_receipts)
        print(f"\nGate 5: Q3 RecallAll = {q3_recall_all:.4f} (threshold 0.75)")
        if q3_recall_all < 0.75:
            print(f"  WARN: Q3 RecallAll below threshold")
            gate_failures.append("Q3_RECALL_ALL_LOW")
        else:
            print(f"  PASS")

    # ---- Per-stratum recall breakdown ----
    print(f"\n{'='*80}")
    print("PER-STRATUM RECALL")
    print(f"{'='*80}")
    strata_order = [
        "t2_conflict_immediate",
        "t2_conflict_late_1",
        "t2_conflict_late_2",
        "t2_conflict_late_3",
        "matched_neg_immediate",
        "matched_neg_late",
        "defer_control",
        "answer_control",
    ]

    per_stratum = {}
    for stratum in strata_order:
        stratum_receipts = [r for r in receipts
                           if stratum in r.get("category", "")
                           and r["retrieval_condition"] in ("Q0_BM25", "Q3_RERANKED")]
        if not stratum_receipts:
            continue
        for level in ("Q0_BM25", "Q3_RERANKED"):
            level_receipts = [r for r in stratum_receipts if r["retrieval_condition"] == level]
            if not level_receipts:
                continue
            mean_recall = sum(r["recall"] for r in level_receipts) / len(level_receipts)
            recall_all = sum(1 for r in level_receipts if r["recall"] == 1.0) / len(level_receipts)
            key = f"{stratum}_{level}"
            per_stratum[key] = {
                "mean_recall": round(mean_recall, 4),
                "recall_all": round(recall_all, 4),
                "n": len(level_receipts),
            }
            print(f"  {stratum:40s} {level:12s}  recall={mean_recall:.4f}  recall_all={recall_all:.4f}  n={len(level_receipts)}")

    # ---- Final qualification ----
    qualified = len(gate_failures) == 0
    print(f"\n{'='*80}")
    print(f"QUALIFICATION: {'PASS' if qualified else 'FAIL'}")
    print(f"{'='*80}")
    if gate_failures:
        print(f"Gate failures: {gate_failures}")
        print("RECEIPT BUNDLE IS NOT QUALIFIED FOR CONFIRMATION EXPERIMENTS.")
    else:
        print("All gates passed. Receipt bundle is qualified.")

    # Write summary with gates
    summary_path = output_path.parent / "retrieval_summary.json"
    summary = {
        "n_tasks": len(tasks),
        "n_retrieval_levels": len(retrieval_levels),
        "retrieval_levels": retrieval_levels,
        "n_receipts": len(receipts),
        "corpus_sha256": corpus_sha,
        "corpus_size": len(chunks),
        "seed": args.seed,
        "n_per_cell": args.n_per_cell,
        "wall_time_s": round(elapsed, 1),
        "mean_recall_by_level": {},
        "qualification": {
            "qualified": qualified,
            "gate_failures": gate_failures,
            "corpus_coverage": round(corpus_coverage, 4),
            "expected_receipts": expected_receipts,
            "actual_receipts": len(receipts),
            "duplicate_canonical_ids": len(duplicate_ids),
        },
        "per_stratum_recall": per_stratum,
    }
    for level in retrieval_levels:
        level_receipts = [r for r in receipts if r["retrieval_condition"] == level]
        mean_recall = sum(r["recall"] for r in level_receipts) / len(level_receipts)
        summary["mean_recall_by_level"][level] = round(mean_recall, 4)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")
    print(f"Mean recall by level: {summary['mean_recall_by_level']}")


if __name__ == "__main__":
    main()
