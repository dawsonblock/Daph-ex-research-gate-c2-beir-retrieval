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
    corpus_by_text = {p.text: p for p in corpus_passages}

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
                "required_ids": sorted(list(required_ids)),
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

    # Write summary
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
    }
    for level in retrieval_levels:
        level_receipts = [r for r in receipts if r["retrieval_condition"] == level]
        mean_recall = sum(r["recall"] for r in level_receipts) / len(level_receipts)
        summary["mean_recall_by_level"][level] = round(mean_recall, 4)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print(f"\nMean recall by level: {summary['mean_recall_by_level']}")


if __name__ == "__main__":
    main()
