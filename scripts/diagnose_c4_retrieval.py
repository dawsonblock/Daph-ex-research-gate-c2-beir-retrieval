#!/usr/bin/env python3
"""Sprint A: retrieval-only diagnosis. No HRM, no mechanism changes.

The qualification failure decomposition put 55.6% of failures upstream of the
selector: the required evidence never reached the candidate pool. This script
answers the question that decides what the repair actually is --

    Is required evidence completely absent from retrieval's ranking, or is it
    ranked and merely falling below the candidate cutoff?

Those need opposite fixes. Below-cutoff evidence is a budget/fusion problem.
Unranked evidence is a representation/query problem, and no amount of budget
will recover it.

Frozen receipts cannot answer this. run_gate_c4.py records bm25_ranked,
bge_ranked and fusion_ranked all truncated to the candidate budget (50), so a
required record at true rank 73 and one at true rank 2000 are
indistinguishable in the receipts -- both simply absent. This script therefore
re-runs the SAME frozen retrieval (same query construction via
run_query_stage, same backends via get_cached_backend, same RRF via
retrieval_stage._rrf) at a much larger probe depth and observes where the
required records actually sit. Nothing about the mechanism is modified; only
the observation window widens.

Two orthogonal axes are reported, which is more precise than one flat failure
list: WHICH evidence role was lost, and WHY it was lost.

  role     IDENTITY / BRIDGE / TERMINAL_ANSWER / TEMPORAL_CURRENT / SUPPORTING
           derived from the task's own proof_edges, answer_node and
           latent_bridge, not guessed from text.

  cause    BELOW_BUDGET        ranked by fusion, just past the cutoff
           FUSION_DISPLACEMENT within budget for BM25 or BGE alone, but RRF
                               pushed it out -- fusion actively lost it
           LEXICAL_MISS        effectively unranked by BM25
           DENSE_MISS          effectively unranked by BGE
           BOTH_RETRIEVERS_MISS neither retriever ranks it anywhere near --
                               a representation failure, not a budget one
           UNRANKED            outside the probe depth entirely

Usage:
    python scripts/diagnose_c4_retrieval.py [--split development]
        [--arm C4_4] [--probe-k 0]   # 0 = whole corpus
        [--ces-at 10,20,50,100,200,500] [--out <path>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_CANDIDATE_BUDGET, C4_RRF_K)
from hrm_adaptive_memory.c4.query_stage import run_query_stage  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import (  # noqa: E402
    _rrf, get_cached_backend)

# Corpus loading and IndexRecord construction are imported from the runner
# rather than reimplemented: a local copy drifted immediately (it dropped
# token_count) and any such drift would mean the backends see different input
# here than in a real run, silently invalidating the diagnosis.
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records)

ROLES = ("IDENTITY", "BRIDGE", "TERMINAL_ANSWER", "TEMPORAL_CURRENT", "SUPPORTING")
CAUSES = ("BELOW_BUDGET", "FUSION_DISPLACEMENT", "LEXICAL_MISS", "DENSE_MISS",
          "BOTH_RETRIEVERS_MISS", "UNRANKED")

# A record ranked beyond this by a single retriever is treated as effectively
# unranked by it. Deliberately generous relative to the budget of 50: the point
# is to separate "just missed the cutoff" from "this retriever has no idea",
# not to draw a fine line.
SINGLE_RETRIEVER_MISS_RANK = 200


def role_of(record_id: str, task: dict, kinds: dict[str, str]) -> str:
    """Label a required record by its function in the task's own proof graph.

    Uses the oracle proof structure rather than string heuristics: an edge that
    terminates at answer_node is answer-support, one terminating at the latent
    bridge is the bridge hop. record_kind disambiguates identity and
    current-state records, which are required but are not proof edges.
    """
    kind = kinds.get(record_id, "")
    if kind == "required_identity":
        return "IDENTITY"
    if kind == "required_current":
        return "TEMPORAL_CURRENT"

    oracle = task.get("_oracle_metadata") or {}
    answer_node = oracle.get("answer_node")
    bridge = oracle.get("latent_bridge")
    for edge in oracle.get("proof_edges") or []:
        if edge.get("record_id") != record_id:
            continue
        if answer_node and edge.get("target") == answer_node:
            return "TERMINAL_ANSWER"
        if bridge and edge.get("target") == bridge:
            return "BRIDGE"
    return "SUPPORTING"


def rank_of(ranked_ids: Sequence[str], record_id: str) -> int | None:
    """1-indexed rank, or None if absent from the probed depth."""
    try:
        return ranked_ids.index(record_id) + 1
    except ValueError:
        return None


def cause_of(fusion_rank: int | None, bm25_rank: int | None,
             dense_rank: int | None, budget: int) -> str:
    """Why a required record failed to reach the candidate pool."""
    if fusion_rank is not None and fusion_rank <= budget:
        raise AssertionError("record is in the pool; it did not fail")

    lexical_ok = bm25_rank is not None and bm25_rank <= SINGLE_RETRIEVER_MISS_RANK
    dense_ok = dense_rank is not None and dense_rank <= SINGLE_RETRIEVER_MISS_RANK

    # Either retriever had it inside the budget, yet fusion did not: RRF
    # actively displaced it. This is the one cause fixable purely in fusion.
    if ((bm25_rank is not None and bm25_rank <= budget)
            or (dense_rank is not None and dense_rank <= budget)):
        return "FUSION_DISPLACEMENT"
    if fusion_rank is not None:
        return "BELOW_BUDGET"
    if not lexical_ok and not dense_ok:
        return "UNRANKED" if (bm25_rank is None and dense_rank is None) \
            else "BOTH_RETRIEVERS_MISS"
    return "DENSE_MISS" if lexical_ok else "LEXICAL_MISS"


def probe_task(task: dict, arm, records, probe_k: int) -> dict[str, Any]:
    """Re-run frozen retrieval at depth probe_k and locate required evidence."""
    _state, query_result = run_query_stage(task["question"], arm)
    query = query_result.rendered_query

    bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    bm25_result = asyncio.run(bm25.search(query, k=probe_k))
    bm25_ids = [e.evidence_id for e in bm25_result.evidence]

    if arm.retrieval_policy == "bm25_only":
        dense_ids: list[str] = []
        fusion_ids = bm25_ids
    else:
        dense = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
        dense_result = asyncio.run(dense.search(query, k=probe_k))
        dense_ids = [e.evidence_id for e in dense_result.evidence]
        fusion_ids = [eid for eid, _ in
                      _rrf([bm25_ids, dense_ids], C4_RRF_K, probe_k)]

    return {"query": query, "bm25": bm25_ids, "dense": dense_ids,
            "fusion": fusion_ids}


def _pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval-only diagnosis for C4 (no HRM)")
    parser.add_argument("--split", default="development")
    parser.add_argument("--arm", default="C4_4",
                        help="arm whose frozen query/retrieval policy to probe")
    parser.add_argument("--probe-k", type=int, default=0,
                        help="ranking depth to observe; 0 = entire corpus")
    parser.add_argument("--ces-at", default="10,20,50,100,200,500",
                        help="candidate budgets to report CES for")
    parser.add_argument("--limit", type=int, default=0, help="first N tasks only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tasks, evidence, _texts = load_split(args.split)
    if args.limit:
        tasks = tasks[:args.limit]
    records = to_index_records(evidence)
    probe_k = args.probe_k or len(records)
    budget = C4_CANDIDATE_BUDGET
    ces_ks = sorted({int(x) for x in args.ces_at.split(",") if x.strip()})
    arm = ARMS[args.arm]
    kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
             for r in evidence}

    print(f"=== C4 retrieval diagnosis: {args.split} ===")
    print(f"  tasks={len(tasks)}  corpus={len(records)}  arm={args.arm} "
          f"({arm.retrieval_policy})")
    print(f"  frozen budget={budget}  probe depth={probe_k}\n")

    per_task: list[dict[str, Any]] = []
    role_totals: Counter = Counter()
    role_in_pool: Counter = Counter()
    cause_counts: Counter = Counter()
    cause_by_role: dict[str, Counter] = defaultdict(Counter)
    missing_ranks: list[int | None] = []
    ces_hits = {k: 0 for k in ces_ks}
    ces_hits_full = 0

    for index, task in enumerate(tasks, 1):
        if index % 25 == 0 or index == len(tasks):
            print(f"  probing {index}/{len(tasks)}...", end="\r", flush=True)
        probe = probe_task(task, arm, records, probe_k)
        fusion, bm25, dense = probe["fusion"], probe["bm25"], probe["dense"]
        required = list(task["required_evidence_ids"])

        rows = []
        for record_id in required:
            f_rank = rank_of(fusion, record_id)
            b_rank = rank_of(bm25, record_id)
            d_rank = rank_of(dense, record_id)
            role = role_of(record_id, task, kinds)
            in_pool = f_rank is not None and f_rank <= budget

            role_totals[role] += 1
            if in_pool:
                role_in_pool[role] += 1
            else:
                cause = cause_of(f_rank, b_rank, d_rank, budget)
                cause_counts[cause] += 1
                cause_by_role[role][cause] += 1
                missing_ranks.append(f_rank)

            rows.append({
                "record_id": record_id, "role": role, "in_pool": in_pool,
                "fusion_rank": f_rank, "bm25_rank": b_rank, "dense_rank": d_rank,
                "cause": None if in_pool else cause_of(f_rank, b_rank, d_rank, budget),
            })

        for k in ces_ks:
            if all(r["fusion_rank"] is not None and r["fusion_rank"] <= k
                   for r in rows):
                ces_hits[k] += 1
        if all(r["fusion_rank"] is not None for r in rows):
            ces_hits_full += 1

        per_task.append({
            "task_id": task["task_id"],
            "family": task["family"],
            "entity_regime": task["metadata"]["entity_regime"],
            "required": rows,
            "complete_at_budget": all(r["in_pool"] for r in rows),
        })

    n = len(tasks)
    print(" " * 40, end="\r")

    ces_curve = {str(k): _pct(v, n) for k, v in sorted(ces_hits.items())}
    ces_curve["unbounded"] = _pct(ces_hits_full, n)

    def grouped(key: str) -> dict[str, Any]:
        agg: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "complete_at_budget": 0, "complete_unbounded": 0})
        for row, task in zip(per_task, tasks):
            group = row[key] if key in row else task["metadata"][key]
            entry = agg[group]
            entry["n"] += 1
            entry["complete_at_budget"] += row["complete_at_budget"]
            entry["complete_unbounded"] += all(
                r["fusion_rank"] is not None for r in row["required"])
        return {
            g: {"n": v["n"],
                "ces_at_budget": _pct(v["complete_at_budget"], v["n"]),
                "ces_unbounded": _pct(v["complete_unbounded"], v["n"]),
                "recoverable_by_budget": _pct(
                    v["complete_unbounded"] - v["complete_at_budget"], v["n"])}
            for g, v in sorted(agg.items(),
                               key=lambda kv: kv[1]["complete_at_budget"] / kv[1]["n"])
        }

    ranked_but_cut = sum(1 for r in missing_ranks if r is not None)
    report = {
        "schema_version": "c4-retrieval-diagnosis-v1",
        "split": args.split,
        "arm": args.arm,
        "retrieval_policy": arm.retrieval_policy,
        "frozen_candidate_budget": budget,
        "probe_depth": probe_k,
        "corpus_records": len(records),
        "task_count": n,
        "diagnostic_only": (
            "Re-runs the frozen retrieval (same query construction, backends "
            "and RRF) at a wider observation depth. No mechanism change, no "
            "HRM, writes no certified artifact."),
        "headline_question": {
            "question": "absent from ranking, or merely below the cutoff?",
            "missing_required_records": len(missing_ranks),
            "ranked_but_below_budget": ranked_but_cut,
            "not_ranked_at_all_within_probe": len(missing_ranks) - ranked_but_cut,
            "share_recoverable_by_budget_alone": _pct(ranked_but_cut,
                                                      len(missing_ranks)),
        },
        "ces_curve_by_candidate_budget": ces_curve,
        "role_recall_at_budget": {
            role: {"required_instances": role_totals[role],
                   "in_pool": role_in_pool[role],
                   "recall": _pct(role_in_pool[role], role_totals[role])}
            for role in ROLES if role_totals[role]
        },
        "missing_cause_counts": {c: cause_counts[c] for c in CAUSES if cause_counts[c]},
        "missing_cause_by_role": {
            role: dict(counter) for role, counter in sorted(cause_by_role.items())},
        "by_family": grouped("family"),
        "by_entity_regime": grouped("entity_regime"),
        "per_task": per_task,
    }

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    head = report["headline_question"]
    print("  THE DECIDING QUESTION")
    print(f"    missing required records      : {head['missing_required_records']}")
    print(f"    ranked, below budget {budget:<9}: {head['ranked_but_below_budget']}"
          f"  ({head['share_recoverable_by_budget_alone']:.1%} of misses)")
    print(f"    not ranked at all             : "
          f"{head['not_ranked_at_all_within_probe']}")

    print("\n  CES by candidate budget (complete required set available):")
    for k, v in ces_curve.items():
        marker = "  <- frozen budget" if k == str(budget) else ""
        print(f"    k={k:<10}{v:>7.1%}{marker}")

    print("\n  role recall at frozen budget:")
    for role, stats in report["role_recall_at_budget"].items():
        print(f"    {role:<18}{stats['in_pool']:>4}/{stats['required_instances']:<4}"
              f" = {stats['recall']:>6.1%}")

    print("\n  why required evidence missed the pool:")
    for cause, count in sorted(report["missing_cause_counts"].items(),
                               key=lambda kv: -kv[1]):
        print(f"    {cause:<22}{count:>4}")

    print("\n  by family (worst first):")
    for family, stats in report["by_family"].items():
        print(f"    {family:<22}CES@{budget}={stats['ces_at_budget']:>6.1%}   "
              f"unbounded={stats['ces_unbounded']:>6.1%}   "
              f"budget-recoverable={stats['recoverable_by_budget']:>6.1%}")

    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
