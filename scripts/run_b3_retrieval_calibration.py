#!/usr/bin/env python3
"""B3 retrieval-only calibration across the frozen multi-scale suite.

Authoritative retriever ONLY. S2 is not involved, no HRM, no rho selected, no
engineering targets set. This produces the scaling-law artifact and stops.

Method
------
Every grid metric derives from ONE exact quantity per task:

    k_i* = min{ k : all required evidence for task i is in the pool }
         = the deepest FUSION rank among task i's required records

So the retriever is run once per task at FULL corpus depth, the fusion rank of
every required record is recorded, and each grid point is then a threshold on
those ranks rather than another retrieval pass. That is exact, not an
approximation of a grid search, and it also yields k* directly -- which is more
informative than any single chosen depth, because it exposes whether k*/N is
constant, sub-linear, or something else. The protocol permits a linear
k = ceil(rho*N) form; this measurement is what decides whether that form is
actually right, so linearity is not assumed here.

Grid
----
absolute depths 25, 50, 75, 100, 150, 200, 300, 400, unioned per scale with the
relative fractions 2.5%, 5%, 7.5%, 10%, 15% converted to integers and
deduplicated.

Recorded at every (N, k)
------------------------
    CES(N, k)
    role availability for identity / bridge / terminal / temporal_current
    complete-set availability
    missing-record taxonomy: CUT_OFF / FUSION_DISPLACED / UNRANKED

The taxonomy matters because it decides whether candidate-budget scaling can
solve the problem at all: CUT_OFF is recoverable by depth, FUSION_DISPLACED by a
fusion rule at the same depth, UNRANKED by neither.

Usage:
    python scripts/run_b3_retrieval_calibration.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.contracts import C4_RRF_K  # noqa: E402
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import run_query_stage  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from scripts.diagnose_c4_retrieval import cause_of, role_of, rank_of  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

SUITE = ROOT / "data/hrm/b3_calibration_v1"
SCALES = ("cal_700", "cal_1000", "cal_1500", "cal_2200", "cal_3000")
ABSOLUTE_DEPTHS = (25, 50, 75, 100, 150, 200, 300, 400)
FRACTIONS = (0.025, 0.05, 0.075, 0.10, 0.15)
ROLES = ("identity", "bridge", "terminal", "temporal_current")


def load_scale(scale: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    base = SUITE / scale
    tasks = [json.loads(l) for l in (base / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (base / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence, {r["evidence_id"]: r["content"] for r in evidence}


def grid_for(corpus_size: int) -> list[int]:
    """Absolute depths unioned with fraction-derived depths, deduplicated."""
    from math import ceil
    points = set(ABSOLUTE_DEPTHS)
    points |= {max(1, ceil(f * corpus_size)) for f in FRACTIONS}
    return sorted(p for p in points if p <= corpus_size)


def role_of_required(task: dict, kinds: dict[str, str]) -> dict[str, set[str]]:
    """Group a task's required records by role, using the oracle proof graph.

    Evaluation-side labelling only -- the retriever never sees this.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for record_id in task["required_evidence_ids"]:
        label = role_of(record_id, task, kinds)
        key = {"IDENTITY": "identity", "BRIDGE": "bridge",
               "TERMINAL_ANSWER": "terminal",
               "TEMPORAL_CURRENT": "temporal_current"}.get(label)
        if key:
            grouped[key].add(record_id)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="B3 retrieval-only calibration")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    print("=== B3 retrieval-only calibration ===")
    print(f"  fusion: frozen_rrf (authoritative), k_rrf={C4_RRF_K}")
    print("  S2 NOT involved; no HRM; no rho selected\n")

    per_scale: dict[str, Any] = {}
    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        depth = corpus_size  # full-depth probe: exact ranks, not a grid search

        # Per task: the fusion/bm25/dense rank of every required record.
        task_rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, 1):
            if index % 25 == 0 or index == len(tasks):
                print(f"  {scale}: {index}/{len(tasks)}", end="\r", flush=True)
            _state, query = run_query_stage(task["question"], arm)
            bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
            bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
            bm25_ids = [e.evidence_id for e in
                        asyncio.run(bm25.search(query.rendered_query, k=depth)).evidence]
            bge_ids = [e.evidence_id for e in
                       asyncio.run(bge.search(query.rendered_query, k=depth)).evidence]
            fusion_ids = [eid for eid, _ in
                          frozen_rrf([bm25_ids, bge_ids], C4_RRF_K, depth)]

            grouped = role_of_required(task, kinds)
            required = list(task["required_evidence_ids"])
            ranks = {
                rid: {"fusion": rank_of(fusion_ids, rid),
                      "bm25": rank_of(bm25_ids, rid),
                      "dense": rank_of(bge_ids, rid)}
                for rid in required}
            fusion_ranks = [r["fusion"] for r in ranks.values()]
            k_star = (max(fusion_ranks) if fusion_ranks and
                      all(r is not None for r in fusion_ranks) else None)
            task_rows.append({
                "task_id": task["task_id"], "family": task["family"],
                "entity_regime": task["metadata"]["entity_regime"],
                "k_star": k_star,
                "ranks": ranks,
                "role_records": {role: sorted(v) for role, v in grouped.items()},
            })
        print(" " * 40, end="\r")

        # Every grid metric is a threshold on the recorded ranks.
        grid = grid_for(corpus_size)
        at_k: dict[str, Any] = {}
        for k in grid:
            complete = sum(
                1 for row in task_rows
                if row["k_star"] is not None and row["k_star"] <= k)
            role_avail: dict[str, dict[str, int]] = {}
            for role in ROLES:
                have = [row for row in task_rows if row["role_records"].get(role)]
                got = sum(
                    1 for row in have
                    if all((row["ranks"][rid]["fusion"] is not None
                            and row["ranks"][rid]["fusion"] <= k)
                           for rid in row["role_records"][role]))
                role_avail[role] = {
                    "tasks_with_role": len(have), "available": got,
                    "p_available": round(got / len(have), 4) if have else None}
            taxonomy: Counter = Counter()
            for row in task_rows:
                for rid, r in row["ranks"].items():
                    if r["fusion"] is not None and r["fusion"] <= k:
                        continue
                    label = cause_of(r["fusion"], r["bm25"], r["dense"], k)
                    taxonomy[{"BELOW_BUDGET": "CUT_OFF",
                              "FUSION_DISPLACEMENT": "FUSION_DISPLACED",
                              "UNRANKED": "UNRANKED",
                              "BOTH_RETRIEVERS_MISS": "UNRANKED",
                              "LEXICAL_MISS": "CUT_OFF",
                              "DENSE_MISS": "CUT_OFF"}.get(label, label)] += 1
            at_k[str(k)] = {
                "k": k, "k_over_n": round(k / corpus_size, 5),
                "ces": round(complete / len(task_rows), 4),
                "role_availability": role_avail,
                "missing_record_taxonomy": dict(taxonomy),
            }

        stars = sorted(r["k_star"] for r in task_rows if r["k_star"] is not None)
        unreachable = sum(1 for r in task_rows if r["k_star"] is None)

        def pct(values: list[int], q: float) -> int | None:
            if not values:
                return None
            idx = min(int(q * len(values)), len(values) - 1)
            return values[idx]

        per_scale[scale] = {
            "corpus_size": corpus_size, "tasks": len(task_rows),
            "grid": grid,
            "k_star": {
                "reachable_tasks": len(stars), "unreachable_tasks": unreachable,
                "median": statistics.median(stars) if stars else None,
                "p75": pct(stars, 0.75), "p90": pct(stars, 0.90),
                "p95": pct(stars, 0.95), "max": stars[-1] if stars else None,
            },
            "rho_star": {
                "median": round(statistics.median(stars) / corpus_size, 5) if stars else None,
                "p75": round(pct(stars, 0.75) / corpus_size, 5) if stars else None,
                "p90": round(pct(stars, 0.90) / corpus_size, 5) if stars else None,
                "p95": round(pct(stars, 0.95) / corpus_size, 5) if stars else None,
            },
            "at_k": at_k,
            "per_task": task_rows,
        }
        ks = per_scale[scale]["k_star"]
        print(f"  {scale}: N={corpus_size}  median k*={ks['median']}  "
              f"p90 k*={ks['p90']}  unreachable={unreachable}")

    report = {
        "schema_version": "b3-retrieval-calibration-v1",
        "suite": "b3_calibration_v1",
        "fusion": "frozen_rrf (authoritative)", "k_rrf": C4_RRF_K,
        "retrieval_only": ("S2 is not involved, no HRM was run, no rho was "
                           "selected and no engineering target was set."),
        "method": ("k* per task is the deepest FUSION rank among its required "
                   "records, measured at full corpus depth. Every grid point is "
                   "a threshold on those exact ranks, so the curves are exact "
                   "rather than sampled, and linearity of k(N) is measured "
                   "rather than assumed."),
        "grid_definition": {"absolute": list(ABSOLUTE_DEPTHS),
                            "fractions": list(FRACTIONS),
                            "rule": "union per scale, deduplicated, k <= N"},
        "scales": per_scale,
    }
    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_b3/calibration/retrieval_scaling.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'Scale':<10}{'N':>6}{'CES@50':>9}{'med k*':>8}{'p90 k*':>8}"
          f"{'p90 k*/N':>10}{'Bridge@50':>11}{'Term@50':>9}")
    for scale in SCALES:
        s = per_scale[scale]
        a50 = s["at_k"].get("50")
        br = a50["role_availability"]["bridge"]["p_available"] if a50 else None
        te = a50["role_availability"]["terminal"]["p_available"] if a50 else None
        ces50 = f"{a50['ces']:.3f}" if a50 else "-"
        bridge50 = f"{br:.3f}" if br is not None else "-"
        term50 = f"{te:.3f}" if te is not None else "-"
        print(f"  {scale:<10}{s['corpus_size']:>6}{ces50:>9}"
              f"{str(s['k_star']['median']):>8}{str(s['k_star']['p90']):>8}"
              f"{s['rho_star']['p90']:>10.4f}{bridge50:>11}{term50:>9}")

    print(f"\n  written: {out}")
    print("  STOP: no rho chosen, no engineering targets set, S2 not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
