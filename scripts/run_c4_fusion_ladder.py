#!/usr/bin/env python3
"""Sprint B1: retrieval-only fusion ladder. No HRM, no budget change.

Runs the arms preregistered in configs/gate_c4_retrieval_fusion_v1.json:

    R0  frozen_rrf                 baseline, delegates to the certified path
    R1  max_reciprocal             remove the consensus bonus (parameter-free)
    R2  reserved_slot_interleave   guarantee constituent depth (parameter-free)
    R3  oracle_fusion              ceiling; what any reordering could reach

Every arm sees the SAME constituent BM25 and BGE rankings for the same query,
so the only thing varying is how those two lists are combined. The candidate
budget stays 50 in every arm -- budget scaling is a separate factor (Sprint
B3), deliberately not confounded with this one.

CES is necessary but not sufficient: this measures whether required evidence
REACHES the pool, not whether the selector then keeps it or whether HRM answers
correctly. A CES win here is a precondition for a mechanism win, not a
mechanism win.

Usage:
    python scripts/run_c4_fusion_ladder.py [--split development]
        [--arm-for-queries C4_4] [--out <path>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_CANDIDATE_BUDGET, C4_RRF_K)
from hrm_adaptive_memory.c4.fusion import (  # noqa: E402
    ORACLE_POLICY, POLICIES, oracle_fusion)
from hrm_adaptive_memory.c4.query_stage import run_query_stage  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from scripts.diagnose_c4_retrieval import role_of  # noqa: E402
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records)

PROTOCOL = ROOT / "configs/gate_c4_retrieval_fusion_v1.json"


def constituent_rankings(task: dict, arm, records, depth: int) -> dict[str, list[str]]:
    """The two constituent lists every arm fuses, retrieved once per task.

    Retrieved to `depth` rather than the budget so each policy can see the same
    material the others do; each policy still emits only `budget` candidates.
    """
    _state, query_result = run_query_stage(task["question"], arm)
    query = query_result.rendered_query
    bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
    return {
        "query": query,
        "bm25": [e.evidence_id for e in asyncio.run(bm25.search(query, k=depth)).evidence],
        "bge": [e.evidence_id for e in asyncio.run(bge.search(query, k=depth)).evidence],
    }


def _pct(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _bootstrap_ci(groups: dict[str, list[float]], iterations: int = 2000,
                  seed: int = 12345) -> tuple[float, float]:
    """Paired grouped bootstrap on per-task deltas, resampling GROUPS.

    Grouped rather than per-task because tasks within a family/cluster are not
    independent -- the same generator template produced them. Fixed seed: this
    must replay identically, like every other number in this project.
    """
    import random
    keys = sorted(groups)
    if not keys:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        picked = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [value for group in picked for value in group]
        if flat:
            means.append(sum(flat) / len(flat))
    if not means:
        return (0.0, 0.0)
    means.sort()
    lower = means[int(0.025 * len(means))]
    upper = means[min(int(0.975 * len(means)), len(means) - 1)]
    return (round(lower, 4), round(upper, 4))


def main() -> int:
    parser = argparse.ArgumentParser(description="C4 fusion ladder (retrieval-only)")
    parser.add_argument("--split", default="development",
                        help="development only for mechanism construction; the "
                             "consumed qualification split must not be used to "
                             "select a winner")
    parser.add_argument("--arm-for-queries", default="C4_4",
                        help="arm whose frozen query policy builds the query")
    parser.add_argument("--depth", type=int, default=0,
                        help="constituent retrieval depth; 0 = whole corpus")
    parser.add_argument("--limit", type=int, default=0, help="first N tasks only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    tasks, evidence, _texts = load_split(args.split)
    if args.limit:
        tasks = tasks[:args.limit]
    records = to_index_records(evidence)
    depth = args.depth or len(records)
    budget = C4_CANDIDATE_BUDGET
    arm = ARMS[args.arm_for_queries]
    kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
             for r in evidence}

    arm_names = list(POLICIES) + [ORACLE_POLICY]
    print(f"=== C4 fusion ladder ({protocol['protocol_id']}) ===")
    print(f"  split={args.split}  tasks={len(tasks)}  corpus={len(records)}")
    print(f"  budget={budget} (FIXED in every arm)  k_rrf={C4_RRF_K}  depth={depth}")
    print(f"  arms={arm_names}\n")

    # per-arm accumulators
    ces: dict[str, int] = defaultdict(int)
    role_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    role_totals: dict[str, int] = defaultdict(int)
    displacement_residual: dict[str, int] = defaultdict(int)
    pool_changed: dict[str, int] = defaultdict(int)
    per_task_ces: dict[str, dict[str, float]] = defaultdict(dict)
    task_meta: dict[str, dict[str, str]] = {}

    for index, task in enumerate(tasks, 1):
        if index % 25 == 0 or index == len(tasks):
            print(f"  {index}/{len(tasks)}...", end="\r", flush=True)
        lists = constituent_rankings(task, arm, records, depth)
        bm25, bge = lists["bm25"], lists["bge"]
        required = list(task["required_evidence_ids"])
        task_id = task["task_id"]
        task_meta[task_id] = {"family": task["family"],
                              "entity_regime": task["metadata"]["entity_regime"]}

        # Was a required record inside a constituent's own top-budget? Then a
        # fusion policy losing it is displacement, not depth.
        inside_constituent = {
            eid for eid in required
            if eid in bm25[:budget] or eid in bge[:budget]}

        pools: dict[str, list[str]] = {}
        for name, policy in POLICIES.items():
            pools[name] = [eid for eid, _ in policy([bm25, bge], C4_RRF_K, budget)]
        pools[ORACLE_POLICY] = [
            eid for eid, _ in
            oracle_fusion([bm25, bge], C4_RRF_K, budget, required=required)]

        baseline_pool = set(pools["R0_frozen_rrf"])
        for name, pool in pools.items():
            pool_set = set(pool)
            complete = set(required) <= pool_set
            ces[name] += complete
            per_task_ces[name][task_id] = 1.0 if complete else 0.0
            pool_changed[name] += len(pool_set ^ baseline_pool)
            for eid in required:
                role = role_of(eid, task, kinds)
                if name == arm_names[0]:
                    role_totals[role] += 1
                if eid in pool_set:
                    role_hits[name][role] += 1
                elif eid in inside_constituent:
                    displacement_residual[name] += 1

    n = len(tasks)
    print(" " * 30, end="\r")

    baseline = "R0_frozen_rrf"
    report: dict[str, Any] = {
        "schema_version": "c4-fusion-ladder-v1",
        "protocol_id": protocol["protocol_id"],
        "split": args.split,
        "task_count": n,
        "corpus_records": len(records),
        "candidate_budget": budget,
        "k_rrf": C4_RRF_K,
        "constituent_depth": depth,
        "retrieval_only": (
            "No HRM. CES measures whether required evidence REACHES the pool, "
            "not whether the selector keeps it or the reader answers "
            "correctly. Necessary, not sufficient."),
        "arms": {},
    }

    for name in arm_names:
        deltas_by_family: dict[str, list[float]] = defaultdict(list)
        deltas_by_regime: dict[str, list[float]] = defaultdict(list)
        for task_id, value in per_task_ces[name].items():
            delta = value - per_task_ces[baseline][task_id]
            deltas_by_family[task_meta[task_id]["family"]].append(delta)
            deltas_by_regime[task_meta[task_id]["entity_regime"]].append(delta)

        all_deltas = [v for group in deltas_by_family.values() for v in group]
        report["arms"][name] = {
            "ces_at_budget": _pct(ces[name], n),
            "ces_delta_vs_R0": round(sum(all_deltas) / n, 4) if n else 0.0,
            "ces_delta_ci_family": list(_bootstrap_ci(deltas_by_family)),
            "ces_delta_ci_regime": list(_bootstrap_ci(deltas_by_regime)),
            "displacement_residual": displacement_residual[name],
            "role_recall": {
                role: _pct(role_hits[name][role], role_totals[role])
                for role in sorted(role_totals) if role_totals[role]},
            "mean_pool_symmetric_difference_vs_R0": round(pool_changed[name] / n, 2),
            "per_family_ces_delta": {
                family: round(sum(v) / len(v), 4)
                for family, v in sorted(deltas_by_family.items(),
                                        key=lambda kv: sum(kv[1]) / len(kv[1]))},
            "per_entity_regime_ces_delta": {
                regime: round(sum(v) / len(v), 4)
                for regime, v in sorted(deltas_by_regime.items())},
            "promotable": name != ORACLE_POLICY,
        }

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_fusion_ladder.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"  {'arm':<30}{'CES@50':>9}{'dCES':>9}{'family CI':>18}"
          f"{'regime CI':>18}{'displaced':>11}")
    for name in arm_names:
        a = report["arms"][name]
        fci, rci = a["ces_delta_ci_family"], a["ces_delta_ci_regime"]
        tag = "" if a["promotable"] else "  (ceiling)"
        print(f"  {name:<30}{a['ces_at_budget']:>8.1%}{a['ces_delta_vs_R0']:>+9.1%}"
              f"{f'[{fci[0]:+.3f},{fci[1]:+.3f}]':>18}"
              f"{f'[{rci[0]:+.3f},{rci[1]:+.3f}]':>18}"
              f"{a['displacement_residual']:>11}{tag}")

    print("\n  role recall @50:")
    roles = sorted(role_totals)
    print(f"  {'arm':<30}" + "".join(f"{r[:11]:>13}" for r in roles))
    for name in arm_names:
        rr = report["arms"][name]["role_recall"]
        print(f"  {name:<30}" + "".join(f"{rr.get(r,0):>12.1%} " for r in roles))

    print("\n  worst per-family CES delta vs R0 (safety axis 1):")
    for name in arm_names:
        pf = report["arms"][name]["per_family_ces_delta"]
        worst = min(pf.items(), key=lambda kv: kv[1]) if pf else ("-", 0.0)
        print(f"  {name:<30}{worst[0]:<22}{worst[1]:>+8.1%}")

    print("\n  per-entity-regime CES delta vs R0 (safety axis 2):")
    for name in arm_names:
        pr = report["arms"][name]["per_entity_regime_ces_delta"]
        print(f"  {name:<30}" + "  ".join(f"{k}={v:+.1%}" for k, v in pr.items()))

    print(f"\n  written: {out}")
    print("\n  NOTE: no winner is declared here. freeze_criteria_pending in the "
          "protocol\n  is unset by design -- the promotion threshold and maximum "
          "subgroup\n  regression are owner decisions, not to be improvised or "
          "back-fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
