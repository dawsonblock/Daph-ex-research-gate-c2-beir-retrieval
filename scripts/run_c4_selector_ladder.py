#!/usr/bin/env python3
"""Sprint B2 selector ladder S0-S4, per configs/gate_c4_selector_v1.json.

Selection-only: no HRM. The PRIMARY frozen criterion -- EXACT-and-bridged
answer-support retention -- is a selection property and needs no generation.
Downstream Q remains a required safety gate and is measured separately, since
it needs the GPU.

Every arm sees IDENTICAL candidate pools (R1_max_reciprocal fusion, budget 50)
and the same packet budget, so the only thing varying is selector treatment.

    S0  frozen s2c_with_s0_fallback (delegated, cannot drift)
    S1  + bridge-aware terminal-answer protection for EXACT identity
    S2  + connectivity constraint
    S3  + temporal-current precedence
    S4  oracle selector ceiling over the same pool

Usage:
    python scripts/run_c4_selector_ladder.py [--split development]
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
    C4_CANDIDATE_BUDGET, C4_PRIMARY_PACKET_BUDGET, C4_RRF_K)
from hrm_adaptive_memory.c4.fusion import max_reciprocal  # noqa: E402
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import run_query_stage  # noqa: E402
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import (  # noqa: E402
    select_s1, select_s2, select_s3)
from hrm_adaptive_memory.c4.contracts import RetrievalResult  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw, s5_oracle  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.diagnose_c4_selector_eligibility import (  # noqa: E402
    task_shape, terminal_records)
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records)

PROTOCOL = ROOT / "configs/gate_c4_selector_v1.json"
ARMS_ORDER = ("S0", "S1", "S2", "S3", "S4")


def build_pool(task: dict, arm, records, depth: int, budget: int):
    """R1_max_reciprocal pool -- identical for every arm."""
    _state, query = run_query_stage(task["question"], arm)
    bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
    bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
    a = [e.evidence_id for e in asyncio.run(bm25.search(query.rendered_query, k=depth)).evidence]
    b = [e.evidence_id for e in asyncio.run(bge.search(query.rendered_query, k=depth)).evidence]
    fused = max_reciprocal([a, b], C4_RRF_K, budget)
    return query, [eid for eid, _ in fused], dict(fused)


def _pct(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def _bootstrap_lcb(groups: dict[str, list[float]], iterations: int = 2000,
                   seed: int = 12345) -> float:
    import random
    keys = sorted(groups)
    if not keys:
        return 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        picked = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [v for g in picked for v in g]
        if flat:
            means.append(sum(flat) / len(flat))
    means.sort()
    return round(means[int(0.025 * len(means))], 4) if means else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="B2 selector ladder (no HRM)")
    parser.add_argument("--split", default="development")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    tasks, evidence, texts = load_split(args.split)
    if args.limit:
        tasks = tasks[:args.limit]
    records = to_index_records(evidence)
    arm = ARMS[args.arm_for_queries]
    cand_budget, packet_budget = C4_CANDIDATE_BUDGET, C4_PRIMARY_PACKET_BUDGET

    print(f"=== B2 selector ladder ({protocol['protocol_id']}) ===")
    print(f"  split={args.split}  tasks={len(tasks)}  fusion=R1_max_reciprocal")
    print(f"  candidate_budget={cand_budget}  packet_budget={packet_budget} "
          f"(FIXED for every arm)\n")

    ces = defaultdict(int)
    answer_ret = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # arm->group->[hit,n]
    role_ret = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    packet_sizes = defaultdict(list)
    disconnected = defaultdict(int)
    receipts: list[dict[str, Any]] = []
    per_task_primary: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict[str, str]] = {}

    for index, task in enumerate(tasks, 1):
        if index % 25 == 0 or index == len(tasks):
            print(f"  {index}/{len(tasks)}...", end="\r", flush=True)

        query, pool, scores = build_pool(task, arm, records, len(records), cand_budget)
        retrieval = RetrievalResult(
            candidate_ids=tuple(pool), candidate_budget=cand_budget,
            retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
            bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
            bm25_ranked=(), bge_ranked=(), fusion_ranked=())
        identity = run_identity_stage(task["question"], arm, retrieval, texts)

        candidates = [{"document_id": eid} for eid in pool]
        resolved_q = task["question"]
        if identity.surface and identity.canonical:
            resolved_q = resolved_q.replace(identity.surface, identity.canonical)

        def frozen(budget: int, _q=resolved_q) -> list[str]:
            """The certified selector, at whatever budget is left."""
            if identity.status in ("EXACT", "RESOLVED") and identity.canonical:
                return s2c_chain_plus_relation(
                    candidates, budget=budget, question=_q, texts=texts)
            return s0_raw(candidates, budget=budget)

        required = set(task["required_evidence_ids"])
        terminals = set(terminal_records(task))
        shape = task_shape(task)
        group = f"{identity.status}_{shape}"
        task_id = task["task_id"]
        meta[task_id] = {"family": task["family"],
                         "entity_regime": task["metadata"]["entity_regime"],
                         "group": group}

        common = dict(identity_status=identity.status, question=task["question"],
                      canonical_subject=identity.canonical, candidate_ids=pool,
                      texts=texts, budget=packet_budget, frozen_select=frozen,
                      fusion_scores=scores)

        selections: dict[str, list[str]] = {"S0": list(frozen(packet_budget))}
        s1_sel, s1_receipt = select_s1(**common)
        selections["S1"] = s1_sel
        s2_sel, _r2, d2 = select_s2(**common)
        selections["S2"] = s2_sel
        s3_sel, _r3, d3 = select_s3(**common)
        selections["S3"] = s3_sel
        selections["S4"] = s5_oracle(
            candidates, budget=packet_budget,
            required=[e for e in task["required_evidence_ids"] if e in set(pool)])

        disconnected["S2"] += d2["disconnected_in_packet"]
        disconnected["S3"] += d3["disconnected_in_packet"]
        if s1_receipt:
            receipts.append({"task_id": task_id, "group": group,
                             "protected_is_required":
                                 s1_receipt.protected_record_id in required,
                             "protected_is_terminal":
                                 s1_receipt.protected_record_id in terminals,
                             **s1_receipt.summary()})

        available = required <= set(pool)
        for name, selected in selections.items():
            chosen = set(selected)
            packet_sizes[name].append(len(selected))
            if available:
                ces[name] += required <= chosen
            # answer-support retention: terminal records kept, on tasks where
            # they were available at all
            if terminals and terminals <= set(pool):
                hit = terminals <= chosen
                answer_ret[name]["ALL"][0] += hit
                answer_ret[name]["ALL"][1] += 1
                answer_ret[name][group][0] += hit
                answer_ret[name][group][1] += 1
                answer_ret[name][identity.status][0] += hit
                answer_ret[name][identity.status][1] += 1
                if group == "EXACT_bridged":
                    per_task_primary[name][task_id] = 1.0 if hit else 0.0
            # identity / bridge retention, for the structural safety gate
            oracle = task.get("_oracle_metadata") or {}
            bridge_records = {e["record_id"] for e in (oracle.get("proof_edges") or [])
                             if e.get("target") == oracle.get("latent_bridge")}
            ident_records = required - terminals - bridge_records
            for label, group_records in (("identity", ident_records),
                                         ("bridge", bridge_records)):
                if group_records and group_records <= set(pool):
                    role_ret[name][label][0] += group_records <= chosen
                    role_ret[name][label][1] += 1

    n = len(tasks)
    print(" " * 30, end="\r")

    report: dict[str, Any] = {
        "schema_version": "c4-selector-ladder-v1",
        "protocol_id": protocol["protocol_id"],
        "split": args.split, "task_count": n,
        "fusion": "R1_max_reciprocal",
        "candidate_budget": cand_budget, "packet_budget": packet_budget,
        "selection_only": (
            "No HRM. The PRIMARY frozen criterion (EXACT_bridged answer-support "
            "retention) is a selection property. Downstream Q remains a required "
            "safety gate and is NOT measured here."),
        "arms": {},
    }
    for name in ARMS_ORDER:
        ar = answer_ret[name]
        report["arms"][name] = {
            "selected_ces": _pct(ces[name], n),
            "answer_retention": {
                g: _pct(v[0], v[1]) for g, v in sorted(ar.items())},
            "identity_retention": _pct(*role_ret[name]["identity"]),
            "bridge_retention": _pct(*role_ret[name]["bridge"]),
            "max_packet_size": max(packet_sizes[name]) if packet_sizes[name] else 0,
            "disconnected_in_packet": disconnected.get(name, None),
            "promotable": name in ("S1", "S2", "S3"),
        }

    # Primary criterion: EXACT_bridged answer retention, S1 vs S0
    prim = {}
    for name in ("S1", "S2", "S3"):
        deltas: dict[str, list[float]] = defaultdict(list)
        for task_id, value in per_task_primary[name].items():
            deltas[meta[task_id]["family"]].append(
                value - per_task_primary["S0"].get(task_id, 0.0))
        flat = [v for g in deltas.values() for v in g]
        prim[name] = {
            "exact_bridged_retention_delta_vs_S0":
                round(sum(flat) / len(flat), 4) if flat else 0.0,
            "bootstrap_lcb": _bootstrap_lcb(deltas),
            "n_exact_bridged": len(flat),
        }
    report["primary_criterion_exact_bridged"] = prim
    report["s1_protection_receipts"] = receipts

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_selector_ladder.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    groups = ["ALL", "EXACT", "RESOLVED", "EXACT_bridged", "EXACT_unbridged",
              "RESOLVED_bridged", "RESOLVED_unbridged"]
    print(f"  {'arm':<5}{'CES':>7}" + "".join(f"{g[:15]:>17}" for g in groups))
    for name in ARMS_ORDER:
        a = report["arms"][name]
        row = "".join(f"{a['answer_retention'].get(g, 0):>16.1%} " for g in groups)
        print(f"  {name:<5}{a['selected_ces']:>6.1%} {row}")

    print(f"\n  {'arm':<5}{'identity ret':>14}{'bridge ret':>12}"
          f"{'max packet':>12}{'disconnected':>14}")
    for name in ARMS_ORDER:
        a = report["arms"][name]
        d = a["disconnected_in_packet"]
        print(f"  {name:<5}{a['identity_retention']:>13.1%}"
              f"{a['bridge_retention']:>12.1%}{a['max_packet_size']:>12}"
              f"{(d if d is not None else '-'):>14}")

    print("\n  PRIMARY criterion -- EXACT_bridged answer retention vs S0:")
    for name, p in prim.items():
        print(f"    {name}: delta={p['exact_bridged_retention_delta_vs_S0']:+.4f}"
              f"  LCB={p['bootstrap_lcb']:+.4f}  n={p['n_exact_bridged']}")

    print(f"\n  S1 protections fired: {len(receipts)}")
    if receipts:
        from collections import Counter
        print(f"    by reason : {dict(Counter(r['protection_reason'] for r in receipts))}")
        print(f"    protected record was REQUIRED: "
              f"{sum(r['protected_is_required'] for r in receipts)}/{len(receipts)}")
        print(f"    protected record was TERMINAL: "
              f"{sum(r['protected_is_terminal'] for r in receipts)}/{len(receipts)}")

    print(f"\n  written: {out}")
    print("\n  Downstream Q is a REQUIRED gate and is not measured here (needs GPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
