#!/usr/bin/env python3
"""B3 selector-pressure gate: can UNCHANGED S2 compress larger pools safely?

No HRM. Retrieval + selection only. Criteria were frozen in the protocol before
this ran.

Retrieval availability is already settled -- both C1 and C2 clear bridge >= 0.70,
terminal >= 0.90 and candidate CES >= 0.75 at every scale. The open question is
the interaction: does the extra availability survive S2, or does the larger pool
overwhelm it?

    baseline    k=50  + S2   (same scale, so the contrast isolates pool size)
    C1          k=300 + S2
    C2          clip(ceil(0.15N), 100, 300) + S2

PRIMARY SAFETY, frozen: delta P(bridge selected | bridge available) >= -0.05
versus the k=50 baseline, required at EVERY scale independently rather than
pooled. Confirmation #1 measured this at -0.0435 against the same bound, and
candidate expansion is the intervention most likely to push it over.

S2 is applied exactly as frozen -- retuning it for bigger pools would confound
the pressure test with a selector change.

Competition diagnostics are recorded so any bridge degradation can be explained
rather than merely observed: if k=50 offers 2 plausible bridge records and k=300
offers 17, S2's local structural rule becomes under-specified.

Usage:
    python scripts/run_b3_selector_pressure.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_PRIMARY_PACKET_BUDGET, C4_RRF_K, RetrievalResult)
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import (  # noqa: E402
    _candidates, connectivity_status, find_protected_record,
    one_hop_bridge_entities, select_s2)
from hrm_adaptive_memory.retrieval.canonicalization import _norm  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.diagnose_c5_confirmation_stopgate import (  # noqa: E402
    bridge_records, identity_records, temporal_current_records)
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

#: Frozen budget policies. Label-free functions of corpus size only.
POLICIES: dict[str, Any] = {
    "k50_baseline": lambda n: 50,
    "C1_constant_300": lambda n: 300,
    "C2_capped_fraction": lambda n: max(100, min(300, math.ceil(0.15 * n))),
}
BASELINE = "k50_baseline"
ROLES = ("identity", "bridge", "terminal", "temporal_current")
BRIDGE_BOUND = -0.05


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def competition_counts(pool: list[str], texts: dict[str, str], question: str,
                       canonical: str | None) -> dict[str, int]:
    """How many candidates plausibly compete for S2's structural slots.

    Uses only runtime-visible signals, matching what S2 itself may read.
    """
    rows = _candidates(pool, texts, None)
    relation = extract_target_relation(question) or ""
    counts = {"candidate_count": len(pool), "connected_candidate_count": 0,
              "bridge_candidate_count": 0,
              "target_relation_candidate_count": 0}
    if not canonical:
        return counts
    bridges = one_hop_bridge_entities(rows, canonical)
    counts["bridge_candidate_count"] = len(bridges)
    for row in rows:
        if connectivity_status(row.content, canonical, bridges) != "DISCONNECTED":
            counts["connected_candidate_count"] += 1
        if relation and _norm(relation) in _norm(row.content):
            counts["target_relation_candidate_count"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="B3 selector-pressure gate")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    packet_budget = C4_PRIMARY_PACKET_BUDGET
    print("=== B3 selector-pressure gate (no HRM) ===")
    print(f"  policies: {list(POLICIES)}  packet_budget={packet_budget} (FIXED)")
    print(f"  PRIMARY SAFETY: delta bridge conditional retention >= "
          f"{BRIDGE_BOUND} at EVERY scale\n")

    results: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        ks = {name: policy(corpus_size) for name, policy in POLICIES.items()}
        depth = max(ks.values())

        acc: dict[str, Any] = {
            name: {"cand_ces": 0, "sel_ces": 0,
                   "role": defaultdict(lambda: [0, 0]),
                   "exact_bridged": [0, 0], "disconnected": 0,
                   "max_packet": 0, "competition": defaultdict(list)}
            for name in POLICIES}

        for index, task in enumerate(tasks, 1):
            if index % 25 == 0 or index == len(tasks):
                print(f"  {scale}: {index}/{len(tasks)}", end="\r", flush=True)
            _state, query = run_query_stage(task["question"], arm)
            bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
            bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
            a = [e.evidence_id for e in
                 asyncio.run(bm25.search(query.rendered_query, k=depth)).evidence]
            b = [e.evidence_id for e in
                 asyncio.run(bge.search(query.rendered_query, k=depth)).evidence]
            fused = frozen_rrf([a, b], C4_RRF_K, depth)

            required = set(task["required_evidence_ids"])
            terminals = set(terminal_records(task))
            bridges_req = bridge_records(task)
            temporal = temporal_current_records(task, kinds)
            idents = identity_records(task, terminals, bridges_req)
            role_map = {"identity": idents, "bridge": bridges_req,
                        "terminal": terminals, "temporal_current": temporal}

            for name, k in ks.items():
                pool = [eid for eid, _ in fused[:k]]
                scores = dict(fused[:k])
                pool_set = set(pool)
                retrieval = RetrievalResult(
                    candidate_ids=tuple(pool), candidate_budget=k,
                    retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                    bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                    bm25_ranked=(), bge_ranked=(), fusion_ranked=())
                identity = run_identity_stage(task["question"], arm, retrieval, texts)
                candidates = [{"document_id": eid} for eid in pool]
                resolved_q = task["question"]
                if identity.surface and identity.canonical:
                    resolved_q = resolved_q.replace(identity.surface, identity.canonical)

                def frozen(budget: int, _q=resolved_q, _c=candidates, _i=identity):
                    if _i.status in ("EXACT", "RESOLVED") and _i.canonical:
                        return s2c_chain_plus_relation(
                            _c, budget=budget, question=_q, texts=texts)
                    return s0_raw(_c, budget=budget)

                selected, receipt, diag = select_s2(
                    identity_status=identity.status, question=task["question"],
                    canonical_subject=identity.canonical, candidate_ids=pool,
                    texts=texts, budget=packet_budget, frozen_select=frozen,
                    fusion_scores=scores)
                chosen = set(selected)

                entry = acc[name]
                entry["cand_ces"] += required <= pool_set
                if required <= pool_set:
                    entry["sel_ces"] += required <= chosen
                entry["max_packet"] = max(entry["max_packet"], len(selected))
                entry["disconnected"] += diag.get("disconnected_in_packet", 0)
                for role, recs in role_map.items():
                    if recs and recs <= pool_set:
                        entry["role"][role][0] += 1
                        if recs <= chosen:
                            entry["role"][role][1] += 1
                if terminals and terminals <= pool_set and bridges_req \
                        and identity.status == "EXACT":
                    entry["exact_bridged"][0] += 1
                    if terminals <= chosen:
                        entry["exact_bridged"][1] += 1
                comp = competition_counts(pool, texts, task["question"],
                                          identity.canonical)
                comp["protection_eligible_count"] = 1 if receipt else 0
                for key, value in comp.items():
                    entry["competition"][key].append(value)

        print(" " * 40, end="\r")
        n = len(tasks)
        scale_out: dict[str, Any] = {"corpus_size": corpus_size, "tasks": n,
                                     "k": ks, "policies": {}}
        for name in POLICIES:
            e = acc[name]
            scale_out["policies"][name] = {
                "k": ks[name],
                "candidate_ces": _rate(e["cand_ces"], n),
                "selected_ces_given_available": _rate(e["sel_ces"], e["cand_ces"]),
                "role_conditional_retention": {
                    role: {"available": e["role"][role][0],
                           "selected": e["role"][role][1],
                           "retention": _rate(*reversed(e["role"][role]))}
                    for role in ROLES if e["role"][role][0]},
                "exact_bridged_retention": _rate(*reversed(e["exact_bridged"])),
                "disconnected_in_packet": e["disconnected"],
                "max_packet_size": e["max_packet"],
                "competition_mean": {k: round(sum(v) / len(v), 2)
                                     for k, v in e["competition"].items() if v},
            }
        # Deltas vs the same-scale k=50 baseline.
        base = scale_out["policies"][BASELINE]
        for name in POLICIES:
            if name == BASELINE:
                continue
            pol = scale_out["policies"][name]
            pol["delta_vs_baseline"] = {
                "candidate_ces": round(pol["candidate_ces"] - base["candidate_ces"], 4),
                "role_conditional_retention": {
                    role: round(pol["role_conditional_retention"][role]["retention"]
                                - base["role_conditional_retention"][role]["retention"], 4)
                    for role in ROLES
                    if role in pol["role_conditional_retention"]
                    and role in base["role_conditional_retention"]},
            }
            bridge_delta = pol["delta_vs_baseline"]["role_conditional_retention"].get("bridge")
            pol["bridge_safety"] = {
                "delta": bridge_delta, "bound": BRIDGE_BOUND,
                "passed": bridge_delta is None or bridge_delta >= BRIDGE_BOUND}
        results[scale] = scale_out
        print(f"  {scale}: N={corpus_size}  k={ks}")

    report = {
        "schema_version": "b3-selector-pressure-v1",
        "no_hrm": True, "s2_unchanged": True,
        "packet_budget": packet_budget,
        "primary_safety": {
            "metric": "delta P(bridge selected | bridge available) vs k=50+S2",
            "bound": BRIDGE_BOUND, "required": "at EVERY scale, not pooled"},
        "scales": results,
    }
    failures = [(s, name) for s, sc in results.items()
                for name, pol in sc["policies"].items()
                if name != BASELINE and not pol["bridge_safety"]["passed"]]
    report["bridge_safety_failures"] = [{"scale": s, "policy": p} for s, p in failures]
    report["VERDICT"] = ("BRIDGE_SAFETY_PASS" if not failures
                         else "BRIDGE_SAFETY_FAIL")

    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_b3/calibration/selector_pressure.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'Scale':<9}{'Policy':<20}{'k':>5}{'candCES':>9}{'selCES|av':>11}"
          f"{'bridge':>8}{'d-bridge':>10}{'term':>7}{'E+br':>7}{'pkt':>5}")
    for scale in SCALES:
        sc = results[scale]
        for name in POLICIES:
            pol = sc["policies"][name]
            rr = pol["role_conditional_retention"]
            br = rr.get("bridge", {}).get("retention")
            te = rr.get("terminal", {}).get("retention")
            db = (pol.get("delta_vs_baseline", {})
                  .get("role_conditional_retention", {}).get("bridge"))
            flag = "" if name == BASELINE or pol["bridge_safety"]["passed"] else " FAIL"
            print(f"  {scale.replace('cal_',''):<9}{name:<20}{pol['k']:>5}"
                  f"{pol['candidate_ces']:>9.3f}"
                  f"{(pol['selected_ces_given_available'] or 0):>11.3f}"
                  f"{(br if br is not None else 0):>8.3f}"
                  f"{(f'{db:+.4f}' if db is not None else '-'):>10}"
                  f"{(te if te is not None else 0):>7.3f}"
                  f"{(pol['exact_bridged_retention'] or 0):>7.3f}"
                  f"{pol['max_packet_size']:>5}{flag}")

    print(f"\n  VERDICT: {report['VERDICT']}")
    if failures:
        for s, p in failures:
            print(f"    FAIL {p} at {s}")
    print(f"\n  written: {out}")
    print("  No HRM run. Stop and report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
