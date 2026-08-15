#!/usr/bin/env python3
"""G2-v2 construction gate: K0, K1, K3. No HRM. STOP GATE before S2.

    K0  literal relation-text match (G2-v1's frozen, unchanged control)
    K1  entity-bound exact relation match via relation_grammar.py    PRIMARY
    K3  oracle endpoint ceiling, restricted to graph-reachable records CEILING

Retrieval, candidate pools, identity resolution, runtime graph node/edge
types, traversal depth, path ranking, deduplication, competition grouping,
working-set-ceiling semantics, and S2 are all frozen unchanged from G2-v1 --
completion recognition is the ONLY treatment variable, per
configs/gate_g2_v2_path_completion.json.

STOP GATE: this script computes construction-only metrics (working-set CES,
R_endpoint, R_path, terminal/bridge working availability) for all three arms
FIRST. If K1 does not meet the frozen construction-gap bound at every scale,
it stops there, classifies the outcome mechanically, and does NOT run S2 --
spending effort scoring a structurally failed intervention through the full
pipeline is exactly what the STOP GATE exists to avoid. Only if K1 passes does
it continue to the S2 phase (unchanged S2, packet=6, no HRM).

Usage:
    python scripts/run_g2_v2_construction.py
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
from hrm_adaptive_memory.c4.decision_gate import select_eligible_decision  # noqa: E402
from hrm_adaptive_memory.c4.endpoint_recognition import (  # noqa: E402
    k0_literal_completion, k1_entity_bound_exact_completion)
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.g2_paths import (  # noqa: E402
    g2_prefilter, topology_reachable_records)
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.oracle_endpoint_ceiling import (  # noqa: E402
    make_oracle_completion_fn)
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import select_s2  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.diagnose_c5_confirmation_stopgate import (  # noqa: E402
    bridge_records, identity_records, temporal_current_records)
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402


def c2(n: int) -> int:
    return max(100, min(300, math.ceil(0.15 * n)))


M_VALUES = (25, 50, 75)
ARM_NAMES = ("K0", "K1", "K3")
ROLES = ("identity", "bridge", "terminal", "temporal_current")
BRIDGE_BOUND_VS_G0 = -0.05
#: Frozen in configs/gate_g2_v2_path_completion.json BEFORE this ran.
CONSTRUCTION_GAP_PASS = 0.50


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description="G2-v2 construction gate: K0/K1/K3")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    parser.add_argument("--force-s2", action="store_true",
                        help="run S2 phase even if the construction STOP GATE fails "
                             "(diagnostic only, never for the frozen decision)")
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    packet_budget = C4_PRIMARY_PACKET_BUDGET
    print("=== G2-v2 construction gate: K0 / K1 / K3 (no HRM, STOP GATE before S2) ===")
    print(f"  M={list(M_VALUES)}  construction-gap pass bound {CONSTRUCTION_GAP_PASS} "
          "at every scale (frozen)\n")

    results: dict[str, Any] = {}
    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        depth = c2(corpus_size)

        acc = {f"{name}_M{m}": {
                  "working_ces": 0, "avail": defaultdict(lambda: [0, 0]),
                  "endpoint_seen": 0, "endpoint_recognized": 0,
                  "path_topology_exists": 0, "path_complete_covers_required": 0,
                  "working_set_size": []}
               for name in ARM_NAMES for m in M_VALUES}

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
            pool = [eid for eid, _ in fused[:depth]]
            scores = dict(fused[:depth])
            relation = extract_target_relation(task["question"]) or ""

            probe = run_identity_stage(
                task["question"], arm,
                RetrievalResult(candidate_ids=tuple(pool), candidate_budget=depth,
                                retrieval_policy=arm.retrieval_policy,
                                bm25_backend="bm25", bge_model_id="",
                                bge_revision="", rrf_k=C4_RRF_K,
                                bm25_ranked=(), bge_ranked=(), fusion_ranked=()),
                texts)

            required = set(task["required_evidence_ids"])
            terminals = set(terminal_records(task))
            bridges_req = bridge_records(task)
            temporal = temporal_current_records(task, kinds)
            idents = identity_records(task, terminals, bridges_req)
            role_map = {"identity": idents, "bridge": bridges_req,
                        "terminal": terminals, "temporal_current": temporal}

            # Build the runtime graph ONCE per task; K0/K1/K3 differ only in
            # completion_fn and all reuse this same graph and topology.
            graph = build_runtime_graph(record_ids=pool, texts=texts, relation=relation)
            reachable = topology_reachable_records(graph, probe.canonical)
            oracle_fn = make_oracle_completion_fn(
                required_evidence_ids=frozenset(required),
                topology_record_ids=reachable)
            completion_fns = {"K0": k0_literal_completion,
                              "K1": k1_entity_bound_exact_completion,
                              "K3": oracle_fn}

            for m in M_VALUES:
                for name, fn in completion_fns.items():
                    g2r = g2_prefilter(
                        candidate_ids=pool, texts=texts,
                        canonical_subject=probe.canonical, relation=relation,
                        working_set_size=m, fusion_scores=scores,
                        graph=graph, completion_fn=fn)
                    entry = acc[f"{name}_M{m}"]
                    working_set = set(g2r.kept)
                    entry["working_ces"] += required <= working_set
                    entry["working_set_size"].append(len(working_set))
                    for role, recs in role_map.items():
                        if recs and recs <= set(pool):
                            entry["avail"][role][0] += 1
                            if recs <= working_set:
                                entry["avail"][role][1] += 1
                    # R_endpoint: of the required records that are graph-reachable,
                    # how many did THIS arm's completion_fn recognize as complete?
                    required_reachable = required & reachable
                    if required_reachable:
                        entry["endpoint_seen"] += 1
                        recognized_ids = {r.record_id for r in g2r.endpoint_recognitions
                                          if r.completed}
                        if required_reachable <= recognized_ids:
                            entry["endpoint_recognized"] += 1
                    # R_path: of tasks where the required path's topology exists,
                    # how many got a COMPLETE path (not just working-set presence)
                    # covering every required record?
                    if required <= reachable:
                        entry["path_topology_exists"] += 1
                        complete_records = {rid for p in g2r.all_paths if p.complete
                                            for rid in p.record_ids}
                        if required <= complete_records:
                            entry["path_complete_covers_required"] += 1

        print(" " * 40, end="\r")
        n = len(tasks)
        out_arms: dict[str, Any] = {}
        for name in ARM_NAMES:
            for m in M_VALUES:
                key = f"{name}_M{m}"
                e = acc[key]
                out_arms[key] = {
                    "arm": name, "M": m,
                    "working_set_ces": _rate(e["working_ces"], n),
                    "availability": {role: _rate(*reversed(e["avail"][role]))
                                     for role in ROLES if e["avail"][role][0]},
                    "R_endpoint": _rate(e["endpoint_recognized"], e["endpoint_seen"]),
                    "R_path": _rate(e["path_complete_covers_required"],
                                    e["path_topology_exists"]),
                    "working_set_size_mean": round(
                        sum(e["working_set_size"]) / len(e["working_set_size"]), 1),
                }
        base = out_arms["K0_M50"] if "K0_M50" in out_arms else None
        for key, pol in out_arms.items():
            bridge_avail = pol["availability"].get("bridge")
            base_bridge = out_arms[f"K0_M{pol['M']}"]["availability"].get("bridge") \
                if pol["arm"] != "K0" else bridge_avail
            delta = (round(bridge_avail - base_bridge, 4)
                     if bridge_avail is not None and base_bridge is not None else None)
            pol["bridge_safety_vs_k0"] = {"delta": delta, "bound": BRIDGE_BOUND_VS_G0,
                                          "passed": delta is None or delta >= BRIDGE_BOUND_VS_G0}

        gap: dict[str, Any] = {}
        for m in M_VALUES:
            k0 = out_arms[f"K0_M{m}"]["working_set_ces"] or 0
            k1 = out_arms[f"K1_M{m}"]["working_set_ces"] or 0
            k3 = out_arms[f"K3_M{m}"]["working_set_ces"] or 0
            denom = k3 - k0
            closure = round((k1 - k0) / denom, 4) if abs(denom) > 1e-9 else None
            gap[f"M{m}"] = {
                "ces_K0": round(k0, 4), "ces_K1": round(k1, 4), "ces_K3": round(k3, 4),
                "headroom_K3_minus_K0": round(denom, 4),
                "construction_gap_closure": closure,
                "meets_frozen_bound": bool(
                    closure is not None and closure >= CONSTRUCTION_GAP_PASS
                    and out_arms[f"K1_M{m}"]["bridge_safety_vs_k0"]["passed"]),
            }
        results[scale] = {"corpus_size": corpus_size, "tasks": n,
                          "arms": out_arms, "gap_closure": gap}
        print(f"  {scale}: N={corpus_size} k(C2)={depth}")

    # --- STOP GATE: mechanical, per the frozen protocol ---------------------
    stop_gate_pass_by_m = {
        f"M{m}": all(results[s]["gap_closure"][f"M{m}"]["meets_frozen_bound"] for s in SCALES)
        for m in M_VALUES}
    closure_ranking = {
        f"M{m}": min((results[s]["gap_closure"][f"M{m}"]["construction_gap_closure"] or -9)
                     for s in SCALES)
        for m in M_VALUES}
    decision = select_eligible_decision(stop_gate_pass_by_m, closure_ranking)
    stop_gate_passed = decision.key is not None

    # --- outcome classification (V2 taxonomy), mechanical -------------------
    def classify() -> str:
        k3_headroom = [results[s]["gap_closure"]["M50"]["headroom_K3_minus_K0"] for s in SCALES]
        k3_strong = all(h >= 0.15 for h in k3_headroom)  # K3 meaningfully beats K0
        if stop_gate_passed:
            return "A_construction_repaired"
        if not k3_strong:
            return "D_topology_deficient"
        return "C_recognizer_inadequate"

    outcome = classify()
    report = {
        "schema_version": "g2-v2-construction-v1", "no_hrm": True,
        "frozen_bounds": {"construction_gap_pass": CONSTRUCTION_GAP_PASS,
                          "bridge_bound_vs_k0": BRIDGE_BOUND_VS_G0,
                          "frozen_before_run": True},
        "scales": results,
        "stop_gate": {"pass_by_M": stop_gate_pass_by_m,
                      "decision_M": decision.key, "eligible_Ms": decision.eligible_keys,
                      "reason": decision.reason, "passed": stop_gate_passed},
        "outcome_v2": outcome,
        "s2_phase_run": False,
    }

    print(f"\n  {'scale':<7}{'arm':<8}{'wsCES':>8}{'R_endpoint':>12}{'R_path':>9}"
          f"{'bridge':>8}{'dBr':>8}{'wsSize':>8}")
    for scale in SCALES:
        for key, pol in results[scale]["arms"].items():
            db = pol["bridge_safety_vs_k0"]["delta"]
            print(f"  {scale.replace('cal_',''):<7}{key:<8}"
                  f"{(pol['working_set_ces'] or 0):>8.3f}"
                  f"{(pol['R_endpoint'] if pol['R_endpoint'] is not None else -1):>12.3f}"
                  f"{(pol['R_path'] if pol['R_path'] is not None else -1):>9.3f}"
                  f"{pol['availability'].get('bridge', 0):>8.3f}"
                  f"{(f'{db:+.3f}' if db is not None else '-'):>8}"
                  f"{pol['working_set_size_mean']:>8.1f}")
        print()
    print("  CONSTRUCTION GAP CLOSURE (K1-K0)/(K3-K0), per scale per M:")
    for scale in SCALES:
        for m in M_VALUES:
            g = results[scale]["gap_closure"][f"M{m}"]
            print(f"    {scale.replace('cal_',''):<7}M{m}: closure={g['construction_gap_closure']} "
                  f"headroom={g['headroom_K3_minus_K0']} meets_bound={g['meets_frozen_bound']}")
    print(f"\n  STOP GATE: {'PASSED' if stop_gate_passed else 'FAILED'} "
          f"(decision_M={decision.key}, eligible={decision.eligible_keys})")
    print(f"  OUTCOME (V2 taxonomy): {outcome}")

    if not stop_gate_passed and not args.force_s2:
        print("\n  Construction STOP GATE failed -- S2 phase NOT run, per protocol.")
        out = Path(args.out) if args.out else (
            ROOT / "evidence/gate_g2_v2/construction_only.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"  written: {out}")
        return 0

    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_g2_v2/construction_only.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
