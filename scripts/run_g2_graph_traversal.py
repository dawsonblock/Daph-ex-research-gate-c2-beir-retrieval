#!/usr/bin/env python3
"""G2: G0, G1_coarse, G1_TYPED_PATH, G2, G3. No HRM.

    G0             k=50 -> S2                              low-pressure baseline
    G1_coarse      C2 -> B4 structural prefilter -> S2      the failed floor
    G1_TYPED_PATH  C2 -> frozen single template -> S2       strongest non-graph baseline
    G2             C2 -> runtime graph path enumeration -> S2   PRIMARY
    G3             C2 -> ORACLE compressor -> S2            the P3 ceiling

G2 differs from G1_TYPED_PATH in three ways, all frozen in
configs/gate_g1_runtime_graph_v1.json before this ran: it enumerates every
distinct subject-anchored path (not one template), it treats distinct bridges
as competing hypotheses ranked lexicographically rather than one flat sort, and
it never pads the working set to M -- if fewer records are structurally
justified, fewer are returned.

Success criteria and the outcome taxonomy (A-E) were frozen in the same
protocol BEFORE this script was scored. This runner computes the classification
mechanically from the recorded instrumentation; it does not narrate a verdict.

Usage:
    python scripts/run_g2_graph_traversal.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.bridge_extraction import (  # noqa: E402
    entity_extractor_config_hash, get_default_boundary_policy,
    set_default_boundary_policy)
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_PRIMARY_PACKET_BUDGET, C4_RRF_K, RetrievalResult)
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.g2_paths import g2_prefilter  # noqa: E402
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.prefilter import (  # noqa: E402
    oracle_prefilter, structural_prefilter)
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import (  # noqa: E402
    _candidates, connectivity_status, one_hop_bridge_entities, select_s2)
from hrm_adaptive_memory.c4.typed_path import typed_path_prefilter  # noqa: E402
from hrm_adaptive_memory.retrieval.canonicalization import _norm  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.diagnose_c5_confirmation_stopgate import (  # noqa: E402
    bridge_records, identity_records, temporal_current_records)
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402


def c2(n: int) -> int:
    """C2, the frozen expanded-retrieval policy carried forward from B3."""
    return max(100, min(300, math.ceil(0.15 * n)))


M_VALUES = (25, 50, 75)
BASELINE = "G0_k50"
COARSE = "G1_coarse_M{}"
TYPED = "G1_TYPED_PATH_M{}"
GRAPH = "G2_M{}"
ORACLE = "G3_oracle_M{}"
ROLES = ("identity", "bridge", "terminal", "temporal_current")
BRIDGE_BOUND_VS_G0 = -0.05
BRIDGE_BOUND_VS_TYPED = -0.02
TERMINAL_BOUND_VS_TYPED = -0.02
#: Frozen in configs/gate_g1_runtime_graph_v1.json BEFORE this ran.
GAP_CLOSURE_PASS = 0.60
INCREMENTAL_CLOSURE_MATERIAL = 0.30


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def competition_counts(pool: list[str], texts: dict[str, str], question: str,
                       canonical: str | None) -> dict[str, int]:
    rows = _candidates(pool, texts, None)
    counts = {"candidate_count": len(pool), "connected_candidate_count": 0,
              "bridge_candidate_count": 0}
    if not canonical:
        return counts
    bridges = one_hop_bridge_entities(rows, canonical)
    counts["bridge_candidate_count"] = len(bridges)
    for row in rows:
        if connectivity_status(row.content, canonical, bridges) != "DISCONNECTED":
            counts["connected_candidate_count"] += 1
    return counts


def build_arms(corpus_size: int) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {BASELINE: {"k": 50, "compressor": None}}
    for m in M_VALUES:
        k = c2(corpus_size)
        arms[COARSE.format(m)] = {"k": k, "compressor": "coarse", "M": m}
        arms[TYPED.format(m)] = {"k": k, "compressor": "typed_path", "M": m}
        arms[GRAPH.format(m)] = {"k": k, "compressor": "g2", "M": m}
        arms[ORACLE.format(m)] = {"k": k, "compressor": "oracle", "M": m}
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description="G2 runtime graph vs typed path vs oracle")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    parser.add_argument("--boundary-policy", default="legacy",
                        choices=("legacy", "grammar_v4"),
                        help="G2-v4E entity-boundary treatment arm. The ONLY "
                             "production change between E0 and E1 runs.")
    args = parser.parse_args()

    # Set ONCE for the whole run: entity extraction happens in runtime_graph,
    # prefilter/structural_signature, typed_path and selector_v2, so threading
    # it per-call would risk an inconsistent mixture across arms.
    set_default_boundary_policy(args.boundary_policy)
    extractor_hash = entity_extractor_config_hash()
    print(f"  entity_extractor_policy={get_default_boundary_policy()} "
          f"config_hash={extractor_hash}")

    arm = ARMS[args.arm_for_queries]
    packet_budget = C4_PRIMARY_PACKET_BUDGET
    print("=== G2 runtime graph path enumeration (no HRM) ===")
    print(f"  M={list(M_VALUES)}  packet={packet_budget} (FIXED)  S2 UNCHANGED")
    print(f"  gap-closure vs coarse pass bound {GAP_CLOSURE_PASS}, "
          f"incremental-over-typed material bound {INCREMENTAL_CLOSURE_MATERIAL} "
          "(both frozen)\n")

    results: dict[str, Any] = {}
    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        arms = build_arms(corpus_size)
        depth = max(a["k"] for a in arms.values())

        acc = {name: {"cand_ces": 0, "pre_ces": 0, "e2e": 0,
                      "role": defaultdict(lambda: [0, 0]),
                      "avail": defaultdict(lambda: [0, 0]),
                      "competition": defaultdict(list),
                      "path": defaultdict(list),
                      "graph_reach": defaultdict(lambda: [0, 0]),
                      "latency_ms": []}
               for name in arms}

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
            relation = extract_target_relation(task["question"]) or ""

            for name, spec in arms.items():
                k = spec["k"]
                pool = [eid for eid, _ in fused[:k]]
                scores = dict(fused[:k])
                pool_set = set(pool)
                entry = acc[name]
                entry["cand_ces"] += required <= pool_set

                probe = run_identity_stage(
                    task["question"], arm,
                    RetrievalResult(candidate_ids=tuple(pool), candidate_budget=k,
                                    retrieval_policy=arm.retrieval_policy,
                                    bm25_backend="bm25", bge_model_id="",
                                    bge_revision="", rrf_k=C4_RRF_K,
                                    bm25_ranked=(), bge_ranked=(),
                                    fusion_ranked=()), texts)

                t0 = time.perf_counter()
                if spec["compressor"] == "coarse":
                    working = structural_prefilter(
                        candidate_ids=pool, texts=texts,
                        question=task["question"],
                        canonical_subject=probe.canonical,
                        working_set_size=spec["M"], fusion_scores=scores).kept
                elif spec["compressor"] == "typed_path":
                    graph = build_runtime_graph(record_ids=pool, texts=texts,
                                                relation=relation)
                    tp = typed_path_prefilter(
                        candidate_ids=pool, texts=texts,
                        canonical_subject=probe.canonical, relation=relation,
                        working_set_size=spec["M"], fusion_scores=scores,
                        graph=graph)
                    working = tp.kept
                elif spec["compressor"] == "g2":
                    # stage 2 of 4: h<=2 graph reachability, measured on the very
                    # graph this arm will compress, before any selection
                    _g = build_runtime_graph(record_ids=pool, texts=texts,
                                             relation=relation)
                    _anchor = _norm(probe.canonical or "")
                    _seen = {_anchor} if _anchor else set()
                    _frontier = set(_seen)
                    for _ in range(2):
                        _nxt = set()
                        for _e in _frontier:
                            for _n in _g.neighbours(_e):
                                if _n not in _seen:
                                    _seen.add(_n); _nxt.add(_n)
                        _frontier = _nxt
                    for _role, _recs in role_map.items():
                        if _recs and _recs <= pool_set:
                            entry["graph_reach"][_role][0] += 1
                            if all(any(_x in _seen for _x in
                                       _g.entities_by_record.get(_r, frozenset()))
                                   for _r in _recs):
                                entry["graph_reach"][_role][1] += 1
                    g2r = g2_prefilter(
                        candidate_ids=pool, texts=texts,
                        canonical_subject=probe.canonical, relation=relation,
                        working_set_size=spec["M"], fusion_scores=scores)
                    working = g2r.kept
                    for key, value in g2r.diagnostics().items():
                        entry["path"][key].append(value)
                elif spec["compressor"] == "oracle":
                    working = oracle_prefilter(
                        candidate_ids=pool,
                        required=list(task["required_evidence_ids"]),
                        working_set_size=spec["M"]).kept
                else:
                    working = pool
                entry["latency_ms"].append((time.perf_counter() - t0) * 1000)

                working_set = set(working)
                entry["pre_ces"] += required <= working_set
                for role, recs in role_map.items():
                    if recs and recs <= pool_set:
                        entry["avail"][role][0] += 1
                        if recs <= working_set:
                            entry["avail"][role][1] += 1

                # --- UNCHANGED S2 -----------------------------------------
                retrieval = RetrievalResult(
                    candidate_ids=tuple(working), candidate_budget=len(working),
                    retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                    bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                    bm25_ranked=(), bge_ranked=(), fusion_ranked=())
                identity = run_identity_stage(task["question"], arm, retrieval, texts)
                candidates = [{"document_id": eid} for eid in working]
                resolved_q = task["question"]
                if identity.surface and identity.canonical:
                    resolved_q = resolved_q.replace(identity.surface, identity.canonical)

                def frozen(budget: int, _q=resolved_q, _c=candidates, _i=identity):
                    if _i.status in ("EXACT", "RESOLVED") and _i.canonical:
                        return s2c_chain_plus_relation(
                            _c, budget=budget, question=_q, texts=texts)
                    return s0_raw(_c, budget=budget)

                selected, _receipt, _diag = select_s2(
                    identity_status=identity.status, question=task["question"],
                    canonical_subject=identity.canonical, candidate_ids=working,
                    texts=texts, budget=packet_budget, frozen_select=frozen,
                    fusion_scores=scores)
                chosen = set(selected)
                entry["e2e"] += required <= chosen
                for role, recs in role_map.items():
                    if recs and recs <= working_set:
                        entry["role"][role][0] += 1
                        if recs <= chosen:
                            entry["role"][role][1] += 1
                for key, value in competition_counts(
                        working, texts, task["question"], identity.canonical).items():
                    entry["competition"][key].append(value)

        print(" " * 40, end="\r")
        n = len(tasks)
        out_arms: dict[str, Any] = {}
        for name, spec in arms.items():
            e = acc[name]
            connected = e["competition"]["connected_candidate_count"]
            out_arms[name] = {
                "k": spec["k"], "M": spec.get("M"), "compressor": spec["compressor"],
                "candidate_ces": _rate(e["cand_ces"], n),
                "working_set_ces": _rate(e["pre_ces"], n),
                "selected_ces_end_to_end": _rate(e["e2e"], n),
                # Explicit stage names. The earlier "candidate_pool_availability"
                # field was a mislabeled duplicate of working-role survival (it
                # reused the same counter); it is REMOVED here rather than left
                # as a known semantic bug. True stage-1 candidate availability is
                # emitted separately below, over task count rather than over the
                # pool-available subset.
                "candidate_role_availability": {
                    role: _rate(e["avail"][role][0], len(tasks))
                    for role in ROLES if e["avail"][role][0]},
                "graph_role_reachability": {role: _rate(*reversed(e["graph_reach"][role]))
                                 for role in ROLES if e["graph_reach"][role][0]},
                "working_role_survival": {role: _rate(*reversed(e["avail"][role]))
                                 for role in ROLES if e["avail"][role][0]},
                "s2_role_retention_given_working": {
                    role: _rate(*reversed(e["role"][role]))
                    for role in ROLES if e["role"][role][0]},
                "availability__DEPRECATED_use_working_role_survival": {
                    role: _rate(*reversed(e["avail"][role]))
                    for role in ROLES if e["avail"][role][0]},
                "conditional_retention": {role: _rate(*reversed(e["role"][role]))
                                          for role in ROLES if e["role"][role][0]},
                "structural_competition_ratio": round(
                    sum(connected) / len(connected) / packet_budget, 2) if connected else None,
                "working_set_size_mean": round(
                    sum(e["competition"]["candidate_count"])
                    / len(e["competition"]["candidate_count"]), 1),
                "construction_latency_ms_mean": round(
                    sum(e["latency_ms"]) / len(e["latency_ms"]), 3) if e["latency_ms"] else None,
                "path_diagnostics_mean": {k2: round(sum(v) / len(v), 3)
                                          for k2, v in e["path"].items() if v},
            }
        base = out_arms[BASELINE]
        for name, pol in out_arms.items():
            if name == BASELINE:
                continue
            deltas = {r: round(pol["conditional_retention"][r]
                               - base["conditional_retention"][r], 4)
                      for r in ROLES
                      if r in pol["conditional_retention"]
                      and r in base["conditional_retention"]}
            pol["delta_vs_baseline"] = deltas
            bd = deltas.get("bridge")
            pol["bridge_safety_vs_g0"] = {"delta": bd, "bound": BRIDGE_BOUND_VS_G0,
                                          "passed": bd is None or bd >= BRIDGE_BOUND_VS_G0}

        # --- gap closure + G2-specific criteria, per matched M -------------
        gap: dict[str, Any] = {}
        for m in M_VALUES:
            coarse = out_arms[COARSE.format(m)]
            typed = out_arms[TYPED.format(m)]
            g2p = out_arms[GRAPH.format(m)]
            oracle = out_arms[ORACLE.format(m)]
            ces_coarse = coarse["selected_ces_end_to_end"] or 0
            ces_typed = typed["selected_ces_end_to_end"] or 0
            ces_g2 = g2p["selected_ces_end_to_end"] or 0
            ces_oracle = oracle["selected_ces_end_to_end"] or 0

            denom_co = ces_oracle - ces_coarse
            gap_closure = round((ces_g2 - ces_coarse) / denom_co, 4) if abs(denom_co) > 1e-9 else None
            denom_ty = ces_oracle - ces_typed
            incremental = round((ces_g2 - ces_typed) / denom_ty, 4) if abs(denom_ty) > 1e-9 else None

            bridge_g2 = g2p["conditional_retention"].get("bridge")
            bridge_typed = typed["conditional_retention"].get("bridge")
            bridge_delta_vs_typed = (round(bridge_g2 - bridge_typed, 4)
                                     if bridge_g2 is not None and bridge_typed is not None else None)
            term_g2 = g2p["availability"].get("terminal")
            term_typed = typed["availability"].get("terminal")
            term_delta_vs_typed = (round(term_g2 - term_typed, 4)
                                   if term_g2 is not None and term_typed is not None else None)
            scr_g2 = g2p["structural_competition_ratio"]
            scr_typed = typed["structural_competition_ratio"]
            competition_ok = (scr_g2 is not None and scr_typed is not None
                              and scr_g2 <= scr_typed)

            criteria = {
                "gap_closure_pass": gap_closure is not None and gap_closure >= GAP_CLOSURE_PASS,
                "incremental_material": incremental is not None and incremental >= INCREMENTAL_CLOSURE_MATERIAL,
                "bridge_non_regression_vs_typed": bridge_delta_vs_typed is None or bridge_delta_vs_typed >= BRIDGE_BOUND_VS_TYPED,
                "bridge_safety_vs_g0": g2p["bridge_safety_vs_g0"]["passed"],
                "terminal_safety_vs_typed": term_delta_vs_typed is None or term_delta_vs_typed >= TERMINAL_BOUND_VS_TYPED,
                "competition_non_increase_vs_typed": competition_ok,
            }
            gap[f"M{m}"] = {
                "ces_coarse": round(ces_coarse, 4), "ces_typed": round(ces_typed, 4),
                "ces_g2": round(ces_g2, 4), "ces_oracle": round(ces_oracle, 4),
                "gap_closure_vs_coarse": gap_closure,
                "incremental_closure_vs_typed": incremental,
                "bridge_delta_vs_typed": bridge_delta_vs_typed,
                "terminal_delta_vs_typed": term_delta_vs_typed,
                "structural_competition_ratio_g2": scr_g2,
                "structural_competition_ratio_typed": scr_typed,
                "criteria": criteria,
                "all_criteria_met": all(criteria.values()),
            }
        results[scale] = {"corpus_size": corpus_size, "tasks": n,
                          "arms": out_arms, "gap_closure": gap}
        print(f"  {scale}: N={corpus_size} k(C2)={c2(corpus_size)}")

    # --- mechanical outcome classification, per the frozen taxonomy --------
    def classify(m: int) -> str:
        per_scale = [results[s]["gap_closure"][f"M{m}"] for s in SCALES]
        if all(g["all_criteria_met"] for g in per_scale):
            return "A_STRONG_CLOSE"
        bridge_avail_g2 = [results[s]["arms"][GRAPH.format(m)]["availability"].get("bridge")
                          for s in SCALES]
        bridge_avail_typed = [results[s]["arms"][TYPED.format(m)]["availability"].get("bridge")
                             for s in SCALES]
        bridge_regressed = any(
            g2v is not None and tv is not None and g2v < tv - 0.02
            for g2v, tv in zip(bridge_avail_g2, bridge_avail_typed))
        if bridge_regressed:
            return "C_CANNOT_PRESERVE_BRIDGE_AVAILABILITY"
        bridge_ret_flat = all(
            g["bridge_delta_vs_typed"] is not None and abs(g["bridge_delta_vs_typed"]) < 0.02
            for g in per_scale)
        bridge_avail_up = all(
            g2v is not None and tv is not None and g2v > tv + 0.02
            for g2v, tv in zip(bridge_avail_g2, bridge_avail_typed))
        if bridge_avail_up and bridge_ret_flat:
            return "B_BRIDGES_FOUND_BUT_S2_DROPS_THEM"
        incrementals = [g["incremental_closure_vs_typed"] for g in per_scale
                        if g["incremental_closure_vs_typed"] is not None]
        if incrementals and max(incrementals) - min(incrementals) < INCREMENTAL_CLOSURE_MATERIAL \
                and all(v < INCREMENTAL_CLOSURE_MATERIAL for v in incrementals):
            return "E_TYPED_APPROX_G2"
        return "D_PATHS_EXIST_BUT_WRONG_ONES_RANKED"

    verdict = {f"M{m}": {"outcome": classify(m),
                         "all_scales_pass": all(
                             results[s]["gap_closure"][f"M{m}"]["all_criteria_met"]
                             for s in SCALES)}
              for m in M_VALUES}
    best = max(M_VALUES, key=lambda m: (
        verdict[f"M{m}"]["all_scales_pass"],
        min((results[s]["gap_closure"][f"M{m}"]["gap_closure_vs_coarse"] or -9)
            for s in SCALES)))

    report = {
        "schema_version": "g2-graph-traversal-v1",
        "no_hrm": True, "s2_unchanged": True, "packet_budget": packet_budget,
        "m_values": list(M_VALUES),
        "frozen_bounds": {
            "gap_closure_pass": GAP_CLOSURE_PASS,
            "incremental_closure_material": INCREMENTAL_CLOSURE_MATERIAL,
            "bridge_bound_vs_g0": BRIDGE_BOUND_VS_G0,
            "bridge_bound_vs_typed": BRIDGE_BOUND_VS_TYPED,
            "terminal_bound_vs_typed": TERMINAL_BOUND_VS_TYPED,
            "frozen_before_run": True},
        "REPORTING_RULE": "selected_ces_end_to_end is primary everywhere",
        "entity_extractor_policy": get_default_boundary_policy(),
        "entity_extractor_config_hash": extractor_hash,
        "scales": results, "verdict_by_M": verdict,
        "decision": verdict[f"M{best}"]["outcome"], "decision_at_M": best,
    }
    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_g1/g2_graph_traversal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'scale':<7}{'arm':<16}{'M':>4}{'candCES':>9}{'wsCES':>8}"
          f"{'E2E':>8}{'bridge':>8}{'SCR':>7}{'wsSize':>8}{'lat(ms)':>9}")
    for scale in SCALES:
        for name, pol in results[scale]["arms"].items():
            print(f"  {scale.replace('cal_',''):<7}{name:<16}{(pol['M'] or 0):>4}"
                  f"{pol['candidate_ces']:>9.3f}{pol['working_set_ces']:>8.3f}"
                  f"{pol['selected_ces_end_to_end']:>8.3f}"
                  f"{pol['conditional_retention'].get('bridge', 0):>8.3f}"
                  f"{(pol['structural_competition_ratio'] or 0):>7.1f}"
                  f"{pol['working_set_size_mean']:>8.1f}"
                  f"{(pol['construction_latency_ms_mean'] or 0):>9.2f}")
        print()
    print(f"  {'GAP CLOSURE (G2 vs coarse->oracle)  |  INCREMENTAL (G2 vs typed->oracle)':<70}")
    for scale in SCALES:
        for m in M_VALUES:
            g = results[scale]["gap_closure"][f"M{m}"]
            print(f"  {scale.replace('cal_',''):<7}M{m:<4}"
                  f"gap={g['gap_closure_vs_coarse']}  incr={g['incremental_closure_vs_typed']}  "
                  f"all_met={g['all_criteria_met']}")
        print()
    for m in M_VALUES:
        print(f"  M{m}: {verdict[f'M{m}']}")
    print(f"\n  DECISION (at M{best}): {report['decision']}")
    print(f"  written: {out}\n  No HRM run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
