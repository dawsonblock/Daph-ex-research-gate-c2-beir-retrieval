#!/usr/bin/env python3
"""G2-v3: Reachability Decomposition. Pure diagnosis, no mechanism changes.

G2-v2's K3 oracle endpoint ceiling showed only 0.0-0.013 absolute headroom
over K0 on working-set CES at every scale: even a PERFECT completion
recognizer, restricted to the current graph topology, could not meaningfully
improve the working set. That falsifies "recognition is the bottleneck" and
narrows the question one layer earlier: the graph generally does not put
required evidence on a reachable path in the first place. This script asks
WHY, decomposed across three independent hypotheses, per
configs/gate_g2_v3_reachability_v1.json (frozen before this ran):

    H1 retrieval availability   -- not even in the C2 candidate pool
    H2 entity fragmentation     -- present, but under a surface form the
                                    graph's own normalization doesn't merge
                                    with the subject's node
    H3 topology/edge-construction -- present and correctly canonicalized,
                                    but the corpus expresses the chain through
                                    a relationship co-mention edges can't
                                    represent

Evaluator-only diagnostic boundary: this script reads
task['_oracle_metadata'] directly (surfaces, proof_edges, latent identities).
That is legitimate HERE specifically because this is a standalone analysis
script that produces a report, not a callable importable by any runtime
mechanism -- unlike K3 (hrm_adaptive_memory/c4/oracle_endpoint_ceiling.py),
which IS gated by the runtime leakage wall because it produces a
completion_fn that could in principle be wired into g2_paths.py.

Runs the REAL, unmodified retrieval + build_runtime_graph + enumerate_paths
pipeline (nothing here is a parallel reimplementation of production logic);
it only adds diagnostic BFS and ceiling comparisons on top, never feeding
back into anything scored. No HRM. No mechanism promoted or rejected.

Usage:
    python scripts/diagnose_g2_v3_reachability.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.bridge_extraction import extract_v4_entities  # noqa: E402
from hrm_adaptive_memory.c4.contracts import C4_RRF_K, RetrievalResult  # noqa: E402
from hrm_adaptive_memory.c4.endpoint_recognition import (  # noqa: E402
    k1_entity_bound_exact_completion)
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.g2_paths import enumerate_paths  # noqa: E402
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.runtime_graph import RuntimeGraph, build_runtime_graph  # noqa: E402
from hrm_adaptive_memory.retrieval.canonicalization import _norm  # noqa: E402
from scripts.diagnose_c5_confirmation_stopgate import (  # noqa: E402
    bridge_records, identity_records, temporal_current_records)
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

R_CODES = ("R0_NOT_IN_CANDIDATE_POOL", "R1_ENTITY_EXTRACTION_MISSING",
          "R2_IDENTITY_FRAGMENTED", "R3_DISCONNECTED_COMPONENT",
          "R4_REQUIRES_GT2_HOPS", "R6_REACHABLE_BUT_NOT_ENUMERATED",
          "R7_ENUMERATED_NOT_COMPLETED", "R8_COMPLETE")
HOP_HORIZONS = (1, 2, 3, 4, "unlimited")
PRODUCTION_MAX_HOPS = 2


def c2(n: int) -> int:
    return max(100, min(300, math.ceil(0.15 * n)))


def diagnostic_bfs_hops(graph: RuntimeGraph, subject_norm: str) -> dict[str, int]:
    """UNBOUNDED hop distance from the subject, diagnostic only -- does not
    touch or reuse the production MAX_HOPS-bounded traversal."""
    if not subject_norm:
        return {}
    hops = {subject_norm: 0}
    frontier = {subject_norm}
    h = 0
    while frontier and h < 25:
        h += 1
        nxt: set[str] = set()
        for entity in frontier:
            for neighbour in graph.neighbours(entity):
                if neighbour not in hops:
                    hops[neighbour] = h
                    nxt.add(neighbour)
        frontier = nxt
    return hops


def oracle_topology_bfs(
    proof_edges: list[dict], pool: set[str], latent_subject: str,
) -> set[str]:
    """Reachable record_ids under the ORACLE's true relation edges, but
    restricted to edges whose OWN record is in the candidate pool -- isolates
    edge/relation-type richness from retrieval availability. Never exposed to
    any runtime mechanism; operates over abstract latent identifiers, not
    extracted surface text, so it is also independent of extraction noise."""
    usable = [pe for pe in proof_edges if pe["record_id"] in pool]
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pe in usable:
        adjacency[pe["source"]].append((pe["target"], pe["record_id"]))
    reachable_records: set[str] = set()
    visited_nodes = {latent_subject}
    frontier = {latent_subject}
    while frontier:
        nxt: set[str] = set()
        for node in frontier:
            for target, record_id in adjacency.get(node, []):
                reachable_records.add(record_id)
                if target not in visited_nodes:
                    visited_nodes.add(target)
                    nxt.add(target)
        frontier = nxt
    return reachable_records


def main() -> int:
    parser = argparse.ArgumentParser(description="G2-v3 reachability decomposition")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    print("=== G2-v3 reachability decomposition (diagnosis only, no mechanism run) ===\n")

    per_scale: dict[str, Any] = {}
    pooled_stage_counts: Counter = Counter()
    pooled_r2_subtype: Counter = Counter()
    pooled_r2_subtype_by_role: dict[str, Counter] = defaultdict(Counter)
    pooled_hop_samples: list[int | None] = []
    pooled_role_stage: dict[str, Counter] = defaultdict(Counter)
    pooled_identity_ceiling = {"current_reachable": 0, "perfect_merge_reachable": 0, "total": 0}
    pooled_topology_ceiling = {"current_reachable": 0, "oracle_topology_reachable": 0, "total": 0}

    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        depth = c2(corpus_size)

        stage_counts: Counter = Counter()
        r2_subtype_counts: Counter = Counter()
        r2_subtype_by_role: dict[str, Counter] = defaultdict(Counter)
        hop_samples: list[int | None] = []
        role_stage: dict[str, Counter] = defaultdict(Counter)
        identity_ceiling = {"current_reachable": 0, "perfect_merge_reachable": 0, "total": 0}
        topology_ceiling = {"current_reachable": 0, "oracle_topology_reachable": 0, "total": 0}

        for index, task in enumerate(tasks, 1):
            if index % 25 == 0 or index == len(tasks):
                print(f"  {scale}: {index}/{len(tasks)}", end="\r", flush=True)
            question = task["question"]
            _state, query = run_query_stage(question, arm)
            bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
            bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
            a = [e.evidence_id for e in
                 asyncio.run(bm25.search(query.rendered_query, k=depth)).evidence]
            b = [e.evidence_id for e in
                 asyncio.run(bge.search(query.rendered_query, k=depth)).evidence]
            fused = frozen_rrf([a, b], C4_RRF_K, depth)
            pool = [eid for eid, _ in fused[:depth]]
            pool_set = set(pool)
            scores = dict(fused[:depth])
            relation = extract_target_relation(question) or ""

            probe = run_identity_stage(
                question, arm,
                RetrievalResult(candidate_ids=tuple(pool), candidate_budget=depth,
                                retrieval_policy=arm.retrieval_policy,
                                bm25_backend="bm25", bge_model_id="",
                                bge_revision="", rrf_k=C4_RRF_K,
                                bm25_ranked=(), bge_ranked=(), fusion_ranked=()),
                texts)
            subject = _norm(probe.canonical or "")

            # REAL, unmodified production mechanism -- reused, not reimplemented.
            graph = build_runtime_graph(record_ids=pool, texts=texts, relation=relation)
            all_paths, _recognitions = enumerate_paths(
                graph=graph, canonical_subject=probe.canonical, relation=relation,
                texts=texts, fusion_scores=scores,
                completion_fn=k1_entity_bound_exact_completion)
            enumerated_records: set[str] = set()
            complete_records: set[str] = set()
            for p in all_paths:
                enumerated_records |= set(p.record_ids)
                if p.complete:
                    complete_records |= set(p.record_ids)

            hop_of_entity = diagnostic_bfs_hops(graph, subject)

            oracle = task.get("_oracle_metadata") or {}
            surfaces = oracle.get("surfaces") or {}
            # BUG FIX vs the first G2-v3 run: surfaces["subject"] is the surface
            # form the QUESTION used (which is the abbreviation itself when
            # entity_regime=abbreviation, e.g. "BCM-3"), not necessarily the true
            # canonical form most evidence records actually use
            # (surfaces["canonical"], e.g. "Bittern control module"). The first
            # run's oracle_targets omitted "canonical" entirely, so every record
            # correctly written using the canonical form was misclassified as
            # R2_IDENTITY_FRAGMENTED against a target it was never trying to match.
            oracle_targets = {_norm(surfaces[k]) for k in ("subject", "bridge", "canonical")
                              if surfaces.get(k)}
            proof_edges = oracle.get("proof_edges") or []
            latent_subject = oracle.get("latent_subject")

            required = task["required_evidence_ids"]
            terminals = set(terminal_records(task))
            bridges_req = bridge_records(task)
            temporal = temporal_current_records(task, kinds)
            idents = identity_records(task, terminals, bridges_req)

            def role_of(rid: str) -> str:
                if rid in terminals:
                    return "terminal"
                if rid in bridges_req:
                    return "bridge"
                if rid in idents:
                    return "identity"
                if rid in temporal:
                    return "temporal_current"
                return "other"

            for rid in required:
                role = role_of(rid)
                r2_subtype = None
                if rid not in pool_set:
                    stage = "R0_NOT_IN_CANDIDATE_POOL"
                    hop = None
                else:
                    content = texts.get(rid, "")
                    extracted_norm = {_norm(e) for e in extract_v4_entities(content)}
                    if not extracted_norm:
                        stage = "R1_ENTITY_EXTRACTION_MISSING"
                        hop = None
                    else:
                        matched = extracted_norm & oracle_targets
                        if not matched:
                            stage = "R2_IDENTITY_FRAGMENTED"
                            hop = None
                            # Sub-classify: is this extraction-boundary noise (the
                            # extractor's greedy regex over/under-captures the SAME
                            # mention, e.g. "Sparrow module service" instead of
                            # "Sparrow module") or genuine cross-surface fragmentation
                            # (an alias/abbreviation with no textual overlap at all,
                            # which extraction boundary tuning cannot fix)?
                            r2_subtype = "genuine_fragmentation"
                            for oracle_form in oracle_targets:
                                for extracted_form in extracted_norm:
                                    if (oracle_form in extracted_form
                                            or extracted_form in oracle_form):
                                        r2_subtype = "extraction_boundary_near_miss"
                                        break
                                if r2_subtype == "extraction_boundary_near_miss":
                                    break
                        else:
                            entity_key = next(iter(matched))
                            hop = hop_of_entity.get(entity_key)
                            if hop is None:
                                stage = "R3_DISCONNECTED_COMPONENT"
                            elif hop > PRODUCTION_MAX_HOPS:
                                stage = "R4_REQUIRES_GT2_HOPS"
                            elif rid not in enumerated_records:
                                stage = "R6_REACHABLE_BUT_NOT_ENUMERATED"
                            elif rid not in complete_records:
                                stage = "R7_ENUMERATED_NOT_COMPLETED"
                            else:
                                stage = "R8_COMPLETE"

                stage_counts[stage] += 1
                role_stage[role][stage] += 1
                hop_samples.append(hop)
                if r2_subtype is not None:
                    r2_subtype_counts[r2_subtype] += 1
                    r2_subtype_by_role[role][r2_subtype] += 1

                # --- D: identity fragmentation ceiling --------------------
                currently_reachable = stage in (
                    "R4_REQUIRES_GT2_HOPS", "R6_REACHABLE_BUT_NOT_ENUMERATED",
                    "R7_ENUMERATED_NOT_COMPLETED", "R8_COMPLETE")
                perfect_merge_reachable = currently_reachable or stage == "R2_IDENTITY_FRAGMENTED"
                identity_ceiling["total"] += 1
                identity_ceiling["current_reachable"] += int(currently_reachable)
                identity_ceiling["perfect_merge_reachable"] += int(perfect_merge_reachable)

            # --- E: topology ceiling, same pool already computed above -----
            if proof_edges and latent_subject:
                oracle_reachable = oracle_topology_bfs(proof_edges, pool_set, latent_subject)
                for pe in proof_edges:
                    rid = pe["record_id"]
                    topology_ceiling["total"] += 1
                    topology_ceiling["current_reachable"] += int(rid in pool_set)
                    topology_ceiling["oracle_topology_reachable"] += int(rid in oracle_reachable)

        print(" " * 40, end="\r")

        per_scale[scale] = {
            "corpus_size": corpus_size, "tasks": len(tasks),
            "stage_counts": dict(stage_counts),
            "role_stage_counts": {r: dict(c) for r, c in role_stage.items()},
            "hop_ceiling": {
                str(h): round(sum(1 for x in hop_samples
                                  if x is not None and (h == "unlimited" or x <= h))
                             / len(hop_samples), 4)
                for h in HOP_HORIZONS} if hop_samples else {},
            "identity_fragmentation_ceiling": {
                "current_reachable_rate": round(
                    identity_ceiling["current_reachable"] / identity_ceiling["total"], 4)
                if identity_ceiling["total"] else None,
                "perfect_merge_reachable_rate": round(
                    identity_ceiling["perfect_merge_reachable"] / identity_ceiling["total"], 4)
                if identity_ceiling["total"] else None,
                "total": identity_ceiling["total"]},
            "topology_ceiling": {
                "current_reachable_rate": round(
                    topology_ceiling["current_reachable"] / topology_ceiling["total"], 4)
                if topology_ceiling["total"] else None,
                "oracle_topology_reachable_rate": round(
                    topology_ceiling["oracle_topology_reachable"] / topology_ceiling["total"], 4)
                if topology_ceiling["total"] else None,
                "total": topology_ceiling["total"]},
            "r2_subtype_counts": dict(r2_subtype_counts),
            "r2_subtype_by_role": {r: dict(c) for r, c in r2_subtype_by_role.items()},
        }
        pooled_stage_counts.update(stage_counts)
        pooled_hop_samples.extend(hop_samples)
        pooled_r2_subtype.update(r2_subtype_counts)
        for r, c in role_stage.items():
            pooled_role_stage[r].update(c)
        for r, c in r2_subtype_by_role.items():
            pooled_r2_subtype_by_role[r].update(c)
        for k in ("current_reachable", "perfect_merge_reachable", "total"):
            pooled_identity_ceiling[k] += identity_ceiling[k]
        for k in ("current_reachable", "oracle_topology_reachable", "total"):
            pooled_topology_ceiling[k] += topology_ceiling[k]
        print(f"  {scale}: N={corpus_size} k(C2)={depth}  "
              f"required_records_classified={sum(stage_counts.values())}")

    total_required = sum(pooled_stage_counts.values())
    pooled_pct = {code: round(pooled_stage_counts.get(code, 0) / total_required, 4)
                 for code in R_CODES}
    non_r8 = total_required - pooled_stage_counts.get("R8_COMPLETE", 0)
    dominant = max(
        (c for c in R_CODES if c != "R8_COMPLETE"),
        key=lambda c: pooled_stage_counts.get(c, 0))

    pooled_hop_ceiling = {
        str(h): round(sum(1 for x in pooled_hop_samples
                          if x is not None and (h == "unlimited" or x <= h))
                     / len(pooled_hop_samples), 4)
        for h in HOP_HORIZONS} if pooled_hop_samples else {}

    report = {
        "schema_version": "g2-v3-reachability-v1", "no_hrm": True, "no_mechanism_run": True,
        "per_scale": per_scale,
        "pooled": {
            "total_required_records": total_required,
            "stage_counts": dict(pooled_stage_counts),
            "stage_percentages": pooled_pct,
            "dominant_non_r8_stage": dominant if non_r8 else None,
            "dominant_share_of_non_r8": round(
                pooled_stage_counts.get(dominant, 0) / non_r8, 4) if non_r8 else None,
            "hop_ceiling": pooled_hop_ceiling,
            "identity_fragmentation_ceiling": {
                "current_reachable_rate": round(
                    pooled_identity_ceiling["current_reachable"] / pooled_identity_ceiling["total"], 4)
                if pooled_identity_ceiling["total"] else None,
                "perfect_merge_reachable_rate": round(
                    pooled_identity_ceiling["perfect_merge_reachable"] / pooled_identity_ceiling["total"], 4)
                if pooled_identity_ceiling["total"] else None},
            "topology_ceiling": {
                "current_reachable_rate": round(
                    pooled_topology_ceiling["current_reachable"] / pooled_topology_ceiling["total"], 4)
                if pooled_topology_ceiling["total"] else None,
                "oracle_topology_reachable_rate": round(
                    pooled_topology_ceiling["oracle_topology_reachable"] / pooled_topology_ceiling["total"], 4)
                if pooled_topology_ceiling["total"] else None},
            "role_stage_counts": {r: dict(c) for r, c in pooled_role_stage.items()},
            "r2_subtype_counts": dict(pooled_r2_subtype),
            "r2_subtype_by_role": {r: dict(c) for r, c in pooled_r2_subtype_by_role.items()},
        },
    }

    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_g2_v3/reachability_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n  POOLED STAGE DECOMPOSITION (share of all required records):")
    for code in R_CODES:
        print(f"    {code:38}{pooled_pct.get(code, 0):>7.2%}"
              f"  (n={pooled_stage_counts.get(code, 0)})")
    print(f"\n  DOMINANT non-R8 stage: {dominant} "
          f"({report['pooled']['dominant_share_of_non_r8']:.1%} of non-complete records)")

    r2_total = sum(pooled_r2_subtype.values())
    if r2_total:
        near_miss = pooled_r2_subtype.get("extraction_boundary_near_miss", 0)
        genuine = pooled_r2_subtype.get("genuine_fragmentation", 0)
        print(f"\n  R2 SUB-CLASSIFICATION (of {r2_total} R2 records):")
        print(f"    extraction_boundary_near_miss (extractor over/under-captures "
              f"the SAME mention): {near_miss/r2_total:.1%} (n={near_miss})")
        print(f"    genuine_fragmentation (alias/abbreviation, no textual overlap "
              f"at all): {genuine/r2_total:.1%} (n={genuine})")
        print("    by role:")
        for role, counts in report["pooled"]["r2_subtype_by_role"].items():
            role_total = sum(counts.values())
            g = counts.get("genuine_fragmentation", 0)
            print(f"      {role:18} n={role_total:<5} genuine_fragmentation={g/role_total:.1%}")

    print("\n  HOP CEILING (pooled, P(reachable within h)):")
    for h in HOP_HORIZONS:
        print(f"    h={h!s:<10}{pooled_hop_ceiling.get(str(h), 0):.2%}")

    ic = report["pooled"]["identity_fragmentation_ceiling"]
    print(f"\n  IDENTITY FRAGMENTATION CEILING: current={ic['current_reachable_rate']}  "
          f"perfect_merge={ic['perfect_merge_reachable_rate']}")
    tc = report["pooled"]["topology_ceiling"]
    print(f"  TOPOLOGY CEILING: current(pool-membership)={tc['current_reachable_rate']}  "
          f"oracle_relation_topology={tc['oracle_topology_reachable_rate']}")

    print(f"\n  written: {out}\n  No mechanism run. No HRM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
