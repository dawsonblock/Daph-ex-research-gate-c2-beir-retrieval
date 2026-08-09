#!/usr/bin/env python3
"""G2-v3.1: corrected anatomy + causal decomposition of TWO bottlenecks.

The original G2-v3 interpretation is VOID (oracle_targets omitted
surfaces['canonical']). The corrected evidence shows two comparably-sized
upstream defects rather than one dominant lever, so this script separates
them and measures their interaction BEFORE any production mechanism changes.
Per configs/gate_g2_v31_causal_decomposition_v1.json, frozen before this ran.

Four analyses:

  1. ROLE-SPECIFIC stage tracing. A single R0-R8 progression conflated
     "completes the requested relation" with "is structurally required" --
     an identity/alias record is not defective for failing to assert the
     target relation, which is why 100% of identity records landed in R7.
     Identity, bridge, and terminal roles now have their own stage machines.

  2. IDENTITY AUDIT: R_parse -> R_link -> R_attachment -> R_connectivity, to
     locate exactly where the EXISTING extract_identity_links/alias_links
     machinery loses the chain. If parse is high and connectivity is much
     lower, the defect is graph integration (small repair); if parse is low,
     the repair belongs upstream (larger job).

  3. BRIDGE R0 DECOMPOSITION, reusing diagnose_c4_retrieval.cause_of()
     unchanged -- not reimplemented. Distinguishes a budget/fusion problem
     (cheap) from genuine non-retrieval (architectural).

  4. THREE ORACLE CEILINGS + 2x2 INTERACTION. Marginal failure counts cannot
     reveal whether identity and bridge failures co-occur on the same tasks;
     only the factorial design can.

Reachability definition, deliberately changed from v3: a required record is
reachable if ANY entity it mentions lands within the hop bound of the anchor.
v3 instead required a match against a specific oracle surface, which is what
made it vulnerable to the surface-target bug in the first place. The new
definition mirrors what the path enumerator actually does (records_by_entity
lookups) and depends on no oracle string at all.

No S2. No HRM. No mechanism modified.

Usage:
    python scripts/diagnose_g2_v31_causal_decomposition.py
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
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.g2_paths import enumerate_paths  # noqa: E402
from hrm_adaptive_memory.c4.endpoint_recognition import (  # noqa: E402
    k1_entity_bound_exact_completion)
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph  # noqa: E402
from hrm_adaptive_memory.retrieval.canonicalization import (  # noqa: E402
    _norm, extract_identity_links)
from scripts.diagnose_c4_retrieval import cause_of, rank_of  # noqa: E402
from scripts.diagnose_c5_confirmation_stopgate import bridge_records  # noqa: E402
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

PRODUCTION_MAX_HOPS = 2
PROBE_DEPTH = 1000  # deep probe so bridge R0 causes are attributable


def c2(n: int) -> int:
    return max(100, min(300, math.ceil(0.15 * n)))


class _Row:
    def __init__(self, record_id: str, content: str):
        self.evidence_id = record_id
        self.content = content


def merged_adjacency(graph, latent_map: dict[str, str]):
    """Adjacency with entity nodes optionally merged through ``latent_map``.

    An empty latent_map yields the CURRENT runtime topology (identity arm I0);
    a populated one yields the oracle-identity-merged topology (arm I1/I3).
    """
    def key(entity: str) -> str:
        return latent_map.get(entity, entity)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, rights in graph.entity_links.items():
        for right in rights:
            adjacency[key(left)].add(key(right))
            adjacency[key(right)].add(key(left))
    for left, rights in graph.alias_links.items():
        for right in rights:
            adjacency[key(left)].add(key(right))
            adjacency[key(right)].add(key(left))
    entities_of_record: dict[str, set[str]] = {}
    for record_id, entities in graph.entities_by_record.items():
        entities_of_record[record_id] = {key(e) for e in entities}
    return adjacency, entities_of_record, key


def hops_from(adjacency: dict[str, set[str]], anchor: str, max_h: int = 25) -> dict[str, int]:
    if not anchor:
        return {}
    hops = {anchor: 0}
    frontier = {anchor}
    h = 0
    while frontier and h < max_h:
        h += 1
        nxt = set()
        for entity in frontier:
            for neighbour in adjacency.get(entity, ()):
                if neighbour not in hops:
                    hops[neighbour] = h
                    nxt.add(neighbour)
        frontier = nxt
    return hops


def record_reachable(record_id, entities_of_record, hops, bound=None) -> bool:
    """A record is reachable if ANY entity it mentions is within ``bound`` hops
    of the anchor -- mirrors what the path enumerator actually does.

    ``bound=None`` means UNLIMITED and is implemented as membership in ``hops``,
    NOT as a large sentinel: an earlier version passed bound=10**9 and compared
    ``hops.get(e, 10**9) <= bound``, which is True for entities BFS never
    reached, so "disconnected" was silently reclassified as "merely beyond the
    hop bound". Membership is the only correct unlimited test here.
    """
    entities = entities_of_record.get(record_id, ())
    if bound is None:
        return any(e in hops for e in entities)
    return any(hops.get(e, 10 ** 9) <= bound for e in entities)


def main() -> int:
    parser = argparse.ArgumentParser(description="G2-v3.1 causal decomposition")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    print("=== G2-v3.1 causal decomposition (no S2, no HRM, no mechanism change) ===\n")

    per_scale: dict[str, Any] = {}
    pooled_role_stage: dict[str, Counter] = defaultdict(Counter)
    pooled_identity_audit = Counter()
    pooled_bridge_r0_cause = Counter()
    pooled_cells = Counter()          # 2x2 per-task Q counts
    pooled_tasks = 0
    pooled_overlap = Counter()

    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        depth = c2(corpus_size)

        role_stage: dict[str, Counter] = defaultdict(Counter)
        identity_audit = Counter()
        bridge_r0_cause = Counter()
        cells = Counter()
        overlap = Counter()

        for index, task in enumerate(tasks, 1):
            if index % 25 == 0 or index == len(tasks):
                print(f"  {scale}: {index}/{len(tasks)}", end="\r", flush=True)
            question = task["question"]
            _state, query = run_query_stage(question, arm)
            bm25 = get_cached_backend(CanonicalRetrievalMode.BM25, records)
            bge = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
            bm25_ids = [e.evidence_id for e in
                        asyncio.run(bm25.search(query.rendered_query, k=PROBE_DEPTH)).evidence]
            dense_ids = [e.evidence_id for e in
                         asyncio.run(bge.search(query.rendered_query, k=PROBE_DEPTH)).evidence]
            fused_deep = frozen_rrf([bm25_ids, dense_ids], C4_RRF_K, PROBE_DEPTH)
            fused_ids = [eid for eid, _ in fused_deep]
            pool = fused_ids[:depth]
            pool_set = set(pool)
            scores = dict(fused_deep[:depth])
            relation = extract_target_relation(question) or ""

            probe = run_identity_stage(
                question, arm,
                RetrievalResult(candidate_ids=tuple(pool), candidate_budget=depth,
                                retrieval_policy=arm.retrieval_policy,
                                bm25_backend="bm25", bge_model_id="", bge_revision="",
                                rrf_k=C4_RRF_K, bm25_ranked=(), bge_ranked=(),
                                fusion_ranked=()), texts)

            oracle = task.get("_oracle_metadata") or {}
            surfaces = oracle.get("surfaces") or {}
            latent_subject = oracle.get("latent_subject") or "LATENT_SUBJ"
            latent_bridge = oracle.get("latent_bridge") or "LATENT_BR"
            # subject-surface AND canonical collapse to ONE latent entity --
            # that is precisely the merge the identity ceiling is about.
            latent_map: dict[str, str] = {}
            for field, latent in (("subject", latent_subject),
                                 ("canonical", latent_subject),
                                 ("bridge", latent_bridge)):
                if surfaces.get(field):
                    latent_map[_norm(surfaces[field])] = latent

            required = list(task["required_evidence_ids"])
            terminals = set(terminal_records(task))
            bridges_req = bridge_records(task)
            idents = set(required) - terminals - bridges_req

            def role_of(rid: str) -> str:
                if rid in terminals:
                    return "terminal"
                if rid in bridges_req:
                    return "bridge"
                if rid in idents:
                    return "identity"
                return "other"

            # ---------- the 2x2: identity x bridge-availability -------------
            missing_bridges = [r for r in bridges_req if r not in pool_set]
            pool_b = pool + [r for r in missing_bridges if r in texts]

            arms_2x2 = {
                ("I0", "B0"): (pool, {}),
                ("I1", "B0"): (pool, latent_map),
                ("I0", "B1"): (pool_b, {}),
                ("I1", "B1"): (pool_b, latent_map),
            }
            q_by_cell: dict[tuple[str, str], bool] = {}
            base_state = None
            for (ident_arm, bridge_arm), (arm_pool, arm_latent) in arms_2x2.items():
                g = build_runtime_graph(record_ids=arm_pool, texts=texts, relation=relation)
                adjacency, entities_of_record, key = merged_adjacency(g, arm_latent)
                anchor = key(_norm(probe.canonical or ""))
                hops = hops_from(adjacency, anchor)
                arm_pool_set = set(arm_pool)
                all_reachable = bool(required) and all(
                    rid in arm_pool_set
                    and record_reachable(rid, entities_of_record, hops, PRODUCTION_MAX_HOPS)
                    for rid in required)
                q_by_cell[(ident_arm, bridge_arm)] = all_reachable
                cells[f"{ident_arm}_{bridge_arm}"] += int(all_reachable)
                if (ident_arm, bridge_arm) == ("I0", "B0"):
                    base_state = (g, adjacency, entities_of_record, hops, anchor)

            # per-task overlap classification, the thing marginals cannot show
            i_helps = q_by_cell[("I1", "B0")] and not q_by_cell[("I0", "B0")]
            b_helps = q_by_cell[("I0", "B1")] and not q_by_cell[("I0", "B0")]
            both_only = (q_by_cell[("I1", "B1")] and not q_by_cell[("I1", "B0")]
                         and not q_by_cell[("I0", "B1")])
            if q_by_cell[("I0", "B0")]:
                overlap["already_reachable"] += 1
            elif i_helps and b_helps:
                overlap["either_repair_suffices"] += 1
            elif i_helps:
                overlap["identity_only"] += 1
            elif b_helps:
                overlap["bridge_only"] += 1
            elif both_only:
                overlap["requires_both"] += 1
            else:
                overlap["neither_suffices"] += 1

            graph, adjacency, entities_of_record, hops, anchor = base_state

            # ---------- role-specific stage tracing (on I0/B0) -------------
            enumerated, completed = set(), set()
            all_paths, _rec = enumerate_paths(
                graph=graph, canonical_subject=probe.canonical, relation=relation,
                texts=texts, fusion_scores=scores,
                completion_fn=k1_entity_bound_exact_completion)
            for p in all_paths:
                enumerated |= set(p.record_ids)
                if p.complete:
                    completed |= set(p.record_ids)

            for rid in required:
                role = role_of(rid)
                if rid not in pool_set:
                    role_stage[role]["S0_not_in_candidate_pool"] += 1
                    continue
                content = texts.get(rid, "")
                entities = {_norm(e) for e in extract_v4_entities(content)}
                if role == "identity":
                    # identity records are graded on whether they CONNECT, not
                    # on whether they assert the target relation
                    links = extract_identity_links([_Row(rid, content)])
                    if not links:
                        role_stage[role]["S1_identity_relation_not_parsed"] += 1
                        continue
                    link = links[0]
                    surf_n, canon_n = _norm(link.surface), _norm(link.canonical)
                    in_alias = (canon_n in graph.alias_links.get(surf_n, set())
                                and surf_n in graph.alias_links.get(canon_n, set()))
                    if not in_alias:
                        role_stage[role]["S2_alias_edge_not_emitted"] += 1
                        continue
                    if not (surf_n in graph.records_by_entity
                            or canon_n in graph.records_by_entity):
                        role_stage[role]["S3_no_graph_node_attachment"] += 1
                        continue
                    if hops.get(canon_n) is None and hops.get(surf_n) is None:
                        role_stage[role]["S4_connectivity_not_improved"] += 1
                        continue
                    role_stage[role]["S5_identity_connected"] += 1
                    continue
                # bridge / terminal
                if not entities:
                    role_stage[role]["S1_entity_extraction_missing"] += 1
                    continue
                if not record_reachable(rid, entities_of_record, hops, PRODUCTION_MAX_HOPS):
                    reachable_unlimited = record_reachable(
                        rid, entities_of_record, hops, bound=None)
                    role_stage[role]["S3_beyond_hop_bound" if reachable_unlimited
                                     else "S2_disconnected_from_anchor"] += 1
                    continue
                if rid not in enumerated:
                    role_stage[role]["S4_reachable_not_enumerated"] += 1
                elif role == "terminal" and rid not in completed:
                    role_stage[role]["S5_enumerated_not_recognized"] += 1
                else:
                    role_stage[role]["S6_ok"] += 1

            # ---------- identity audit rates -------------------------------
            if idents and surfaces.get("subject") and surfaces.get("canonical"):
                oracle_surf, oracle_canon = (_norm(surfaces["subject"]),
                                            _norm(surfaces["canonical"]))
                if oracle_surf != oracle_canon:  # only abbreviation/alias regimes
                    for rid in idents:
                        if rid not in pool_set:
                            identity_audit["identity_record_absent_from_pool"] += 1
                            continue
                        identity_audit["in_pool"] += 1
                        links = extract_identity_links([_Row(rid, texts.get(rid, ""))])
                        if not links:
                            continue
                        identity_audit["parser_fired"] += 1
                        pair = {_norm(links[0].surface), _norm(links[0].canonical)}
                        if pair != {oracle_surf, oracle_canon}:
                            continue
                        identity_audit["parse_correct"] += 1
                        s_n, c_n = _norm(links[0].surface), _norm(links[0].canonical)
                        if not (c_n in graph.alias_links.get(s_n, set())
                                and s_n in graph.alias_links.get(c_n, set())):
                            continue
                        identity_audit["link_emitted_bidirectional"] += 1
                        if s_n in graph.records_by_entity and c_n in graph.records_by_entity:
                            identity_audit["both_nodes_have_records"] += 1
                        if hops.get(c_n) is not None:
                            identity_audit["canonical_reachable_from_anchor"] += 1

            # ---------- bridge R0 cause decomposition ----------------------
            for rid in missing_bridges:
                f_rank = rank_of(fused_ids, rid)
                b_rank = rank_of(bm25_ids, rid)
                d_rank = rank_of(dense_ids, rid)
                try:
                    bridge_r0_cause[cause_of(f_rank, b_rank, d_rank, depth)] += 1
                except AssertionError:
                    bridge_r0_cause["IN_POOL_UNEXPECTED"] += 1

        print(" " * 44, end="\r")
        n = len(tasks)
        per_scale[scale] = {
            "corpus_size": corpus_size, "tasks": n,
            "role_stage_counts": {r: dict(c) for r, c in role_stage.items()},
            "identity_audit": dict(identity_audit),
            "bridge_r0_causes": dict(bridge_r0_cause),
            "q_2x2_counts": dict(cells),
            "q_2x2_rates": {k: round(v / n, 4) for k, v in cells.items()},
            "overlap": dict(overlap),
        }
        for r, c in role_stage.items():
            pooled_role_stage[r].update(c)
        pooled_identity_audit.update(identity_audit)
        pooled_bridge_r0_cause.update(bridge_r0_cause)
        pooled_cells.update(cells)
        pooled_overlap.update(overlap)
        pooled_tasks += n
        print(f"  {scale}: N={corpus_size} k(C2)={depth}  tasks={n}")

    q00 = pooled_cells["I0_B0"] / pooled_tasks
    q10 = pooled_cells["I1_B0"] / pooled_tasks
    q01 = pooled_cells["I0_B1"] / pooled_tasks
    q11 = pooled_cells["I1_B1"] / pooled_tasks
    ia = pooled_identity_audit

    def rate(num: str, den: str) -> float | None:
        return round(ia[num] / ia[den], 4) if ia[den] else None

    report = {
        "schema_version": "g2-v3.1-causal-decomposition-v1",
        "no_s2": True, "no_hrm": True, "no_mechanism_change": True,
        "supersedes": "original G2-v3 interpretation (VOID)",
        "per_scale": per_scale,
        "pooled": {
            "tasks": pooled_tasks,
            "role_stage_counts": {r: dict(c) for r, c in pooled_role_stage.items()},
            "identity_audit_counts": dict(ia),
            "identity_audit_rates": {
                "R_parse": rate("parse_correct", "in_pool"),
                "R_link": rate("link_emitted_bidirectional", "parse_correct"),
                "R_attachment": rate("both_nodes_have_records", "link_emitted_bidirectional"),
                "R_connectivity": rate("canonical_reachable_from_anchor",
                                       "link_emitted_bidirectional")},
            "bridge_r0_causes": dict(pooled_bridge_r0_cause),
            "factorial_2x2": {
                "Q_I0_B0": round(q00, 4), "Q_I1_B0": round(q10, 4),
                "Q_I0_B1": round(q01, 4), "Q_I1_B1": round(q11, 4),
                "E_identity": round(q10 - q00, 4),
                "E_bridge": round(q01 - q00, 4),
                "interaction": round(q11 - q10 - q01 + q00, 4)},
            "per_task_overlap": dict(pooled_overlap),
        },
    }
    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_g2_v3/v31_causal_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n  ROLE-SPECIFIC STAGES (pooled):")
    for role in ("identity", "bridge", "terminal"):
        counts = pooled_role_stage.get(role)
        if not counts:
            continue
        total = sum(counts.values())
        print(f"    {role} (n={total}):")
        for stage, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"      {stage:36}{cnt/total:>7.1%}  (n={cnt})")

    print("\n  IDENTITY AUDIT (alias/abbreviation-regime tasks):")
    for label, value in report["pooled"]["identity_audit_rates"].items():
        print(f"    {label:18}{value if value is not None else 'n/a'}")
    print(f"    raw counts: {dict(ia)}")

    print("\n  BRIDGE R0 CAUSES (pooled):")
    tot_r0 = sum(pooled_bridge_r0_cause.values()) or 1
    for cause, cnt in sorted(pooled_bridge_r0_cause.items(), key=lambda x: -x[1]):
        print(f"    {cause:26}{cnt/tot_r0:>7.1%}  (n={cnt})")

    f = report["pooled"]["factorial_2x2"]
    print("\n  FACTORIAL 2x2 (per-task: ALL required records reachable at h<=2):")
    print(f"    Q(I0,B0) baseline        {f['Q_I0_B0']:.4f}")
    print(f"    Q(I1,B0) identity only   {f['Q_I1_B0']:.4f}   E_identity = {f['E_identity']:+.4f}")
    print(f"    Q(I0,B1) bridge only     {f['Q_I0_B1']:.4f}   E_bridge   = {f['E_bridge']:+.4f}")
    print(f"    Q(I1,B1) both            {f['Q_I1_B1']:.4f}   interaction= {f['interaction']:+.4f}")
    print("\n  PER-TASK OVERLAP:")
    for k, v in sorted(pooled_overlap.items(), key=lambda x: -x[1]):
        print(f"    {k:26}{v/pooled_tasks:>7.1%}  (n={v})")
    print(f"\n  written: {out}\n  STOP -- no S2, no HRM, no mechanism modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
