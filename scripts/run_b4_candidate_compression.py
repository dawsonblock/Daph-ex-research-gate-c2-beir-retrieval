#!/usr/bin/env python3
"""B4 structural candidate compression: P0-P3. No HRM.

Retrieval + selection only, criteria frozen in gate_b4_candidate_compression_v1
before this ran.

    P0  k=50 -> S2                                    historical baseline
    P1  C2 expanded -> S2 directly                    known overload control
    P2  C2 expanded -> structural prefilter to M -> S2  PRIMARY
    P3  C2 expanded -> ORACLE prefilter to M -> S2      PREFILTER CEILING

The question: can compression preserve the availability B3 bought while cutting
structural competition enough that UNCHANGED S2 regains its discrimination?

S2 is untouched in every arm. The prefilter is the only new mechanism, and it
deliberately does a DIFFERENT job -- structure as admissibility, retrieval score
as ranking -- because ranking by S2's own one-hop heuristic would reproduce the
B3 failure one stage earlier.

Packet budget stays 6. Raising it might improve results while erasing the
question.

Usage:
    python scripts/run_b3_selector_pressure.py   (B3, kept for reference)
    python scripts/run_b4_candidate_compression.py
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
from hrm_adaptive_memory.c4.prefilter import (  # noqa: E402
    oracle_prefilter, structural_prefilter)
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

#: C2, the frozen expanded-retrieval policy under test.
def c2(n: int) -> int:
    return max(100, min(300, math.ceil(0.15 * n)))


#: Working-set sizes. Small prospective set, not a grid.
M_VALUES = (25, 50, 75)
BASELINE = "P0_k50"
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


def build_arms(corpus_size: int) -> dict[str, dict[str, Any]]:
    """P0-P3 for one scale. P2/P3 are instantiated once per M."""
    arms: dict[str, dict[str, Any]] = {
        "P0_k50": {"k": 50, "prefilter": None},
        "P1_C2_direct": {"k": c2(corpus_size), "prefilter": None},
    }
    for m in M_VALUES:
        arms[f"P2_prefilter_M{m}"] = {"k": c2(corpus_size),
                                      "prefilter": "structural", "M": m}
        arms[f"P3_oracle_M{m}"] = {"k": c2(corpus_size),
                                   "prefilter": "oracle", "M": m}
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description="B4 candidate compression")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    packet_budget = C4_PRIMARY_PACKET_BUDGET
    print("=== B4 structural candidate compression (no HRM) ===")
    print(f"  M values: {list(M_VALUES)}   packet_budget={packet_budget} (FIXED)")
    print(f"  bridge safety bound {BRIDGE_BOUND} vs {BASELINE}, at EVERY scale\n")

    results: dict[str, Any] = {}
    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        arms = build_arms(corpus_size)
        depth = max(a["k"] for a in arms.values())

        acc = {name: {"cand_ces": 0, "pre_ces": 0, "sel_ces": 0,
                      "role": defaultdict(lambda: [0, 0]),
                      "pre_role": defaultdict(lambda: [0, 0]),
                      "exact_bridged": [0, 0], "max_packet": 0,
                      "competition": defaultdict(list),
                      "prefilter": defaultdict(list)}
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

            for name, spec in arms.items():
                k = spec["k"]
                pool = [eid for eid, _ in fused[:k]]
                scores = dict(fused[:k])
                pool_set = set(pool)
                entry = acc[name]
                entry["cand_ces"] += required <= pool_set

                # --- compression -------------------------------------------
                if spec["prefilter"] == "structural":
                    identity_probe = run_identity_stage(
                        task["question"], arm,
                        RetrievalResult(candidate_ids=tuple(pool),
                                        candidate_budget=k,
                                        retrieval_policy=arm.retrieval_policy,
                                        bm25_backend="bm25", bge_model_id="",
                                        bge_revision="", rrf_k=C4_RRF_K,
                                        bm25_ranked=(), bge_ranked=(),
                                        fusion_ranked=()), texts)
                    pf = structural_prefilter(
                        candidate_ids=pool, texts=texts,
                        question=task["question"],
                        canonical_subject=identity_probe.canonical,
                        working_set_size=spec["M"], fusion_scores=scores)
                    working = pf.kept
                    for key, value in pf.summary().items():
                        entry["prefilter"][key].append(value)
                elif spec["prefilter"] == "oracle":
                    pf = oracle_prefilter(
                        candidate_ids=pool,
                        required=list(task["required_evidence_ids"]),
                        working_set_size=spec["M"])
                    working = pf.kept
                    for key, value in pf.summary().items():
                        entry["prefilter"][key].append(value)
                else:
                    working = pool

                working_set = set(working)
                entry["pre_ces"] += required <= working_set
                for role, recs in role_map.items():
                    if recs and recs <= pool_set:
                        entry["pre_role"][role][0] += 1
                        if recs <= working_set:
                            entry["pre_role"][role][1] += 1

                # --- UNCHANGED S2 on the working set ----------------------
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

                selected, receipt, diag = select_s2(
                    identity_status=identity.status, question=task["question"],
                    canonical_subject=identity.canonical, candidate_ids=working,
                    texts=texts, budget=packet_budget, frozen_select=frozen,
                    fusion_scores=scores)
                chosen = set(selected)
                entry["max_packet"] = max(entry["max_packet"], len(selected))
                if required <= working_set:
                    entry["sel_ces"] += required <= chosen
                for role, recs in role_map.items():
                    if recs and recs <= working_set:
                        entry["role"][role][0] += 1
                        if recs <= chosen:
                            entry["role"][role][1] += 1
                if terminals and terminals <= working_set and bridges_req \
                        and identity.status == "EXACT":
                    entry["exact_bridged"][0] += 1
                    if terminals <= chosen:
                        entry["exact_bridged"][1] += 1
                comp = competition_counts(working, texts, task["question"],
                                          identity.canonical)
                comp["protection_eligible_count"] = 1 if receipt else 0
                for key, value in comp.items():
                    entry["competition"][key].append(value)

        print(" " * 40, end="\r")
        n = len(tasks)
        out_arms: dict[str, Any] = {}
        for name, spec in arms.items():
            e = acc[name]
            out_arms[name] = {
                "k": spec["k"], "M": spec.get("M"),
                "prefilter": spec["prefilter"],
                "candidate_ces": _rate(e["cand_ces"], n),
                "prefilter_ces": _rate(e["pre_ces"], n),
                "prefilter_role_retention": {
                    role: _rate(*reversed(e["pre_role"][role]))
                    for role in ROLES if e["pre_role"][role][0]},
                "selected_ces_given_working": _rate(e["sel_ces"], e["pre_ces"]),
                "role_conditional_retention": {
                    role: _rate(*reversed(e["role"][role]))
                    for role in ROLES if e["role"][role][0]},
                "exact_bridged_retention": _rate(*reversed(e["exact_bridged"])),
                "max_packet_size": e["max_packet"],
                "competition_mean": {k2: round(sum(v) / len(v), 2)
                                     for k2, v in e["competition"].items() if v},
                "structural_competition_ratio": round(
                    sum(e["competition"]["connected_candidate_count"])
                    / len(e["competition"]["connected_candidate_count"])
                    / packet_budget, 2),
                "bridge_competition_ratio": round(
                    sum(e["competition"]["bridge_candidate_count"])
                    / len(e["competition"]["bridge_candidate_count"])
                    / packet_budget, 2),
                "prefilter_mean": {k2: round(sum(v) / len(v), 2)
                                   for k2, v in e["prefilter"].items() if v},
            }
        base = out_arms[BASELINE]
        for name in out_arms:
            if name == BASELINE:
                continue
            pol = out_arms[name]
            deltas = {}
            for role in ROLES:
                if role in pol["role_conditional_retention"] and \
                        role in base["role_conditional_retention"]:
                    deltas[role] = round(
                        pol["role_conditional_retention"][role]
                        - base["role_conditional_retention"][role], 4)
            pol["delta_vs_baseline"] = deltas
            bd = deltas.get("bridge")
            pol["bridge_safety"] = {
                "delta": bd, "bound": BRIDGE_BOUND,
                "passed": bd is None or bd >= BRIDGE_BOUND}
        results[scale] = {"corpus_size": corpus_size, "tasks": n, "arms": out_arms}
        print(f"  {scale}: N={corpus_size} k(C2)={c2(corpus_size)}")

    report = {
        "schema_version": "b4-candidate-compression-v1",
        "no_hrm": True, "s2_unchanged": True, "packet_budget": packet_budget,
        "m_values": list(M_VALUES),
        "primary_safety": {"metric": "delta bridge conditional retention vs P0",
                           "bound": BRIDGE_BOUND,
                           "required": "at EVERY scale, not pooled"},
        "scales": results,
    }
    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_b3/calibration/b4_compression.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'scale':<7}{'arm':<22}{'k':>5}{'M':>5}{'candCES':>9}"
          f"{'preCES':>8}{'selCES|w':>10}{'bridge':>8}{'dBr':>9}{'SCR':>7}{'BCR':>6}")
    for scale in SCALES:
        for name, pol in results[scale]["arms"].items():
            rr = pol["role_conditional_retention"]
            db = pol.get("bridge_safety", {}).get("delta")
            flag = "" if name == BASELINE or pol["bridge_safety"]["passed"] else " F"
            print(f"  {scale.replace('cal_',''):<7}{name:<22}{pol['k']:>5}"
                  f"{(pol['M'] or 0):>5}{pol['candidate_ces']:>9.3f}"
                  f"{pol['prefilter_ces']:>8.3f}"
                  f"{(pol['selected_ces_given_working'] or 0):>10.3f}"
                  f"{rr.get('bridge', 0):>8.3f}"
                  f"{(f'{db:+.3f}' if db is not None else '-'):>9}"
                  f"{pol['structural_competition_ratio']:>7.1f}"
                  f"{pol['bridge_competition_ratio']:>6.1f}{flag}")
        print()
    print(f"  written: {out}\n  No HRM run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
