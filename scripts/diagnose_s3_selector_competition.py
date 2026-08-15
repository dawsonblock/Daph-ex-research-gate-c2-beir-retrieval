#!/usr/bin/env python3
"""S3: why does S2 pick the wrong six records from a working set that already
contains the required evidence?

G2-v4E removed the availability bottleneck cleanly enough to expose this: at
cal_3000/M50 bridge working-set survival is 1.000 while P(bridge selected | in
working set) is 0.015. The evidence arrives and dies in selection.

Everything upstream is pinned: grammar_v4 extraction, C2 retrieval, G2 path
enumeration, M=50 working set, packet 6. The ONLY thing that varies is the
selector. Per configs/gate_s3_selector_competition_v1.json, frozen before this ran.

Separates record RANKING from packet COMPOSITION, because S2 may be scoring
records sensibly while composing an incoherent packet. Diagnostic arms S1/S2/S3
are NOT promotable -- they exist to attribute the failure, not to ship.

No HRM.

Usage:
    python scripts/diagnose_s3_selector_competition.py
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
from hrm_adaptive_memory.c4.bridge_extraction import (  # noqa: E402
    set_default_boundary_policy)
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_PRIMARY_PACKET_BUDGET, C4_RRF_K, RetrievalResult)
from hrm_adaptive_memory.c4.endpoint_recognition import (  # noqa: E402
    k1_entity_bound_exact_completion)
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.g2_paths import g2_prefilter  # noqa: E402
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.prefilter import structural_signature  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import select_s2  # noqa: E402
from hrm_adaptive_memory.retrieval.canonicalization import (  # noqa: E402
    _norm, extract_identity_links)
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.diagnose_c5_confirmation_stopgate import bridge_records  # noqa: E402
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

PACKET = C4_PRIMARY_PACKET_BUDGET
M = 50
ARMS_S3 = ("S0_current", "S1_diversity", "S2_role_reserved", "S3_path_coherent", "S4_oracle")
FAILURES = ("F0_SELECTED", "F1_DROPPED_BY_CAPACITY", "F2_REDUNDANCY_COLLISION",
            "F3_WRONG_PATH_PREFERRED", "F4_CONNECTIVITY_BIAS", "F5_ROLE_IMBALANCE",
            "F6_TIE_OR_ORDER", "F7_OTHER")


def c2(n: int) -> int:
    return max(100, min(300, math.ceil(0.15 * n)))


class _Row:
    def __init__(self, rid, content):
        self.evidence_id = rid
        self.content = content


def runtime_role(rid, texts, canonical, relation) -> str:
    """RUNTIME-inferred packet role -- no oracle. Used only by the S2_role_reserved
    diagnostic arm, which is explicitly not promotable."""
    content = texts.get(rid, "")
    if extract_identity_links([_Row(rid, content)]):
        return "identity"
    norm = _norm(content)
    has_subject = bool(canonical) and _norm(canonical) in norm
    has_relation = bool(relation) and _norm(relation) in norm
    if has_relation:
        return "terminal"
    if has_subject:
        return "bridge"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="S3 selector competition diagnosis")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_default_boundary_policy("grammar_v4")  # pinned upstream, per protocol
    arm = ARMS[args.arm_for_queries]
    print("=== S3 selector competition (grammar_v4 pinned, G2/M50, packet 6, no HRM) ===\n")

    per_scale: dict[str, Any] = {}
    pooled_sel = {a: Counter() for a in ARMS_S3}
    pooled_fail = Counter()
    pooled_comp: dict[str, list] = defaultdict(list)
    pooled_tasks = 0

    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        depth = c2(len(records))
        sel = {a: Counter() for a in ARMS_S3}
        fail = Counter()
        comp: dict[str, list] = defaultdict(list)

        for i, task in enumerate(tasks, 1):
            if i % 25 == 0 or i == len(tasks):
                print(f"  {scale}: {i}/{len(tasks)}", end="\r", flush=True)
            q = task["question"]
            _s, qr = run_query_stage(q, arm)
            bm = get_cached_backend(CanonicalRetrievalMode.BM25, records)
            bg = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
            a = [e.evidence_id for e in asyncio.run(bm.search(qr.rendered_query, k=depth)).evidence]
            b = [e.evidence_id for e in asyncio.run(bg.search(qr.rendered_query, k=depth)).evidence]
            fused = frozen_rrf([a, b], C4_RRF_K, depth)
            pool = [e for e, _ in fused[:depth]]
            scores = dict(fused[:depth])
            relation = extract_target_relation(q) or ""
            probe = run_identity_stage(q, arm, RetrievalResult(
                candidate_ids=tuple(pool), candidate_budget=depth,
                retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                bm25_ranked=(), bge_ranked=(), fusion_ranked=()), texts)
            canonical = probe.canonical

            g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                               canonical_subject=canonical, relation=relation,
                               working_set_size=M, fusion_scores=scores,
                               completion_fn=k1_entity_bound_exact_completion)
            working = g2r.kept
            wset = set(working)
            if not working:
                continue

            required = set(task["required_evidence_ids"])
            terms = set(terminal_records(task)) & wset
            brs = bridge_records(task) & wset
            req_in_ws = required & wset

            identity = run_identity_stage(q, arm, RetrievalResult(
                candidate_ids=tuple(working), candidate_budget=len(working),
                retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                bm25_ranked=(), bge_ranked=(), fusion_ranked=()), texts)
            rq = q
            if identity.surface and identity.canonical:
                rq = rq.replace(identity.surface, identity.canonical)
            cands = [{"document_id": e} for e in working]

            def frozen(budget, _q=rq, _c=cands, _i=identity):
                if _i.status in ("EXACT", "RESOLVED") and _i.canonical:
                    return s2c_chain_plus_relation(_c, budget=budget, question=_q, texts=texts)
                return s0_raw(_c, budget=budget)

            def _doc_id(c):
                """The frozen selectors return bare id strings; earlier callers
                passed their output through untouched so the shape never
                mattered. This diagnostic filters it (to restrict the deduped
                arm to its own subset), so it must handle both shapes."""
                return c["document_id"] if isinstance(c, dict) else c

            def run_s2(cand_ids, budget=PACKET):
                allowed = set(cand_ids)
                s, _r, _d = select_s2(
                    identity_status=identity.status, question=q,
                    canonical_subject=identity.canonical, candidate_ids=cand_ids,
                    texts=texts, budget=budget,
                    frozen_select=lambda bb: [
                        c for c in frozen(bb) if _doc_id(c) in allowed],
                    fusion_scores=scores)
                return [_doc_id(x) for x in s]

            # ---- full S2 ordering (ranking, separated from composition) -----
            full_order = run_s2(working, budget=len(working))
            rank_of = {r: idx + 1 for idx, r in enumerate(full_order)}

            packets: dict[str, list[str]] = {}
            packets["S0_current"] = run_s2(working)

            # S1: suppress duplicate structural signatures, then S2
            seen_sig, deduped = set(), []
            for r in working:
                sig = structural_signature(texts.get(r, ""), canonical, relation)
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                deduped.append(r)
            packets["S1_diversity"] = run_s2(deduped)

            # S2 arm: reserve one slot per runtime-inferred role, then fill with S2
            reserved: list[str] = []
            for want in ("identity", "bridge", "terminal"):
                for r in full_order:
                    if r in reserved:
                        continue
                    if runtime_role(r, texts, canonical, relation) == want:
                        reserved.append(r)
                        break
            rest = [r for r in run_s2(working, budget=len(working)) if r not in reserved]
            packets["S2_role_reserved"] = (reserved + rest)[:PACKET]

            # S3 arm: highest-ranked COMPLETE path's records first, then fill
            complete_paths = [p for p in g2r.retained_paths if p.complete] or \
                             [p for p in g2r.all_paths if p.complete]
            path_first: list[str] = []
            if complete_paths:
                for r in complete_paths[0].record_ids:
                    if r in wset and r not in path_first:
                        path_first.append(r)
            fill = [r for r in full_order if r not in path_first]
            packets["S3_path_coherent"] = (path_first + fill)[:PACKET]

            # S4 oracle ceiling
            oracle_first = [r for r in working if r in required]
            packets["S4_oracle"] = (oracle_first + [
                r for r in full_order if r not in set(oracle_first)])[:PACKET]

            complete_req_path_in_ws = bool(
                complete_paths and required and
                any(required <= set(p.record_ids) for p in complete_paths))

            for name, pk in packets.items():
                chosen = set(pk)
                c = sel[name]
                c["tasks"] += 1
                c["selected_ces"] += int(bool(required) and required <= chosen)
                if brs:
                    c["bridge_den"] += 1
                    c["bridge_num"] += int(brs <= chosen)
                if terms:
                    c["term_den"] += 1
                    c["term_num"] += int(terms <= chosen)
                if complete_req_path_in_ws:
                    c["cpath_den"] += 1
                    c["cpath_num"] += int(any(
                        required <= chosen and required <= set(p.record_ids)
                        for p in complete_paths))
                roles = {runtime_role(r, texts, canonical, relation) for r in pk}
                c["role_coverage"] += len(roles & {"identity", "bridge", "terminal"})
                sigs = [structural_signature(texts.get(r, ""), canonical, relation) for r in pk]
                c["redundancy"] += (len(sigs) - len(set(sigs)))

            # ---- failure attribution on the BASELINE packet ---------------
            base = set(packets["S0_current"])
            base_sigs = [structural_signature(texts.get(r, ""), canonical, relation)
                         for r in packets["S0_current"]]
            base_roles = Counter(runtime_role(r, texts, canonical, relation)
                                 for r in packets["S0_current"])
            for rid in sorted(req_in_ws):
                if rid in base:
                    fail["F0_SELECTED"] += 1
                    continue
                rid_sig = structural_signature(texts.get(rid, ""), canonical, relation)
                rk = rank_of.get(rid)
                if rid_sig in base_sigs:
                    fail["F2_REDUNDANCY_COLLISION"] += 1
                elif complete_paths and any(
                        set(p.record_ids) & base and rid not in set(p.record_ids)
                        for p in complete_paths):
                    fail["F3_WRONG_PATH_PREFERRED"] += 1
                elif base_roles and max(base_roles.values()) >= PACKET - 1:
                    fail["F5_ROLE_IMBALANCE"] += 1
                elif rk is not None and rk <= 2 * PACKET:
                    fail["F1_DROPPED_BY_CAPACITY"] += 1
                elif base_roles.get("other", 0) >= PACKET // 2:
                    fail["F4_CONNECTIVITY_BIAS"] += 1
                elif rk is None:
                    fail["F7_OTHER"] += 1
                else:
                    fail["F6_TIE_OR_ORDER"] += 1

            comp["working_set_size"].append(len(working))
            comp["complete_paths"].append(len(complete_paths))

        print(" " * 44, end="\r")
        def rates(c):
            n = c["tasks"] or 1
            return {"tasks": c["tasks"],
                    "selected_ces": round(c["selected_ces"] / n, 4),
                    "bridge_retention": (round(c["bridge_num"] / c["bridge_den"], 4)
                                         if c["bridge_den"] else None),
                    "terminal_retention": (round(c["term_num"] / c["term_den"], 4)
                                           if c["term_den"] else None),
                    "complete_path_retention": (round(c["cpath_num"] / c["cpath_den"], 4)
                                                if c["cpath_den"] else None),
                    "mean_role_coverage": round(c["role_coverage"] / n, 3),
                    "mean_redundancy": round(c["redundancy"] / n, 3)}
        per_scale[scale] = {"arms": {a: rates(sel[a]) for a in ARMS_S3},
                            "failure_attribution": dict(fail),
                            "mean_working_set": round(
                                sum(comp["working_set_size"]) / max(1, len(comp["working_set_size"])), 1)}
        for a in ARMS_S3:
            pooled_sel[a].update(sel[a])
        pooled_fail.update(fail)
        for k, v in comp.items():
            pooled_comp[k].extend(v)
        pooled_tasks += len(tasks)
        print(f"  {scale}: done")

    def prates(c):
        n = c["tasks"] or 1
        return {"selected_ces": round(c["selected_ces"] / n, 4),
                "bridge_retention": (round(c["bridge_num"] / c["bridge_den"], 4)
                                     if c["bridge_den"] else None),
                "terminal_retention": (round(c["term_num"] / c["term_den"], 4)
                                       if c["term_den"] else None),
                "complete_path_retention": (round(c["cpath_num"] / c["cpath_den"], 4)
                                            if c["cpath_den"] else None),
                "mean_role_coverage": round(c["role_coverage"] / n, 3),
                "mean_redundancy": round(c["redundancy"] / n, 3)}

    pooled = {a: prates(pooled_sel[a]) for a in ARMS_S3}
    base_ces = pooled["S0_current"]["selected_ces"]
    ceil_ces = pooled["S4_oracle"]["selected_ces"]
    for a in ARMS_S3:
        d = ceil_ces - base_ces
        pooled[a]["oracle_gap_closure"] = (
            round((pooled[a]["selected_ces"] - base_ces) / d, 4) if abs(d) > 1e-9 else None)

    total_fail = sum(pooled_fail.values()) or 1
    report = {"schema_version": "s3-selector-competition-v1", "no_hrm": True,
              "upstream_pinned": {"extractor": "grammar_v4", "compressor": "G2", "M": M,
                                  "packet": PACKET},
              "per_scale": per_scale,
              "pooled": {"arms": pooled,
                         "failure_attribution": dict(pooled_fail),
                         "failure_shares": {k: round(v / total_fail, 4)
                                            for k, v in pooled_fail.items()},
                         "mean_working_set": round(
                             sum(pooled_comp["working_set_size"])
                             / max(1, len(pooled_comp["working_set_size"])), 1)}}
    out = Path(args.out) if args.out else ROOT / "evidence/gate_s3/selector_competition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'arm':<20}{'selCES':>8}{'bridge':>8}{'term':>8}{'cpath':>8}"
          f"{'roles':>7}{'redun':>7}{'closure':>9}")
    for a in ARMS_S3:
        p = pooled[a]
        print(f"  {a:<20}{p['selected_ces']:>8.3f}"
              f"{(p['bridge_retention'] or 0):>8.3f}{(p['terminal_retention'] or 0):>8.3f}"
              f"{(p['complete_path_retention'] or 0):>8.3f}"
              f"{p['mean_role_coverage']:>7.2f}{p['mean_redundancy']:>7.2f}"
              f"{(p['oracle_gap_closure'] if p['oracle_gap_closure'] is not None else 0):>9.3f}")
    print("\n  FAILURE ATTRIBUTION (required records present in working set):")
    for k in FAILURES:
        if pooled_fail.get(k):
            print(f"    {k:28}{pooled_fail[k] / total_fail:>7.1%}  (n={pooled_fail[k]})")
    print(f"\n  written: {out}\n  No HRM run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
