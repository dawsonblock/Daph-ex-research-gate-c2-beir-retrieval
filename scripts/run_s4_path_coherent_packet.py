#!/usr/bin/env python3
"""S4: path-coherent packet composition qualification. Matched ALL-TASK.

The architectural question, stated directly:

    does G2 + path-coherent composition beat the SIMPLER typed-path architecture?

S3 conditioned on non-empty working sets; this gate deliberately does not, so
every arm is pooled over an identical population. Per
configs/gate_s4_path_coherent_packet_v1.json, frozen before this ran.

    T0  G1_TYPED_PATH + current S2      the simpler architecture to beat
    T1  G2 + current S2
    T2  G2 + path-coherent packet       the proposition
    T3  oracle ceiling

    delta_graph_value = CES(T2) - CES(T0)   <- the architectural decision
    delta_composition = CES(T2) - CES(T1)   <- does coherence work at all

Upstream is pinned (grammar_v4, C2, G2 enumeration, M=50, packet 6) and S2's
record ranking is CONSUMED, not replaced. No HRM.

Usage:
    python scripts/run_s4_path_coherent_packet.py
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
from hrm_adaptive_memory.c4.packet_composition import (  # noqa: E402
    NOT_COMPUTABLE, complete_path_packet, complete_paths_represented,
    compose_path_coherent_packet, packet_coherence_ratio)
from hrm_adaptive_memory.c4.prefilter import structural_signature  # noqa: E402
from hrm_adaptive_memory.c4.typed_path import typed_path_prefilter  # noqa: E402
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
ARMS_S3 = ("T0_typed_S2", "T1_g2_S2", "T2_g2_pathcoherent", "T3_oracle")
COMPETITION_BUCKETS = ("1", "2-3", "4-6", "7+")


def competition_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    return "7+"
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
    ap = argparse.ArgumentParser(description="S4 path-coherent packet qualification")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_default_boundary_policy("grammar_v4")  # pinned upstream, per protocol
    arm = ARMS[args.arm_for_queries]
    print("=== S4 path-coherent packet qualification "
          "(matched all-task; grammar_v4 pinned, M50, packet 6, no HRM) ===\n")

    per_scale: dict[str, Any] = {}
    pooled_sel = {a: Counter() for a in ARMS_S3}
    pooled_fail = Counter()
    pooled_strata: dict = defaultdict(lambda: [0, 0])
    pooled_comp: dict[str, list] = defaultdict(list)
    pooled_tasks = 0

    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        depth = c2(len(records))
        sel = {a: Counter() for a in ARMS_S3}
        strata: dict = defaultdict(lambda: [0, 0])
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

            # ---- working sets: typed path (T0) and G2 (T1/T2/T3) ---------
            tp = typed_path_prefilter(
                candidate_ids=pool, texts=texts, canonical_subject=canonical,
                relation=relation, working_set_size=M, fusion_scores=scores)
            g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                               canonical_subject=canonical, relation=relation,
                               working_set_size=M, fusion_scores=scores,
                               completion_fn=k1_entity_bound_exact_completion)
            typed_ws, g2_ws = tp.kept, g2r.kept
            required = set(task["required_evidence_ids"])
            complete_paths = [pp for pp in g2r.all_paths if pp.complete]
            n_paths = len(complete_paths)
            bucket = competition_bucket(n_paths)
            fam = task.get("family", "?")
            regime = (task.get("metadata") or {}).get("entity_regime", "?")

            def s2_order(ws):
                """Unchanged S2 ranking over a working set. MATCHED ALL-TASK:
                an empty working set yields an empty packet and scores 0 rather
                than being skipped, so populations stay identical across arms."""
                if not ws:
                    return []
                ident = run_identity_stage(q, arm, RetrievalResult(
                    candidate_ids=tuple(ws), candidate_budget=len(ws),
                    retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                    bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                    bm25_ranked=(), bge_ranked=(), fusion_ranked=()), texts)
                rq2 = q
                if ident.surface and ident.canonical:
                    rq2 = rq2.replace(ident.surface, ident.canonical)
                cds = [{"document_id": e} for e in ws]
                allowed = set(ws)

                def fz(bb):
                    if ident.status in ("EXACT", "RESOLVED") and ident.canonical:
                        out = s2c_chain_plus_relation(cds, budget=bb, question=rq2,
                                                      texts=texts)
                    else:
                        out = s0_raw(cds, budget=bb)
                    return [c for c in out
                            if (c["document_id"] if isinstance(c, dict) else c) in allowed]

                sel_, _r, _d = select_s2(
                    identity_status=ident.status, question=q,
                    canonical_subject=ident.canonical, candidate_ids=ws,
                    texts=texts, budget=len(ws), frozen_select=fz,
                    fusion_scores=scores)
                return [x["document_id"] if isinstance(x, dict) else x for x in sel_]

            typed_order = s2_order(typed_ws)
            g2_order = s2_order(g2_ws)

            packets = {
                "T0_typed_S2": typed_order[:PACKET],
                "T1_g2_S2": g2_order[:PACKET],
                "T2_g2_pathcoherent": compose_path_coherent_packet(
                    complete_paths=complete_paths, s2_ordering=g2_order,
                    working_set=g2_ws, packet_budget=PACKET).packet,
                "T3_oracle": ([r for r in g2_ws if r in required]
                              + [r for r in g2_order
                                 if r not in required])[:PACKET],
            }

            terms_all = set(terminal_records(task))
            brs_all = bridge_records(task)
            idents_all = required - terms_all - brs_all

            for name, pk in packets.items():
                chosen = set(pk)
                c = sel[name]
                c["tasks"] += 1
                hit = int(bool(required) and required <= chosen)
                c["selected_ces"] += hit
                ws_for_arm = typed_ws if name == "T0_typed_S2" else g2_ws
                for lbl, recs in (("bridge", brs_all), ("term", terms_all),
                                  ("ident", idents_all)):
                    present = recs & set(ws_for_arm)
                    if present:
                        c[f"{lbl}_den"] += 1
                        c[f"{lbl}_num"] += int(present <= chosen)
                if complete_paths:
                    c["cpath_den"] += 1
                    c["cpath_num"] += complete_path_packet(pk, complete_paths)
                    pcr = packet_coherence_ratio(pk, complete_paths)
                    if pcr != NOT_COMPUTABLE:
                        c["pcr_sum"] += pcr
                        c["pcr_den"] += 1
                    c["paths_repr"] += complete_paths_represented(pk, complete_paths)
                sigs = [structural_signature(texts.get(r, ""), canonical, relation)
                        for r in pk]
                c["redundancy"] += (len(sigs) - len(set(sigs)))
                strata[(name, "bucket", bucket)][0] += 1
                strata[(name, "bucket", bucket)][1] += hit
                strata[(name, "family", fam)][0] += 1
                strata[(name, "family", fam)][1] += hit
                strata[(name, "regime", regime)][0] += 1
                strata[(name, "regime", regime)][1] += hit

            comp["n_complete_paths"].append(n_paths)

        print(" " * 44, end="\r")

        def rates(c):
            n = c["tasks"] or 1
            def r(a, b):
                return round(c[a] / c[b], 4) if c[b] else None
            return {"tasks": c["tasks"],
                    "selected_ces": round(c["selected_ces"] / n, 4),
                    "bridge_retention": r("bridge_num", "bridge_den"),
                    "terminal_retention": r("term_num", "term_den"),
                    "identity_retention": r("ident_num", "ident_den"),
                    "complete_path_packet": r("cpath_num", "cpath_den"),
                    "PCR": (round(c["pcr_sum"] / c["pcr_den"], 4)
                            if c["pcr_den"] else None),
                    "mean_paths_represented": round(c["paths_repr"] / n, 3),
                    "mean_redundancy": round(c["redundancy"] / n, 3)}

        per_scale[scale] = {
            "arms": {a: rates(sel[a]) for a in ARMS_S3},
            "mean_complete_paths": round(
                sum(comp["n_complete_paths"]) / max(1, len(comp["n_complete_paths"])), 2)}
        for a in ARMS_S3:
            pooled_sel[a].update(sel[a])
        for k, v in strata.items():
            pooled_strata[k][0] += v[0]
            pooled_strata[k][1] += v[1]
        for k, v in comp.items():
            pooled_comp[k].extend(v)
        pooled_tasks += len(tasks)
        print(f"  {scale}: done  tasks={len(tasks)}")

    def prates(c):
        n = c["tasks"] or 1
        def r(a, b):
            return round(c[a] / c[b], 4) if c[b] else None
        return {"tasks": c["tasks"],
                "selected_ces": round(c["selected_ces"] / n, 4),
                "bridge_retention": r("bridge_num", "bridge_den"),
                "terminal_retention": r("term_num", "term_den"),
                "identity_retention": r("ident_num", "ident_den"),
                "complete_path_packet": r("cpath_num", "cpath_den"),
                "PCR": (round(c["pcr_sum"] / c["pcr_den"], 4) if c["pcr_den"] else None),
                "mean_paths_represented": round(c["paths_repr"] / n, 3),
                "mean_redundancy": round(c["redundancy"] / n, 3)}

    pooled = {a: prates(pooled_sel[a]) for a in ARMS_S3}
    t0, t1, t2, t3 = (pooled[a]["selected_ces"] for a in ARMS_S3)
    deltas = {"delta_graph_value_T2_minus_T0": round(t2 - t0, 4),
              "delta_composition_T2_minus_T1": round(t2 - t1, 4),
              "oracle_headroom_T3_minus_T2": round(t3 - t2, 4)}

    strat_out: dict = {}
    for (arm, kind, key), (den, num) in sorted(pooled_strata.items()):
        strat_out.setdefault(kind, {}).setdefault(key, {})[arm] = {
            "n": den, "ces": round(num / den, 4) if den else None}
    # safety gates
    def g(a, f):
        return pooled[a][f]
    safety = {
      "identity_no_material_regression": (
          g("T2_g2_pathcoherent","identity_retention") is None
          or g("T1_g2_S2","identity_retention") is None
          or g("T2_g2_pathcoherent","identity_retention")
             >= g("T1_g2_S2","identity_retention") - 0.05),
      "bridge_no_regression_vs_T1": (
          g("T2_g2_pathcoherent","bridge_retention")
          >= g("T1_g2_S2","bridge_retention") - 0.02),
      "terminal_no_regression_vs_T1": (
          g("T2_g2_pathcoherent","terminal_retention")
          >= g("T1_g2_S2","terminal_retention") - 0.02)}
    worst_sub = None
    for kind in ("family", "regime"):
        for key, arms_ in strat_out.get(kind, {}).items():
            a2 = arms_.get("T2_g2_pathcoherent", {}).get("ces")
            a1 = arms_.get("T1_g2_S2", {}).get("ces")
            if a2 is not None and a1 is not None:
                d = a2 - a1
                if worst_sub is None or d < worst_sub[1]:
                    worst_sub = (f"{kind}:{key}", round(d, 4))
    safety["no_pathological_subgroup"] = bool(worst_sub is None or worst_sub[1] >= -0.10)
    safety["worst_subgroup"] = worst_sub

    if deltas["delta_composition_T2_minus_T1"] <= 0:
        decision = "REJECT_COMPOSITION_MECHANISM"
    elif deltas["delta_graph_value_T2_minus_T0"] <= 0:
        decision = "KEEP_TYPED_PATH_ARCHITECTURE__GRAPH_NOT_YET_EARNED"
    elif not all(v for k, v in safety.items() if isinstance(v, bool)):
        decision = "DO_NOT_PROMOTE__SAFETY_GATE_FAILED"
    else:
        decision = "PROMOTE_PATH_COHERENT_COMPOSITION__PROVISIONALLY_PROMOTE_G2_STACK"

    report = {"schema_version": "s4-path-coherent-packet-v1", "no_hrm": True,
              "matched_all_task_pooling": True,
              "upstream_pinned": {"extractor": "grammar_v4", "compressor_T0": "typed_path",
                                  "compressor_T1_T2_T3": "G2", "M": M, "packet": PACKET},
              "per_scale": per_scale, "pooled": {"arms": pooled, **deltas},
              "strata": strat_out, "safety_gates": safety, "decision": decision}
    out = Path(args.out) if args.out else ROOT / "evidence/gate_s4/path_coherent_packet.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'arm':<22}{'selCES':>8}{'bridge':>8}{'term':>8}{'ident':>8}"
          f"{'cpathPkt':>10}{'PCR':>7}{'paths':>7}{'redun':>7}")
    for a in ARMS_S3:
        p_ = pooled[a]
        def f(v):
            return 0.0 if v is None else v
        print(f"  {a:<22}{p_['selected_ces']:>8.3f}{f(p_['bridge_retention']):>8.3f}"
              f"{f(p_['terminal_retention']):>8.3f}{f(p_['identity_retention']):>8.3f}"
              f"{f(p_['complete_path_packet']):>10.3f}{f(p_['PCR']):>7.3f}"
              f"{p_['mean_paths_represented']:>7.2f}{p_['mean_redundancy']:>7.2f}")
    print(f"\n  delta_graph_value  (T2-T0) = {deltas['delta_graph_value_T2_minus_T0']:+.4f}   <- architecture")
    print(f"  delta_composition  (T2-T1) = {deltas['delta_composition_T2_minus_T1']:+.4f}")
    print(f"  oracle headroom    (T3-T2) = {deltas['oracle_headroom_T3_minus_T2']:+.4f}")
    print("\n  PATH-COMPETITION STRATIFICATION (registered prediction: gain grows with ambiguity):")
    for key in COMPETITION_BUCKETS:
        arms_ = strat_out.get("bucket", {}).get(key)
        if not arms_:
            continue
        a0 = arms_.get("T0_typed_S2", {}).get("ces")
        a1 = arms_.get("T1_g2_S2", {}).get("ces")
        a2 = arms_.get("T2_g2_pathcoherent", {}).get("ces")
        n = arms_.get("T2_g2_pathcoherent", {}).get("n")
        if None in (a1, a2):
            continue
        print(f"    paths={key:<5} n={n:<5} T0={a0:.3f} T1={a1:.3f} T2={a2:.3f}  "
              f"dComp={a2-a1:+.3f}  dGraph={a2-(a0 or 0):+.3f}")
    print("\n  SAFETY GATES:")
    for k, v in safety.items():
        print(f"    {k:36}{v}")
    print(f"\n  DECISION: {decision}")
    print(f"  written: {out}\n  No HRM run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
