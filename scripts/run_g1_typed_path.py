#!/usr/bin/env python3
"""G1: G0, G1_coarse, G1_TYPED_PATH, G3. No HRM. G2 deliberately NOT run.

    G0             k=50 -> S2                              low-pressure baseline
    G1_coarse      C2 -> B4 structural prefilter -> S2      the failed floor
    G1_TYPED_PATH  C2 -> frozen typed path -> S2            FIRST SCORED ARM
    G3             C2 -> ORACLE compressor -> S2            the P3 ceiling

The question this answers, and the only one: is the missing middle layer really a
graph substrate, or merely typed path discrimination? G2 is withheld until this
reports, so that the graph cannot become architecture-by-assumption.

Reporting obeys the permanent rule: end-to-end selected CES is primary, and the
conditional term is never reported alone. B4 proved a compressor can post the
best conditional number in the table by discarding evidence before S2 sees it.

Usage:
    python scripts/run_g1_typed_path.py
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
ORACLE = "G3_oracle_M{}"
ROLES = ("identity", "bridge", "terminal", "temporal_current")
BRIDGE_BOUND = -0.05
#: Frozen in configs/gate_g1_runtime_graph_v1.json BEFORE this ran.
GAP_CLOSURE_PASS = 0.60
GAP_CLOSURE_PARTIAL = 0.30


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def competition_counts(pool: list[str], texts: dict[str, str], question: str,
                       canonical: str | None) -> dict[str, int]:
    """How many candidates plausibly compete for S2's structural slots."""
    rows = _candidates(pool, texts, None)
    relation = extract_target_relation(question) or ""
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
        arms[ORACLE.format(m)] = {"k": k, "compressor": "oracle", "M": m}
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description="G1 typed path vs coarse vs oracle")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm = ARMS[args.arm_for_queries]
    packet_budget = C4_PRIMARY_PACKET_BUDGET
    print("=== G1 runtime evidence graph: typed-path arm (no HRM, G2 withheld) ===")
    print(f"  M={list(M_VALUES)}  packet={packet_budget} (FIXED)  S2 UNCHANGED")
    print(f"  gap-closure pass bound {GAP_CLOSURE_PASS} at EVERY scale (frozen)\n")

    results: dict[str, Any] = {}
    for scale in SCALES:
        tasks, evidence, texts = load_scale(scale)
        records = to_index_records(evidence)
        corpus_size = len(records)
        kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
                 for r in evidence}
        arms = build_arms(corpus_size)
        depth = max(a["k"] for a in arms.values())

        acc = {name: {"cand_ces": 0, "pre_ces": 0, "sel_ces": 0, "e2e": 0,
                      "role": defaultdict(lambda: [0, 0]),
                      "avail": defaultdict(lambda: [0, 0]),
                      "competition": defaultdict(list),
                      "path": defaultdict(list)}
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

                # --- compression -------------------------------------------
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
                    for key, value in tp.diagnostics().items():
                        entry["path"][key].append(value)
                elif spec["compressor"] == "oracle":
                    working = oracle_prefilter(
                        candidate_ids=pool,
                        required=list(task["required_evidence_ids"]),
                        working_set_size=spec["M"]).kept
                else:
                    working = pool

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
                # END-TO-END, measured directly. Never the conditional alone.
                entry["e2e"] += required <= chosen
                if required <= working_set:
                    entry["sel_ces"] += required <= chosen
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
                "selected_ces_given_working__NEVER_ALONE": _rate(e["sel_ces"], e["pre_ces"]),
                "availability": {role: _rate(*reversed(e["avail"][role]))
                                 for role in ROLES if e["avail"][role][0]},
                "conditional_retention": {role: _rate(*reversed(e["role"][role]))
                                          for role in ROLES if e["role"][role][0]},
                "structural_competition_ratio": round(
                    sum(connected) / len(connected) / packet_budget, 2) if connected else None,
                "working_set_size_mean": round(
                    sum(e["competition"]["candidate_count"])
                    / len(e["competition"]["candidate_count"]), 1),
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
            pol["bridge_safety"] = {"delta": bd, "bound": BRIDGE_BOUND,
                                    "passed": bd is None or bd >= BRIDGE_BOUND}

        # --- gap closure, per matched M -----------------------------------
        gap: dict[str, Any] = {}
        for m in M_VALUES:
            coarse = out_arms[COARSE.format(m)]["selected_ces_end_to_end"]
            typed = out_arms[TYPED.format(m)]["selected_ces_end_to_end"]
            oracle = out_arms[ORACLE.format(m)]["selected_ces_end_to_end"]
            denominator = (oracle or 0) - (coarse or 0)
            closure = (round(((typed or 0) - (coarse or 0)) / denominator, 4)
                       if abs(denominator) > 1e-9 else None)
            gap[f"M{m}"] = {
                "ces_coarse_P2": coarse, "ces_typed": typed, "ces_oracle_P3": oracle,
                "headroom": round(denominator, 4), "gap_closure": closure,
                "bridge_safety_passed": out_arms[TYPED.format(m)]["bridge_safety"]["passed"],
                "meets_frozen_bound": bool(
                    closure is not None and closure >= GAP_CLOSURE_PASS
                    and out_arms[TYPED.format(m)]["bridge_safety"]["passed"]),
            }
        results[scale] = {"corpus_size": corpus_size, "tasks": n,
                          "arms": out_arms, "gap_closure": gap}
        print(f"  {scale}: N={corpus_size} k(C2)={c2(corpus_size)}")

    # --- frozen decision rule ---------------------------------------------
    verdict: dict[str, Any] = {}
    for m in M_VALUES:
        per_scale = {s: results[s]["gap_closure"][f"M{m}"]["gap_closure"]
                     for s in SCALES}
        passed = all(results[s]["gap_closure"][f"M{m}"]["meets_frozen_bound"]
                     for s in SCALES)
        values = [v for v in per_scale.values() if v is not None]
        worst = min(values) if values else None
        if passed:
            outcome = "TYPED_PATH_SUFFICIENT__NO_GRAPH_SUBSTRATE_YET"
        elif worst is not None and worst >= GAP_CLOSURE_PARTIAL:
            outcome = "PARTIAL__BUILD_G2__TYPED_PATH_RETAINED_AS_SEED_TEMPLATE"
        else:
            outcome = "TYPED_PATH_INSUFFICIENT__BUILD_G2"
        verdict[f"M{m}"] = {"gap_closure_per_scale": per_scale,
                            "worst_scale": worst, "all_scales_pass": passed,
                            "outcome": outcome}
    best = max(M_VALUES, key=lambda m: (verdict[f"M{m}"]["all_scales_pass"],
                                        verdict[f"M{m}"]["worst_scale"] or -9))

    report = {
        "schema_version": "g1-typed-path-v1",
        "no_hrm": True, "s2_unchanged": True, "g2_run": False,
        "packet_budget": packet_budget, "m_values": list(M_VALUES),
        "frozen_bounds": {"gap_closure_pass": GAP_CLOSURE_PASS,
                          "gap_closure_partial_floor": GAP_CLOSURE_PARTIAL,
                          "bridge_bound": BRIDGE_BOUND,
                          "frozen_before_run": True},
        "REPORTING_RULE": "selected_ces_end_to_end is primary; the conditional "
                          "term is never reported alone",
        "scales": results, "verdict_by_M": verdict,
        "decision": verdict[f"M{best}"]["outcome"], "decision_at_M": best,
    }
    out = Path(args.out) if args.out else (
        ROOT / "evidence/gate_g1/typed_path.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  {'scale':<7}{'arm':<22}{'M':>4}{'candCES':>9}{'wsCES':>8}"
          f"{'E2E':>8}{'cond':>8}{'bridge':>8}{'dBr':>8}{'SCR':>7}")
    for scale in SCALES:
        for name, pol in results[scale]["arms"].items():
            db = pol.get("bridge_safety", {}).get("delta")
            flag = "" if name == BASELINE or pol["bridge_safety"]["passed"] else " F"
            print(f"  {scale.replace('cal_',''):<7}{name:<22}{(pol['M'] or 0):>4}"
                  f"{pol['candidate_ces']:>9.3f}{pol['working_set_ces']:>8.3f}"
                  f"{pol['selected_ces_end_to_end']:>8.3f}"
                  f"{(pol['selected_ces_given_working__NEVER_ALONE'] or 0):>8.3f}"
                  f"{pol['conditional_retention'].get('bridge', 0):>8.3f}"
                  f"{(f'{db:+.3f}' if db is not None else '-'):>8}"
                  f"{(pol['structural_competition_ratio'] or 0):>7.1f}{flag}")
        print()
    print(f"  {'GAP CLOSURE (typed - coarse) / (oracle - coarse)':<50}")
    print(f"  {'scale':<9}" + "".join(f"{'M'+str(m):>12}" for m in M_VALUES))
    for scale in SCALES:
        row = "".join(
            f"{(results[scale]['gap_closure'][f'M{m}']['gap_closure']):>12.3f}"
            if results[scale]["gap_closure"][f"M{m}"]["gap_closure"] is not None
            else f"{'n/a':>12}" for m in M_VALUES)
        print(f"  {scale.replace('cal_',''):<9}{row}")
    print()
    for m in M_VALUES:
        v = verdict[f"M{m}"]
        print(f"  M{m}: worst={v['worst_scale']}  {v['outcome']}")
    print(f"\n  DECISION (at M{best}): {report['decision']}")
    print(f"  written: {out}\n  G2 not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
