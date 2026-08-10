#!/usr/bin/env python3
"""HRM qualification: does S4's better packet improve HRM answer quality?

    H0  typed_path working set -> unchanged S2                architecture to beat
    H1  G2 working set -> unchanged S2                         diagnostic only
    H2  G2 working set -> path-coherent composer (S4)          PRIMARY
    H3  required-first, G2 working set, S2 fill                selector ceiling
    H4  required records directly                              global ceiling

Per configs/gate_hrm_qualification_v1.json, frozen before this ran. Reuses
existing, already-certified machinery rather than reimplementing it:
run_packet_stage (packet ordering + prompt composition), _run_hrm/_run_hrm_batch
and _assert_prompt_binding (from run_gate_c4.py), and S4's frozen path-composer
contract. This is a development/qualification gate on calibration-informed
data, NOT untouched confirmation.

Defaults to --dry-run: builds every packet, prompt, and hash-chain link, and
writes the receipt WITHOUT calling the model -- lets the full pipeline be
validated on CPU before any GPU compute is spent. Pass --execute to actually
run HRM generation.

Usage:
    python scripts/run_hrm_qualification.py --dry-run   # validate, no GPU
    python scripts/run_hrm_qualification.py --execute    # real run, needs GPU
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
    entity_extractor_config_hash, set_default_boundary_policy)
from hrm_adaptive_memory.c4.contracts import (  # noqa: E402
    C4_PRIMARY_PACKET_BUDGET, C4_RRF_K, RetrievalResult, SelectionResult)
from hrm_adaptive_memory.c4.endpoint_recognition import (  # noqa: E402
    k1_entity_bound_exact_completion)
from hrm_adaptive_memory.c4.fusion import frozen_rrf  # noqa: E402
from hrm_adaptive_memory.c4.g2_paths import g2_prefilter  # noqa: E402
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage  # noqa: E402
from hrm_adaptive_memory.c4.packet_composition import (  # noqa: E402
    compose_path_coherent_packet, composed_packet_hash, generation_hash,
    graph_hash, path_set_hash)
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import select_s2  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.execution_identity import (  # noqa: E402
    ExecutionIdentity)
from hrm_adaptive_memory.retrieval.canonicalization import _norm  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from hrm_adaptive_memory.c4.typed_path import typed_path_prefilter  # noqa: E402
from scripts.diagnose_c5_confirmation_stopgate import bridge_records  # noqa: E402
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import (  # noqa: E402
    _assert_prompt_binding, _load_hrm, _run_hrm_batch,
    _to_index_records as to_index_records)

M = 50
PACKET = C4_PRIMARY_PACKET_BUDGET
H_ARMS = ("H0_typed_path", "H1_g2_S2", "H2_g2_pathcoherent",
          "H3_oracle_selector", "H4_oracle_evidence")
PIPELINE_VERSION = "hrm_qualification_v1"


def c2(n: int) -> int:
    return max(100, min(300, math.ceil(0.15 * n)))


def competition_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    return "7+"


def main() -> int:
    ap = argparse.ArgumentParser(description="HRM qualification: H0-H4")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scales", nargs="*", default=None,
                    help="subset of SCALES, for a bounded smoke run")
    ap.add_argument("--limit-tasks", type=int, default=None,
                    help="cap tasks per scale, for a bounded smoke run")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="build packets/prompts/hashes only, no GPU, no generation")
    mode.add_argument("--execute", action="store_true",
                      help="real HRM generation -- needs GPU")
    args = ap.parse_args()

    set_default_boundary_policy("grammar_v4")  # pinned, per protocol
    extractor_hash = entity_extractor_config_hash()
    arm = ARMS[args.arm_for_queries]
    scales = args.scales or list(SCALES)

    print(f"=== HRM qualification: H0-H4 ({'DRY RUN, no GPU' if args.dry_run else 'EXECUTE'}) ===")
    print(f"  entity_extractor_config_hash={extractor_hash}  M={M}  packet={PACKET}\n")

    adapter, condition = (None, None)
    if args.execute:
        adapter, condition = _load_hrm()

    per_scale: dict[str, Any] = {}
    pooled = {a: Counter() for a in H_ARMS}
    pooled_strata: dict = defaultdict(lambda: [0, 0])
    pooled_tasks = 0
    receipts: list[dict[str, Any]] = []

    for scale in scales:
        tasks, evidence, texts = load_scale(scale)
        if args.limit_tasks:
            tasks = tasks[:args.limit_tasks]
        records = to_index_records(evidence)
        depth = c2(len(records))
        sel = {a: Counter() for a in H_ARMS}
        strata: dict = defaultdict(lambda: [0, 0])

        for i, task in enumerate(tasks, 1):
            if i % 10 == 0 or i == len(tasks):
                print(f"  {scale}: {i}/{len(tasks)}", end="\r", flush=True)
            q = task["question"]
            _s, qr = run_query_stage(q, arm)
            bm = get_cached_backend(CanonicalRetrievalMode.BM25, records)
            bg = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
            a_ids = [e.evidence_id for e in
                     asyncio.run(bm.search(qr.rendered_query, k=depth)).evidence]
            b_ids = [e.evidence_id for e in
                     asyncio.run(bg.search(qr.rendered_query, k=depth)).evidence]
            fused = frozen_rrf([a_ids, b_ids], C4_RRF_K, depth)
            pool = [e for e, _ in fused[:depth]]
            scores = dict(fused[:depth])
            relation = extract_target_relation(q) or ""
            probe = run_identity_stage(q, arm, RetrievalResult(
                candidate_ids=tuple(pool), candidate_budget=depth,
                retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                bm25_ranked=(), bge_ranked=(), fusion_ranked=()), texts)
            canonical = probe.canonical
            required = set(task["required_evidence_ids"])

            tp = typed_path_prefilter(candidate_ids=pool, texts=texts,
                                      canonical_subject=canonical, relation=relation,
                                      working_set_size=M, fusion_scores=scores)
            g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                               canonical_subject=canonical, relation=relation,
                               working_set_size=M, fusion_scores=scores,
                               completion_fn=k1_entity_bound_exact_completion)
            typed_ws, g2_ws = tp.kept, g2r.kept
            complete_paths = [p for p in g2r.all_paths if p.complete]
            n_paths = len(complete_paths)
            bucket = competition_bucket(n_paths)
            fam = task.get("family", "?")
            regime = (task.get("metadata") or {}).get("entity_regime", "?")

            def s2_order(ws, _q=q, _scores=scores):
                if not ws:
                    return []
                ident = run_identity_stage(_q, arm, RetrievalResult(
                    candidate_ids=tuple(ws), candidate_budget=len(ws),
                    retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                    bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                    bm25_ranked=(), bge_ranked=(), fusion_ranked=()), texts)
                rq = _q
                if ident.surface and ident.canonical:
                    rq = rq.replace(ident.surface, ident.canonical)
                cds = [{"document_id": e} for e in ws]
                allowed = set(ws)

                def fz(bb):
                    out = (s2c_chain_plus_relation(cds, budget=bb, question=rq, texts=texts)
                          if ident.status in ("EXACT", "RESOLVED") and ident.canonical
                          else s0_raw(cds, budget=bb))
                    return [c for c in out
                            if (c["document_id"] if isinstance(c, dict) else c) in allowed]

                s_, _r, _d = select_s2(
                    identity_status=ident.status, question=_q,
                    canonical_subject=ident.canonical, candidate_ids=ws,
                    texts=texts, budget=len(ws), frozen_select=fz,
                    fusion_scores=_scores)
                return [x["document_id"] if isinstance(x, dict) else x for x in s_]

            typed_order = s2_order(typed_ws)
            g2_order = s2_order(g2_ws)

            selected_ids_by_arm = {
                "H0_typed_path": typed_order[:PACKET],
                "H1_g2_S2": g2_order[:PACKET],
                "H2_g2_pathcoherent": compose_path_coherent_packet(
                    complete_paths=complete_paths, s2_ordering=g2_order,
                    working_set=g2_ws, packet_budget=PACKET).packet,
                "H3_oracle_selector": ([r for r in g2_ws if r in required]
                                       + [r for r in g2_order if r not in required])[:PACKET],
                "H4_oracle_evidence": list(required)[:PACKET],
            }

            g2_graph = build_runtime_graph(record_ids=pool, texts=texts, relation=relation)
            ghash = graph_hash(g2_graph)
            pshash = path_set_hash(complete_paths)

            for name, selected_ids in selected_ids_by_arm.items():
                ws_for_arm = typed_ws if name == "H0_typed_path" else g2_ws
                retrieval = RetrievalResult(
                    candidate_ids=tuple(pool), candidate_budget=depth,
                    retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                    bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                    bm25_ranked=(), bge_ranked=(),
                    fusion_ranked=tuple(fused[:depth]))
                selection = SelectionResult(
                    selector="s4_hrm_qual", selected_ids=tuple(selected_ids),
                    selector_policy=name, identity_status=probe.status)
                prompt, packet = run_packet_stage(arm, q, selection, texts, retrieval)

                identity = ExecutionIdentity(
                    task_id=task["task_id"], arm_id=name, prompt_hash=packet.prompt_hash,
                    retrieval_config_hash="C2", selector_config_hash="s2_v2+s4_composer_v1",
                    graph_compressor_config_hash=extractor_hash,
                    model_revision="sapientinc/HRM-Text-1B@9f082d68",
                    pipeline_version=PIPELINE_VERSION,
                    source_commit="pending",
                    extra_config_hashes={
                        "graph_hash": ghash, "path_set_hash": pshash,
                        "composed_packet_hash": composed_packet_hash(tuple(selected_ids))})

                receipt: dict[str, Any] = {
                    "task_id": task["task_id"], "arm": name, "scale": scale,
                    "family": fam, "entity_regime": regime,
                    "path_competition_bucket": bucket,
                    "packet_ids": list(packet.packet_ids),
                    "candidate_pool_hash": packet.candidate_pool_hash,
                    "graph_hash": ghash, "path_set_hash": pshash,
                    "working_set_hash": composed_packet_hash(tuple(sorted(ws_for_arm))),
                    "composed_packet_hash": composed_packet_hash(tuple(selected_ids)),
                    "membership_hash": packet.membership_hash,
                    "order_hash": packet.order_hash, "prompt_hash": packet.prompt_hash,
                    "execution_identity_sha256": identity.canonical_sha256(),
                    "required_in_packet": bool(required) and required <= set(packet.packet_ids),
                }

                if args.execute:
                    hrm_result = _run_hrm_batch(adapter, condition, [prompt])[0]

                    class _Pre:
                        pass
                    pre = _Pre(); pre.task_id = task["task_id"]; pre.arm_id = name; pre.packet = packet
                    _assert_prompt_binding(pre, hrm_result)
                    correct = task.get("answer", "").strip().lower() in hrm_result.output.strip().lower()
                    receipt.update({
                        "output": hrm_result.output, "correct": correct,
                        "generation_hash": generation_hash(hrm_result.output),
                        "prompt_tokens": hrm_result.prompt_tokens,
                        "completion_tokens": hrm_result.completion_tokens,
                        "latency_seconds": hrm_result.latency_seconds})
                    c = sel[name]
                    c["tasks"] += 1
                    c["quality"] += int(correct)
                    c["latency_sum"] += hrm_result.latency_seconds
                    c["tokens_sum"] += hrm_result.completion_tokens
                    strata[(name, "bucket", bucket)][0] += 1
                    strata[(name, "bucket", bucket)][1] += int(correct)
                    strata[(name, "family", fam)][0] += 1
                    strata[(name, "family", fam)][1] += int(correct)
                    strata[(name, "regime", regime)][0] += 1
                    strata[(name, "regime", regime)][1] += int(correct)
                else:
                    receipt["output"] = None  # dry run: no generation performed
                receipts.append(receipt)

        print(" " * 44, end="\r")
        if args.execute:
            def rates(c):
                n = c["tasks"] or 1
                return {"tasks": c["tasks"], "quality": round(c["quality"] / n, 4),
                        "mean_latency_s": round(c["latency_sum"] / n, 3),
                        "mean_completion_tokens": round(c["tokens_sum"] / n, 2)}
            per_scale[scale] = {a: rates(sel[a]) for a in H_ARMS}
            for a in H_ARMS:
                pooled[a].update(sel[a])
            for k, v in strata.items():
                pooled_strata[k][0] += v[0]
                pooled_strata[k][1] += v[1]
        pooled_tasks += len(tasks)
        print(f"  {scale}: done  tasks={len(tasks)}")

    # _assert_prompt_binding fails CLOSED via AssertionError, not a soft counter
    # -- reaching this line at all means every binding check that ran, passed.
    binding_checks_passed = "all (fail-closed: a violation raises, not counts)"

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_hrm/qualification_{'dry_run' if args.dry_run else 'execute'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "hrm-qualification-v1", "mode": "dry_run" if args.dry_run else "execute",
        "entity_extractor_config_hash": extractor_hash,
        "scales_run": scales, "tasks_total": pooled_tasks,
        "receipts_written": len(receipts),
        "binding_assertion_failures": binding_checks_passed,
    }
    if args.execute:
        def prates(c):
            n = c["tasks"] or 1
            return {"quality": round(c["quality"] / n, 4),
                    "mean_latency_s": round(c["latency_sum"] / n, 3),
                    "mean_completion_tokens": round(c["tokens_sum"] / n, 2)}
        pr = {a: prates(pooled[a]) for a in H_ARMS}
        q0, q2 = pr["H0_typed_path"]["quality"], pr["H2_g2_pathcoherent"]["quality"]
        strat_out: dict = {}
        for (a, kind, key), (den, num) in sorted(pooled_strata.items()):
            strat_out.setdefault(kind, {}).setdefault(key, {})[a] = {
                "n": den, "quality": round(num / den, 4) if den else None}
        report["pooled"] = pr
        report["delta_HRM_graph_value"] = round(q2 - q0, 4)
        report["delta_composition"] = round(q2 - pr["H1_g2_S2"]["quality"], 4)
        report["strata"] = strat_out
        report["s4_packet_delta_for_reference"] = 0.1533
        denom = report["s4_packet_delta_for_reference"]
        report["conversion_efficiency_diagnostic"] = (
            round(report["delta_HRM_graph_value"] / denom, 4) if abs(denom) > 1e-9 else None)

    receipts_path = out.with_suffix(".receipts.jsonl")
    receipts_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in receipts) + "\n")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.execute:
        print(f"\n  {'arm':<22}{'quality':>9}{'lat(s)':>9}{'tok':>7}")
        for a in H_ARMS:
            p_ = pr[a]
            print(f"  {a:<22}{p_['quality']:>9.3f}{p_['mean_latency_s']:>9.2f}{p_['mean_completion_tokens']:>7.1f}")
        print(f"\n  delta_HRM_graph_value (H2-H0) = {report['delta_HRM_graph_value']:+.4f}")
        print(f"  delta_composition     (H2-H1) = {report['delta_composition']:+.4f}")
        print(f"  conversion_efficiency (diag)  = {report['conversion_efficiency_diagnostic']}")
    else:
        print(f"\n  DRY RUN complete: {len(receipts)} receipts built "
              "(no generation performed, no prompt-binding check exercised).")
        print("  Re-run with --execute on GPU to generate.")
    print(f"\n  written: {out}\n  receipts: {receipts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
