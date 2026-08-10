#!/usr/bin/env python3
"""Data collection for Executive v0 training: for every task in
data/hrm/exec_training_v1/, compute Q(ANSWER_NOW) WITH confidence features
(hrm_adaptive_memory.executive.confidence) and Q(USE_CERTIFIED_MEMORY) via
the unmodified certified pipeline -- zero bypass anywhere, since every task
in this suite natively parses through the real extract_subject/
extract_target_relation (verified at suite-build time for ANSWER_NOW_viable;
MEMORY_required is genuine b3-style data, always native).

This script does NOT train or evaluate Executive v0 -- it only collects the
per-task (features, Q_A0, Q_A1) records that a separate, GPU-free analysis
script (scripts/train_exec_v0.py) consumes. Keeping data collection
(expensive, GPU) and policy fitting (cheap, CPU, iterable) as separate
scripts means re-running the ladder/regret analysis never requires a new
GPU run.

Defaults to --dry-run (builds prompts/hashes, does not call the model).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.backends import CanonicalRetrievalMode  # noqa: E402
from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
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
from hrm_adaptive_memory.executive.confidence import generate_with_confidence  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.certified_memory import (  # noqa: E402
    CertifiedMemoryDriftError, assert_certified_memory_v1_unchanged,
    pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.experiment_integrity.execution_identity import (  # noqa: E402
    ExecutionIdentity)
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from scripts.run_gate_c4 import (  # noqa: E402
    HRM_MAX_NEW_TOKENS, _assert_prompt_binding, _load_hrm, _run_hrm_batch,
    _to_index_records as to_index_records)

M = 50
PACKET = C4_PRIMARY_PACKET_BUDGET
PIPELINE_VERSION = "exec_training_v1_collection"
SUITE_ROOT = ROOT / "data/hrm/exec_training_v1"
FAMILIES = ("ANSWER_NOW_viable", "MEMORY_required")


def load_family(family: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    base = SUITE_ROOT / family
    tasks = [json.loads(l) for l in (base / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (base / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence, {r["evidence_id"]: r["content"] for r in evidence}


def c2(n: int) -> int:
    return max(1, min(300, math.ceil(0.15 * n))) if n else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Executive v0 training data collection")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--limit-tasks", type=int, default=None)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    pin_certified_memory_v1_boundary_policy()
    try:
        certified_identity = assert_certified_memory_v1_unchanged()
    except CertifiedMemoryDriftError as e:
        print(f"ABORT: CERTIFIED_MEMORY_V1 drift detected before any task ran: {e}")
        return 1
    extractor_hash = certified_identity.graph_compressor_config_hash
    arm = ARMS[args.arm_for_queries]
    families = args.families or list(FAMILIES)

    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_digest = hashlib.sha256()
        for path in sorted(ROOT.rglob("*.py")):
            if any(part in {"evidence", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            source_digest.update(str(path.relative_to(ROOT)).encode())
            source_digest.update(path.read_bytes())
        source_commit = f"non-git-tree:{source_digest.hexdigest()}"

    print(f"=== Executive v0 training data collection ({'DRY RUN' if args.dry_run else 'EXECUTE'}) ===")
    print(f"  CERTIFIED_MEMORY_V1 identity OK  extractor_hash={extractor_hash}  M={M}  packet={PACKET}\n")

    adapter, condition = (None, None)
    if args.execute:
        adapter, condition = _load_hrm()

    receipts: list[dict[str, Any]] = []

    for family in families:
        tasks, evidence, texts = load_family(family)
        if args.limit_tasks:
            tasks = tasks[:args.limit_tasks]
        records = to_index_records(evidence)
        depth = c2(len(records))

        for i, task in enumerate(tasks, 1):
            if i % 10 == 0 or i == len(tasks):
                print(f"  {family}: {i}/{len(tasks)}", end="\r", flush=True)
            q = task["question"]
            fam = task.get("family", family)
            required = set(task.get("required_evidence_ids", []))

            # --- A0: ANSWER_NOW, with confidence -----------------------------
            a0_prompt = f"[OBJECTIVE]\n{q}\n[EVIDENCE]\n[NO EXTERNAL EVIDENCE]\n[RESPONSE REQUIREMENT]\nAnswer directly, concisely."
            a0_identity = ExecutionIdentity(
                task_id=task["task_id"], arm_id="A0_ANSWER_NOW",
                prompt_hash=hashlib.sha256(a0_prompt.encode()).hexdigest(),
                retrieval_config_hash="NONE", selector_config_hash="NONE",
                graph_compressor_config_hash=extractor_hash,
                model_revision="sapientinc/HRM-Text-1B@9f082d68",
                pipeline_version=PIPELINE_VERSION, source_commit=source_commit,
                extra_config_hashes={})
            a0_receipt: dict[str, Any] = {
                "task_id": task["task_id"], "action": "A0_ANSWER_NOW", "family": fam,
                "suite_family": family, "prompt_hash": a0_identity.prompt_hash,
                "execution_identity_sha256": a0_identity.canonical_sha256(),
            }
            if args.execute:
                conf = generate_with_confidence(adapter, condition, a0_prompt,
                                                max_new_tokens=HRM_MAX_NEW_TOKENS)
                correct = task.get("answer", "").strip().lower() in conf.text.strip().lower()
                a0_receipt.update({
                    "output": conf.text, "correct": correct,
                    "prompt_tokens": conf.prompt_tokens, "completion_tokens": conf.completion_tokens,
                    "mean_token_confidence": conf.mean_token_confidence,
                    "min_token_confidence": conf.min_token_confidence,
                    "sequence_confidence": conf.sequence_confidence,
                    "mean_entropy": conf.mean_entropy,
                    "answer_length": conf.answer_length,
                })
            else:
                a0_receipt["output"] = None
            receipts.append(a0_receipt)

            # --- A1: USE_CERTIFIED_MEMORY, full pipeline, zero bypass -------
            if records:
                if extract_target_relation(q):
                    _s, qr = run_query_stage(q, arm)
                    rendered_query = qr.rendered_query
                else:
                    rendered_query = q
                bm = get_cached_backend(CanonicalRetrievalMode.BM25, records)
                bg = get_cached_backend(CanonicalRetrievalMode.DENSE_BGE, records)
                a_ids = [e.evidence_id for e in
                        asyncio.run(bm.search(rendered_query, k=depth)).evidence]
                b_ids = [e.evidence_id for e in
                        asyncio.run(bg.search(rendered_query, k=depth)).evidence]
                fused = frozen_rrf([a_ids, b_ids], C4_RRF_K, depth)
                pool = [e for e, _ in fused[:depth]]
                scores = dict(fused[:depth])
            else:
                fused, pool, scores = [], [], {}

            relation = extract_target_relation(q) or ""
            retrieval = RetrievalResult(
                candidate_ids=tuple(pool), candidate_budget=depth,
                retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                bm25_ranked=(), bge_ranked=(), fusion_ranked=tuple(fused[:depth]))

            probe = run_identity_stage(q, arm, retrieval, texts)  # REAL parser, zero bypass
            canonical = probe.canonical

            g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                               canonical_subject=canonical, relation=relation,
                               working_set_size=M, fusion_scores=scores,
                               completion_fn=k1_entity_bound_exact_completion)
            g2_ws = g2r.kept
            complete_paths = [p for p in g2r.all_paths if p.complete]

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

            g2_order = s2_order(g2_ws)
            a1_packet_ids = compose_path_coherent_packet(
                complete_paths=complete_paths, s2_ordering=g2_order,
                working_set=g2_ws, packet_budget=PACKET).packet

            g2_graph = build_runtime_graph(record_ids=pool, texts=texts, relation=relation)
            ghash = graph_hash(g2_graph)
            pshash = path_set_hash(complete_paths)

            selection = SelectionResult(
                selector="exec_training_v1", selected_ids=tuple(a1_packet_ids),
                selector_policy="A1_USE_CERTIFIED_MEMORY", identity_status=probe.status)
            prompt, packet = run_packet_stage(arm, q, selection, texts, retrieval)

            a1_identity = ExecutionIdentity(
                task_id=task["task_id"], arm_id="A1_USE_CERTIFIED_MEMORY",
                prompt_hash=packet.prompt_hash, retrieval_config_hash="C2",
                selector_config_hash="s2_v2+s4_composer_v1",
                graph_compressor_config_hash=extractor_hash,
                model_revision="sapientinc/HRM-Text-1B@9f082d68",
                pipeline_version=PIPELINE_VERSION, source_commit=source_commit,
                extra_config_hashes={
                    "graph_hash": ghash, "path_set_hash": pshash,
                    "composed_packet_hash": composed_packet_hash(tuple(a1_packet_ids))})

            a1_receipt: dict[str, Any] = {
                "task_id": task["task_id"], "action": "A1_USE_CERTIFIED_MEMORY", "family": fam,
                "suite_family": family,
                "packet_ids": list(packet.packet_ids),
                "candidate_pool_hash": packet.candidate_pool_hash,
                "graph_hash": ghash, "path_set_hash": pshash,
                "composed_packet_hash": composed_packet_hash(tuple(a1_packet_ids)),
                "membership_hash": packet.membership_hash, "order_hash": packet.order_hash,
                "prompt_hash": packet.prompt_hash,
                "execution_identity_sha256": a1_identity.canonical_sha256(),
                "required_in_packet": bool(required) and required <= set(packet.packet_ids),
                "identity_status": probe.status,  # legitimate here: POST_RETRIEVAL, not fed to Executive v0
            }
            if args.execute:
                hrm_result = _run_hrm_batch(adapter, condition, [prompt])[0]

                class _Pre:
                    pass
                pre = _Pre(); pre.task_id = task["task_id"]; pre.arm_id = "A1_USE_CERTIFIED_MEMORY"; pre.packet = packet
                _assert_prompt_binding(pre, hrm_result)
                correct = task.get("answer", "").strip().lower() in hrm_result.output.strip().lower()
                a1_receipt.update({
                    "output": hrm_result.output, "correct": correct,
                    "generation_hash": generation_hash(hrm_result.output),
                    "prompt_tokens": hrm_result.prompt_tokens,
                    "completion_tokens": hrm_result.completion_tokens,
                    "latency_seconds": hrm_result.latency_seconds})
            else:
                a1_receipt["output"] = None
            receipts.append(a1_receipt)

        print(" " * 44, end="\r")
        print(f"  {family}: done  tasks={len(tasks)}")

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_executive/exec_training_v1_{'dry_run' if args.dry_run else 'collection'}.receipts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in receipts) + "\n")

    report = {
        "schema_version": "exec-training-v1-collection", "mode": "dry_run" if args.dry_run else "execute",
        "certified_memory_v1_identity_hash": certified_identity.canonical_sha256(),
        "source_commit": source_commit, "families_run": families,
        "receipts_written": len(receipts),
        "binding_assertion_failures": "all (fail-closed: a violation raises, not counts)",
    }
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n  written: {report_path}\n  receipts: {out}")
    if args.dry_run:
        print(f"\n  DRY RUN complete: {len(receipts)} receipts (no generation performed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
