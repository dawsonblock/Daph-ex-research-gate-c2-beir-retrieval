#!/usr/bin/env python3
"""Executive Opportunity Study: does observable runtime state predict when
invoking CERTIFIED_MEMORY_V1 beats answering directly?

    A0  ANSWER_NOW              empty evidence packet (frozen composer's
                                 existing '[NO EXTERNAL EVIDENCE]' path)
    A1  USE_CERTIFIED_MEMORY    exactly H2_g2_pathcoherent, gated by
                                 assert_certified_memory_v1_unchanged()

Per configs/gate_executive_opportunity_v1.json, frozen before this ran. This
is NOT controller training -- it establishes whether ExecutiveOpportunity =
U(E3) - max(U(E0), U(E1)) is real, statistically supported (grouped bootstrap
LCB>0), and operationally meaningful (both actions >=15% of oracle picks)
before any executive is built.

Reuses run_hrm_qualification.py's retrieval/identity/graph/composition
pipeline for the A1 arm (same C2 -> G2 -> composer sequence that earned
CONFIRMED_GRAPH_PLUS_PATH_COMPOSITION) rather than reimplementing it, and
run_packet_stage's existing empty-packet path for A0 rather than inventing a
new prompt format.

Defaults to --dry-run: builds every packet/prompt/hash/state-feature and
writes receipts WITHOUT calling the model. Pass --execute for real generation.

Usage:
    python scripts/run_executive_opportunity_study.py --dry-run
    python scripts/run_executive_opportunity_study.py --execute
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
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
    graph_hash, packet_coherence_ratio, path_set_hash)
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation, run_query_stage)
from hrm_adaptive_memory.c4.retrieval_stage import get_cached_backend  # noqa: E402
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph  # noqa: E402
from hrm_adaptive_memory.c4.selector_v2 import select_s2  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.certified_memory import (  # noqa: E402
    CertifiedMemoryDriftError, assert_certified_memory_v1_unchanged,
    pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.experiment_integrity.execution_identity import (  # noqa: E402
    ExecutionIdentity)
from hrm_adaptive_memory.experiment_integrity.executive_bootstrap import (  # noqa: E402
    grouped_lcb_executive_opportunity)
from hrm_adaptive_memory.retrieval_bench.selectors import s0_raw  # noqa: E402
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (  # noqa: E402
    s2c_chain_plus_relation)
from hrm_adaptive_memory.c4.typed_path import typed_path_prefilter  # noqa: E402,F401 (parity import, unused: A0/A1 only, no H0 arm here)
from scripts.diagnose_c4_selector_eligibility import terminal_records  # noqa: E402
from scripts.diagnose_c5_confirmation_stopgate import bridge_records  # noqa: E402
from scripts.run_b3_retrieval_calibration import SCALES, load_scale  # noqa: E402
from scripts.run_gate_c4 import (  # noqa: E402
    _assert_prompt_binding, _load_hrm, _run_hrm_batch,
    _to_index_records as to_index_records)

M = 50
PACKET = C4_PRIMARY_PACKET_BUDGET
PIPELINE_VERSION = "executive_opportunity_v1"
MATERIAL_OPPORTUNITY_MARGIN = 0.05  # frozen, configs/gate_executive_opportunity_v1.json
ACTION_DIVERSITY_MIN_SHARE = 0.15   # frozen, ditto


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


def e2_heuristic(identity_status: str, n_complete_paths: int) -> str:
    """Frozen per configs/gate_executive_opportunity_v1.json E2_FROZEN_HEURISTIC.
    Pre-generation, non-evaluator state only."""
    if identity_status in ("EXACT", "RESOLVED") and n_complete_paths >= 1:
        return "A1_USE_CERTIFIED_MEMORY"
    return "A0_ANSWER_NOW"


def main() -> int:
    ap = argparse.ArgumentParser(description="Executive Opportunity Study: A0 vs A1")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scales", nargs="*", default=None)
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
    scales = args.scales or list(SCALES)

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

    print(f"=== Executive Opportunity Study: A0 vs A1 ({'DRY RUN' if args.dry_run else 'EXECUTE'}) ===")
    print(f"  CERTIFIED_MEMORY_V1 identity OK  extractor_hash={extractor_hash}  M={M}  packet={PACKET}\n")

    adapter, condition = (None, None)
    if args.execute:
        adapter, condition = _load_hrm()

    per_scale: dict[str, Any] = {}
    receipts: list[dict[str, Any]] = []
    triples: list[tuple[str, float, float]] = []  # (family, Q_A0, Q_A1) for the LCB
    e2_correct_total = 0
    e2_n_total = 0
    oracle_choice_counts: Counter = Counter()
    pooled_tasks = 0
    pooled_correct = {"A0_ANSWER_NOW": 0, "A1_USE_CERTIFIED_MEMORY": 0}
    pooled_n = 0

    for scale in scales:
        tasks, evidence, texts = load_scale(scale)
        if args.limit_tasks:
            tasks = tasks[:args.limit_tasks]
        records = to_index_records(evidence)
        depth = c2(len(records))
        scale_correct = {"A0_ANSWER_NOW": 0, "A1_USE_CERTIFIED_MEMORY": 0}
        scale_n = 0

        for i, task in enumerate(tasks, 1):
            if i % 10 == 0 or i == len(tasks):
                print(f"  {scale}: {i}/{len(tasks)}", end="\r", flush=True)
            q = task["question"]
            fam = task.get("family", "?")
            regime = (task.get("metadata") or {}).get("entity_regime", "?")

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
            top_scores = sorted(scores.values(), reverse=True)
            retrieval_score_margin = (top_scores[0] - top_scores[1]) if len(top_scores) >= 2 else (
                top_scores[0] if top_scores else 0.0)
            relation = extract_target_relation(q) or ""

            retrieval = RetrievalResult(
                candidate_ids=tuple(pool), candidate_budget=depth,
                retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                bm25_ranked=(), bge_ranked=(), fusion_ranked=tuple(fused[:depth]))

            probe = run_identity_stage(q, arm, retrieval, texts)
            canonical = probe.canonical
            required = set(task["required_evidence_ids"])

            g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                               canonical_subject=canonical, relation=relation,
                               working_set_size=M, fusion_scores=scores,
                               completion_fn=k1_entity_bound_exact_completion)
            g2_ws = g2r.kept
            complete_paths = [p for p in g2r.all_paths if p.complete]
            n_paths = len(complete_paths)
            bucket = competition_bucket(n_paths)

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

            # --- state features (pre-generation, non-evaluator) ---
            sources_with_edges = {e.source_record_id for e in g2_graph.edges}
            graph_reachability = (len(sources_with_edges & set(pool)) / len(pool)) if pool else 0.0
            working_set_size = len(g2_ws)
            structural_competition_ratio = n_paths / max(1, working_set_size)
            bridge_ids = bridge_records(task)
            terminal_ids = set(terminal_records(task))
            bridge_availability = bool(bridge_ids & set(g2_ws))
            terminal_availability = bool(terminal_ids & set(g2_ws))
            packet_coherence = packet_coherence_ratio(a1_packet_ids, complete_paths)

            selected_ids_by_action = {
                "A0_ANSWER_NOW": [],
                "A1_USE_CERTIFIED_MEMORY": list(a1_packet_ids),
            }

            per_action_correct: dict[str, int] = {}
            per_action_receipt: dict[str, dict[str, Any]] = {}
            for name, selected_ids in selected_ids_by_action.items():
                selection = SelectionResult(
                    selector="executive_opportunity_v1", selected_ids=tuple(selected_ids),
                    selector_policy=name, identity_status=probe.status)
                prompt, packet = run_packet_stage(arm, q, selection, texts, retrieval)

                identity = ExecutionIdentity(
                    task_id=task["task_id"], arm_id=name, prompt_hash=packet.prompt_hash,
                    retrieval_config_hash="C2", selector_config_hash="s2_v2+s4_composer_v1",
                    graph_compressor_config_hash=extractor_hash,
                    model_revision="sapientinc/HRM-Text-1B@9f082d68",
                    pipeline_version=PIPELINE_VERSION,
                    source_commit=source_commit,
                    extra_config_hashes={
                        "graph_hash": ghash, "path_set_hash": pshash,
                        "composed_packet_hash": composed_packet_hash(tuple(selected_ids))})

                receipt: dict[str, Any] = {
                    "task_id": task["task_id"], "action": name, "scale": scale,
                    "family": fam, "entity_regime": regime,
                    "path_competition_bucket": bucket,
                    "packet_ids": list(packet.packet_ids),
                    "candidate_pool_hash": packet.candidate_pool_hash,
                    "graph_hash": ghash, "path_set_hash": pshash,
                    "composed_packet_hash": composed_packet_hash(tuple(selected_ids)),
                    "membership_hash": packet.membership_hash,
                    "order_hash": packet.order_hash, "prompt_hash": packet.prompt_hash,
                    "execution_identity_sha256": identity.canonical_sha256(),
                    "required_in_packet": bool(required) and required <= set(packet.packet_ids),
                    "state_features": {
                        "identity_status": probe.status,
                        "retrieval_score_margin": round(retrieval_score_margin, 6),
                        "graph_reachability": round(graph_reachability, 4),
                        "working_set_size": working_set_size,
                        "n_complete_paths": n_paths,
                        "path_competition_bucket": bucket,
                        "structural_competition_ratio": round(structural_competition_ratio, 4),
                        "bridge_availability_estimate": bridge_availability,
                        "terminal_availability_estimate": terminal_availability,
                        "packet_coherence": (packet_coherence if isinstance(packet_coherence, str)
                                             else round(packet_coherence, 4)),
                    },
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
                    per_action_correct[name] = int(correct)
                else:
                    receipt["output"] = None
                per_action_receipt[name] = receipt
                receipts.append(receipt)

            if args.execute:
                q_a0 = per_action_correct["A0_ANSWER_NOW"]
                q_a1 = per_action_correct["A1_USE_CERTIFIED_MEMORY"]
                triples.append((fam, float(q_a0), float(q_a1)))
                pooled_correct["A0_ANSWER_NOW"] += q_a0
                pooled_correct["A1_USE_CERTIFIED_MEMORY"] += q_a1
                scale_correct["A0_ANSWER_NOW"] += q_a0
                scale_correct["A1_USE_CERTIFIED_MEMORY"] += q_a1
                pooled_n += 1
                scale_n += 1

                oracle_action = ("A0_ANSWER_NOW" if q_a0 >= q_a1 else "A1_USE_CERTIFIED_MEMORY")
                oracle_choice_counts[oracle_action] += 1

                e2_action = e2_heuristic(probe.status, n_paths)
                e2_correct_total += per_action_correct[e2_action]
                e2_n_total += 1

        print(" " * 44, end="\r")
        if args.execute:
            per_scale[scale] = {
                "tasks": scale_n,
                "A0_ANSWER_NOW_quality": round(scale_correct["A0_ANSWER_NOW"] / max(1, scale_n), 4),
                "A1_USE_CERTIFIED_MEMORY_quality": round(scale_correct["A1_USE_CERTIFIED_MEMORY"] / max(1, scale_n), 4),
            }
        pooled_tasks += len(tasks)
        print(f"  {scale}: done  tasks={len(tasks)}")

    binding_checks_passed = "all (fail-closed: a violation raises, not counts)"

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_executive/opportunity_{'dry_run' if args.dry_run else 'execute'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "executive-opportunity-v1", "mode": "dry_run" if args.dry_run else "execute",
        "certified_memory_v1_identity_hash": certified_identity.canonical_sha256(),
        "source_commit": source_commit,
        "scales_run": scales, "tasks_total": pooled_tasks,
        "receipts_written": len(receipts),
        "binding_assertion_failures": binding_checks_passed,
    }

    if args.execute:
        u_a0 = pooled_correct["A0_ANSWER_NOW"] / pooled_n
        u_a1 = pooled_correct["A1_USE_CERTIFIED_MEMORY"] / pooled_n
        u_e2 = e2_correct_total / max(1, e2_n_total)
        u_e3 = sum(max(q0, q1) for _, q0, q1 in triples) / len(triples)
        best_fixed = max(u_a0, u_a1)
        executive_opportunity = round(u_e3 - best_fixed, 4)
        lcb = grouped_lcb_executive_opportunity(triples)

        oracle_a0_share = oracle_choice_counts["A0_ANSWER_NOW"] / pooled_n
        oracle_a1_share = oracle_choice_counts["A1_USE_CERTIFIED_MEMORY"] / pooled_n
        diversity_ok = (oracle_a0_share >= ACTION_DIVERSITY_MIN_SHARE
                        and oracle_a1_share >= ACTION_DIVERSITY_MIN_SHARE)
        margin_met = executive_opportunity >= MATERIAL_OPPORTUNITY_MARGIN
        lcb_met = lcb is not None and lcb > 0.0

        report["per_scale"] = per_scale
        report["pooled_quality"] = {
            "E0_always_answer_now": round(u_a0, 4),
            "E1_always_use_memory": round(u_a1, 4),
            "E2_simple_heuristic": round(u_e2, 4),
            "E3_oracle": round(u_e3, 4),
        }
        report["ExecutiveOpportunity"] = executive_opportunity
        report["ExecutiveOpportunity_lcb_2p5"] = lcb
        report["frozen_promotion_criteria"] = {
            "material_opportunity_margin": MATERIAL_OPPORTUNITY_MARGIN,
            "lcb_must_exceed": 0.0,
            "action_diversity_min_share": ACTION_DIVERSITY_MIN_SHARE,
        }
        report["oracle_action_shares"] = {
            "A0_ANSWER_NOW": round(oracle_a0_share, 4),
            "A1_USE_CERTIFIED_MEMORY": round(oracle_a1_share, 4),
        }
        report["margin_met"] = margin_met
        report["lcb_met"] = lcb_met
        report["diversity_ok"] = diversity_ok

        if not margin_met or not lcb_met:
            decision = "NO_EXECUTIVE_OPPORTUNITY"
        elif not diversity_ok:
            decision = "EXPANSION_NOT_WARRANTED"
        else:
            decision = "EXECUTIVE_OPPORTUNITY_CONFIRMED"
        report["decision"] = decision

    receipts_path = out.with_suffix(".receipts.jsonl")
    receipts_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in receipts) + "\n")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.execute:
        print(f"\n  {'policy':<24}{'quality':>9}")
        for label, key in [("E0 always ANSWER_NOW", "E0_always_answer_now"),
                           ("E1 always USE_MEMORY", "E1_always_use_memory"),
                           ("E2 simple heuristic", "E2_simple_heuristic"),
                           ("E3 oracle", "E3_oracle")]:
            print(f"  {label:<24}{report['pooled_quality'][key]:>9.4f}")
        print(f"\n  ExecutiveOpportunity = {executive_opportunity:+.4f}  LCB2.5={lcb}")
        print(f"  oracle shares: A0={oracle_a0_share:.3f}  A1={oracle_a1_share:.3f}")
        print(f"\n  DECISION: {decision}")
    else:
        print(f"\n  DRY RUN complete: {len(receipts)} receipts built "
              "(no generation performed, no prompt-binding check exercised).")
    print(f"\n  written: {out}\n  receipts: {receipts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
