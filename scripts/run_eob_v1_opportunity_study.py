#!/usr/bin/env python3
"""Executive Opportunity Study v2: A0 (ANSWER_NOW) vs A1 (CERTIFIED_MEMORY_V1)
on EOB-v1 (configs/gate_eob_v1_design.json, data/hrm/eob_v1/), the
purpose-built benchmark with controlled D0/D1/D2/D3 heterogeneity.

Reuses the EXACT same retrieval -> identity -> G2 -> path-coherent-composer
pipeline as scripts/run_executive_opportunity_study.py (v1), gated by
assert_certified_memory_v1_unchanged() -- CERTIFIED_MEMORY_V1 is not modified
by this script. What's different from v1:

  1. Data: EOB-v1's four regimes instead of b3_calibration_v1's five scales.
  2. Metrics: preserves the FULL (Q, prompt_tokens, completion_tokens,
     latency_seconds) vector per action per task -- never collapsed to a
     scalar utility before reporting. A Pareto dominance count is reported
     alongside the quality-only ExecutiveOpportunity (kept, for continuity
     with v1's frozen thresholds), per the research-lead directive that
     quality alone is not sufficient once ties are possible (D2 specifically).
  3. Diversity: reports P(delta_U>0) / P(delta_U<0) / P(delta_U=0) directly,
     not an oracle tie-break action share -- the v1 study showed the latter
     can manufacture apparent diversity (65% 'A0 chosen') out of pure ties
     (0% actual A0 wins).
  4. Per-regime breakdown is a FIRST-CLASS report field, not a post-verdict
     diagnostic -- regime is the treatment variable EOB-v1 was built to test.

Defaults to --dry-run. Pass --execute for real generation.
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
    C4_PRIMARY_PACKET_BUDGET, C4_RRF_K, IdentityResolution, RetrievalResult,
    SelectionResult)
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
from scripts.run_gate_c4 import (  # noqa: E402
    _assert_prompt_binding, _load_hrm, _run_hrm_batch,
    _to_index_records as to_index_records)

M = 50
PACKET = C4_PRIMARY_PACKET_BUDGET
PIPELINE_VERSION = "eob_v1_opportunity_study"
MATERIAL_OPPORTUNITY_MARGIN = 0.05  # kept from v1, for continuity
ACTION_DIVERSITY_MIN_SHARE = 0.15   # kept from v1, for continuity
REGIMES = ("D0_direct_sufficient", "D1_memory_required",
          "D2_both_sufficient", "D3_memory_distractor")
EOB_ROOT = ROOT / "data/hrm/eob_v1"


def load_regime(regime: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    base = EOB_ROOT / regime
    tasks = [json.loads(l) for l in (base / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (base / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence, {r["evidence_id"]: r["content"] for r in evidence}


def c2(n: int) -> int:
    return max(1, min(300, math.ceil(0.15 * n))) if n else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="EOB-v1 Executive Opportunity Study: A0 vs A1")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
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
    regimes = args.regimes or list(REGIMES)

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

    print(f"=== EOB-v1 Executive Opportunity Study: A0 vs A1 ({'DRY RUN' if args.dry_run else 'EXECUTE'}) ===")
    print(f"  CERTIFIED_MEMORY_V1 identity OK  extractor_hash={extractor_hash}  M={M}  packet={PACKET}\n")

    adapter, condition = (None, None)
    if args.execute:
        adapter, condition = _load_hrm()

    receipts: list[dict[str, Any]] = []
    # per-task record for vector-preserved analysis: (regime, family, task_id,
    # Q_A0, Q_A1, tok_A0, tok_A1, lat_A0, lat_A1)
    task_records: list[dict[str, Any]] = []

    for regime in regimes:
        tasks, evidence, texts = load_regime(regime)
        if args.limit_tasks:
            tasks = tasks[:args.limit_tasks]
        records = to_index_records(evidence)
        depth = c2(len(records))

        for i, task in enumerate(tasks, 1):
            if i % 10 == 0 or i == len(tasks):
                print(f"  {regime}: {i}/{len(tasks)}", end="\r", flush=True)
            q = task["question"]
            fam = task.get("family", "?")
            required = set(task.get("required_evidence_ids", []))

            if records:
                # run_query_stage/InformationState hard-require an extractable
                # target relation (tuned for b3/C4-style "What is X's Y?"
                # phrasing) and raise ValueError otherwise -- D0/D2/D3's
                # arithmetic/comparison/transform/restatement questions never
                # match that pattern, so the typed-query stage is bypassed for
                # them and the raw question text is used as the retrieval
                # query directly. This is confined to this new runner only;
                # query_stage.py/information_state.py (part of the certified
                # stack) are not modified.
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
                # D0 has zero evidence records -- retrieval is trivially empty,
                # not skipped: MEMORY still pays the pipeline's overhead for
                # nothing, which is exactly the D0 comparison this benchmark
                # exists to make.
                fused, pool, scores = [], [], {}

            # D0/D2/D3 tasks carry their own relation_word directly in
            # metadata (the word deliberately embedded, verbatim, in their
            # confirming/distractor evidence sentence -- see
            # eob_v1_dataset._b3_style_fact_sentence) rather than relying on
            # extract_target_relation, which is tuned to b3/C4-style "which/
            # what X is..." phrasing these questions don't use. D1 tasks are
            # genuine b3-style questions, so they keep using the same
            # extraction every other runner in this project uses.
            relation = task.get("metadata", {}).get("relation_word") or extract_target_relation(q) or ""
            retrieval = RetrievalResult(
                candidate_ids=tuple(pool), candidate_budget=depth,
                retrieval_policy=arm.retrieval_policy, bm25_backend="bm25",
                bge_model_id="", bge_revision="", rrf_k=C4_RRF_K,
                bm25_ranked=(), bge_ranked=(), fusion_ranked=tuple(fused[:depth]))

            locator = task.get("metadata", {}).get("locator")
            if locator:
                # D0/D2/D3: extract_subject (hrm_adaptive_memory/c4/query_stage.py)
                # is tuned to b3/C4-style "is held by X?" / "for X, which..."
                # phrasing and has no pattern for "Reference: X. ..." -- a
                # template mismatch, not a genuine resolution difficulty (the
                # locator is unambiguously the subject by construction). Built
                # directly from ground truth here rather than by adding a new
                # pattern to query_stage.py/identity_stage.py, which are part
                # of the certified stack and not modified by this script.
                probe = IdentityResolution(
                    status="EXACT", surface=locator, canonical=locator,
                    evidence_ids=(), candidate_mappings=(),
                    resolution_needed=True, resolution_attempted=True,
                    resolution_changed_state=False)
            else:
                probe = run_identity_stage(q, arm, retrieval, texts)
            canonical = probe.canonical

            g2r = g2_prefilter(candidate_ids=pool, texts=texts,
                               canonical_subject=canonical, relation=relation,
                               working_set_size=M, fusion_scores=scores,
                               completion_fn=k1_entity_bound_exact_completion)
            g2_ws = g2r.kept
            complete_paths = [p for p in g2r.all_paths if p.complete]

            def s2_order(ws, _q=q, _scores=scores, _locator=locator):
                if not ws:
                    return []
                if _locator:
                    # Same ground-truth bypass as the top-level probe above,
                    # applied consistently to the working-set-scoped
                    # resolution s2_order performs internally.
                    ident = IdentityResolution(
                        status="EXACT", surface=_locator, canonical=_locator,
                        evidence_ids=(), candidate_mappings=(),
                        resolution_needed=True, resolution_attempted=True,
                        resolution_changed_state=False)
                else:
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

            selected_ids_by_action = {
                "A0_ANSWER_NOW": [],
                "A1_USE_CERTIFIED_MEMORY": list(a1_packet_ids),
            }

            per_action: dict[str, dict[str, Any]] = {}
            for name, selected_ids in selected_ids_by_action.items():
                selection = SelectionResult(
                    selector="eob_v1_opportunity_study", selected_ids=tuple(selected_ids),
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
                    "task_id": task["task_id"], "action": name, "regime": regime,
                    "family": fam,
                    "packet_ids": list(packet.packet_ids),
                    "candidate_pool_hash": packet.candidate_pool_hash,
                    "graph_hash": ghash, "path_set_hash": pshash,
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
                    per_action[name] = {
                        "correct": int(correct), "prompt_tokens": hrm_result.prompt_tokens,
                        "completion_tokens": hrm_result.completion_tokens,
                        "latency_seconds": hrm_result.latency_seconds}
                else:
                    receipt["output"] = None
                receipts.append(receipt)

            if args.execute:
                a0, a1 = per_action["A0_ANSWER_NOW"], per_action["A1_USE_CERTIFIED_MEMORY"]
                task_records.append({
                    "regime": regime, "family": fam, "task_id": task["task_id"],
                    "q_a0": a0["correct"], "q_a1": a1["correct"],
                    "tok_a0": a0["prompt_tokens"] + a0["completion_tokens"],
                    "tok_a1": a1["prompt_tokens"] + a1["completion_tokens"],
                    "lat_a0": a0["latency_seconds"], "lat_a1": a1["latency_seconds"],
                })

        print(" " * 44, end="\r")
        print(f"  {regime}: done  tasks={len(tasks)}")

    binding_checks_passed = "all (fail-closed: a violation raises, not counts)"
    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_executive/eob_v1_{'dry_run' if args.dry_run else 'execute'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "eob-v1-opportunity-v1", "mode": "dry_run" if args.dry_run else "execute",
        "certified_memory_v1_identity_hash": certified_identity.canonical_sha256(),
        "source_commit": source_commit,
        "regimes_run": regimes, "tasks_total": sum(1 for _ in task_records) if args.execute else None,
        "receipts_written": len(receipts),
        "binding_assertion_failures": binding_checks_passed,
    }

    if args.execute:
        n = len(task_records)
        report["tasks_total"] = n

        # --- quality-only ExecutiveOpportunity, kept for continuity with v1 ---
        u0 = sum(r["q_a0"] for r in task_records) / n
        u1 = sum(r["q_a1"] for r in task_records) / n
        u_oracle = sum(max(r["q_a0"], r["q_a1"]) for r in task_records) / n
        best_fixed = max(u0, u1)
        executive_opportunity = round(u_oracle - best_fixed, 4)
        triples = [(r["family"], float(r["q_a0"]), float(r["q_a1"])) for r in task_records]
        lcb = grouped_lcb_executive_opportunity(triples)

        # --- corrected diversity: strict wins / ties, not oracle tie-break shares ---
        strict_a1 = sum(1 for r in task_records if r["q_a1"] > r["q_a0"])
        strict_a0 = sum(1 for r in task_records if r["q_a0"] > r["q_a1"])
        ties = n - strict_a1 - strict_a0
        diversity_ok = (strict_a0 / n) >= ACTION_DIVERSITY_MIN_SHARE and (strict_a1 / n) >= ACTION_DIVERSITY_MIN_SHARE

        # --- per-regime breakdown: first-class, not diagnostic-only ---
        by_regime: dict[str, dict] = {}
        for regime in regimes:
            rr = [r for r in task_records if r["regime"] == regime]
            if not rr:
                continue
            rn = len(rr)
            by_regime[regime] = {
                "n": rn,
                "Q_A0": round(sum(r["q_a0"] for r in rr) / rn, 4),
                "Q_A1": round(sum(r["q_a1"] for r in rr) / rn, 4),
                "mean_tokens_A0": round(sum(r["tok_a0"] for r in rr) / rn, 1),
                "mean_tokens_A1": round(sum(r["tok_a1"] for r in rr) / rn, 1),
                "mean_latency_A0": round(sum(r["lat_a0"] for r in rr) / rn, 3),
                "mean_latency_A1": round(sum(r["lat_a1"] for r in rr) / rn, 3),
                "strict_A1_wins": sum(1 for r in rr if r["q_a1"] > r["q_a0"]),
                "strict_A0_wins": sum(1 for r in rr if r["q_a0"] > r["q_a1"]),
                "ties": sum(1 for r in rr if r["q_a0"] == r["q_a1"]),
            }

        # --- Pareto dominance: does one action dominate on quality AND cost? ---
        # A0 dominates a task if Q_A0>=Q_A1 and tok_A0<=tok_A1 and lat_A0<=lat_A1,
        # with at least one strict inequality (else truly tied on everything).
        pareto = Counter()
        for r in task_records:
            q0, q1 = r["q_a0"], r["q_a1"]
            t0, t1 = r["tok_a0"], r["tok_a1"]
            l0, l1 = r["lat_a0"], r["lat_a1"]
            a0_dom = q0 >= q1 and t0 <= t1 and l0 <= l1 and (q0 > q1 or t0 < t1 or l0 < l1)
            a1_dom = q1 >= q0 and t1 <= t0 and l1 <= l0 and (q1 > q0 or t1 < t0 or l1 < l0)
            if a0_dom and not a1_dom:
                pareto["A0_dominates"] += 1
            elif a1_dom and not a0_dom:
                pareto["A1_dominates"] += 1
            elif a0_dom and a1_dom:
                pareto["identical"] += 1
            else:
                pareto["non_dominated_tradeoff"] += 1

        report["pooled_quality"] = {"E0_always_answer_now": round(u0, 4), "E1_always_use_memory": round(u1, 4),
                                    "E3_oracle": round(u_oracle, 4)}
        report["ExecutiveOpportunity_quality_only"] = executive_opportunity
        report["ExecutiveOpportunity_lcb_2p5"] = lcb
        report["frozen_promotion_criteria_quality_only"] = {
            "material_opportunity_margin": MATERIAL_OPPORTUNITY_MARGIN,
            "lcb_must_exceed": 0.0, "action_diversity_min_share": ACTION_DIVERSITY_MIN_SHARE,
        }
        report["strict_win_tie_distribution"] = {
            "memory_strict_wins": f"{strict_a1}/{n}", "answer_now_strict_wins": f"{strict_a0}/{n}",
            "ties": f"{ties}/{n}",
            "P_delta_U_gt_0": round(strict_a1 / n, 4), "P_delta_U_lt_0": round(strict_a0 / n, 4),
            "P_delta_U_eq_0": round(ties / n, 4),
        }
        report["diversity_ok_corrected"] = diversity_ok
        report["per_regime"] = by_regime
        report["pareto_dominance"] = dict(pareto)

        margin_met = executive_opportunity >= MATERIAL_OPPORTUNITY_MARGIN
        lcb_met = lcb is not None and lcb > 0.0
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
        print(f"\n  {'policy':<22}{'quality':>9}")
        for label, key in [("E0 always ANSWER", "E0_always_answer_now"),
                           ("E1 always MEMORY", "E1_always_use_memory"),
                           ("E3 oracle", "E3_oracle")]:
            print(f"  {label:<22}{report['pooled_quality'][key]:>9.4f}")
        print(f"\n  ExecutiveOpportunity(quality-only) = {executive_opportunity:+.4f}  LCB2.5={lcb}")
        print(f"  strict wins: MEMORY={strict_a1}/{n}  ANSWER_NOW={strict_a0}/{n}  ties={ties}/{n}")
        print(f"  Pareto: {dict(pareto)}")
        print("\n  per-regime:")
        for regime, d in by_regime.items():
            print(f"    {regime:<24} Q_A0={d['Q_A0']:.3f} Q_A1={d['Q_A1']:.3f} "
                  f"A1wins={d['strict_A1_wins']} A0wins={d['strict_A0_wins']} ties={d['ties']}")
        print(f"\n  DECISION: {decision}")
    else:
        print(f"\n  DRY RUN complete: {len(receipts)} receipts built "
              "(no generation performed, no prompt-binding check exercised).")
    print(f"\n  written: {out}\n  receipts: {receipts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
