#!/usr/bin/env python3
"""Gate C4-BRIDGE — Runtime Information-State Acquisition.

Qualification gate for the one remaining unqualified C4 mechanism:
runtime bridge discovery from first-pass retrieval evidence.

Question: Can runtime-visible first-pass evidence recover the correct
bridge/state needed by the already-qualified subject+bridge+relation
query formulation?

No HRM. Uses the existing frozen 120-task development corpus.

Arms:
    B0: one-pass baseline (no bridge, no second retrieval)
    B1: current heuristic bridge detector (regex-based)
    B2: deterministic relation/link parser (relational_state)
    B3: entity-connectivity / relational-state inference (relational_state + connectivity)
    B4: oracle bridge + same real second retrieval (ceiling)

B4 is evaluator-only and supplies only the correct bridge/state, then
uses the same real retriever. It does NOT inject required evidence.

Usage:
    python scripts/run_gate_c4_bridge.py
    python scripts/run_gate_c4_bridge.py --output-dir evidence/gate_c4/bridge
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.c4.contracts import C4Arm, C4_CANDIDATE_BUDGET, C4_RRF_K
from hrm_adaptive_memory.c4.query_stage import run_query_stage, extract_subject, extract_target_relation
from hrm_adaptive_memory.c4.retrieval_stage import run_retrieval_stage, clear_backend_cache
from hrm_adaptive_memory.c4.bridge_extraction import extract_bridge, extract_v4_entities
from hrm_adaptive_memory.c4.relational_state import (
    build_relational_state, get_bridge, is_target_bound, parse_relation_edges,
    RelationalState, RelationFact,
)
from hrm_adaptive_memory.c4.provenance import (
    build_manifest, write_results_hash, sha256_corpus)
from hrm_adaptive_memory.retrieval.information_state import (
    InformationState, formulate_followup, FOLLOWUP_FORMULATION)

CORPUS = ROOT / "data/hrm/controlled_gate_a_v4"
OUT = ROOT / "evidence/gate_c4/bridge"

# Use C4-1 arm configuration (subject_preserving, bm25_only)
_ARM = C4Arm(
    arm_id="bridge_test", description="bridge qualification",
    query_policy="subject_preserving", retrieval_policy="bm25_only",
    identity_policy="none", selector_policy="s0",
    evidence_policy="bounded_packet", packet_budget=6,
)

BRIDGE_ARMS = ["B0", "B1", "B2", "B3", "B4"]


def _load_split(split: str = "development"):
    tasks = [json.loads(l) for l in (CORPUS / split / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / split / "evidence.jsonl").read_text().splitlines() if l.strip()]
    texts = {r["evidence_id"]: r["content"] for r in evidence}
    records = [IndexRecord(
        evidence_id=r["evidence_id"], source_id=r.get("source_id", ""),
        content=r["content"], token_count=max(1, len(r["content"].split())),
        source_type=r.get("source_type", ""), metadata=r.get("metadata", {}),
    ) for r in evidence]
    return tasks, evidence, texts, records


def _required_evidence_recall(required_ids: list[str], candidate_ids: tuple[str, ...]) -> float:
    """Fraction of required evidence IDs in the candidate pool."""
    if not required_ids:
        return 1.0
    pool = set(candidate_ids)
    found = sum(1 for eid in required_ids if eid in pool)
    return found / len(required_ids)


def _complete_evidence_set(required_ids: list[str], candidate_ids: tuple[str, ...]) -> bool:
    """All required evidence IDs are in the candidate pool."""
    if not required_ids:
        return True
    pool = set(candidate_ids)
    return all(eid in pool for eid in required_ids)


def _merge_candidates(first: tuple[str, ...], second: tuple[str, ...], budget: int) -> tuple[str, ...]:
    """Merge first and second pass candidates, deduplicated, bounded."""
    seen = set()
    merged = []
    # Interleave for fairness
    for i in range(max(len(first), len(second))):
        if i < len(first) and first[i] not in seen:
            seen.add(first[i])
            merged.append(first[i])
        if i < len(second) and second[i] not in seen:
            seen.add(second[i])
            merged.append(second[i])
        if len(merged) >= budget:
            break
    return tuple(merged)


def _get_oracle_bridge(task: dict, texts: dict[str, str]) -> str | None:
    """Get the correct bridge from oracle metadata (evaluator-only)."""
    meta = task.get("_oracle_metadata", {})
    return meta.get("surfaces", {}).get("bridge")


def run_bridge_arm(
    arm_id: str,
    task: dict,
    texts: dict[str, str],
    records: list[IndexRecord],
) -> dict:
    """Run one bridge arm for one task. Returns a result dict."""
    question = task["question"]
    subject = extract_subject(question)
    target_rel = extract_target_relation(question) or ""
    required_ids = task.get("required_evidence_ids", [])

    # First-pass query + retrieval (same for all arms)
    state_q, query_result = run_query_stage(question, _ARM)
    first_retrieval = run_retrieval_stage(query_result.rendered_query, _ARM, records)
    first_candidates = first_retrieval.candidate_ids

    # First-pass metrics
    recall_before = _required_evidence_recall(required_ids, first_candidates)
    ces_before = _complete_evidence_set(required_ids, first_candidates)

    # Arm-specific bridge selection
    bridge = None
    second_query = None
    second_candidates: tuple[str, ...] = ()
    merged_candidates = first_candidates

    if arm_id == "B0":
        # No bridge, no second retrieval
        pass
    elif arm_id == "B1":
        # Heuristic bridge detector (regex-based, from bridge_extraction.py)
        bridge = extract_bridge(subject, question, first_candidates, texts)
    elif arm_id == "B2":
        # Deterministic relation/link parser (relational_state)
        rstate = build_relational_state(subject, target_rel, first_candidates, texts, question=question)
        bridge = get_bridge(rstate)
    elif arm_id == "B3":
        # Entity-connectivity / relational-state inference
        # Same as B2 but with connectivity-weighted bridge selection
        rstate = build_relational_state(subject, target_rel, first_candidates, texts, question=question)
        # B3 uses the connectivity-sorted bridges (already done in build_relational_state)
        bridge = get_bridge(rstate)
    elif arm_id == "B4":
        # Oracle bridge (evaluator-only)
        bridge = _get_oracle_bridge(task, texts)

    # Second-pass retrieval if bridge found
    if bridge:
        state_with_bridge = state_q.with_bridge(bridge)
        second_query = formulate_followup(state_with_bridge, formulation=FOLLOWUP_FORMULATION)
        second_retrieval = run_retrieval_stage(second_query, _ARM, records)
        second_candidates = second_retrieval.candidate_ids
        merged_candidates = _merge_candidates(first_candidates, second_candidates, C4_CANDIDATE_BUDGET)

    # Second-pass metrics
    recall_after = _required_evidence_recall(required_ids, merged_candidates)
    ces_after = _complete_evidence_set(required_ids, merged_candidates)

    # Oracle bridge (for scoring, all arms)
    oracle_bridge = _get_oracle_bridge(task, texts)
    bridge_needed = oracle_bridge is not None
    bridge_correct = (bridge is not None and oracle_bridge is not None and
                      bridge.lower() == oracle_bridge.lower())
    bridge_false = (bridge is not None and (oracle_bridge is None or
                     bridge.lower() != oracle_bridge.lower()))
    bridge_missed = (bridge is None and bridge_needed)

    # Second pass classification
    if bridge is None:
        second_pass_class = "neutral"  # no second pass
    elif recall_after > recall_before:
        second_pass_class = "positive"
    elif recall_after < recall_before:
        second_pass_class = "negative"
    else:
        second_pass_class = "neutral"

    return {
        "task_id": task["task_id"],
        "arm_id": arm_id,
        "question": question,
        "subject": subject,
        "target_relation": target_rel,
        "bridge_needed": bridge_needed,
        "bridge_predicted": bridge is not None,
        "bridge_correct": bridge_correct,
        "bridge_false": bridge_false,
        "bridge_missed": bridge_missed,
        "extracted_bridge": bridge,
        "oracle_bridge": oracle_bridge,
        "second_pass_performed": bridge is not None,
        "second_query": second_query,
        "first_pass_ids": list(first_candidates),
        "second_pass_ids": list(second_candidates),
        "merged_ids": list(merged_candidates),
        "required_evidence_ids": required_ids,
        "recall_before": recall_before,
        "recall_after": recall_after,
        "ces_before": ces_before,
        "ces_after": ces_after,
        "delta_recall": recall_after - recall_before,
        "delta_ces": int(ces_after) - int(ces_before),
        "second_pass_class": second_pass_class,
        "family": task.get("family", "unknown"),
        "source_cluster_id": task.get("source_cluster_id", "unknown"),
        "template_id": task.get("template_id", "unknown"),
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics for one arm."""
    n = len(results)
    if n == 0:
        return {}

    bridge_needed = sum(1 for r in results if r["bridge_needed"])
    bridge_predicted = sum(1 for r in results if r["bridge_predicted"])
    bridge_correct = sum(1 for r in results if r["bridge_correct"])
    bridge_false = sum(1 for r in results if r["bridge_false"])
    bridge_missed = sum(1 for r in results if r["bridge_missed"])

    second_pass = sum(1 for r in results if r["second_pass_performed"])
    sp_positive = sum(1 for r in results if r["second_pass_class"] == "positive")
    sp_negative = sum(1 for r in results if r["second_pass_class"] == "negative")
    sp_neutral = sum(1 for r in results if r["second_pass_class"] == "neutral")

    recall_before = sum(r["recall_before"] for r in results) / n
    recall_after = sum(r["recall_after"] for r in results) / n
    ces_before = sum(1 for r in results if r["ces_before"]) / n
    ces_after = sum(1 for r in results if r["ces_after"]) / n

    return {
        "n": n,
        "BridgeNeededRate": bridge_needed / n,
        "BridgePredictionRate": bridge_predicted / n,
        "CorrectBridgeRate": bridge_correct / n,
        "FalseBridgeRate": bridge_false / n,
        "MissedBridgeRate": bridge_missed / n,
        "PrecisionAmongPredictedBridges": bridge_correct / bridge_predicted if bridge_predicted else 0,
        "RecallAmongNeededBridges": bridge_correct / bridge_needed if bridge_needed else 0,
        "SecondPassRate": second_pass / n,
        "UsefulSecondPassRate": sp_positive / n,
        "HarmfulSecondPassRate": sp_negative / n,
        "NeutralSecondPassRate": sp_neutral / n,
        "RequiredEvidenceRecall_before": recall_before,
        "RequiredEvidenceRecall_after": recall_after,
        "CompleteEvidenceSet_before": ces_before,
        "CompleteEvidenceSet_after": ces_after,
        "DeltaRecall": recall_after - recall_before,
        "DeltaCES": ces_after - ces_before,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate C4-BRIDGE: Runtime Information-State Acquisition")
    parser.add_argument("--split", default="development")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    tasks, evidence, texts, records = _load_split(args.split)
    print(f"C4-BRIDGE: {len(tasks)} tasks × {len(BRIDGE_ARMS)} arms on {args.split}")

    all_results: dict[str, list[dict]] = {}
    for arm_id in BRIDGE_ARMS:
        print(f"  {arm_id}...", end=" ", flush=True)
        t0 = time.time()
        results = []
        for task in tasks:
            r = run_bridge_arm(arm_id, task, texts, records)
            results.append(r)
        all_results[arm_id] = results
        print(f"{time.time()-t0:.1f}s")

    clear_backend_cache()

    # Compute metrics
    metrics = {}
    for arm_id in BRIDGE_ARMS:
        metrics[arm_id] = compute_metrics(all_results[arm_id])

    # Print summary
    print(f"\n=== C4-BRIDGE Results ({args.split}) ===")
    print(f"{'Arm':<6} {'BndR':>6} {'PredR':>6} {'CorrR':>6} {'FalseR':>6} {'MissR':>6} "
          f"{'Prec':>6} {'Recall':>6} {'SP+':>6} {'SP-':>6} {'RclBef':>7} {'RclAft':>7} {'CESBef':>7} {'CESAft':>7}")
    for arm_id in BRIDGE_ARMS:
        m = metrics[arm_id]
        print(f"{arm_id:<6} {m['BridgeNeededRate']:>6.3f} {m['BridgePredictionRate']:>6.3f} "
              f"{m['CorrectBridgeRate']:>6.3f} {m['FalseBridgeRate']:>6.3f} {m['MissedBridgeRate']:>6.3f} "
              f"{m['PrecisionAmongPredictedBridges']:>6.3f} {m['RecallAmongNeededBridges']:>6.3f} "
              f"{m['UsefulSecondPassRate']:>6.3f} {m['HarmfulSecondPassRate']:>6.3f} "
              f"{m['RequiredEvidenceRecall_before']:>7.3f} {m['RequiredEvidenceRecall_after']:>7.3f} "
              f"{m['CompleteEvidenceSet_before']:>7.3f} {m['CompleteEvidenceSet_after']:>7.3f}")

    # Promotion check: B2/B3 must beat B0
    print("\n=== Promotion Check ===")
    b0_ces = metrics["B0"]["CompleteEvidenceSet_after"]
    b0_recall = metrics["B0"]["RequiredEvidenceRecall_after"]
    for arm_id in ["B1", "B2", "B3"]:
        m = metrics[arm_id]
        ces_gain = m["CompleteEvidenceSet_after"] - b0_ces
        recall_gain = m["RequiredEvidenceRecall_after"] - b0_recall
        useful_gt_harmful = m["UsefulSecondPassRate"] > m["HarmfulSecondPassRate"]
        beats_b0 = ces_gain > 0 or recall_gain > 0
        print(f"  {arm_id}: CES_gain={ces_gain:+.4f} Recall_gain={recall_gain:+.4f} "
              f"Useful>Harmful={useful_gt_harmful} BeatsB0={beats_b0}")

    # Oracle gap capture
    b4_ces = metrics["B4"]["CompleteEvidenceSet_after"]
    b4_recall = metrics["B4"]["RequiredEvidenceRecall_after"]
    if b4_ces - b0_ces > 0:
        for arm_id in ["B1", "B2", "B3"]:
            ces_gap = (metrics[arm_id]["CompleteEvidenceSet_after"] - b0_ces) / (b4_ces - b0_ces)
            print(f"  BridgeGapCapture({arm_id}) = {ces_gap:.4f}")
    if b4_recall - b0_recall > 0:
        for arm_id in ["B1", "B2", "B3"]:
            recall_gap = (metrics[arm_id]["RequiredEvidenceRecall_after"] - b0_recall) / (b4_recall - b0_recall)
            print(f"  RecallGapCapture({arm_id}) = {recall_gap:.4f}")

    # Write output
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-task results
    for arm_id in BRIDGE_ARMS:
        (out_dir / f"{arm_id}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in all_results[arm_id]))

    # Metrics
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    # Manifest
    import hashlib
    protocol_hash = hashlib.sha256(
        (ROOT / "configs/gate_c4_protocol.json").read_bytes()).hexdigest()
    manifest = build_manifest(
        repo=ROOT, mode="c4_bridge", split=args.split, arm_ids=BRIDGE_ARMS,
        task_count=len(tasks),
        protocol_sha256=protocol_hash,
        task_corpus_sha256=sha256_corpus(CORPUS / args.split / "oracle_tasks.jsonl"),
        evidence_corpus_sha256=sha256_corpus(CORPUS / args.split / "evidence.jsonl"),
        candidate_budget=C4_CANDIDATE_BUDGET,
        rrf_k=C4_RRF_K,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # RESULTS.sha256
    write_results_hash(out_dir)

    print(f"\n  results: {out_dir}")


if __name__ == "__main__":
    main()
