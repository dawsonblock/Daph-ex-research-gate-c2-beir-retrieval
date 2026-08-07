#!/usr/bin/env python3
"""Gate C4 runner — integrated non-oracle memory pipeline.

Orchestrates the stage-composition harness. Each arm is a configuration over
shared stages, not a separate code path.

Modes:
    dry-run: CPU-only pre-HRM stages for all tasks, validate parity + leakage.
    smoke:   5 tasks × 7 arms with HRM (model load check).
    full:    All tasks × all arms with HRM, resumable.

Protocol hash is embedded in every manifest to prevent config drift.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.c4.contracts import *
from hrm_adaptive_memory.c4.arms import ARMS, PRIMARY_ORDER
from hrm_adaptive_memory.c4.query_stage import run_query_stage
from hrm_adaptive_memory.c4.retrieval_stage import run_retrieval_stage
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage
from hrm_adaptive_memory.c4.selection_stage import run_selection_stage
from hrm_adaptive_memory.c4.packet_stage import run_packet_stage
from hrm_adaptive_memory.c4.receipts import build_pre_hrm_receipt, assert_runtime_clean
from hrm_adaptive_memory.c4.parity import (
    validate_all_parity, validate_no_leakage,
    validate_selected_in_pool, validate_packet_budgets)

CORPUS = ROOT / "data/hrm/controlled_gate_a_v4"
OUT = ROOT / "evidence/gate_c4"


def _protocol_hash() -> str:
    return hashlib.sha256(
        (ROOT / "configs/gate_c4_protocol.json").read_bytes()).hexdigest()


def _load_split(split: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    tasks = [json.loads(l) for l in (CORPUS / split / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / split / "evidence.jsonl").read_text().splitlines() if l.strip()]
    texts = {r["evidence_id"]: r["content"] for r in evidence}
    return tasks, evidence, texts


def _to_index_records(evidence: list[dict]) -> list[IndexRecord]:
    return [IndexRecord(
        evidence_id=r["evidence_id"], source_id=r["source_id"],
        content=r["content"], token_count=max(1, len(r["content"].split())),
        source_type=r["source_type"], metadata=r["metadata"],
    ) for r in evidence]


def _state_to_dict(state) -> dict:
    return {
        "subject": state.subject,
        "target_relation": state.target_relation,
        "bridge": state.bridge,
        "canonical_subject": state.canonical_subject,
        "resolved_identities": list(state.resolved_identities),
        "hop": state.hop,
    }


def run_pre_hrm_stages(task: dict, arm: C4Arm, records: list[IndexRecord],
                       texts: dict[str, str]) -> PreHRMResult:
    """Run all pre-HRM stages for one task under one arm."""
    question = task["question"]
    meta = task["_oracle_metadata"]
    split = task.get("split", "development")

    # Query stage
    state_before, query_result = run_query_stage(question, arm)

    # Retrieval stage
    retrieval_result = run_retrieval_stage(query_result.rendered_query, arm, records)

    # Identity stage
    identity_result = run_identity_stage(question, arm, retrieval_result, texts)

    # If identity resolved, update state and potentially reformulate query
    state_after = state_before
    canonical_question = None
    if identity_result.status == "RESOLVED" and identity_result.canonical:
        state_after = state_before.with_identity(
            identity_result.surface or state_before.subject,
            identity_result.canonical)
        canonical_question = question.replace(
            identity_result.surface or "",
            identity_result.canonical)

    # Selection stage
    selection_result = run_selection_stage(
        arm, question, retrieval_result, identity_result, texts,
        task.get("required_evidence_ids", []),
        canonical_question=canonical_question)

    # Packet stage
    prompt, packet_result = run_packet_stage(arm, question, selection_result, texts)

    return PreHRMResult(
        task_id=task["task_id"],
        arm_id=arm.arm_id,
        split=split,
        query=query_result,
        retrieval=retrieval_result,
        identity=identity_result,
        selection=selection_result,
        packet=packet_result,
        information_state_before=_state_to_dict(state_before),
        information_state_after=_state_to_dict(state_after),
    )


def run_dry_run(split: str = "development", arm_ids: list[str] | None = None):
    """CPU-only dry run: all pre-HRM stages, validate parity + leakage."""
    if arm_ids is None:
        arm_ids = [a for a in PRIMARY_ORDER if a != "C4_6"]  # C4-6 needs oracle evidence
    tasks, evidence, texts = _load_split(split)
    records = _to_index_records(evidence)
    print(f"Dry run: {len(tasks)} tasks × {len(arm_ids)} arms on {split}")

    all_results: dict[str, list[PreHRMResult]] = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        print(f"  {arm_id}...", end=" ", flush=True)
        t0 = time.time()
        results = []
        for task in tasks:
            r = run_pre_hrm_stages(task, arm, records, texts)
            results.append(r)
        all_results[arm_id] = results
        print(f"{time.time()-t0:.1f}s")

    # Validation gates
    print("\n=== Validation ===")

    ok, violations = validate_no_leakage(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: no oracle leakage ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")

    ok, violations = validate_all_parity(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: arm parity ({len(violations)} violations)")
    for v in violations[:10]: print(f"    {v}")

    ok, violations = validate_selected_in_pool(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: selected IDs in pool ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")

    ok, violations = validate_packet_budgets(all_results)
    print(f"  {'PASS' if ok else 'FAIL'}: packet budgets ({len(violations)} violations)")
    for v in violations[:5]: print(f"    {v}")

    # Write dry-run receipts
    out_dir = OUT / "dry_run" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm_id, results in all_results.items():
        receipts = [build_pre_hrm_receipt(r) for r in results]
        (out_dir / f"{arm_id}_dry.jsonl").write_text(
            "".join(json.dumps({
                "task_id": r.task_id, "arm_id": r.arm_id, "split": r.split,
                "runtime_payload": r.runtime_payload,
                "evaluator_annotation": r.evaluator_annotation,
            }, sort_keys=True) + "\n" for r in receipts))

    # Manifest
    manifest = {
        "mode": "dry_run",
        "split": split,
        "protocol_sha256": _protocol_hash(),
        "arm_ids": arm_ids,
        "task_count": len(tasks),
        "validation": {
            "no_leakage": validate_no_leakage(all_results)[0],
            "parity": validate_all_parity(all_results)[0],
            "selected_in_pool": validate_selected_in_pool(all_results)[0],
            "packet_budgets": validate_packet_budgets(all_results)[0],
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n  receipts: {out_dir}")

    all_pass = all(manifest["validation"].values())
    if all_pass:
        print("  ALL VALIDATION GATES PASSED — safe to proceed to HRM")
    else:
        print("  VALIDATION FAILED — do NOT start HRM")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Gate C4 runner")
    parser.add_argument("mode", choices=["dry-run", "smoke", "full"])
    parser.add_argument("--split", default="development")
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()

    if args.mode == "dry-run":
        run_dry_run(args.split, args.arms)
    elif args.mode == "smoke":
        print("Smoke mode not yet implemented (requires HRM model loading)")
        sys.exit(1)
    elif args.mode == "full":
        print("Full mode not yet implemented (requires HRM model loading)")
        sys.exit(1)


if __name__ == "__main__":
    main()
