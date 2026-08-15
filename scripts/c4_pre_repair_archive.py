#!/usr/bin/env python3
"""Phase 0: Freeze current evidence and generate comparison receipt.

Archives both existing C4 development runs and produces a machine-readable
comparison receipt proving the nondeterminism defect existed before repair.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evidence/c4_pre_repair"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=30).decode().strip()


def _protocol_hash() -> str:
    p = ROOT / "configs/gate_c4_protocol.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _compare_arms(run_a: Path, run_b: Path, arm_id: str) -> dict:
    """Compare two runs for a specific arm."""
    file_a = run_a / f"{arm_id}.jsonl"
    file_b = run_b / f"{arm_id}.jsonl"
    
    if not file_a.exists() or not file_b.exists():
        return {"arm_id": arm_id, "error": "missing file"}
    
    receipts_a = {r["task_id"]: r for r in _load_jsonl(file_a)}
    receipts_b = {r["task_id"]: r for r in _load_jsonl(file_b)}
    
    common = sorted(set(receipts_a) & set(receipts_b))
    
    same_candidates = 0
    same_identities = 0
    diff_selected_order = 0
    diff_selected_membership = 0
    diff_packet = 0
    
    for tid in common:
        a = receipts_a[tid]
        b = receipts_b[tid]
        
        # Compare candidate pools
        ca = set(a["runtime_payload"]["retrieval"]["candidate_ids"])
        cb = set(b["runtime_payload"]["retrieval"]["candidate_ids"])
        if ca == cb:
            same_candidates += 1
        
        # Compare identities
        ia = a["runtime_payload"]["identity"]["status"]
        ib = b["runtime_payload"]["identity"]["status"]
        if ia == ib:
            same_identities += 1
        
        # Compare selected IDs
        sa = list(a["runtime_payload"]["selection"]["selected_ids"])
        sb = list(b["runtime_payload"]["selection"]["selected_ids"])
        
        if sa == sb:
            pass  # identical
        else:
            diff_packet += 1
            if set(sa) == set(sb):
                diff_selected_order += 1
            else:
                diff_selected_membership += 1
    
    n = len(common)
    return {
        "arm_id": arm_id,
        "n_common": n,
        "same_candidates": same_candidates,
        "same_candidates_rate": same_candidates / n if n else 0,
        "same_identities": same_identities,
        "same_identities_rate": same_identities / n if n else 0,
        "different_selected_order": diff_selected_order,
        "different_selected_membership": diff_selected_membership,
        "total_packet_differences": diff_packet,
        "total_packet_diff_rate": diff_packet / n if n else 0,
    }


def main():
    run_a = ARCHIVE / "run_A"
    run_b = ARCHIVE / "run_B"
    comparison_dir = ARCHIVE / "comparison"
    
    print("=== Phase 0: Freeze Current Evidence ===\n")
    
    # Record metadata
    git_commit = _git_commit()
    protocol_hash = _protocol_hash()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Compute archive hashes
    archive_info = {}
    for run_name, run_dir in [("run_A", run_a), ("run_B", run_b)]:
        archive_info[run_name] = {
            "source": "development_v1_superseded" if run_name == "run_A" else "development_evaluator_v2",
            "git_commit": git_commit,
            "protocol_sha256": protocol_hash,
            "timestamp": timestamp,
            "files": {},
        }
        for f in sorted(run_dir.glob("*.jsonl")):
            archive_info[run_name]["files"][f.name] = _file_sha256(f)
        for f in sorted(run_dir.glob("*.json")):
            archive_info[run_name]["files"][f.name] = _file_sha256(f)
    
    # Compare all arms
    print("Comparing runs...")
    arm_comparisons = {}
    for arm_id in ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]:
        cmp = _compare_arms(run_a, run_b, arm_id)
        arm_comparisons[arm_id] = cmp
        if "error" not in cmp:
            print(f"  {arm_id}: same_cand={cmp['same_candidates']}/{cmp['n_common']} "
                  f"same_id={cmp['same_identities']}/{cmp['n_common']} "
                  f"diff_order={cmp['different_selected_order']} "
                  f"diff_membership={cmp['different_selected_membership']} "
                  f"total_diff={cmp['total_packet_differences']}")
    
    # Focus on C4_4
    c4_4_cmp = arm_comparisons.get("C4_4", {})
    
    # Write comparison receipt
    receipt = {
        "schema_version": "c4-pre-repair-comparison-v1",
        "created_utc": timestamp,
        "git_commit": git_commit,
        "protocol_sha256": protocol_hash,
        "python_version": sys.version,
        "archives": archive_info,
        "arm_comparisons": arm_comparisons,
        "summary": {
            "c4_4_same_candidates": c4_4_cmp.get("same_candidates", 0),
            "c4_4_same_candidates_rate": c4_4_cmp.get("same_candidates_rate", 0),
            "c4_4_same_identities": c4_4_cmp.get("same_identities", 0),
            "c4_4_same_identities_rate": c4_4_cmp.get("same_identities_rate", 0),
            "c4_4_different_selected_order": c4_4_cmp.get("different_selected_order", 0),
            "c4_4_different_selected_membership": c4_4_cmp.get("different_selected_membership", 0),
            "c4_4_total_packet_differences": c4_4_cmp.get("total_packet_differences", 0),
            "defect_proven": c4_4_cmp.get("total_packet_differences", 0) > 0,
        },
    }
    
    receipt_path = comparison_dir / "comparison_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"\nComparison receipt: {receipt_path}")
    print(f"\n=== Summary ===")
    print(f"  C4_4 same candidates: {receipt['summary']['c4_4_same_candidates']}/120")
    print(f"  C4_4 same identities: {receipt['summary']['c4_4_same_identities']}/120")
    print(f"  C4_4 different selected order: {receipt['summary']['c4_4_different_selected_order']}/120")
    print(f"  C4_4 different selected membership: {receipt['summary']['c4_4_different_selected_membership']}/120")
    print(f"  C4_4 total packet differences: {receipt['summary']['c4_4_total_packet_differences']}/120")
    print(f"  Defect proven: {receipt['summary']['defect_proven']}")


if __name__ == "__main__":
    main()
