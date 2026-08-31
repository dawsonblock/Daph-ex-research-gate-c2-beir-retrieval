#!/usr/bin/env python3
"""DAPH Qualification Ledger — chained hash records for every run.

Every run emits an immutable qualification record chained by SHA256:
  H_n = SHA256(H_{n-1} || Record_n)

This prevents silent post-hoc alteration of historical qualification evidence.

Usage:
    python scripts/qualification_ledger.py record --release-id daph_v3r2_terminal_authority
    python scripts/qualification_ledger.py verify
    python scripts/qualification_ledger.py list
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "releases" / "qualification_ledger.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_last_hash() -> str:
    """Get the hash of the last record in the ledger."""
    if not LEDGER_PATH.exists():
        return "0" * 64  # Genesis hash

    last_line = None
    with open(LEDGER_PATH) as f:
        for line in f:
            if line.strip():
                last_line = line.strip()

    if last_line:
        record = json.loads(last_line)
        return record.get("record_hash", "0" * 64)
    return "0" * 64


def compute_record_hash(previous_hash: str, record: dict) -> str:
    """Compute the chained hash: H_n = SHA256(H_{n-1} || Record_n)."""
    payload = previous_hash + json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cmd_record(args):
    """Record a new qualification entry."""
    release_id = args.release_id
    release_dir = REPO_ROOT / "releases" / release_id

    if not release_dir.exists():
        print(f"ERROR: Release directory not found: {release_dir}")
        sys.exit(1)

    manifest_path = release_dir / "RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        print(f"ERROR: Release manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get git identity
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()

    # Compute manifest hash
    manifest_sha = sha256_file(manifest_path)

    # Build record
    previous_hash = get_last_hash()
    record = {
        "release_id": release_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "manifest_sha256": manifest_sha,
        "source_commit": manifest.get("source_commit"),
        "source_tag": manifest.get("source_tag"),
        "dirty_worktree": manifest.get("dirty_worktree"),
        "file_count": len(manifest.get("file_hashes", {})),
        "key_results": manifest.get("key_results"),
        "qualification_gates": manifest.get("qualification_gates"),
        "claim_level": manifest.get("claim_level"),
        "promotion_status": manifest.get("promotion_status"),
        "previous_hash": previous_hash,
    }

    # Compute chained hash
    record_hash = compute_record_hash(previous_hash, record)
    record["record_hash"] = record_hash

    # Verify chain
    verify_hash = compute_record_hash(previous_hash, {k: v for k, v in record.items() if k != "record_hash"})
    assert verify_hash == record_hash, f"Hash computation mismatch: {verify_hash} != {record_hash}"

    # Append to ledger
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    print(f"Recorded qualification entry:")
    print(f"  Release: {release_id}")
    print(f"  Timestamp: {record['timestamp']}")
    print(f"  Git commit: {git_commit[:12]}")
    print(f"  Manifest SHA: {manifest_sha[:16]}...")
    print(f"  Record hash: {record_hash[:16]}...")
    print(f"  Previous hash: {previous_hash[:16]}...")
    print(f"  Chain length: {sum(1 for _ in open(LEDGER_PATH))}")


def cmd_verify(args):
    """Verify the entire ledger chain."""
    if not LEDGER_PATH.exists():
        print("No ledger found.")
        return

    records = [json.loads(line) for line in open(LEDGER_PATH) if line.strip()]

    print(f"Verifying {len(records)} records...")

    previous_hash = "0" * 64
    all_valid = True

    for i, record in enumerate(records):
        expected_previous = previous_hash
        actual_previous = record.get("previous_hash")

        if expected_previous != actual_previous:
            print(f"  Record {i}: CHAIN BROKEN")
            print(f"    Expected previous: {expected_previous[:16]}...")
            print(f"    Actual previous:   {actual_previous[:16]}...")
            all_valid = False
            break

        # Verify record hash
        record_without_hash = {k: v for k, v in record.items() if k != "record_hash"}
        computed_hash = compute_record_hash(previous_hash, record_without_hash)
        actual_hash = record.get("record_hash")

        if computed_hash != actual_hash:
            print(f"  Record {i}: HASH MISMATCH")
            print(f"    Computed: {computed_hash[:16]}...")
            print(f"    Actual:   {actual_hash[:16]}...")
            all_valid = False
            break

        previous_hash = actual_hash
        print(f"  Record {i}: OK ({record['release_id']}, hash={actual_hash[:16]}...)")

    if all_valid:
        print(f"\n  Chain valid. {len(records)} records verified.")
    else:
        print(f"\n  CHAIN INVALID at record {i}.")
        sys.exit(1)


def cmd_list(args):
    """List all ledger entries."""
    if not LEDGER_PATH.exists():
        print("No ledger found.")
        return

    records = [json.loads(line) for line in open(LEDGER_PATH) if line.strip()]

    print(f"{'#':>3} {'Release':>40} {'Timestamp':>28} {'Hash':>16}")
    print("-" * 92)
    for i, record in enumerate(records):
        print(f"{i:>3} {record['release_id']:>40} {record['timestamp'][:19]:>28} {record['record_hash'][:16]:>16}...")


def cmd_gates(args):
    """Check all 15 qualification gates for a release."""
    release_id = args.release_id
    release_dir = REPO_ROOT / "releases" / release_id
    manifest_path = release_dir / "RELEASE_MANIFEST.json"

    if not manifest_path.exists():
        print(f"ERROR: Release manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    gates = manifest.get("qualification_gates", {})

    gate_names = {
        "G1_clean_source_identity": "Clean source identity (dirty=false)",
        "G2_dependency_hashes": "Complete dependency hashes",
        "G3_benchmark_frozen": "Benchmark frozen before run",
        "G4_treatment_purity": "Zero treatment contamination",
        "G5_zero_runtime_failures": "Zero runtime failures",
        "G6_positive_ci_lower": "Positive paired utility CI lower bound",
        "G7_rescues_gt_breaks": "Rescues > breaks",
        "G8_zero_false_terminal": "Zero false terminal forces",
        "G9_semantic_conformance": "Semantic conformance",
        "G10_nonzero_coverage": "Nonzero effective intervention coverage",
        "G11_authority_receipts": "Complete authority receipts",
        "G12_trajectory_recomputability": "Raw trajectory recomputability",
        "G13_novelty_verified": "Novelty claim verified",
        "G14_no_post_hoc_changes": "No post-hoc benchmark changes",
        "G15_model_identity_fixed": "Model/runtime identity fixed",
    }

    print(f"\n{'='*70}")
    print(f"QUALIFICATION GATES: {release_id}")
    print(f"{'='*70}")
    print(f"\n{'Gate':>40} {'Result':>8} {'Description'}")
    print("-" * 90)

    all_pass = True
    for key, name in gate_names.items():
        result = gates.get(key, None)
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"{key:>40} {status:>8} {name}")

    print(f"\n{'='*70}")
    if all_pass:
        print(f"  ALL 15 GATES PASS → PROMOTION ELIGIBLE")
    else:
        failed = sum(1 for v in gates.values() if not v)
        print(f"  {failed} GATE(S) FAILED → NOT ELIGIBLE FOR PROMOTION")
    print(f"{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DAPH Qualification Ledger")
    subparsers = parser.add_subparsers(dest="command")

    # Record
    p_record = subparsers.add_parser("record", help="Record a new qualification entry")
    p_record.add_argument("--release-id", required=True)

    # Verify
    subparsers.add_parser("verify", help="Verify the ledger chain")

    # List
    subparsers.add_parser("list", help="List all ledger entries")

    # Gates
    p_gates = subparsers.add_parser("gates", help="Check qualification gates")
    p_gates.add_argument("--release-id", required=True)

    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "gates":
        cmd_gates(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
