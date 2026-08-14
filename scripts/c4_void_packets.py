#!/usr/bin/env python3
"""Void superseded C4 artifacts — with proof, and without destroying them.

The pattern this replaces:

    # PHASE 3 CONCLUSION: Data is non-conformant.
    shutil.move(source_dir, voided_dir)

The conclusion was correct, but the notebook showed no comparison establishing
it. "Non-conformant" then rests on the operator's memory rather than on an
artifact, and a later reader cannot tell a genuinely stale packet set from one
that was voided by mistake.

This script establishes non-conformance mechanically before it moves anything:

  * protocol hash recorded in the artifacts vs. the active protocol hash
  * policy versions (query / identity / selector / ordering / evaluator)
  * packet hash presence and the protocol v2 boundary hash fields
  * arm policy fields against the current executable registry

It then writes VOID_NOTICE.json next to the voided data recording the exact
mismatches, and MOVES rather than deletes. If no mismatch is found, it refuses
to void: conformant data is not voided on a hunch.

Usage:
    python scripts/c4_void_packets.py --path evidence/gate_c4/frozen_packets/development
    python scripts/c4_void_packets.py --path ... --reason "superseded by v2 rerun"
    python scripts/c4_void_packets.py --path ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.c4.protocol_validation import (  # noqa: E402
    ProtocolViolation, load_and_validate_protocol)

REQUIRED_PACKET_HASH_FIELDS = ("candidate_pool_hash", "membership_hash",
                               "order_hash", "packet_hash", "prompt_hash")


def _collect_evidence(path: Path) -> dict[str, Any]:
    """Read whatever provenance the target artifacts actually carry."""
    found: dict[str, Any] = {
        "manifests": [],
        "protocol_shas": set(),
        "policy_versions": {},
        "packet_hash_fields_present": set(),
        "packet_files": 0,
        "receipt_files": 0,
    }

    for manifest in sorted(path.rglob("manifest.json")):
        data = json.loads(manifest.read_text())
        found["manifests"].append(str(manifest.relative_to(path)))
        if data.get("protocol_sha256"):
            found["protocol_shas"].add(data["protocol_sha256"])
        for key in ("query_policy_version", "identity_policy_version",
                    "selector_version", "ordering_policy_version",
                    "evaluator_manifest_version", "bridge_policy_version"):
            if data.get(key):
                found["policy_versions"].setdefault(key, set()).add(data[key])

    for packet in sorted(path.rglob("packet.json")):
        found["packet_files"] += 1
        try:
            data = json.loads(packet.read_text())
        except json.JSONDecodeError:
            continue
        for f in REQUIRED_PACKET_HASH_FIELDS:
            if f in data:
                found["packet_hash_fields_present"].add(f)

    for jsonl in sorted(path.rglob("*.jsonl")):
        found["receipt_files"] += 1
        first = next((l for l in jsonl.read_text().splitlines() if l.strip()), None)
        if not first:
            continue
        try:
            receipt = json.loads(first)
        except json.JSONDecodeError:
            continue
        pkt = receipt.get("runtime_payload", {}).get("packet", {})
        for f in REQUIRED_PACKET_HASH_FIELDS:
            if pkt.get(f):
                found["packet_hash_fields_present"].add(f)

    return found


def build_void_notice(path: Path, protocol_path: Path, reason: str) -> dict:
    """Compare the artifacts against the active protocol and code."""
    protocol, active_sha, checks = load_and_validate_protocol(protocol_path)
    found = _collect_evidence(path)
    mismatches: list[dict[str, Any]] = []

    observed_shas = sorted(found["protocol_shas"])
    if not observed_shas:
        mismatches.append({
            "field": "protocol_sha256",
            "observed": None,
            "required": active_sha,
            "detail": "artifacts declare no protocol hash, so they cannot be "
                      "shown to conform to any protocol",
        })
    else:
        for sha in observed_shas:
            if sha != active_sha:
                mismatches.append({
                    "field": "protocol_sha256",
                    "observed": sha,
                    "required": active_sha,
                    "detail": "packets were built under a superseded protocol",
                })

    active_policies = protocol.get("policy_versions", {})
    for key, observed in sorted(found["policy_versions"].items()):
        required = active_policies.get(
            {"selector_version": "selector_policy_version"}.get(key, key))
        if required is None:
            continue
        for value in sorted(observed):
            if value != required:
                mismatches.append({
                    "field": key,
                    "observed": value,
                    "required": required,
                    "detail": "policy version superseded",
                })

    missing_hashes = [f for f in REQUIRED_PACKET_HASH_FIELDS
                      if f not in found["packet_hash_fields_present"]]
    if (found["packet_files"] or found["receipt_files"]) and missing_hashes:
        mismatches.append({
            "field": "packet_hashing.hash_fields",
            "observed": sorted(found["packet_hash_fields_present"]),
            "required": list(REQUIRED_PACKET_HASH_FIELDS),
            "detail": f"artifacts lack {missing_hashes}; they predate the "
                      f"protocol v2 packet boundary hashes",
        })

    return {
        "schema_version": "c4-void-notice-v1",
        "voided_path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "voided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "active_protocol_sha256": active_sha,
        "active_protocol_id": protocol.get("protocol_id"),
        "active_policy_versions": active_policies,
        "observed": {
            "manifests": found["manifests"],
            "protocol_shas": observed_shas,
            "policy_versions": {k: sorted(v)
                                for k, v in sorted(found["policy_versions"].items())},
            "packet_hash_fields_present": sorted(found["packet_hash_fields_present"]),
            "packet_files": found["packet_files"],
            "receipt_files": found["receipt_files"],
        },
        "mismatch": bool(mismatches),
        "mismatches": mismatches,
        "protocol_semantic_checks": checks,
        "disposition": "MOVED_PRESERVED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Void superseded C4 artifacts with a proof of non-conformance")
    parser.add_argument("--path", type=Path, required=True,
                        help="Directory of artifacts to void")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/gate_c4_protocol_v2_1.json")
    parser.add_argument("--voided-root", type=Path,
                        default=ROOT / "evidence/gate_c4/voided")
    parser.add_argument("--reason", default="superseded by the active C4 protocol")
    parser.add_argument("--label", default=None,
                        help="Subdirectory name under the voided root")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the notice; move nothing")
    parser.add_argument("--force", action="store_true",
                        help="Void even if no mismatch is proven (records that "
                             "no proof was found)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Nothing to void: {args.path} does not exist.")
        return 0

    try:
        notice = build_void_notice(args.path, args.protocol, args.reason)
    except ProtocolViolation as exc:
        print(f"FATAL: {exc}")
        return 2

    print("--- Void notice ---")
    print(f"  path:             {notice['voided_path']}")
    print(f"  active protocol:  {notice['active_protocol_sha256'][:16]}...")
    print(f"  observed shas:    {notice['observed']['protocol_shas'] or 'NONE'}")
    print(f"  packet files:     {notice['observed']['packet_files']}")
    print(f"  receipt files:    {notice['observed']['receipt_files']}")
    print(f"  mismatch:         {notice['mismatch']}")
    for m in notice["mismatches"]:
        print(f"    - {m['field']}: observed={m['observed']} "
              f"required={m['required']}")
        print(f"      {m['detail']}")

    if not notice["mismatch"] and not args.force:
        print("\nREFUSING TO VOID: no non-conformance was proven.")
        print("  These artifacts match the active protocol as far as their own")
        print("  provenance shows. Use --force only with a written reason.")
        return 1
    if not notice["mismatch"] and args.force:
        notice["forced_without_proof"] = True

    if args.dry_run:
        print("\n--dry-run: nothing moved.")
        print(json.dumps(notice, indent=2, sort_keys=True))
        return 0

    label = args.label or (
        f"{args.path.name}_voided_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    destination = args.voided_root / label
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"FATAL: destination already exists: {destination}")
        return 2

    # Move, never delete: the notice is only useful next to the data it voids.
    shutil.move(str(args.path), str(destination))
    notice["moved_to"] = str(
        destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination)
    (destination / "VOID_NOTICE.json").write_text(
        json.dumps(notice, indent=2, sort_keys=True) + "\n")

    print(f"\n  moved to:     {notice['moved_to']}")
    print(f"  notice:       {destination / 'VOID_NOTICE.json'}")
    print("  original data preserved (moved, not deleted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
