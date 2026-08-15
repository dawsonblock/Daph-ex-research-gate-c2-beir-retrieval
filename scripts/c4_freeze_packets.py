#!/usr/bin/env python3
"""Phase 12: Freeze deterministic packets before HRM.

Once pre-HRM qualification passes, generate the 120 C4 arm packets once.
Write immutable packet artifacts:

    packet.json   — canonical packet representation
    packet.sha256 — SHA-256 of packet.json
    prompt.txt    — the full HRM prompt
    prompt.sha256 — SHA-256 of prompt.txt

The HRM stage must consume these frozen packets. It should not rerun
retrieval or selection internally.

Architecture:
    PRE-HRM FREEZE
          ↓
    immutable packets
          ↓
    HRM generation

Usage:
    python scripts/c4_freeze_packets.py [--split development] [--arms C4_0,C4_1,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.c4.provenance import (
    canonical_packet_hash, build_canonical_packet, hash_text)
from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt


def freeze_packets(split: str = "development", arm_ids: list[str] | None = None):
    """Generate frozen packet artifacts for all tasks and arms."""
    from scripts.run_gate_c4 import (
        run_pre_hrm_stages, _load_split, _to_index_records, ARMS, PRIMARY_ORDER)

    if arm_ids is None:
        arm_ids = PRIMARY_ORDER

    tasks, evidence, texts = _load_split(split)
    records = _to_index_records(evidence)

    output_dir = ROOT / "evidence/gate_c4/frozen_packets" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 12: Freeze Deterministic Packets ===")
    print(f"Split: {split}")
    print(f"Arms: {arm_ids}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {output_dir}")
    print()

    all_hashes = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        arm_dir = output_dir / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Freezing {arm_id}...", end=" ", flush=True)
        arm_hashes = []

        for i, task in enumerate(tasks):
            tid = task["task_id"]
            task_dir = arm_dir / tid
            task_dir.mkdir(parents=True, exist_ok=True)

            # Run pre-HRM stages
            pre_hrm = run_pre_hrm_stages(task, arm, records, texts)

            # Build canonical packet
            selected_ids = list(pre_hrm.selection.selected_ids)
            ordered_text_hashes = [hash_text(texts.get(eid, "")) for eid in selected_ids]

            packet = build_canonical_packet(
                task_id=tid,
                query_hash=hashlib.sha256(
                    pre_hrm.query.rendered_query.encode()).hexdigest(),
                canonical_subject=pre_hrm.identity.canonical or "",
                candidate_pool_hash=hashlib.sha256(
                    json.dumps(
                        list(pre_hrm.retrieval.candidate_ids),
                        separators=(",", ":")).encode()).hexdigest(),
                selector_policy_id=pre_hrm.selection.selector,
                ordered_selected_ids=selected_ids,
                ordered_text_sha256=ordered_text_hashes,
            )

            packet_hash = canonical_packet_hash(packet)

            # Write packet.json
            packet_path = task_dir / "packet.json"
            packet_path.write_text(
                json.dumps(packet, indent=2, sort_keys=True) + "\n")

            # Write packet.sha256
            (task_dir / "packet.sha256").write_text(packet_hash + "\n")

            # Build and write prompt
            full_prompt = compose_evidence_prompt(
                task["question"],
                [texts.get(eid, "") for eid in selected_ids if eid in texts])

            prompt_path = task_dir / "prompt.txt"
            prompt_path.write_text(full_prompt)

            # Write prompt.sha256
            (task_dir / "prompt.sha256").write_text(
                hash_text(full_prompt) + "\n")

            arm_hashes.append(packet_hash)

        all_hashes[arm_id] = arm_hashes
        print(f"{len(arm_hashes)} packets frozen")

    # Write manifest
    manifest = {
        "schema_version": "c4-frozen-packets-v1",
        "split": split,
        "arm_ids": arm_ids,
        "task_count": len(tasks),
        "arm_packet_hashes": {
            arm_id: {
                "count": len(hashes),
                "all_hashes": hashes,
            }
            for arm_id, hashes in all_hashes.items()
        },
    }

    manifest_path = output_dir / "frozen_packets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\n=== Summary ===")
    for arm_id, hashes in all_hashes.items():
        print(f"  {arm_id}: {len(hashes)} packets frozen")
    print(f"\n  Manifest: {manifest_path}")
    print(f"\n  Packets are immutable. HRM must consume these frozen packets.")
    print(f"  Do not rerun retrieval or selection internally.")


def main():
    parser = argparse.ArgumentParser(description="Freeze C4 deterministic packets")
    parser.add_argument("--split", default="development")
    parser.add_argument("--arms", default=None,
                        help="Comma-separated arm IDs (default: all primary)")
    args = parser.parse_args()

    arm_ids = args.arms.split(",") if args.arms else None
    freeze_packets(split=args.split, arm_ids=arm_ids)


if __name__ == "__main__":
    main()
