#!/usr/bin/env python3
"""Freeze the candidate pools of a scored run. Retrieval/selection only.

No HRM. No mechanism change. This exists to close a provenance gap, not to
produce a new result.

The gap
-------
B3-B needs per-task availability predicates -- was the required bridge in the
pool? the terminal record? the complete set? -- and those depend on pool
MEMBERSHIP. The scored run's receipts persisted selections and hashes but not
the pools themselves, so availability had to be reconstructed by replay. On a
CPU replay of a CUDA-scored run, 22 of 500 order-sensitive candidate_pool_hash
values differed.

Two arguments were offered that the replay was nonetheless faithful, and BOTH
are insufficient:

  * "all 500 selections reproduce" -- selection equality does NOT imply pool
    equality. Two pools differing in low-ranked members can yield identical
    top-6 selections.
  * "aggregate candidate CES is 0.4120 on both" -- an aggregate match only
    means the COUNT of complete tasks agreed. Two tasks could swap, one gaining
    and one losing completeness.

The correct invariant is unordered candidate-MEMBERSHIP identity per task.
Establishing it requires the pool IDs from the scored run's own platform, which
is what this script captures.

What it does
------------
Reproduces every task's candidate pool through the same authoritative path the
run used (run_query_stage -> cached BM25/BGE backends -> frozen fusion), and
writes a frozen artifact containing, per task:

    candidate IDs in rank order
    fusion scores
    candidate_order_hash        (order-sensitive)
    candidate_membership_hash   (order-independent)

It verifies its order hashes against the scored receipts and reports any
divergence. Run it ON THE PLATFORM THAT SCORED THE RUN and 500/500 should
reproduce exactly; then any later diagnostic reads the artifact instead of
re-running retrieval, so no future analysis depends on cross-platform
bit-reproducibility.

Usage:
    python scripts/replay_c5_candidate_pools.py --split confirmation
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.packet_ordering import (  # noqa: E402
    canonical_candidate_membership_hash, canonical_candidate_pool_hash)
from scripts.run_c5_integrated_ladder import (  # noqa: E402
    evaluate_task, use_ladder)
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records)


def environment_fingerprint() -> dict[str, Any]:
    """Recorded with the artifact: a replay contract is platform-scoped."""
    fingerprint: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        fingerprint["torch"] = torch.__version__
        fingerprint["cuda_available"] = torch.cuda.is_available()
        fingerprint["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception as exc:  # noqa: BLE001
        fingerprint["torch_error"] = str(exc)
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze candidate pools")
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--ladder", default="J")
    parser.add_argument("--arm-for-queries", default="C4_4")
    parser.add_argument("--receipts", default=None,
                        help="scored receipts to verify order hashes against")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    use_ladder(args.ladder)
    receipts_path = Path(args.receipts) if args.receipts else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_c5_"
               f"{args.ladder}ladder_hrm.receipts.jsonl")
    scored: dict[str, dict] = {}
    if receipts_path.is_file():
        for line in receipts_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                scored[row["task_id"]] = row
        print(f"  verifying against {len(scored)} scored receipts")
    else:
        print(f"  NOTE: no scored receipts at {receipts_path}; capturing only")

    tasks, evidence, texts = load_split(args.split)
    records = to_index_records(evidence)
    arm = ARMS[args.arm_for_queries]
    fingerprint = environment_fingerprint()
    print(f"  platform: {fingerprint.get('device_name')} "
          f"torch={fingerprint.get('torch')} "
          f"cuda={fingerprint.get('cuda_available')}")

    pools: list[dict[str, Any]] = []
    order_mismatch: list[str] = []
    for index, task in enumerate(tasks, 1):
        if index % 25 == 0 or index == len(tasks):
            print(f"  {index}/{len(tasks)}...", end="\r", flush=True)
        row = evaluate_task(task, arm, records, texts, len(records))
        # Every J arm on the frozen fusion shares one pool; capture it once from
        # the baseline arm and record which arms it covers.
        baseline = row["arms"]["J0"] if "J0" in row["arms"] else next(
            iter(row["arms"].values()))
        candidate_ids = list(baseline["pool"])
        entry = {
            "task_id": task["task_id"],
            "candidate_ids": candidate_ids,
            "candidate_order_hash": canonical_candidate_pool_hash(candidate_ids),
            "candidate_membership_hash":
                canonical_candidate_membership_hash(candidate_ids),
            "recorded_pool_hash_in_receipt":
                scored.get(task["task_id"], {}).get("arms", {})
                .get("J0", {}).get("candidate_pool_hash"),
            "selections": {name: list(a["selected"])
                           for name, a in row["arms"].items()},
        }
        if entry["recorded_pool_hash_in_receipt"] and \
                entry["recorded_pool_hash_in_receipt"] != entry["candidate_order_hash"]:
            order_mismatch.append(task["task_id"])
        pools.append(entry)
    print(" " * 30, end="\r")

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_b3/pools/{args.split}_candidate_pools.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": "c5-candidate-pools-v1",
        "split": args.split, "ladder": args.ladder,
        "task_count": len(pools),
        "environment_fingerprint": fingerprint,
        "verified_against_receipts": str(receipts_path.name) if scored else None,
        "order_hash_mismatches": order_mismatch,
        "order_hash_mismatch_count": len(order_mismatch),
        "reproduces_scored_run_exactly": bool(scored) and not order_mismatch,
        "why_two_hashes": (
            "order hash answers 'did retrieval rank the pool identically'; "
            "membership hash answers 'was record X in the pool at all'. "
            "Availability predicates depend only on membership, and a "
            "cross-platform replay can permute ranks at numerical ties without "
            "changing membership."),
        "pools": pools,
    }, indent=2, sort_keys=True) + "\n")

    print(f"  captured {len(pools)} pools")
    if scored:
        print(f"  order-hash mismatches vs scored receipts: {len(order_mismatch)}")
        if order_mismatch:
            print(f"    e.g. {order_mismatch[:5]}")
            print("  => this platform does NOT reproduce the scored run exactly")
        else:
            print("  => 500/500 exact: this platform reproduces the scored run")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
