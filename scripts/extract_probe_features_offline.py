#!/usr/bin/env python3
"""RETRIEVAL_PROBE_GATE_V1 PHASE_1 (GPU-free variant): extract probe features.

The cheap retrieval probe performs NO model generation -- it is C2 retrieval
(BM25 + dense fusion) plus the regex/parser identity-binding stage. Its
features are therefore computable on CPU, without the HRM checkpoint and
without any GPU.

The quality labels the PHASE_2 stop-gate needs (Q_direct, Q_memory) and the
confidence features already exist in the frozen exec_training_v2 receipts,
collected on A100 at commit a697cf5. Joining those to locally-computed probe
features yields everything the PHASE_2 incremental-information diagnostic
requires -- which means the decisive stop-gate can be run with zero GPU
spend.

What this does NOT produce is the latency decomposition (T_A0_generation /
T_A1_generation are meaningless without the real GPU generations). Latency
is a PHASE_3+ concern for cost-based promotion, not an input to the PHASE_2
stop condition, so it is deferred rather than faked.

DEVELOPMENT USE ONLY -- operates on a consumed split; no promotion claim.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.executive.retrieval_probe import (  # noqa: E402
    PROBE_FEATURE_NAMES, retrieval_probe_features, run_retrieval_probe)
from scripts.run_exec_training_v2_collection import load_groups  # noqa: E402
from scripts.run_gate_c4 import _to_index_records as to_index_records  # noqa: E402

FAMILIES = ("ANSWER_NOW_viable", "MEMORY_required")


def c2(n: int) -> int:
    return max(1, min(300, math.ceil(0.15 * n))) if n else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "evidence/gate_executive/retrieval_probe_v1_features.jsonl"))
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--limit-tasks", type=int, default=None)
    args = ap.parse_args()

    arm = ARMS[args.arm_for_queries]
    rows = []
    print("=== RETRIEVAL_PROBE_GATE_V1 PHASE_1 -- offline probe feature extraction ===")
    print("    CPU only: the probe performs no model generation.\n")

    for family in FAMILIES:
        for group_label, tasks, evidence, texts in load_groups(family):
            if args.limit_tasks:
                tasks = tasks[:args.limit_tasks]
            records = to_index_records(evidence)
            depth = c2(len(records))
            prefix = f"{group_label}: " if family == "MEMORY_required" else "ANSWER_NOW_viable: "
            for i, task in enumerate(tasks, 1):
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {prefix}{i}/{len(tasks)}", end="\r", flush=True)
                rid = (f"{group_label}:{task['task_id']}"
                       if family == "MEMORY_required" else task["task_id"])
                probe = run_retrieval_probe(task["question"], arm, records, texts, depth)
                rows.append({
                    "task_id": rid, "suite_family": family, "scale": group_label,
                    "family": task.get("family", family),
                    "identity_status": probe.identity_status,
                    "probe_handoff_hash": probe.handoff_hash(),
                    "evidence_pool_size": len(evidence),
                    **retrieval_probe_features(probe),
                })
            print(" " * 48, end="\r")
            print(f"  {prefix}done  tasks={len(tasks)}  pool={len(evidence)}  depth={depth}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    print(f"\n  {len(rows)} rows -> {out}")
    print(f"  features: {', '.join(PROBE_FEATURE_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
