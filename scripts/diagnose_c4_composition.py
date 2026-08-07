#!/usr/bin/env python3
"""C4 composition diagnostic — diagnose S2c selection behavior.

Analyzes the dry-run receipts to understand:
1. How S2c selection affects CES (complete evidence set) by identity status
2. Which tasks improve vs regress under S2c
3. What S2c selects that displaces required evidence
4. The composition effect: identity → S2c → packet

Usage:
    python scripts/diagnose_c4_composition.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DRY_RUN_DIR = ROOT / "evidence/gate_c4/dry_run/development"
CORPUS = ROOT / "data/hrm/controlled_gate_a_v4/development"


def main():
    tasks = [json.loads(l) for l in (CORPUS / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    task_by_id = {t["task_id"]: t for t in tasks}

    arms = ["C4_0", "C4_3", "C4_4", "C4_5"]
    results = {}
    for arm_id in arms:
        path = DRY_RUN_DIR / f"{arm_id}_dry.jsonl"
        if path.exists():
            results[arm_id] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # 1. CES by identity status
    print("=== CES by Identity Status ===")
    print(f"{'Arm':<8} {'Status':<12} {'N':>4} {'Recall':>8} {'CES':>8}")
    for arm_id in arms:
        if arm_id not in results:
            continue
        by_status = defaultdict(lambda: {"n": 0, "recall_sum": 0.0, "complete": 0})
        for r in results[arm_id]:
            payload = r["runtime_payload"]
            status = payload["identity"]["status"]
            task = task_by_id.get(r["task_id"], {})
            required = set(task.get("required_evidence_ids", []))
            selected = set(payload["selection"]["selected_ids"])
            recall = len(required & selected) / len(required) if required else 1.0
            by_status[status]["n"] += 1
            by_status[status]["recall_sum"] += recall
            if required.issubset(selected):
                by_status[status]["complete"] += 1
        for status, stats in sorted(by_status.items()):
            avg_recall = stats["recall_sum"] / stats["n"]
            ces = stats["complete"] / stats["n"]
            print(f"{arm_id:<8} {status:<12} {stats['n']:>4} {avg_recall:>8.3f} {ces:>8.3f}")

    # 2. S2c task-level changes (C4_3 S0 vs C4_4 S2c)
    if "C4_3" not in results or "C4_4" not in results:
        print("\nC4_3 or C4_4 not found")
        return

    c4_3 = {r["task_id"]: r for r in results["C4_3"]}
    c4_4 = {r["task_id"]: r for r in results["C4_4"]}

    print("\n=== S2c Task-Level Changes (C4_3 S0 → C4_4 S2c) ===")
    for status_filter in ["EXACT", "RESOLVED"]:
        improved = []; regressed = []; same = []
        for tid in c4_3:
            r3, r4 = c4_3[tid], c4_4[tid]
            if r3["runtime_payload"]["identity"]["status"] != status_filter:
                continue
            task = task_by_id[tid]
            required = set(task.get("required_evidence_ids", []))
            s3 = set(r3["runtime_payload"]["selection"]["selected_ids"])
            s4 = set(r4["runtime_payload"]["selection"]["selected_ids"])
            ces3 = required.issubset(s3)
            ces4 = required.issubset(s4)
            if ces4 and not ces3:
                improved.append(tid)
            elif ces3 and not ces4:
                regressed.append(tid)
            else:
                same.append(tid)
        print(f"\n  {status_filter}: improved={len(improved)} regressed={len(regressed)} same={len(same)}")

        # Show what S2c selects that displaces required evidence (regressed cases)
        if regressed:
            print(f"  Regressed tasks (S2c displaced required evidence):")
            for tid in regressed[:3]:
                task = task_by_id[tid]
                required = set(task.get("required_evidence_ids", []))
                s3 = set(c4_3[tid]["runtime_payload"]["selection"]["selected_ids"])
                s4 = set(c4_4[tid]["runtime_payload"]["selection"]["selected_ids"])
                missing_s2c = required - s4
                gained_s2c = s4 - s3
                print(f"    {tid}: missing={missing_s2c}  gained={gained_s2c}")

    # 3. S2c selection pattern analysis
    print("\n=== S2c Selection Pattern Analysis ===")
    s2c_record_types = defaultdict(int)
    s0_record_types = defaultdict(int)
    for tid in c4_3:
        r3, r4 = c4_3[tid], c4_4[tid]
        for eid in r3["runtime_payload"]["selection"]["selected_ids"]:
            rtype = eid.split("/")[-1] if "/" in eid else "unknown"
            s0_record_types[rtype] += 1
        for eid in r4["runtime_payload"]["selection"]["selected_ids"]:
            rtype = eid.split("/")[-1] if "/" in eid else "unknown"
            s2c_record_types[rtype] += 1

    print(f"{'RecordType':<20} {'S0_count':>10} {'S2c_count':>10} {'Delta':>8}")
    all_types = sorted(set(s0_record_types) | set(s2c_record_types))
    for rtype in all_types:
        s0_count = s0_record_types[rtype]
        s2c_count = s2c_record_types[rtype]
        delta = s2c_count - s0_count
        print(f"{rtype:<20} {s0_count:>10} {s2c_count:>10} {delta:>+8}")

    # 4. Composition effect summary
    print("\n=== Composition Effect Summary ===")
    c4_0 = {r["task_id"]: r for r in results.get("C4_0", [])}
    c4_5 = {r["task_id"]: r for r in results.get("C4_5", [])}

    for label, ref_dict in [("C4_0 (no identity, S0)", c4_0), ("C4_5 (oracle)", c4_5)]:
        if not ref_dict:
            continue
        total_recall = 0; total_ces = 0; n = 0
        for tid in c4_4:
            if tid not in ref_dict:
                continue
            task = task_by_id[tid]
            required = set(task.get("required_evidence_ids", []))
            ref_selected = set(ref_dict[tid]["runtime_payload"]["selection"]["selected_ids"])
            ref_recall = len(required & ref_selected) / len(required) if required else 1.0
            ref_ces = required.issubset(ref_selected)
            total_recall += ref_recall
            total_ces += int(ref_ces)
            n += 1
        if n:
            print(f"  {label}: n={n} recall={total_recall/n:.3f} CES={total_ces/n:.3f}")

    # C4_4 metrics
    total_recall = 0; total_ces = 0; n = 0
    for tid in c4_4:
        task = task_by_id[tid]
        required = set(task.get("required_evidence_ids", []))
        s4 = set(c4_4[tid]["runtime_payload"]["selection"]["selected_ids"])
        recall = len(required & s4) / len(required) if required else 1.0
        total_recall += recall
        total_ces += int(required.issubset(s4))
        n += 1
    print(f"  C4_4 (identity, S2c): n={n} recall={total_recall/n:.3f} CES={total_ces/n:.3f}")


if __name__ == "__main__":
    main()
