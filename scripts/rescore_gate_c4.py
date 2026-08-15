#!/usr/bin/env python3
"""Rescore existing C4 development receipts with the corrected verifier.

The original C4 evaluator (evaluator_v1) failed to strip HRM control tokens
(e.g. ``<|box_end|>``) before comparing answers, systematically marking correct
canonical/symbolic outputs wrong. This script:

1. Reads each receipt from evidence/gate_c4/full/development/<ARM>.jsonl
2. Re-runs verify_answer using the shared (corrected) verifier
3. Recomputes quality scores
4. Writes corrected receipts to evidence/gate_c4/full/development_evaluator_v2/
5. Emits a rescore manifest and summary report

No HRM rerun is required — only evaluator-side annotations change. The
runtime_payload (including hrm.output) is preserved byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.evaluation.verifiers import verify_answer

DEV_DIR = ROOT / "evidence/gate_c4/full/development"
OUT_DIR = ROOT / "evidence/gate_c4/full/development_evaluator_v2"
ARMS = ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]


def _compute_quality(task_ea: dict, selected_ids: list[str], correct: bool) -> float:
    """Recompute quality score (matches run_gate_c4._compute_quality)."""
    required = set(task_ea["required_evidence_ids"])
    selected = set(selected_ids)
    complete = required <= selected
    if complete and correct:
        return 1.0
    elif complete and not correct:
        return 0.5
    elif not complete and correct:
        return 0.25
    return 0.0


def rescore_arm(arm_id: str) -> dict:
    """Rescore one arm's receipts. Returns per-arm summary stats."""
    in_path = DEV_DIR / f"{arm_id}.jsonl"
    out_path = OUT_DIR / f"{arm_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "arm_id": arm_id,
        "n_tasks": 0,
        "v1_correct": 0,
        "v2_correct": 0,
        "v1_quality_sum": 0.0,
        "v2_quality_sum": 0.0,
        "flips_false_to_true": 0,
        "flips_true_to_false": 0,
        "by_verifier": {},
        "by_regime": {},
    }

    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            receipt = json.loads(line)
            ea = receipt["evaluator_annotation"]
            hrm_output = receipt["runtime_payload"]["hrm"]["output"]
            selected_ids = receipt["runtime_payload"]["selection"]["selected_ids"]

            # Save v1 values BEFORE any modification
            v1_correct = ea["correct"]
            v1_quality = ea["quality"]

            # Re-verify with corrected verifier
            v2_score, v2_correct = verify_answer(
                ea.get("verifier", "exact"), ea["answer"], hrm_output
            )
            v2_quality = _compute_quality(ea, selected_ids, v2_correct)

            # Track flips
            if v2_correct and not v1_correct:
                stats["flips_false_to_true"] += 1
            elif not v2_correct and v1_correct:
                stats["flips_true_to_false"] += 1

            # By verifier type
            vtype = ea.get("verifier", "exact")
            if vtype not in stats["by_verifier"]:
                stats["by_verifier"][vtype] = {
                    "n": 0, "v1_correct": 0, "v2_correct": 0}
            stats["by_verifier"][vtype]["n"] += 1
            stats["by_verifier"][vtype]["v1_correct"] += int(v1_correct)
            stats["by_verifier"][vtype]["v2_correct"] += int(v2_correct)

            # By entity regime
            regime = ea.get("metadata", {}).get("entity_regime", "unknown")
            if regime not in stats["by_regime"]:
                stats["by_regime"][regime] = {
                    "n": 0, "v1_correct": 0, "v2_correct": 0,
                    "v1_quality": 0.0, "v2_quality": 0.0}
            stats["by_regime"][regime]["n"] += 1
            stats["by_regime"][regime]["v1_correct"] += int(v1_correct)
            stats["by_regime"][regime]["v2_correct"] += int(v2_correct)
            stats["by_regime"][regime]["v1_quality"] += v1_quality
            stats["by_regime"][regime]["v2_quality"] += v2_quality

            # Update evaluator annotation
            ea["correct"] = v2_correct
            ea["quality"] = v2_quality
            ea["evaluator_version"] = "v2_corrected"
            ea["evaluator_v1_correct"] = v1_correct
            ea["evaluator_v1_quality"] = v1_quality

            stats["n_tasks"] += 1
            stats["v1_correct"] += int(v1_correct)
            stats["v2_correct"] += int(v2_correct)
            stats["v1_quality_sum"] += v1_quality
            stats["v2_quality_sum"] += v2_quality

            f_out.write(json.dumps(receipt) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rescore C4 development receipts")
    parser.add_argument("--arms", nargs="*", default=ARMS, help="Arms to rescore")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for arm in args.arms:
        stats = rescore_arm(arm)
        all_stats.append(stats)
        q_v1 = stats["v1_quality_sum"] / stats["n_tasks"] if stats["n_tasks"] else 0
        q_v2 = stats["v2_quality_sum"] / stats["n_tasks"] if stats["n_tasks"] else 0
        print(f"{arm}: Q_v1={q_v1:.4f}  Q_v2={q_v2:.4f}  "
              f"correct_v1={stats['v1_correct']}/{stats['n_tasks']}  "
              f"correct_v2={stats['v2_correct']}/{stats['n_tasks']}  "
              f"flips F->T={stats['flips_false_to_true']}  T->F={stats['flips_true_to_false']}")

    print("\n--- By verifier type ---")
    for arm_stats in all_stats:
        for vtype, vs in arm_stats["by_verifier"].items():
            print(f"  {arm_stats['arm_id']} {vtype}: "
                  f"v1={vs['v1_correct']}/{vs['n']}  v2={vs['v2_correct']}/{vs['n']}")

    print("\n--- By entity regime ---")
    for arm_stats in all_stats:
        for regime, rs in arm_stats["by_regime"].items():
            q1 = rs["v1_quality"] / rs["n"] if rs["n"] else 0
            q2 = rs["v2_quality"] / rs["n"] if rs["n"] else 0
            print(f"  {arm_stats['arm_id']} {regime}: "
                  f"Q_v1={q1:.4f}  Q_v2={q2:.4f}  "
                  f"correct_v1={rs['v1_correct']}/{rs['n']}  "
                  f"correct_v2={rs['v2_correct']}/{rs['n']}")

    # Write rescore manifest
    manifest = {
        "evaluator_version": "v2_corrected",
        "description": "Rescored with shared verifier that strips HRM control tokens",
        "source_dir": "evidence/gate_c4/full/development",
        "output_dir": "evidence/gate_c4/full/development_evaluator_v2",
        "arms": args.arms,
        "summary": all_stats,
    }
    manifest_path = OUT_DIR / "rescore_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nRescore manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
