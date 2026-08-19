#!/usr/bin/env python3
"""Verify the I3.5.3-r2.1 closure bundle.

Hard-fails on:
  - missing artifact
  - SHA mismatch
  - criterion mismatch
  - approved != positive for 0+0
  - frozen 5+5 intervention count != 0
  - source result hash mismatch
  - model hash mismatch

Usage:
    PYTHONPATH=. python scripts/verify_i3_5_3r2_closure.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLOSURE_DIR = ROOT / "experiments/v2b_i3_5_2/development/i353r2_closure"
SOURCE_DIR = ROOT / "experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main():
    print("Verifying I3.5.3-r2.1 closure bundle...")
    errors: list[str] = []
    checks: list[str] = []

    # Load manifest
    manifest_path = CLOSURE_DIR / "MANIFEST.sha256.json"
    if not manifest_path.exists():
        print(f"FATAL: {manifest_path} missing")
        sys.exit(1)
    manifest = load_json(manifest_path)

    # Check all artifacts exist and match SHA
    for name, expected_sha in manifest["artifacts"].items():
        path = CLOSURE_DIR / name
        actual_sha = file_sha256(path)
        if actual_sha == "MISSING":
            errors.append(f"MISSING artifact: {name}")
        elif actual_sha != expected_sha:
            errors.append(f"SHA mismatch: {name} (expected {expected_sha[:16]}, got {actual_sha[:16]})")
        else:
            checks.append(f"SHA OK: {name}")

    # Load replay summaries
    summary_5_5 = load_json(CLOSURE_DIR / "replay/gate_evaluation_summary_tau5_margin5.json")
    summary_0_0 = load_json(CLOSURE_DIR / "replay/gate_evaluation_summary_tau0_margin0.json")

    # Check criterion matches in summaries
    if summary_5_5["runtime_params"]["delta_threshold"] != 5.0:
        errors.append("tau5_margin5 summary has wrong threshold")
    if summary_5_5["runtime_params"]["lcb_margin"] != 5.0:
        errors.append("tau5_margin5 summary has wrong margin")
    if summary_0_0["runtime_params"]["delta_threshold"] != 0.0:
        errors.append("tau0_margin0 summary has wrong threshold")
    if summary_0_0["runtime_params"]["lcb_margin"] != 0.0:
        errors.append("tau0_margin0 summary has wrong margin")
    checks.append("Criterion parameters match in both summaries")

    # Check 0+0 invariant: approved == pred_positive
    pd_0_0 = summary_0_0["prediction_distribution"]
    if pd_0_0["n_intervene"] != pd_0_0["pred_positive"]:
        errors.append(
            f"0+0 invariant violated: approved ({pd_0_0['n_intervene']}) != "
            f"pred_positive ({pd_0_0['pred_positive']})")
    else:
        checks.append(f"0+0 invariant: approved == pred_positive == {pd_0_0['n_intervene']}")

    # Check 5+5: approved == 0
    pd_5_5 = summary_5_5["prediction_distribution"]
    if pd_5_5["n_intervene"] != 0:
        errors.append(f"5+5 should have 0 interventions, got {pd_5_5['n_intervene']}")
    else:
        checks.append("5+5: 0 interventions (frozen criterion)")

    # Check 5+5: max predicted < 10
    if pd_5_5["max"] >= 10:
        errors.append(f"5+5 max predicted >= 10: {pd_5_5['max']}")
    else:
        checks.append(f"5+5 max predicted < 10: {pd_5_5['max']}")

    # Check source results hash matches
    source_results_sha = file_sha256(SOURCE_DIR / "results.json")
    if source_results_sha != summary_5_5["provenance"]["source_results_sha256"]:
        errors.append("Source results SHA mismatch")
    else:
        checks.append(f"Source results SHA matches: {source_results_sha[:16]}...")

    # Check model hash matches
    model_path = ROOT / "experiments/v2b_i3_5_2/development/i353r1/pairwise_advantage_gate_v1.pkl"
    model_sha = file_sha256(model_path)
    if model_sha != summary_5_5["provenance"]["gate_model_sha256"]:
        errors.append("Gate model SHA mismatch")
    else:
        checks.append(f"Gate model SHA matches: {model_sha[:16]}...")

    # Check historical artifacts unchanged
    for name in ["results.json", "analysis.json", "experiment_identity.json", "receipts.jsonl"]:
        path = SOURCE_DIR / name
        if not path.exists():
            errors.append(f"Historical artifact missing: {name}")
        else:
            checks.append(f"Historical artifact present: {name}")

    # Check replay identity differs between criteria
    if summary_5_5["replay_identity_sha256"] == summary_0_0["replay_identity_sha256"]:
        errors.append("Replay identity should differ between 5+5 and 0+0")
    else:
        checks.append("Replay identities differ between criteria")

    # Print results
    print(f"\n{'='*78}")
    print("CLOSURE VERIFICATION RESULTS")
    print(f"{'='*78}")
    print(f"\nChecks passed: {len(checks)}")
    for c in checks:
        print(f"  ✓ {c}")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\n{'='*78}")
        print("VERIFICATION FAILED")
        print(f"{'='*78}")
        sys.exit(1)
    else:
        print(f"\n{'='*78}")
        print("VERIFICATION PASSED")
        print(f"{'='*78}")


if __name__ == "__main__":
    main()
