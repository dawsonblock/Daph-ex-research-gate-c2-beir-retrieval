#!/usr/bin/env python3
"""Verify that a confirmation bundle is self-consistent.

Walks every frozen component in the confirmation manifest and checks
that the current source tree still matches. Fails if even one SHA differs.

This is the tool that should have existed from the start. The evaluator's
G11 checks the confirmation manifest against the development preregistration,
which is the wrong comparison for bundle self-consistency.

Usage:
    python scripts/verify_confirmation_bundle.py

Exit code:
    0 = all components match
    1 = one or more components differ (bundle is not self-consistent)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Map manifest keys to file paths
MANIFEST_KEY_TO_PATH = {
    "runner_sha256": "scripts/run_i3_30r3_confirmation.py",
    "evaluator_sha256": "scripts/evaluate_i3_30r3_authority_isolation.py",
    "authority_isolation_sha256": "daph/authority/isolation.py",
    "authority_policy_v2_sha256": "daph/authority/policy.py",
    "authority_policy_v3_sha256": "daph/authority/policy_v3.py",
    "checkpoint_sha256": "daph/intervention/checkpoint.py",
    "restore_sha256": "daph/intervention/restore.py",
    "topology_sha256": "daph/epistemic/topology.py",
    "v3_features_sha256": "daph/epistemic/v3_features.py",
    "confirmation_generator_sha256": "hrm_adaptive_memory/executive/evidence_benchmark/i3_30r3_confirmation_generator.py",
    "schema_grammar_sha256": "scripts/r2_schema.py",
    "r2_allowed_actions_sha256": "scripts/r2_allowed_actions.py",
    "i3_7e_snapshot_builder_sha256": "scripts/run_i3_7e_compact_governor.py",
    "model_backend_sha256": "hrm_adaptive_memory/executive/model_backend.py",
    "utility_config_sha256": "configs/v2b_i3_1_utility_v1.json",
    "Q_V3R_model_sha256": "experiments/i3_30r/Q_V3R2_A.pkl",
    "Q_V3R_schema_sha256": "experiments/i3_30r/v3r2_feature_schema.json",
    "Q_V1_model_sha256": "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
    "Q_V1_schema_sha256": "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
}


def main():
    manifest_path = REPO_ROOT / "experiments/i3_30r3/confirmation/frozen_manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("=" * 60)
    print("Confirmation Bundle Self-Consistency Verification")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"Experiment: {manifest.get('experiment', '?')}")
    print()

    mismatches = []
    checked = 0
    skipped = 0

    for key, rel_path in sorted(MANIFEST_KEY_TO_PATH.items()):
        if key not in manifest:
            print(f"  SKIP (not in manifest): {key}")
            skipped += 1
            continue

        expected_sha = manifest[key]
        file_path = REPO_ROOT / rel_path

        if not file_path.exists():
            print(f"  MISSING: {key}")
            print(f"    path: {rel_path}")
            mismatches.append((key, expected_sha, "FILE_MISSING"))
            continue

        actual_sha = sha256_file(file_path)
        checked += 1

        if actual_sha == expected_sha:
            print(f"  OK: {key} = {actual_sha[:16]}...")
        else:
            print(f"  MISMATCH: {key}")
            print(f"    manifest: {expected_sha[:16]}...")
            print(f"    current:  {actual_sha[:16]}...")
            print(f"    path:     {rel_path}")
            mismatches.append((key, expected_sha, actual_sha))

    # Also check GGUF if path is in manifest
    gguf_path = manifest.get("qwen_gguf_path")
    if gguf_path and Path(gguf_path).exists():
        expected_gguf = manifest.get("qwen_gguf_sha256")
        if expected_gguf:
            actual_gguf = sha256_file(Path(gguf_path))
            checked += 1
            if actual_gguf == expected_gguf:
                print(f"  OK: qwen_gguf_sha256 = {actual_gguf[:16]}...")
            else:
                print(f"  MISMATCH: qwen_gguf_sha256")
                mismatches.append(("qwen_gguf_sha256", expected_gguf, actual_gguf))

    print()
    print(f"Components checked: {checked}")
    print(f"Components skipped: {skipped}")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        print()
        print("*** BUNDLE IS NOT SELF-CONSISTENT ***")
        print()
        print("The current source tree does not match the frozen confirmation")
        print("manifest. The confirmation result is still valid historical")
        print("evidence (the manifest records what was run), but the archive")
        print("is not clean-checkout reproducible from the current tree.")
        print()
        print("To reproduce the confirmation, use git tag v3r2-confirmed or")
        print("the confirmed_release/ directory.")
        sys.exit(1)
    else:
        print()
        print("*** BUNDLE IS SELF-CONSISTENT ***")
        sys.exit(0)


if __name__ == "__main__":
    main()
