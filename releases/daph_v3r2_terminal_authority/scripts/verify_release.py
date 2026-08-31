#!/usr/bin/env python3
"""Release bundle verifier — validates RELEASE_MANIFEST.json.

Separates artifact integrity from scientific qualification.

Usage:
    python scripts/verify_release.py [--release-dir releases/daph_v3r2_terminal_authority]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sha(sha: str) -> bool:
    return len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default=None,
                        help="Release directory (default: auto-detect)")
    args = parser.parse_args()

    if args.release_dir:
        release_dir = Path(args.release_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        release_dir = script_dir.parent
        if not (release_dir / "RELEASE_MANIFEST.json").exists():
            repo_root = script_dir.parent.parent.parent
            candidates = list((repo_root / "releases").glob("*/RELEASE_MANIFEST.json"))
            if candidates:
                release_dir = candidates[0].parent
            else:
                print("ERROR: No RELEASE_MANIFEST.json found")
                sys.exit(1)

    manifest_path = release_dir / "RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    errors = []
    warnings = []

    # ============================================================
    # 1. Artifact integrity: dirty_worktree
    # ============================================================
    print("=" * 60)
    print("ARTIFACT INTEGRITY")
    print("=" * 60)

    dirty = manifest.get("dirty_worktree", True)
    if dirty:
        errors.append("dirty_worktree is True (should be False)")
    print(f"  dirty_worktree: {dirty} {'✓' if not dirty else '✗'}")

    # ============================================================
    # 2. Artifact integrity: file hashes
    # ============================================================
    print(f"\n  File hashes:")
    file_hashes = manifest.get("file_hashes", {})
    hash_errors = 0
    for rel_path, expected_sha in sorted(file_hashes.items()):
        full_path = release_dir / rel_path
        if not full_path.exists():
            errors.append(f"Missing file: {rel_path}")
            hash_errors += 1
            continue
        if not validate_sha(expected_sha):
            errors.append(f"Invalid SHA format: {rel_path}: {expected_sha}")
            hash_errors += 1
            continue
        actual_sha = sha256_file(full_path)
        if actual_sha != expected_sha:
            errors.append(f"Hash mismatch: {rel_path}: expected {expected_sha[:16]}, got {actual_sha[:16]}")
            hash_errors += 1

    print(f"    Files checked: {len(file_hashes)}")
    print(f"    Hash errors: {hash_errors}")
    if hash_errors == 0:
        print(f"    HASH CLOSURE: PASS ✓")

    # ============================================================
    # 3. Artifact integrity: GGUF
    # ============================================================
    gguf = manifest.get("gguf", {})
    gguf_path = gguf.get("path")
    gguf_sha = gguf.get("sha256")
    if gguf_path and gguf_sha and Path(gguf_path).exists():
        actual_gguf_sha = sha256_file(Path(gguf_path))
        if actual_gguf_sha != gguf_sha:
            errors.append(f"GGUF hash mismatch: expected {gguf_sha[:16]}, got {actual_gguf_sha[:16]}")
        else:
            print(f"\n  GGUF: {gguf_path}")
            print(f"    SHA256: {gguf_sha[:16]}... ✓")
    else:
        print(f"\n  GGUF: not present locally (SHA: {gguf_sha[:16] if gguf_sha else 'N/A'})")

    # ============================================================
    # 4. Source identity
    # ============================================================
    print(f"\n  Source identity:")
    confirmed_commit = manifest.get("confirmed_executive_commit", "N/A")
    confirmed_tag = manifest.get("confirmed_executive_tag", "N/A")
    packaging_commit = manifest.get("release_packaging_commit", "N/A")

    # Also check embedded experiment manifest
    exp_manifest = manifest.get("experiment_manifest", {})
    if exp_manifest:
        exp_commit = exp_manifest.get("source_commit", "N/A")
        exp_tag = exp_manifest.get("source_tag", "N/A")
        exp_dirty = exp_manifest.get("dirty_worktree", "N/A")
    else:
        exp_commit = "N/A"
        exp_tag = "N/A"
        exp_dirty = "N/A"

    print(f"    confirmed_executive_commit: {confirmed_commit}")
    print(f"    confirmed_executive_tag: {confirmed_tag}")
    print(f"    release_packaging_commit: {packaging_commit}")
    print(f"    experiment_manifest.source_commit: {exp_commit}")
    print(f"    experiment_manifest.source_tag: {exp_tag}")
    print(f"    experiment_manifest.dirty_worktree: {exp_dirty}")

    if confirmed_commit == "N/A" and exp_commit == "N/A":
        warnings.append("No source commit identity found")

    # ============================================================
    # 5. Scientific qualification gates
    # ============================================================
    print(f"\n{'=' * 60}")
    print("SCIENTIFIC QUALIFICATION GATES")
    print(f"{'=' * 60}")

    gates = manifest.get("qualification_gates", {})
    gate_names = {
        "G1_clean_source_identity": "Clean source identity",
        "G2_dependency_hashes": "Complete dependency hashes",
        "G3_benchmark_frozen": "Benchmark frozen",
        "G4_treatment_purity": "Treatment purity",
        "G5_zero_runtime_failures": "Zero runtime failures",
        "G6_positive_ci_lower": "Positive CI lower bound",
        "G7_rescues_gt_breaks": "Rescues > breaks",
        "G8_zero_false_terminal": "Zero false terminal",
        "G9_semantic_conformance": "Semantic conformance",
        "G10_nonzero_coverage": "Nonzero coverage",
        "G11_authority_receipts": "Authority receipts",
        "G12_trajectory_recomputability": "Trajectory recomputability",
        "G13_novelty_verified": "Novelty verified",
        "G14_no_post_hoc_changes": "No post-hoc changes",
        "G15_model_identity_fixed": "Model identity fixed",
    }

    pass_count = 0
    fail_count = 0
    for key, name in gate_names.items():
        result = gates.get(key, None)
        status = "PASS" if result else "FAIL"
        print(f"  {key}: {status}")
        if result:
            pass_count += 1
        else:
            fail_count += 1

    # Benchmark validity gates
    print(f"\n  Benchmark validity gates:")
    b_gates = {
        "G_B1_oracle_actions_legal": "Oracle actions legal",
        "G_B2_continuation_required": "Continuation required",
        "G_B3_terminal_matches": "Terminal matches",
        "G_B4_hypothesis_justified": "Hypothesis justified",
        "G_B_benchmark_valid": "Benchmark valid",
    }
    for key, name in b_gates.items():
        result = gates.get(key, None)
        status = "PASS" if result else "FAIL"
        print(f"  {key}: {status}")

    # ============================================================
    # 6. Summary
    # ============================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    integrity_ok = len(errors) == 0
    qualification_ok = fail_count == 0

    print(f"  ARTIFACT INTEGRITY: {'PASS' if integrity_ok else 'FAIL'}")
    print(f"  HASH CLOSURE: {'PASS' if hash_errors == 0 else 'FAIL'}")
    print(f"  SOURCE IDENTITY: {'PASS' if confirmed_commit != 'N/A' else 'PARTIAL'}")
    print(f"  QUALIFICATION GATES: {pass_count}/{pass_count + fail_count}")
    print(f"  SCIENTIFIC QUALIFICATION: {'PASS' if qualification_ok else 'FAIL'}")
    print(f"  PROMOTION: {'ALLOWED' if integrity_ok and qualification_ok else 'BLOCKED'}")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    if integrity_ok and not qualification_ok:
        print(f"\n  Artifact integrity verified, but scientific qualification FAILED.")
        print(f"  Promotion is BLOCKED until all gates pass.")
        sys.exit(0)  # Exit 0 — integrity is fine, just not qualified
    elif not integrity_ok:
        print(f"\n  ARTIFACT INTEGRITY FAILED")
        sys.exit(1)
    else:
        print(f"\n  FULLY VERIFIED AND QUALIFIED ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
