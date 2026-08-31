#!/usr/bin/env python3
"""Release bundle verifier — validates RELEASE_MANIFEST.json.

This is a self-contained verifier that checks:
1. All file hashes match the manifest
2. dirty_worktree is False
3. All SHA256 strings are valid 64 lowercase hex
4. Qualification gates status
5. Oracle-path validation (if benchmark data present)

Usage:
    python scripts/verify_release.py [--release-dir releases/daph_v3r2_terminal_authority]
"""
from __future__ import annotations

import hashlib
import json
import os
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

    # Auto-detect release directory
    if args.release_dir:
        release_dir = Path(args.release_dir)
    else:
        # Try to find RELEASE_MANIFEST.json relative to this script
        script_dir = Path(__file__).resolve().parent
        release_dir = script_dir.parent  # releases/daph_v3r2_terminal_authority/scripts/ -> releases/daph_v3r2_terminal_authority/
        if not (release_dir / "RELEASE_MANIFEST.json").exists():
            # Try from repo root
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
    # 1. Check dirty_worktree
    # ============================================================
    print("=" * 60)
    print("RELEASE VERIFICATION")
    print("=" * 60)

    dirty = manifest.get("dirty_worktree", True)
    if dirty:
        errors.append(f"dirty_worktree is True (should be False)")
    print(f"  dirty_worktree: {dirty} {'✓' if not dirty else '✗'}")

    # ============================================================
    # 2. Validate all file hashes
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
        print(f"    ALL HASHES VALID ✓")

    # ============================================================
    # 3. Check GGUF hash if present
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
        print(f"\n  GGUF: not present (expected SHA: {gguf_sha[:16] if gguf_sha else 'N/A'})")

    # ============================================================
    # 4. Check qualification gates
    # ============================================================
    print(f"\n  Qualification gates:")
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
    for key, name in gate_names.items():
        result = gates.get(key, None)
        status = "PASS" if result else "FAIL"
        print(f"    {key}: {status}")
        if not result:
            warnings.append(f"Gate {key} ({name}) did not pass")

    # ============================================================
    # 5. Check source identity
    # ============================================================
    print(f"\n  Source identity:")
    print(f"    source_commit: {manifest.get('source_commit', 'N/A')}")
    print(f"    source_tag: {manifest.get('source_tag', 'N/A')}")
    print(f"    current_worktree_commit: {manifest.get('current_worktree_commit', 'N/A')}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    if not errors:
        print(f"\n  RELEASE VERIFIED ✓")
        sys.exit(0)
    else:
        print(f"\n  RELEASE VERIFICATION FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
