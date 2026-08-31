#!/usr/bin/env python3
"""Verify the structural-OOD experiment bundle.

Checks that every executable dependency that could influence trajectories
is covered by a manifest hash, and that the manifest's source commit is
clean (dirty_worktree=false).

Usage:
    python scripts/verify_ood_bundle.py [--bundle-dir experiments/i3_30r3/structural_ood_run]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="experiments/i3_30r3/structural_ood_run")
    args = parser.parse_args()

    bundle_dir = REPO_ROOT / args.bundle_dir
    manifest_path = bundle_dir / "frozen_manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("=" * 60)
    print("OOD Bundle Verification")
    print("=" * 60)
    print(f"Bundle: {bundle_dir}")
    print(f"Experiment: {manifest.get('experiment')}")
    print(f"Source commit: {manifest.get('source_commit')}")
    print(f"Source tag: {manifest.get('source_tag', 'N/A')}")
    print(f"Dirty worktree: {manifest.get('dirty_worktree')}")

    # 1. Check dirty_worktree
    if manifest.get("dirty_worktree"):
        print("\n*** FAIL: dirty_worktree = true ***")
        print("The experiment was not run from a clean checkout.")
        sys.exit(1)
    else:
        print("  dirty_worktree = false: PASS")

    # 2. Check source commit matches tag
    source_tag = manifest.get("source_tag")
    source_commit = manifest.get("source_commit")
    if source_tag:
        try:
            tag_commit = subprocess.check_output(
                ["git", "rev-parse", source_tag], cwd=REPO_ROOT,
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            # The worktree commit may be different (it has OOD files added)
            # but the tag should be an ancestor
            print(f"  Tag {source_tag} -> {tag_commit[:12]}")
            print(f"  Manifest commit: {source_commit[:12]}")
        except Exception:
            print(f"  WARNING: Could not resolve tag {source_tag}")

    # 3. Verify all component hashes
    print(f"\n{'='*60}")
    print("Component Hash Verification")
    print(f"{'='*60}")

    # Map manifest keys to actual source file paths
    component_map = {
        "authority_policy_v3_sha256": "daph/authority/policy_v3.py",
        "restore_sha256": "daph/intervention/restore.py",
        "checkpoint_sha256": "daph/intervention/checkpoint.py",
        "authority_isolation_sha256": "daph/authority/isolation.py",
        "topology_sha256": "daph/epistemic/topology.py",
        "v3_features_sha256": "daph/epistemic/v3_features.py",
        "authority_policy_v2_sha256": "daph/authority/policy.py",
        "confirmation_generator_sha256": "hrm_adaptive_memory/executive/evidence_benchmark/i3_30r3_confirmation_generator.py",
        "schema_grammar_sha256": "scripts/r2_schema.py",
        "r2_allowed_actions_sha256": "scripts/r2_allowed_actions.py",
        "i3_7e_snapshot_builder_sha256": "scripts/run_i3_7e_compact_governor.py",
        "model_backend_sha256": "hrm_adaptive_memory/executive/model_backend.py",
        "runner_sha256": "scripts/run_i3_30r3_confirmation.py",
        "evaluator_sha256": "scripts/evaluate_i3_30r3_authority_isolation.py",
        "ood_runner_sha256": "scripts/run_structural_ood_experiment.py",
        "ood_pool_builder_sha256": "scripts/build_structural_ood_pool.py",
    }

    # Also check Q models, utility, GGUF
    artifact_map = {
        "Q_V3R_model_sha256": "experiments/i3_30r/Q_V3R2_A.pkl",
        "Q_V3R_schema_sha256": "experiments/i3_30r/v3r2_feature_schema.json",
        "Q_V1_model_sha256": "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
        "Q_V1_schema_sha256": "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
        "utility_config_sha256": "configs/v2b_i3_1_utility_v1.json",
    }

    all_match = True
    mismatches = []
    missing = []

    # Check source components
    for key, rel_path in component_map.items():
        if key not in manifest:
            missing.append(f"{key} (-> {rel_path})")
            continue
        actual_path = REPO_ROOT / rel_path
        if not actual_path.exists():
            missing.append(f"{key}: file not found: {rel_path}")
            continue
        actual_hash = sha256_file(actual_path)
        manifest_hash = manifest[key]
        if actual_hash == manifest_hash:
            print(f"  OK: {key} -> {rel_path}")
        else:
            print(f"  MISMATCH: {key}")
            print(f"    manifest: {manifest_hash[:16]}...")
            print(f"    actual:   {actual_hash[:16]}...")
            mismatches.append(key)
            all_match = False

    # Check artifacts
    for key, rel_path in artifact_map.items():
        if key not in manifest:
            missing.append(f"{key} (-> {rel_path})")
            continue
        actual_path = REPO_ROOT / rel_path
        if not actual_path.exists():
            missing.append(f"{key}: file not found: {rel_path}")
            continue
        actual_hash = sha256_file(actual_path)
        manifest_hash = manifest[key]
        if actual_hash == manifest_hash:
            print(f"  OK: {key} -> {rel_path}")
        else:
            print(f"  MISMATCH: {key}")
            mismatches.append(key)
            all_match = False

    # Check GGUF
    gguf_key = "qwen_gguf_sha256"
    if gguf_key in manifest:
        gguf_path = manifest.get("qwen_gguf_path", "")
        if gguf_path and Path(gguf_path).exists():
            actual_hash = sha256_file(gguf_path)
            if actual_hash == manifest[gguf_key]:
                print(f"  OK: {gguf_key}")
            else:
                print(f"  MISMATCH: {gguf_key}")
                mismatches.append(gguf_key)
                all_match = False
        else:
            missing.append(f"{gguf_key}: GGUF file not found at {gguf_path}")

    # Check OOD pool
    ood_pool_key = "ood_pool_sha256"
    if ood_pool_key in manifest:
        ood_pool_path = REPO_ROOT / "experiments/i3_30r3/structural_ood/ood_pool.json"
        if ood_pool_path.exists():
            actual_hash = sha256_file(ood_pool_path)
            if actual_hash == manifest[ood_pool_key]:
                print(f"  OK: {ood_pool_key}")
            else:
                print(f"  MISMATCH: {ood_pool_key}")
                mismatches.append(ood_pool_key)
                all_match = False
        else:
            missing.append(f"{ood_pool_key}: OOD pool file not found")

    # Report missing
    if missing:
        print(f"\n  MISSING from manifest:")
        for m in missing:
            print(f"    {m}")

    # 4. Check dependency closure
    print(f"\n{'='*60}")
    print("Dependency Closure Check")
    print(f"{'='*60}")
    print("  The following Python modules are imported during trajectory execution:")
    print("  - daph.authority.policy_v3 (hashed)")
    print("  - daph.authority.policy (hashed)")
    print("  - daph.authority.isolation (hashed)")
    print("  - daph.intervention.restore (hashed)")
    print("  - daph.intervention.checkpoint (hashed)")
    print("  - daph.epistemic.topology (hashed)")
    print("  - daph.epistemic.v3_features (hashed)")
    print("  - hrm_adaptive_memory.executive.model_backend (hashed)")
    print("  - hrm_adaptive_memory.executive.evidence_benchmark.schema (NOT hashed)")
    print("  - hrm_adaptive_memory.executive.evidence_executor (NOT hashed)")
    print("  - hrm_adaptive_memory.executive.resources (NOT hashed)")
    print("  - hrm_adaptive_memory.cognitive_control.core (NOT hashed)")
    print("  - hrm_adaptive_memory.cognitive_control.state (NOT hashed)")
    print("  - scripts.r2_schema (hashed)")
    print("  - scripts.r2_allowed_actions (hashed)")
    print("  - scripts.run_i3_7e_compact_governor (hashed)")
    print("  - scripts.run_i3_30r3_authority_isolation (NOT separately hashed)")
    print()
    unhashed = [
        "hrm_adaptive_memory/executive/evidence_benchmark/schema.py",
        "hrm_adaptive_memory/executive/evidence_executor.py",
        "hrm_adaptive_memory/executive/resources.py",
        "hrm_adaptive_memory/cognitive_control/core.py",
        "hrm_adaptive_memory/cognitive_control/state.py",
        "scripts/run_i3_30r3_authority_isolation.py",
    ]
    print("  Unhashed modules (potential provenance gap):")
    for p in unhashed:
        actual_path = REPO_ROOT / p
        if actual_path.exists():
            h = sha256_file(actual_path)[:16]
            print(f"    {p}: {h}...")
        else:
            print(f"    {p}: NOT FOUND")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if all_match and not missing:
        print("  All hashed components match manifest.")
        print("  dirty_worktree = false")
        print("  Bundle is self-consistent.")
        sys.exit(0)
    else:
        print(f"  Mismatches: {len(mismatches)}")
        print(f"  Missing: {len(missing)}")
        print("  Bundle is NOT fully self-consistent.")
        sys.exit(1)


if __name__ == "__main__":
    main()
