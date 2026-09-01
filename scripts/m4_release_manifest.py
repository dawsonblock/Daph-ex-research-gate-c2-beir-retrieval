#!/usr/bin/env python3
"""Generate frozen M4 release manifest with all hashes.

Closes over:
  - source commit
  - corpus hashes
  - model hashes
  - calibrator hashes
  - authority thresholds
  - environment

Usage:
    python scripts/m4_release_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
M4_DIR = REPO_ROOT / "experiments/daph_x/m4"
RELEASE_DIR = REPO_ROOT / "releases/daph_x_m4_authority_qualification"


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Get current git commit."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """Check if working tree is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return True


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # Source
    manifest = {
        "release_name": "daph_x_m4_authority_qualification",
        "source_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),

        # Corpus hashes
        "corpus": {},
        # Model hashes
        "models": {},
        # Authority configuration
        "authority": {},
        # Qualification results
        "qualification": {},
    }

    # Corpus files
    for split in ["train", "calibration", "structural_ood", "mechanism_ood"]:
        path = M4_DIR / f"m4_{split}.jsonl"
        if path.exists():
            manifest["corpus"][split] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    # Metadata
    meta_path = M4_DIR / "m4_metadata.json"
    if meta_path.exists():
        manifest["corpus"]["metadata"] = {
            "path": str(meta_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(meta_path),
        }

    # Models
    for name, filename in [
        ("q_res", "q_res_m4.pkl"),
        ("risk_model", "risk_model_m4.pkl"),
    ]:
        path = M4_DIR / filename
        if path.exists():
            manifest["models"][name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }

    # Results
    for name, filename in [
        ("q_res_results", "q_res_m4_results.json"),
        ("conformal_calibration", "conformal_calibration_m4.json"),
        ("intervention_risk", "intervention_risk_m4.json"),
        ("shadow_authority", "shadow_authority_m4.json"),
        ("v3r2_proxy_comparison", "v3r2_proxy_comparison_m4.json"),
    ]:
        path = M4_DIR / filename
        if path.exists():
            manifest["qualification"][name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }

    # Authority configuration (frozen thresholds)
    manifest["authority"] = {
        "force_applied": 0,
        "force_mode": "shadow",
        "tau_delta": 0.0,
        "rho": 0.3,
        "alpha": 0.90,
        "note": "FORCE is shadow-only. Not qualified for live authority.",
    }

    # Scientific qualification status
    manifest["qualification_status"] = {
        "artifact_integrity_pass": True,
        "scientific_qualification_pass": False,  # NOT qualified — 7/8 gates pass
        "promotion_gates_passed": 7,
        "promotion_gates_total": 8,
        "failed_gates": [
            "N_effective_intervention = 287 (need >= 300 — 13 short)",
        ],
        "reasons": [
            "N_effective_intervention = 287, just 13 short of 300 threshold",
            "Mechanism OOD has 1 break (not zero) — need to investigate",
            "V3R2 comparison uses rule proxy, not frozen V3R2 release",
            "Need fresh confirmation corpus before any hard authority promotion",
        ],
        "passed_gates": [
            "Regret_hybrid^structOOD (11.29) < Regret_MB^structOOD (19.94)",
            "Regret_hybrid^mechOOD (15.06) < Regret_MB^mechOOD (31.66)",
            "Stratified conformal coverage_90^structOOD = 0.904 >= 0.88",
            "Mechanism OOD harm FNR = 8.3% < 10%",
            "Breaks == 0 on structural OOD (74 interventions)",
            "UCB_95(break_rate) = 4.05% < 5% on structural OOD",
            "Rescue recall > 0 on both splits (struct=11.9%, mech=40.5%)",
        ],
        "improvements_over_m4": [
            "Corpus expanded 2.5x: 1248→3168 train, 300→750 per OOD split",
            "Added ambiguous_competition mechanism for near-boundary cases",
            "Boundary-weighted Q_res training prioritizes decision-boundary examples",
            "Pairwise advantage model trained directly on ΔU (85% sign acc on struct OOD)",
            "Stratified conformal calibration (6 strata by action/entropy/competition)",
            "Struct OOD coverage_90: 82% → 86.7% (global) → 90.4% (stratified)",
            "Struct OOD regret: 14.57 → 11.29",
            "Struct OOD top-1: 0.480 → 0.552",
            "Risk model struct OOD AUROC: 0.949 → 0.974",
            "Risk model struct OOD harm FNR: 16.2% → 7.5%",
            "Shadow authority struct OOD: 12 force/0 breaks → 74 force/0 breaks",
            "Shadow authority mech OOD: 0 force (abstained) → 213 force/1 break",
            "Shadow authority rescue recall: struct 5.7% → 11.9%, mech 0% → 40.5%",
            "V3R2 proxy: DAPH-X +13.98 utility on struct OOD, +13.52 on mech OOD",
        ],
    }

    # Save manifest
    manifest_path = RELEASE_DIR / "MANIFEST.sha256"
    with open(manifest_path, "w") as f:
        f.write(f"# DAPH-X M4 Release Manifest\n")
        f.write(f"# Generated from commit {manifest['source_commit']}\n")
        f.write(f"# Working tree dirty: {manifest['dirty_worktree']}\n\n")
        for category, items in manifest.items():
            if isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict) and "sha256" in info:
                        f.write(f"{info['sha256']}  {info['path']}\n")

    # Save full manifest as JSON
    json_path = RELEASE_DIR / "experiment_manifest.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Release manifest saved to {RELEASE_DIR}")
    print(f"  MANIFEST.sha256: {manifest_path}")
    print(f"  experiment_manifest.json: {json_path}")
    print(f"\nScientific qualification: {'PASS' if manifest['qualification_status']['scientific_qualification_pass'] else 'FAIL'}")
    print(f"Reasons:")
    for r in manifest["qualification_status"]["reasons"]:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
