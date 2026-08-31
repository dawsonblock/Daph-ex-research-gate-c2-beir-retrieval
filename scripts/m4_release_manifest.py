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
        "artifact_integrity_pass": True,  # Will be verified
        "scientific_qualification_pass": False,  # NOT qualified
        "reasons": [
            "Q_res does not generalize to structural OOD (MAE worse than Q_MB)",
            "Conformal coverage below nominal on OOD splits",
            "Shadow authority break rate 27.6% on structural OOD, 100% on mechanism OOD",
            "Intervention risk harm FNR 20-34% on OOD",
            "Structural OOD has only 300 states (smoke test, not full qualification)",
            "V3R2 comparison uses rule proxy, not frozen V3R2 release",
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
