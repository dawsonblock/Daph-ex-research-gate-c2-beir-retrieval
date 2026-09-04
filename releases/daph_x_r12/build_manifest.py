#!/usr/bin/env python3
"""R12 Release: Freeze all R12 artifacts with hash verification.

Creates an immutable manifest of all R12 artifacts.
R13 consumes R12 artifacts as immutable inputs.
Do not regenerate R12 data after this point.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
R12_DIR = REPO_ROOT / "experiments/daph_x/r12"
RELEASE_DIR = REPO_ROOT / "releases/daph_x_r12"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def git_commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


def git_branch() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


# Artifacts to freeze
ARTIFACTS = [
    # Raw data
    ("raw_candidates", "experiments/daph_x/r12/r12_raw_candidates.jsonl"),
    ("enriched_corpus", "experiments/daph_x/r12/r12_enriched_corpus.jsonl"),
    # Counterfactual records
    ("counterfactuals_full", "experiments/daph_x/r12/r12_counterfactuals_full.jsonl"),
    ("counterfactual_summary", "experiments/daph_x/r12/r12_counterfactual_summary.json"),
    # Evaluation results
    ("r12_results", "experiments/daph_x/r12/r12_results.json"),
    ("r12_eval_full_log", "experiments/daph_x/r12/r12_eval_full.log"),
    ("r12_eval_multiseed_log", "experiments/daph_x/r12/r12_eval_multiseed.log"),
    # Reports
    ("r12_results_report", "experiments/daph_x/r12/R12_RESULTS_REPORT.md"),
    # Collection logs
    ("stage1_log", "experiments/daph_x/r12/stage1.log"),
    ("stage2_log", "experiments/daph_x/r12/stage2.log"),
    # Source code (key files)
    ("reasoning_tasks_py", "daph_x/coding/reasoning_tasks.py"),
    ("r12_collection_py", "scripts/run_r12_collection.py"),
    ("r12_evaluation_py", "scripts/run_r12_evaluation.py"),
    ("r12_counterfactual_py", "scripts/run_r12_counterfactual.py"),
    ("r12_data_collection_py", "scripts/run_r12_data_collection.py"),
    # Tests
    ("test_r12_corpus", "tests/unit/test_r12_reasoning_corpus.py"),
]

# Configuration to freeze
CONFIG = {
    "base_model": "Qwen2.5-7B-Instruct Q4_K_M",
    "model_path": "/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "n_tasks": 500,
    "n_candidates": 12,
    "max_tokens": 256,
    "temperature_schedule": [0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.2],
    "verification_rounds": 1,
    "verification_version": "v2",
    "split": {"train": 300, "cal": 75, "dev": 25, "test": 100},
    "seeds": [42, 123, 7, 99, 2024],
    "lambda_costs": [0.1, 0.2],
    "checkpoints": [2, 4, 6, 8, 10],
    "cost_per_step": 0.2,
    "evaluator_version": "r12_v1",
}

# Key results to freeze
KEY_RESULTS = {
    "base_correct": 0.520,
    "oracle_at_8": 0.726,
    "oracle_at_12": 0.752,
    "rescue_available": 0.232,
    "counterfactual_rescue_rate": 0.029,
    "counterfactual_break_rate": 0.001,
    "counterfactual_waste_rate": 0.970,
    "best_r12_policy": "r12_dq_t0.01",
    "best_r12_accuracy": 0.684,
    "best_r12_avg_k": 5.4,
    "best_r12_j_0_1": 0.650,
    "maxcal_8_accuracy": 0.676,
    "maxcal_8_k": 8.0,
    "uncertainty_p50_accuracy": 0.682,
    "uncertainty_p50_k": 5.0,
    "oracle_lookahead6_accuracy": 0.684,
    "oracle_lookahead6_k": 2.3,
    "oracle_regret_accuracy": 0.0,
    "oracle_regret_compute": 3.1,
    "rescue_auroc_compact": 0.932,
    "rescue_auroc_r12": 0.898,
}


def build_manifest() -> dict:
    """Build the R12 release manifest."""
    manifest = {
        "release": "daph_x_r12",
        "release_date": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "git_branch": git_branch(),
        "description": (
            "R12: Decision-aligned adaptive compute on 500-task × 12-candidate corpus. "
            "Two-stage pipeline (raw generation + enrichment). "
            "ΔQ value model with trajectory features. "
            "Matches MaxCal@8 accuracy with 33% less compute."
        ),
        "config": CONFIG,
        "key_results": KEY_RESULTS,
        "artifacts": {},
    }

    for name, rel_path in ARTIFACTS:
        path = REPO_ROOT / rel_path
        if path.exists():
            manifest["artifacts"][name] = {
                "path": rel_path,
                "sha256": sha256_file(path),
                "size_bytes": file_size(path),
            }
        else:
            manifest["artifacts"][name] = {
                "path": rel_path,
                "sha256": None,
                "size_bytes": 0,
                "error": "FILE NOT FOUND",
            }
            print(f"  WARNING: {rel_path} not found")

    return manifest


def verify_manifest(manifest: dict) -> bool:
    """Verify all artifact hashes match."""
    all_ok = True
    for name, info in manifest["artifacts"].items():
        if info.get("error"):
            print(f"  FAIL: {name} — {info['error']}")
            all_ok = False
            continue

        path = REPO_ROOT / info["path"]
        if not path.exists():
            print(f"  FAIL: {name} — file missing")
            all_ok = False
            continue

        current_hash = sha256_file(path)
        if current_hash != info["sha256"]:
            print(f"  FAIL: {name} — hash mismatch")
            print(f"    expected: {info['sha256']}")
            print(f"    actual:   {current_hash}")
            all_ok = False
        else:
            print(f"  OK: {name} ({info['size_bytes']} bytes)")

    return all_ok


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    print("R12 Release: Freezing artifacts...")
    manifest = build_manifest()

    manifest_path = RELEASE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    print(f"\nVerifying {len(manifest['artifacts'])} artifacts...")
    ok = verify_manifest(manifest)

    if ok:
        print("\n  ALL ARTIFACTS VERIFIED")
        print(f"  Git commit: {manifest['git_commit']}")
        print(f"  Release date: {manifest['release_date']}")
        print(f"\n  R12 is now frozen. Do not regenerate R12 data.")
    else:
        print("\n  VERIFICATION FAILED — some artifacts missing or changed")
        sys.exit(1)


if __name__ == "__main__":
    main()
