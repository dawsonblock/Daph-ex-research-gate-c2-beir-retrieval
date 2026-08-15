"""C4 evaluator provenance — cryptographic binding of evaluator outputs to sources.

Phase 9 of the C4 determinism repair.

Every evaluator output must declare its exact input. Before rescoring,
verify that the source hash matches. Before loading an existing evaluator
result, verify that its declared source hash equals the actual raw arm hash.

This would have immediately caught the two different C4-4 raw files sharing
byte-identical evaluator-v2 outputs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "c4-evaluator-manifest-v3"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _code_sha256(repo: Path) -> str:
    """Hash the evaluator-relevant source files."""
    files = [
        "hrm_adaptive_memory/c4/metrics.py",
        "hrm_adaptive_memory/c4/provenance.py",
        "hrm_adaptive_memory/c4/evaluator_manifest.py",
        "hrm_adaptive_memory/evaluation/verifiers.py",
        "scripts/analyze_gate_c4.py",
    ]
    h = hashlib.sha256()
    for f in files:
        p = repo / f
        if p.exists():
            h.update(_sha256_file(p).encode())
    return h.hexdigest()


def build_evaluator_manifest(
    *,
    repo: Path,
    run_id: str,
    arm_id: str,
    jsonl_path: Path,
    output_dir: Path,
    scoring_policy: str = "c4_quality_v2",
) -> dict:
    """Build an evaluator manifest that cryptographically binds output to source.

    Args:
        repo: Repository root (for git commit and code hash).
        run_id: Identifier for the run.
        arm_id: Arm ID (e.g. "C4_4").
        jsonl_path: Path to the raw arm JSONL file.
        output_dir: Directory where evaluator outputs will be written.
        scoring_policy: Scoring policy identifier.

    Returns:
        Manifest dict to be written as 'evaluator_manifest.json' in output_dir.
    """
    jsonl_sha256 = _sha256_file(jsonl_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "run_id": run_id,
            "arm_id": arm_id,
            "jsonl_path": str(jsonl_path.relative_to(repo)) if jsonl_path.is_relative_to(repo) else str(jsonl_path),
            "jsonl_sha256": jsonl_sha256,
        },
        "evaluator": {
            "version": sys.version.split()[0],
            "git_commit": _git_commit(repo),
            "code_sha256": _code_sha256(repo),
            "scoring_policy": scoring_policy,
        },
        "output": {
            "output_dir": str(output_dir.relative_to(repo)) if output_dir.is_relative_to(repo) else str(output_dir),
        },
    }


def finalize_evaluator_manifest(
    manifest: dict,
    *,
    output_jsonl_path: Path,
    analysis_path: Path | None = None,
) -> dict:
    """Add output hashes to a manifest after evaluation is complete."""
    manifest["output"]["jsonl_sha256"] = _sha256_file(output_jsonl_path)
    if analysis_path and analysis_path.exists():
        manifest["output"]["analysis_sha256"] = _sha256_file(analysis_path)
    return manifest


def verify_evaluator_manifest(manifest: dict, jsonl_path: Path) -> None:
    """Verify that a manifest's declared source hash matches the actual file.

    Raises AssertionError if the hash does not match. This is a hard fail —
    do not silently continue.
    """
    expected = manifest["source"]["jsonl_sha256"]
    actual = _sha256_file(jsonl_path)
    if expected != actual:
        raise AssertionError(
            f"Evaluator provenance mismatch:\n"
            f"  Manifest declares source hash: {expected}\n"
            f"  Actual file hash:              {actual}\n"
            f"  File: {jsonl_path}\n"
            f"This means the evaluator output is NOT bound to this source.\n"
            f"FAIL CLOSED — do not use this evaluator result."
        )


def load_and_verify_evaluator_manifest(
    manifest_path: Path,
    expected_jsonl_path: Path,
) -> dict:
    """Load an evaluator manifest and verify it against the expected source.

    Raises AssertionError if the manifest is missing or the hash doesn't match.
    """
    if not manifest_path.exists():
        raise AssertionError(
            f"Evaluator manifest missing: {manifest_path}\n"
            f"Cannot verify provenance. FAIL CLOSED."
        )
    manifest = json.loads(manifest_path.read_text())
    verify_evaluator_manifest(manifest, expected_jsonl_path)
    return manifest


def write_evaluator_manifest(
    manifest: dict,
    output_dir: Path,
) -> Path:
    """Write the evaluator manifest to the output directory."""
    path = output_dir / "evaluator_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
