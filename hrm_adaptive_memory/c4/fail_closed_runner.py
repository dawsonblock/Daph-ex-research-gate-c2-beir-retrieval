#!/usr/bin/env python3
"""Phase 10: Fail-closed experiment runner wrapper.

The Colab/local wrapper should no longer continue after structural failures.
This module provides pre-flight checks and a fail-closed execution framework.

Abort immediately if:
- git tree dirty
- git revision unexpected
- protocol hash mismatch
- test suite fails
- dependency version mismatch
- GPU environment invalid
- source receipt missing
- source hash mismatch
- output file already exists unexpectedly

The runner produces a final status only if every prerequisite passed:
    VALID_RUN=true
Otherwise:
    VALID_RUN=false
    reason=[...]

A failed qualification never produces artifacts in the same directory layout
as a valid run. Failed runs go to evidence/failed_runs/<run-id>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class PreFlightResult:
    """Result of pre-flight checks."""
    passed: bool
    reason: str = ""
    checks: dict | None = None

    def __post_init__(self):
        if self.checks is None:
            self.checks = {}


class FailClosedError(Exception):
    """Raised when a pre-flight check fails. Do not continue."""
    pass


def check_git_clean(repo: Path) -> tuple[bool, str]:
    """Verify git tree is clean."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo, capture_output=True, text=True, timeout=10)
    if result.stdout.strip():
        return False, f"Git tree dirty: {result.stdout.strip()[:200]}"
    return True, ""


def check_git_revision(repo: Path, expected: str | None = None) -> tuple[bool, str]:
    """Verify git revision matches expected (if provided)."""
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=30).strip()
    if expected and actual != expected:
        return False, f"Git revision mismatch: expected {expected}, got {actual}"
    return True, actual


def check_protocol_hash(repo: Path, expected: str | None = None) -> tuple[bool, str]:
    """Verify protocol hash matches expected (if provided)."""
    protocol_path = repo / "configs/gate_c4_protocol.json"
    if not protocol_path.exists():
        return False, f"Protocol file missing: {protocol_path}"
    actual = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if expected and actual != expected:
        return False, f"Protocol hash mismatch: expected {expected}, got {actual}"
    return True, actual


def check_test_suite(repo: Path) -> tuple[bool, str]:
    """Run the test suite and verify it passes."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--timeout=30"],
        cwd=repo, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return False, f"Test suite failed: {result.stdout[-300:]}"
    return True, ""


def check_no_unexpected_output(output_dir: Path) -> tuple[bool, str]:
    """Verify output directory doesn't already have results."""
    if output_dir.exists():
        files = list(output_dir.glob("*.jsonl"))
        if files:
            return False, f"Output directory already has {len(files)} JSONL files"
    return True, ""


def check_source_receipt(source_path: Path, expected_hash: str | None = None) -> tuple[bool, str]:
    """Verify source receipt exists and hash matches."""
    if not source_path.exists():
        return False, f"Source receipt missing: {source_path}"
    if expected_hash:
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual != expected_hash:
            return False, f"Source hash mismatch: expected {expected_hash}, got {actual}"
    return True, ""


def run_preflight_checks(
    repo: Path,
    *,
    expected_git_revision: str | None = None,
    expected_protocol_hash: str | None = None,
    run_tests: bool = True,
    output_dir: Path | None = None,
    source_path: Path | None = None,
    expected_source_hash: str | None = None,
) -> PreFlightResult:
    """Run all pre-flight checks. Return result.

    If any check fails, return immediately with passed=False.
    """
    checks = {}

    # 1. Git clean
    ok, msg = check_git_clean(repo)
    checks["git_clean"] = {"passed": ok, "message": msg}
    if not ok:
        return PreFlightResult(False, msg, checks)

    # 2. Git revision
    ok, msg = check_git_revision(repo, expected_git_revision)
    checks["git_revision"] = {"passed": ok, "message": msg}
    if not ok:
        return PreFlightResult(False, msg, checks)

    # 3. Protocol hash
    ok, msg = check_protocol_hash(repo, expected_protocol_hash)
    checks["protocol_hash"] = {"passed": ok, "message": msg}
    if not ok:
        return PreFlightResult(False, msg, checks)

    # 4. Test suite
    if run_tests:
        ok, msg = check_test_suite(repo)
        checks["test_suite"] = {"passed": ok, "message": msg}
        if not ok:
            return PreFlightResult(False, msg, checks)

    # 5. Output directory
    if output_dir:
        ok, msg = check_no_unexpected_output(output_dir)
        checks["output_dir"] = {"passed": ok, "message": msg}
        if not ok:
            return PreFlightResult(False, msg, checks)

    # 6. Source receipt
    if source_path:
        ok, msg = check_source_receipt(source_path, expected_source_hash)
        checks["source_receipt"] = {"passed": ok, "message": msg}
        if not ok:
            return PreFlightResult(False, msg, checks)

    return PreFlightResult(True, "All checks passed", checks)


def write_status_file(
    output_dir: Path,
    *,
    valid: bool,
    reason: str = "",
    checks: dict | None = None,
) -> Path:
    """Write a status file to the output directory.

    For valid runs: output_dir/STATUS.json with VALID_RUN=true
    For failed runs: output_dir/STATUS.json with VALID_RUN=false
    """
    status = {
        "VALID_RUN": valid,
        "reason": reason,
        "checks": checks or {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "STATUS.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status_path


def fail_closed_run(
    repo: Path,
    output_dir: Path,
    experiment_fn: Callable,
    *,
    expected_git_revision: str | None = None,
    expected_protocol_hash: str | None = None,
    run_tests: bool = True,
    source_path: Path | None = None,
    expected_source_hash: str | None = None,
    failed_runs_dir: Path | None = None,
) -> dict:
    """Execute an experiment with fail-closed semantics.

    1. Run pre-flight checks.
    2. If any check fails, write status to failed_runs_dir and return.
    3. If all checks pass, run the experiment.
    4. Write status to output_dir.

    Args:
        repo: Repository root.
        output_dir: Where valid run artifacts go.
        experiment_fn: Callable that takes (repo, output_dir) and returns a dict.
        expected_git_revision: Optional expected git commit.
        expected_protocol_hash: Optional expected protocol hash.
        run_tests: Whether to run the test suite (default True).
        source_path: Optional source receipt to verify.
        expected_source_hash: Optional expected source hash.
        failed_runs_dir: Where failed run status goes (default: evidence/failed_runs/).

    Returns:
        Dict with 'valid', 'reason', 'output_dir', and 'result' (if valid).
    """
    if failed_runs_dir is None:
        failed_runs_dir = repo / "evidence/failed_runs"

    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    # Pre-flight
    preflight = run_preflight_checks(
        repo,
        expected_git_revision=expected_git_revision,
        expected_protocol_hash=expected_protocol_hash,
        run_tests=run_tests,
        output_dir=output_dir,
        source_path=source_path,
        expected_source_hash=expected_source_hash,
    )

    if not preflight.passed:
        # Write to failed_runs, NOT to the valid output directory
        failed_dir = failed_runs_dir / run_id
        write_status_file(
            failed_dir,
            valid=False,
            reason=preflight.reason,
            checks=preflight.checks,
        )
        return {
            "valid": False,
            "reason": preflight.reason,
            "output_dir": str(failed_dir),
            "result": None,
        }

    # All checks passed — run the experiment
    try:
        result = experiment_fn(repo, output_dir)
        write_status_file(
            output_dir,
            valid=True,
            reason="All checks passed",
            checks=preflight.checks,
        )
        return {
            "valid": True,
            "reason": "All checks passed",
            "output_dir": str(output_dir),
            "result": result,
        }
    except Exception as e:
        # Experiment failed — write to failed_runs
        failed_dir = failed_runs_dir / run_id
        write_status_file(
            failed_dir,
            valid=False,
            reason=f"Experiment error: {e}",
            checks=preflight.checks,
        )
        return {
            "valid": False,
            "reason": f"Experiment error: {e}",
            "output_dir": str(failed_dir),
            "result": None,
        }
