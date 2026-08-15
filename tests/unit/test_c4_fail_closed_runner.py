"""Tests for the fail-closed experiment runner.

Phase 10 of the C4 determinism repair.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.fail_closed_runner import (
    run_preflight_checks,
    write_status_file,
    fail_closed_run,
    check_git_clean,
    check_protocol_hash,
    FailClosedError,
)


class TestPreFlightChecks:
    """Pre-flight checks must catch structural failures before they propagate."""

    def test_git_clean_check_passes_in_repo(self, tmp_path):
        # Initialize a clean git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        ok, msg = check_git_clean(tmp_path)
        assert ok

    def test_git_clean_check_fails_with_changes(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("changed")  # dirty

        ok, msg = check_git_clean(tmp_path)
        assert not ok
        assert "dirty" in msg.lower()

    def test_protocol_hash_check(self, tmp_path):
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        protocol = config_dir / "gate_c4_protocol.json"
        protocol.write_text('{"version": "test"}')

        ok, msg = check_protocol_hash(tmp_path)
        assert ok

        # With expected hash
        import hashlib
        expected = hashlib.sha256(protocol.read_bytes()).hexdigest()
        ok, msg = check_protocol_hash(tmp_path, expected=expected)
        assert ok

        # With wrong expected hash
        ok, msg = check_protocol_hash(tmp_path, expected="wrong")
        assert not ok
        assert "mismatch" in msg.lower()


class TestStatusFile:
    def test_valid_status(self, tmp_path):
        path = write_status_file(tmp_path, valid=True, reason="OK")
        status = json.loads(path.read_text())
        assert status["VALID_RUN"] is True
        assert status["reason"] == "OK"

    def test_invalid_status(self, tmp_path):
        path = write_status_file(tmp_path, valid=False, reason="Failed")
        status = json.loads(path.read_text())
        assert status["VALID_RUN"] is False
        assert status["reason"] == "Failed"


class TestFailClosedRun:
    """The runner must not produce artifacts in the valid directory on failure."""

    def test_failed_run_goes_to_failed_dir(self, tmp_path):
        # Create a dirty git repo so pre-flight fails
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("changed")  # dirty

        output_dir = tmp_path / "valid_output"
        failed_dir = tmp_path / "failed_runs"

        def experiment(repo, out):
            out.mkdir(parents=True, exist_ok=True)
            (out / "result.json").write_text("{}")
            return {"status": "done"}

        result = fail_closed_run(
            tmp_path, output_dir, experiment,
            run_tests=False,
            failed_runs_dir=failed_dir,
        )

        assert result["valid"] is False
        # The valid output directory should NOT have a STATUS file
        assert not (output_dir / "STATUS.json").exists()
        # The failed directory SHOULD have a STATUS file
        assert (failed_dir).exists()
        status_files = list(failed_dir.glob("*/STATUS.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text())
        assert status["VALID_RUN"] is False

    def test_successful_run_goes_to_valid_dir(self, tmp_path):
        # Clean git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
        (tmp_path / "file.txt").write_text("hello")
        # Create the protocol file that the runner expects
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "gate_c4_protocol.json").write_text('{"version": "test"}')
        subprocess.run(["git", "add", "-A"], cwd=tmp_path)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        output_dir = tmp_path / "valid_output"
        failed_dir = tmp_path / "failed_runs"

        def experiment(repo, out):
            out.mkdir(parents=True, exist_ok=True)
            (out / "result.json").write_text('{"quality": 0.5}')
            return {"quality": 0.5}

        result = fail_closed_run(
            tmp_path, output_dir, experiment,
            run_tests=False,
            failed_runs_dir=failed_dir,
        )

        assert result["valid"] is True
        assert (output_dir / "STATUS.json").exists()
        assert (output_dir / "result.json").exists()
        status = json.loads((output_dir / "STATUS.json").read_text())
        assert status["VALID_RUN"] is True
