"""Tests for the C4 environment lock.

The point of the lock is that an unrecorded version is a failure, not a
wildcard. "pip install transformers" under a heading reading INSTALL EXACT
DEPENDENCIES is what these tests exist to make impossible to certify.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.environment_lock import (
    LOCK_SCHEMA_VERSION,
    RESULT_AFFECTING_PACKAGES,
    EnvironmentLockViolation,
    capture_environment,
    load_lock,
    pip_requirements,
    verify_environment,
)

ROOT = Path(__file__).resolve().parents[2]


def _lock(**overrides) -> dict:
    base = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "python": "3.12.13",
        "packages": {"torch": "2.11.0+cu128", "transformers": "5.13.1"},
        "accelerator": {"cuda": "12.8", "gpu_name": "Tesla T4", "available": True},
    }
    base.update(overrides)
    return base


def _observed(**overrides) -> dict:
    base = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "python": "3.12.13",
        "packages": {"torch": "2.11.0+cu128", "transformers": "5.13.1"},
        "accelerator": {"cuda": "12.8", "gpu_name": "Tesla T4", "available": True},
    }
    base.update(overrides)
    return base


class TestCapture:
    def test_captures_schema_and_python(self):
        env = capture_environment()
        assert env["schema_version"] == LOCK_SCHEMA_VERSION
        assert env["python"].count(".") == 2
        assert set(env["packages"]) == set(RESULT_AFFECTING_PACKAGES)

    def test_does_not_pin_unused_packages(self):
        """BM25 is implemented in-repo; pinning rank-bm25 is false precision."""
        assert "rank_bm25" not in RESULT_AFFECTING_PACKAGES
        assert "faiss" not in RESULT_AFFECTING_PACKAGES
        assert "sentence_transformers" not in RESULT_AFFECTING_PACKAGES


class TestVerify:
    def test_matching_environment_passes(self):
        ok, violations, _ = verify_environment(_lock(), _observed())
        assert ok
        assert violations == []

    def test_version_mismatch_fails(self):
        ok, violations, _ = verify_environment(
            _lock(), _observed(packages={"torch": "2.10.0",
                                         "transformers": "5.13.1"}))
        assert not ok
        assert any("torch: lock=2.11.0+cu128 observed=2.10.0" in v for v in violations)

    def test_python_mismatch_fails(self):
        ok, violations, _ = verify_environment(_lock(), _observed(python="3.11.9"))
        assert not ok
        assert any(v.startswith("python:") for v in violations)

    def test_missing_package_fails(self):
        ok, violations, _ = verify_environment(
            _lock(), _observed(packages={"transformers": "5.13.1"}))
        assert not ok
        assert any("NOT_INSTALLED" in v for v in violations)

    def test_null_pin_is_must_record_not_wildcard(self):
        """An unrecorded version must FAIL, not match anything."""
        lock = _lock(packages={"torch": None, "transformers": "5.13.1"})
        ok, violations, _ = verify_environment(lock, _observed())
        assert not ok
        assert any("MUST_RECORD" in v and "torch" in v for v in violations)

    def test_empty_packages_fails(self):
        ok, violations, _ = verify_environment(_lock(packages={}), _observed())
        assert not ok
        assert any("MUST_RECORD" in v for v in violations)

    def test_accelerator_mismatch_fails(self):
        ok, violations, _ = verify_environment(
            _lock(), _observed(accelerator={"cuda": "11.8", "gpu_name": "Tesla T4",
                                            "available": True}))
        assert not ok
        assert any("accelerator.cuda" in v for v in violations)

    def test_unpinned_accelerator_is_allowed(self):
        """A CPU-only pre-HRM replay is legitimately device-independent."""
        lock = _lock(accelerator={"cuda": None, "gpu_name": None, "available": False})
        ok, violations, _ = verify_environment(
            lock, _observed(accelerator={"cuda": None, "gpu_name": None,
                                         "available": False}))
        assert ok, violations


class TestLoad:
    def test_missing_lock_raises(self, tmp_path):
        with pytest.raises(EnvironmentLockViolation, match="not found"):
            load_lock(tmp_path / "absent.lock")

    def test_bad_schema_raises(self, tmp_path):
        p = tmp_path / "x.lock"
        p.write_text(json.dumps({"schema_version": "nope"}))
        with pytest.raises(EnvironmentLockViolation, match="schema"):
            load_lock(p)

    def test_malformed_json_raises(self, tmp_path):
        p = tmp_path / "x.lock"
        p.write_text("{oops")
        with pytest.raises(EnvironmentLockViolation, match="not valid JSON"):
            load_lock(p)

    def test_shipped_lock_is_loadable(self):
        """The repo's lock must parse."""
        lock = load_lock(ROOT / "configs/c4_requirements.lock")
        assert lock["schema_version"] == LOCK_SCHEMA_VERSION
        assert lock["python"]

    def test_shipped_lock_has_no_unrecorded_pins(self):
        """The shipped lock is now CAPTURED (RunPod), not transcribed: no nulls.

        A transcribed lock with null pins previously shipped here on purpose --
        it could never certify, by design, until regenerated in a real run
        environment. It has since been replaced by one frozen live via
        scripts/c4_freeze_environment.py, so every package must be recorded.
        """
        lock = load_lock(ROOT / "configs/c4_requirements.lock")
        nulls = [k for k, v in lock["packages"].items() if v is None]
        assert not nulls, f"shipped lock has unrecorded pins: {nulls}"
        assert "CAPTURED" in lock.get("note", "")

    def test_transcribed_lock_with_nulls_still_fails_must_record(self):
        """The MUST_RECORD behavior itself stays covered, independent of
        whatever the shipped lock currently looks like."""
        transcribed = _lock(packages={"torch": "2.11.0+cu128",
                                      "transformers": "5.13.1",
                                      "numpy": None})
        ok, violations, _ = verify_environment(transcribed, _observed())
        assert not ok
        assert any("MUST_RECORD" in v for v in violations)


class TestPipRendering:
    def test_renders_pinned_packages(self):
        body = pip_requirements(_lock())
        assert "torch==2.11.0+cu128" in body
        assert "transformers==5.13.1" in body

    def test_unrecorded_package_is_commented_not_unpinned(self):
        """An unrecorded pin must never render as a bare install line."""
        body = pip_requirements(_lock(packages={"torch": None}))
        assert "# torch==UNRECORDED" in body
        assert "\ntorch\n" not in body
        assert "torch==" not in body.replace("# torch==UNRECORDED", "")
