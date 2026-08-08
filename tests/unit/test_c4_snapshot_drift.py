"""Tests for hrm_adaptive_memory.c4.snapshot_drift.

The invariant under test: every file present in a certified snapshot must
still hash identically for a later run to be trusted as drift-free. Files
NOT in the snapshot are ignored -- that is precisely what allows new,
purely-additive files (a qualification launcher, new tests) without
weakening the check for anything that mattered at certification time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.snapshot_drift import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotLoadError,
    load_source_snapshot,
    verify_no_drift,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _snapshot(files: dict[str, str]) -> dict:
    """A snapshot dict from {relpath: content} -- hashes computed here."""
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "files": {path: _sha256(content) for path, content in files.items()},
    }


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    return tmp_path


class TestVerifyNoDrift:
    def test_unchanged_files_pass(self, repo):
        snap = _snapshot({"a.py": "x = 1\n", "sub/b.py": "y = 2\n"})
        result = verify_no_drift(snap, repo)
        assert result.ok
        assert result.changed == []
        assert result.missing == []
        assert result.files_checked == 2

    def test_changed_file_is_detected(self, repo):
        snap = _snapshot({"a.py": "x = 1\n"})
        (repo / "a.py").write_text("x = 999\n")
        result = verify_no_drift(snap, repo)
        assert not result.ok
        assert result.changed == ["a.py"]

    def test_deleted_file_is_detected_as_missing(self, repo):
        snap = _snapshot({"a.py": "x = 1\n"})
        (repo / "a.py").unlink()
        result = verify_no_drift(snap, repo)
        assert not result.ok
        assert result.missing == ["a.py"]

    def test_new_file_not_in_snapshot_is_ignored(self, repo):
        """The exact property that lets a qualification launcher be added
        without failing drift checks against an older development snapshot."""
        snap = _snapshot({"a.py": "x = 1\n"})
        (repo / "new_launcher.py").write_text("print('new')\n")
        result = verify_no_drift(snap, repo)
        assert result.ok
        assert result.changed == []
        assert result.missing == []

    def test_subdirectory_files_are_checked(self, repo):
        snap = _snapshot({"sub/b.py": "y = 2\n"})
        (repo / "sub" / "b.py").write_text("y = 999\n")
        result = verify_no_drift(snap, repo)
        assert not result.ok
        assert result.changed == ["sub/b.py"]

    def test_multiple_violations_all_reported(self, repo):
        snap = _snapshot({"a.py": "x = 1\n", "sub/b.py": "y = 2\n"})
        (repo / "a.py").write_text("changed\n")
        (repo / "sub" / "b.py").unlink()
        result = verify_no_drift(snap, repo)
        assert not result.ok
        assert result.changed == ["a.py"]
        assert result.missing == ["sub/b.py"]

    def test_empty_snapshot_files_trivially_passes(self, repo):
        snap = {"schema_version": SNAPSHOT_SCHEMA_VERSION, "files": {}}
        result = verify_no_drift(snap, repo)
        assert result.ok
        assert result.files_checked == 0

    def test_summary_is_json_serializable(self, repo):
        snap = _snapshot({"a.py": "x = 1\n"})
        (repo / "a.py").write_text("changed\n")
        result = verify_no_drift(snap, repo)
        json.dumps(result.summary())  # must not raise


class TestLoadSourceSnapshot:
    def test_loads_valid_snapshot(self, tmp_path):
        p = tmp_path / "SOURCE_SNAPSHOT.json"
        p.write_text(json.dumps(_snapshot({"a.py": "x\n"})))
        snap = load_source_snapshot(p)
        assert snap["schema_version"] == SNAPSHOT_SCHEMA_VERSION

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SnapshotLoadError, match="not found"):
            load_source_snapshot(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        with pytest.raises(SnapshotLoadError, match="not valid JSON"):
            load_source_snapshot(p)

    def test_wrong_schema_version_raises(self, tmp_path):
        p = tmp_path / "wrong.json"
        p.write_text(json.dumps({"schema_version": "v0", "files": {"a": "b"}}))
        with pytest.raises(SnapshotLoadError, match="schema_version"):
            load_source_snapshot(p)

    def test_empty_files_map_raises(self, tmp_path):
        """An empty files map would make every drift check vacuously pass;
        that must be treated as a load error, not a clean bill of health."""
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"schema_version": SNAPSHOT_SCHEMA_VERSION,
                                 "files": {}}))
        with pytest.raises(SnapshotLoadError, match="empty"):
            load_source_snapshot(p)


class TestAgainstTheRealCommittedSnapshot:
    """End-to-end: the actual development certificate's snapshot."""

    def test_real_snapshot_loads_and_is_well_formed(self):
        root = Path(__file__).resolve().parents[2]
        snap_path = (root / "evidence/gate_c4/full/development/certification"
                    / "SOURCE_SNAPSHOT.json")
        if not snap_path.is_file():
            pytest.skip("development certification bundle not present")
        snap = load_source_snapshot(snap_path)
        assert len(snap["files"]) > 100

    def test_real_snapshot_detects_the_known_post_certification_changes(self):
        """Files legitimately modified after the certified run (this repair
        sprint's own fixes) must show up as drift -- proving the checker is
        actually sensitive to real changes, not just self-consistent."""
        root = Path(__file__).resolve().parents[2]
        snap_path = (root / "evidence/gate_c4/full/development/certification"
                    / "SOURCE_SNAPSHOT.json")
        if not snap_path.is_file():
            pytest.skip("development certification bundle not present")
        snap = load_source_snapshot(snap_path)
        result = verify_no_drift(snap, root)
        # As of this test's writing, run_gate_c4.py and certify_c4_run.py
        # have both been modified since the certified snapshot was taken.
        assert not result.ok
        assert result.missing == []  # nothing certified was deleted
