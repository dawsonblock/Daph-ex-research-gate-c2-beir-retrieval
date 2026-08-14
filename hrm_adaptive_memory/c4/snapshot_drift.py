"""Verify a live source tree matches a prior certified SOURCE_SNAPSHOT.json.

Built for one specific problem: qualification must run the EXACT mechanism
that development certified, with zero silent drift, while still allowing new
orchestration files (a qualification launcher, new tests) to be added on top.

The invariant this enforces is simple and complete: every file present in the
snapshot must still hash identically. Files not in the snapshot are ignored --
that is precisely how new, purely-additive files (the launcher itself, new
tests) are allowed without weakening the check. Nothing that existed and
mattered at certification time may change without the change being visible.

This is deliberately NOT scoped to "mechanism files only" (e.g. just
hrm_adaptive_memory/c4/*.py). A hand-picked file list is exactly the kind of
thing that quietly misses a real dependency -- the S2c selector lives in
hrm_adaptive_memory/retrieval_bench/selectors/chain.py, BGE embedding in
hrm_adaptive_memory/retrieval/embedding.py, BM25 in hrm_adaptive_memory/
backends/ -- all outside any single directory someone might guess. Checking
the WHOLE recorded snapshot instead of a curated subset means there is no list
to keep in sync and no way to forget a dependency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA_VERSION = "c4-source-snapshot-v1"


class SnapshotLoadError(Exception):
    """Raised when a snapshot file is missing, malformed, or the wrong schema."""


@dataclass
class DriftResult:
    """Outcome of comparing a live tree against a certified snapshot."""
    ok: bool
    files_checked: int
    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    new_files_ignored: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "files_checked": self.files_checked,
            "changed": self.changed[:20],
            "changed_count": len(self.changed),
            "missing": self.missing[:20],
            "missing_count": len(self.missing),
            "new_files_ignored": self.new_files_ignored,
        }


def load_source_snapshot(path: str | Path) -> dict:
    """Load and schema-check a SOURCE_SNAPSHOT.json.

    A missing or malformed snapshot is a load error, not an empty dict that
    would silently make every subsequent drift check vacuously pass.
    """
    p = Path(path)
    if not p.is_file():
        raise SnapshotLoadError(
            f"source snapshot not found: {p}. Qualification cannot verify "
            f"'no development-side changes' without the certified snapshot "
            f"it is compared against.")
    try:
        snapshot = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotLoadError(f"snapshot {p} is not valid JSON: {exc}") from exc

    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotLoadError(
            f"snapshot {p} has schema_version "
            f"{snapshot.get('schema_version')!r}, expected "
            f"{SNAPSHOT_SCHEMA_VERSION!r}")
    if not snapshot.get("files"):
        raise SnapshotLoadError(f"snapshot {p} has an empty files map")
    return snapshot


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_no_drift(snapshot: dict, repo_root: Path) -> DriftResult:
    """Compare every file recorded in ``snapshot`` against the live tree.

    Every recorded (path, hash) pair must still match exactly. A file the
    snapshot recorded but that no longer exists is a violation (deletion is a
    development-side change too). A file that exists now but was NOT in the
    snapshot is not checked at all -- it is new, and by construction nothing
    the snapshot certified could have depended on it.
    """
    files: dict[str, str] = snapshot["files"]
    changed: list[str] = []
    missing: list[str] = []

    for rel_path, expected_hash in sorted(files.items()):
        full_path = repo_root / rel_path
        if not full_path.is_file():
            missing.append(rel_path)
            continue
        actual_hash = _sha256_file(full_path)
        if actual_hash != expected_hash:
            changed.append(rel_path)

    return DriftResult(
        ok=not changed and not missing,
        files_checked=len(files),
        changed=changed,
        missing=missing,
    )
