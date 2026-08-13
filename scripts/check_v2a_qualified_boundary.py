#!/usr/bin/env python3
"""Fail closed if a checkout misrepresents or loses the frozen V2A baseline."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = ROOT / "configs" / "v2a_qualified_boundary.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check() -> None:
    boundary = json.loads(BOUNDARY_PATH.read_text())
    commit = str(boundary["qualified_commit"])
    tree = str(boundary["qualified_tree"])
    tag = str(boundary["required_tag"])
    if _git("rev-parse", f"{commit}^{{tree}}") != tree:
        raise RuntimeError("V2A qualified commit no longer resolves to the recorded tree")
    if _git("rev-parse", f"{tag}^{{commit}}") != commit:
        raise RuntimeError(f"{tag} must resolve exactly to the V2A qualified commit")

    archive = ROOT / str(boundary["build_manifest"]["qualified_archive"])
    if not archive.is_file() or _sha256(archive) != boundary["build_manifest"]["qualified_archive_sha256"]:
        raise RuntimeError("V2A qualified archive is absent or does not match its recorded hash")
    for receipt in boundary["qualification_receipts"]:
        if not (ROOT / receipt).is_file():
            raise RuntimeError(f"V2A qualification receipt is absent: {receipt}")

    changed = set(_git("diff", "--name-only", f"{commit}...HEAD").splitlines())
    protected = set(boundary["qualification_inputs"])
    changed_protected = sorted(changed & protected)
    if not changed_protected:
        return

    declaration = boundary["required_v2b_declaration"]
    candidate = ROOT / declaration["path"]
    if not candidate.is_file():
        raise RuntimeError("V2A qualification inputs changed without a V2B declaration")
    v2b = json.loads(candidate.read_text())
    if (v2b.get("protocol_id") != declaration["protocol_id"]
            or v2b.get("status") != declaration["status"]):
        raise RuntimeError("V2A qualification inputs changed without an explicit V2B development status")
    print("V2A boundary preserved; V2B-declared changes:", ", ".join(changed_protected))


if __name__ == "__main__":
    check()
