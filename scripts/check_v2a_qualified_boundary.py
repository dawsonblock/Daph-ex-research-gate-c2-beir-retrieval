#!/usr/bin/env python3
"""Fail closed if a checkout misrepresents or loses the frozen V2A baseline.

Boundary schema: DAPH_V2A_QUALIFIED_BOUNDARY_V2 (see configs/v2a_qualified_boundary.json).

The V2 boundary is derived entirely from immutable, version-controlled evidence
and requires no local git tag and no untracked dist/ archive, so it reproduces on
a clean checkout with full history. It fails closed (raises, never warns) when:

  * the qualified commit is absent from the cloned history
  * the qualified commit no longer maps to the recorded tree
  * any required evidence receipt file is missing
  * any required evidence receipt's on-disk SHA-256 differs from the recorded hash
  * any receipt's recorded source commit/tree disagrees with the boundary
  * any receipt's qualification_identity source commit/tree disagrees with the boundary
  * any receipt's recorded component hash disagrees with the qualified commit's blob
  * any receipt's qualification_lock hash disagrees with the qualified commit's blob
  * V2A qualification inputs changed on this branch without an explicit V2B declaration
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = ROOT / "configs" / "v2a_qualified_boundary.json"


class BoundaryError(RuntimeError):
    """Raised for every fail-closed condition. Never caught to warn-and-continue."""


def _git(*args: str, root: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _git_bytes(revpath: str, root: Path = ROOT) -> bytes:
    return subprocess.check_output(["git", "show", revpath], cwd=root)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_receipt(name: str, meta: dict, commit: str, tree: str, root: Path = ROOT) -> None:
    rel = str(meta["path"])
    receipt_path = root / rel
    if not receipt_path.is_file():
        raise BoundaryError(f"V2A qualification receipt is absent: {rel}")
    actual = _sha256_file(receipt_path)
    if actual != meta["sha256"]:
        raise BoundaryError(
            f"V2A receipt '{name}' ({rel}) SHA-256 mismatch: "
            f"expected {meta['sha256']}, got {actual}"
        )
    receipt = json.loads(receipt_path.read_text())
    # Receipt must bind the same source identity as the boundary.
    if str(receipt.get("source_commit")) != commit:
        raise BoundaryError(
            f"V2A receipt '{name}' binds source_commit {receipt.get('source_commit')!r}, "
            f"expected {commit}"
        )
    if str(receipt.get("source_tree_hash")) != tree:
        raise BoundaryError(
            f"V2A receipt '{name}' binds source_tree_hash {receipt.get('source_tree_hash')!r}, "
            f"expected {tree}"
        )
    qi = receipt.get("qualification_identity") or {}
    if str(qi.get("source_commit")) != commit:
        raise BoundaryError(
            f"V2A receipt '{name}' qualification_identity.source_commit "
            f"{qi.get('source_commit')!r}, expected {commit}"
        )
    if str(qi.get("source_tree_hash")) != tree:
        raise BoundaryError(
            f"V2A receipt '{name}' qualification_identity.source_tree_hash "
            f"{qi.get('source_tree_hash')!r}, expected {tree}"
        )
    # Every recorded component hash must match the qualified commit's immutable blob.
    for comp, cmeta in (qi.get("component_hashes") or {}).items():
        cpath = str(cmeta["path"])
        expected = str(cmeta["sha256"])
        try:
            blob = _git_bytes(f"{commit}:{cpath}", root=root)
        except subprocess.CalledProcessError:
            raise BoundaryError(
                f"V2A receipt '{name}' component '{comp}' path {cpath!r} "
                f"absent at qualified commit {commit}"
            )
        if _sha256_bytes(blob) != expected:
            raise BoundaryError(
                f"V2A receipt '{name}' component '{comp}' ({cpath}) hash "
                f"disagrees with qualified commit blob"
            )
    lock = qi.get("qualification_lock") or {}
    if lock:
        lpath = str(lock["path"])
        try:
            blob = _git_bytes(f"{commit}:{lpath}", root=root)
        except subprocess.CalledProcessError:
            raise BoundaryError(
                f"V2A receipt '{name}' qualification_lock path {lpath!r} "
                f"absent at qualified commit {commit}"
            )
        if _sha256_bytes(blob) != str(lock["sha256"]):
            raise BoundaryError(
                f"V2A receipt '{name}' qualification_lock ({lpath}) hash "
                f"disagrees with qualified commit blob"
            )


def check(boundary_path: Path = BOUNDARY_PATH, root: Path = ROOT) -> None:
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("schema") != "DAPH_V2A_QUALIFIED_BOUNDARY_V2":
        raise BoundaryError(
            f"Unexpected V2A boundary schema {boundary.get('schema')!r}; "
            "expected DAPH_V2A_QUALIFIED_BOUNDARY_V2"
        )
    commit = str(boundary["commit"])
    tree = str(boundary["tree"])

    # The qualified commit must be present in the cloned history (requires
    # fetch-depth: 0, which the CI workflow already uses). No tag is required.
    try:
        resolved_commit = _git("rev-parse", "--verify", f"{commit}^{{commit}}", root=root)
    except subprocess.CalledProcessError:
        raise BoundaryError(
            f"V2A qualified commit {commit} is absent from the cloned history "
            "(checkout must use full history: actions/checkout fetch-depth: 0)"
        )
    if resolved_commit != commit:
        raise BoundaryError(
            f"V2A qualified commit resolved to {resolved_commit}, expected {commit}"
        )
    resolved_tree = _git("rev-parse", f"{commit}^{{tree}}", root=root)
    if resolved_tree != tree:
        raise BoundaryError(
            f"V2A qualified commit maps to tree {resolved_tree}, expected {tree}"
        )

    receipts = boundary.get("required_receipts") or {}
    if not receipts:
        raise BoundaryError("V2A boundary declares no required receipts")
    for name, meta in receipts.items():
        _verify_receipt(name, meta, commit, tree, root=root)

    # V2B declaration guard: qualification inputs may only change on a V2B branch
    # if an explicit V2B development-status declaration is present.
    changed = set(_git("diff", "--name-only", f"{commit}...HEAD", root=root).splitlines())
    protected = set(boundary.get("qualification_inputs") or [])
    changed_protected = sorted(changed & protected)
    if not changed_protected:
        print(f"V2A boundary preserved; verified {len(receipts)} receipt(s).")
        return

    declaration = boundary.get("required_v2b_declaration") or {}
    candidate = root / str(declaration.get("path", ""))
    if not candidate.is_file():
        raise BoundaryError(
            "V2A qualification inputs changed without a V2B declaration: "
            + ", ".join(changed_protected)
        )
    v2b = json.loads(candidate.read_text())
    if (v2b.get("protocol_id") != declaration.get("protocol_id")
            or v2b.get("status") != declaration.get("status")):
        raise BoundaryError(
            "V2A qualification inputs changed without an explicit V2B development status"
        )
    print(
        f"V2A boundary preserved; verified {len(receipts)} receipt(s); "
        "V2B-declared changes: " + ", ".join(changed_protected)
    )


if __name__ == "__main__":
    check()
