#!/usr/bin/env python3
"""R12 Release Verifier: Check that all frozen R12 artifacts are intact.

Usage:
    python releases/daph_x_r12/verify_release.py

Exits 0 if all artifacts match, 1 otherwise.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RELEASE_DIR = REPO_ROOT / "releases/daph_x_r12"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest_path = RELEASE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("FAIL: manifest.json not found")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"R12 Release Verifier")
    print(f"  Release: {manifest['release']}")
    print(f"  Git commit: {manifest['git_commit']}")
    print(f"  Frozen: {manifest['release_date']}")
    print()

    all_ok = True
    for name, info in manifest["artifacts"].items():
        if info.get("error"):
            print(f"  FAIL: {name} — {info['error']}")
            all_ok = False
            continue

        path = REPO_ROOT / info["path"]
        if not path.exists():
            print(f"  FAIL: {name} — file missing at {info['path']}")
            all_ok = False
            continue

        current_hash = sha256_file(path)
        if current_hash != info["sha256"]:
            print(f"  FAIL: {name} — hash mismatch")
            print(f"    expected: {info['sha256']}")
            print(f"    actual:   {current_hash}")
            all_ok = False
        else:
            size_mb = info["size_bytes"] / 1e6
            print(f"  OK: {name} ({size_mb:.1f} MB)")

    print()
    if all_ok:
        print("  ALL ARTIFACTS VERIFIED — R12 release is intact.")
        sys.exit(0)
    else:
        print("  VERIFICATION FAILED — R12 release has been modified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
