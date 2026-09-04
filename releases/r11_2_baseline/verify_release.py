#!/usr/bin/env python3
"""Verify that the R11.2 baseline release is intact.

Checks that all file hashes in the manifest match the current files.
The R12 qualification scripts should refuse to run if this verifier fails.

Usage:
    python releases/r11_2_baseline/verify_release.py

Exit code 0 = all hashes match.
Exit code 1 = one or more hashes mismatch (release is corrupted or files were modified).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "releases/r11_2_baseline/manifest.json"


def verify() -> int:
    if not MANIFEST_PATH.exists():
        print("FAIL: manifest.json not found")
        return 1

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"Verifying release: {manifest['release_name']}")
    print(f"  Git commit: {manifest['git_commit']}")
    print()

    all_ok = True

    for fpath, info in manifest["files"].items():
        full = REPO_ROOT / fpath
        if "error" in info:
            print(f"  SKIP {fpath}: {info['error']}")
            continue
        if not full.exists():
            print(f"  FAIL {fpath}: file missing")
            all_ok = False
            continue
        actual = hashlib.sha256(full.read_bytes()).hexdigest()
        expected = info["sha256"]
        if actual == expected:
            print(f"  OK   {fpath}")
        else:
            print(f"  FAIL {fpath}: hash mismatch")
            print(f"       expected: {expected[:32]}...")
            print(f"       actual:   {actual[:32]}...")
            all_ok = False

    # Check model file
    model_info = manifest.get("model", {})
    if "sha256" in model_info:
        model_path = Path(model_info["path"])
        if model_path.exists():
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual == model_info["sha256"]:
                print(f"  OK   model: {model_info['name']}")
            else:
                print(f"  FAIL model: hash mismatch")
                all_ok = False
        else:
            print(f"  FAIL model: file missing at {model_path}")
            all_ok = False

    print()
    if all_ok:
        print("PASS: All release hashes verified.")
        return 0
    else:
        print("FAIL: One or more hashes mismatch. Release is not intact.")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
