#!/usr/bin/env python3
"""Freeze the current environment into the C4 environment lock.

Run this on the machine that will produce the certified run, BEFORE the run:

    python scripts/c4_freeze_environment.py

Then certification compares the live environment against the lock. Without a
lock generated from a real environment, "dependency version mismatch" is an
abort condition with nothing to compare against.

Options:
    --output PATH     lock destination (default configs/c4_requirements.lock)
    --pip PATH        also emit a pip-installable requirements file
    --check           do not write; verify the live environment against the
                      existing lock and exit non-zero on any mismatch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.c4.environment_lock import (  # noqa: E402
    capture_environment, load_lock, pip_requirements, verify_environment)

DEFAULT_LOCK = ROOT / "configs/c4_requirements.lock"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze/verify the C4 environment lock")
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--pip", type=Path, default=None,
                        help="Also write a pip-installable requirements file")
    parser.add_argument("--check", action="store_true",
                        help="Verify the live environment against the lock")
    parser.add_argument("--note", default="",
                        help="Free-text provenance note stored in the lock")
    args = parser.parse_args()

    if args.check:
        try:
            lock = load_lock(args.output)
        except AssertionError as exc:
            print(f"FAIL: {exc}")
            return 1
        ok, violations, observed = verify_environment(lock)
        print(f"--- C4 environment check against {args.output} ---")
        print(f"  python:   lock={lock.get('python')} observed={observed['python']}")
        for name, version in sorted((lock.get("packages") or {}).items()):
            print(f"  {name}: lock={version} "
                  f"observed={observed['packages'].get(name)}")
        if ok:
            print("\nPASS: environment matches the lock.")
            return 0
        print(f"\nFAIL: {len(violations)} environment violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1

    lock = capture_environment()
    if args.note:
        lock["note"] = args.note
    unrecorded = [n for n, v in lock["packages"].items() if v is None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(f"  python: {lock['python']}")
    for name, version in sorted(lock["packages"].items()):
        print(f"  {name}: {version}")
    accel = lock["accelerator"]
    print(f"  accelerator: cuda={accel['cuda']} gpu={accel['gpu_name']}")

    if args.pip:
        args.pip.write_text(pip_requirements(lock))
        print(f"Wrote {args.pip}")

    if unrecorded:
        print(f"\nWARNING: not installed in this environment: {unrecorded}")
        print("  These stay null in the lock and will FAIL certification.")
        print("  Regenerate the lock in the full run environment.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
