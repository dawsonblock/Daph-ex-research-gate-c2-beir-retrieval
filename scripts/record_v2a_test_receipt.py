#!/usr/bin/env python3
"""Run the complete repository suite and record a hashed V2A receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hrm_adaptive_memory.external_verification.qualification import qualification_identity

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "evidence/background_verification_v2a/tests/full_suite.json"))
    parser.add_argument("--skip-litlogger", action="store_true")
    args = parser.parse_args()
    if _git("status", "--porcelain"):
        raise RuntimeError("test receipt refuses a dirty source tree")
    source_commit = _git("rev-parse", "HEAD")
    source_tree_hash = _git("rev-parse", "HEAD^{tree}")
    identity = qualification_identity(ROOT)
    experiment = None
    if not args.skip_litlogger:
        import litlogger
        experiment = litlogger.init(
            name=f"v2a-full-suite-{source_commit[:7]}",
            teamspace="deep-gpu-acceleration-project",
            metadata={"protocol": "BACKGROUND_VERIFICATION_V2A",
                      "source_commit": source_commit,
                      "source_tree_hash": source_tree_hash,
                      "location": "US", "altitude": "1334"})
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    combined = completed.stdout + completed.stderr
    passed_match = re.search(r"(\d+) passed", combined)
    skipped_match = re.search(r"(\d+) skipped", combined)
    failed_match = re.search(r"(\d+) failed", combined)
    passed = int(passed_match.group(1)) if passed_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    run_valid = completed.returncode == 0 and passed > 0 and failed == 0
    receipt = {
        "artifact": "BACKGROUND_VERIFICATION_V2A_FULL_TEST_RECEIPT",
        "run_valid": run_valid,
        "source_commit": source_commit,
        "source_tree_hash": source_tree_hash,
        "qualification_identity": identity,
        "branch": _git("branch", "--show-current"),
        "command": [sys.executable, "-m", "pytest", "-q"],
        "returncode": completed.returncode,
        "passed": passed, "failed": failed, "skipped": skipped,
        "elapsed_s": elapsed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": combined[-4000:],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {out.name}\n")
    if experiment is not None:
        experiment["passed"].append(passed)
        experiment["skipped"].append(skipped)
        experiment["failed"].append(failed)
        experiment["elapsed_s"].append(elapsed)
        experiment["run_valid"] = str(run_valid).lower()
        experiment.finalize()
    print(combined, end="")
    print(json.dumps({"run_valid": run_valid, "passed": passed,
                      "failed": failed, "skipped": skipped,
                      "receipt": str(out)}, sort_keys=True))
    return 0 if run_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
