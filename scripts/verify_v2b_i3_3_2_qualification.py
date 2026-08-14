#!/usr/bin/env python3
"""Self-verify the I3.3.2 frozen-benchmark qualification evidence bundle.

A clean checkout plus the committed bundle under
evidence/v2b_i3_3_2/qualification/ is enough to prove:

  * this qualification bundle corresponds to this exact commit/tree
  * the benchmark/oracle identities match the frozen evidence

No hidden local files are required. Fails closed on any inconsistency.

Usage:
  python scripts/verify_v2b_i3_3_2_qualification.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "evidence/v2b_i3_3_2/qualification"
BASELINE = ROOT / "configs/v2b_i3_3_3_baseline.json"
CACHE_MANIFEST = ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_oracle_cache_manifest_v1.json"


class VerificationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _require(path: Path) -> dict:
    if not path.is_file():
        raise VerificationError(f"Bundle file absent: {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def verify() -> None:
    baseline = json.loads(BASELINE.read_text())
    cache = json.loads(CACHE_MANIFEST.read_text())
    commit = baseline["source_commit"]
    tree = baseline["source_tree"]

    if not BUNDLE_DIR.is_dir():
        raise VerificationError(f"Qualification bundle directory absent: {BUNDLE_DIR}")

    identity = _require(BUNDLE_DIR / "identity.json")
    source = _require(BUNDLE_DIR / "source_identity.json")
    closure = _require(BUNDLE_DIR / "benchmark_closure.json")
    oracle = _require(BUNDLE_DIR / "oracle_set_hashes.json")
    environment = _require(BUNDLE_DIR / "environment.json")
    tests = _require(BUNDLE_DIR / "tests.json")
    receipt = _require(BUNDLE_DIR / "qualification_receipt.json")
    bundle_manifest = _require(BUNDLE_DIR / "BUNDLE.sha256")

    # 1. The bundle binds the exact commit/tree, and the checkout resolves it.
    if source["source_commit"] != commit or source["source_tree"] != tree:
        raise VerificationError("Bundle source_identity disagrees with baseline")
    actual_commit = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    actual_tree = _git("rev-parse", f"{commit}^{{tree}}")
    if actual_commit != commit or actual_tree != tree:
        raise VerificationError(
            f"Qualified commit/tree resolves to {actual_commit}/{actual_tree}, "
            f"expected {commit}/{tree}")
    if source["baseline_config_sha256"] != _sha256_file(BASELINE):
        raise VerificationError("baseline config hash changed since bundle creation")

    # 2. Receipt binds the same source identity.
    if receipt["source_commit"] != commit or receipt["source_tree"] != tree:
        raise VerificationError("Receipt source identity disagrees with baseline")

    # 3. Oracle identities match the frozen cache manifest (the frozen evidence).
    if oracle["latent_oracle"]["table_set_sha256"] != cache["latent_oracles"]["table_set_sha256"]:
        raise VerificationError("Latent oracle set hash disagrees with frozen manifest")
    for cond, meta in cache["sequential_observable_oracles"].items():
        got = oracle["sequential_observable_oracles"].get(cond, {}).get("set_sha256")
        if got != meta["set_sha256"]:
            raise VerificationError(f"Sequential oracle '{cond}' set hash disagrees with frozen manifest")
    if oracle["latent_oracle"]["table_set_sha256"] != receipt["latent_oracle_set_sha256"]:
        raise VerificationError("Receipt latent oracle hash disagrees with bundle oracle file")
    for cond, expected in receipt["sequential_observable_oracle_set_shas"].items():
        got = oracle["sequential_observable_oracles"].get(cond, {}).get("set_sha256")
        if got != expected:
            raise VerificationError(f"Receipt sequential oracle '{cond}' hash disagrees with bundle")

    # 4. Benchmark closure matches.
    if closure["benchmark_closure_sha256"] != cache["benchmark_closure_sha256"]:
        raise VerificationError("Bundle benchmark closure disagrees with frozen manifest")
    if closure["benchmark_closure_sha256"] != receipt["benchmark_closure_sha256"]:
        raise VerificationError("Receipt benchmark closure disagrees with bundle")

    # 5. Qualification runtime + test corpus bind the baseline.
    if identity["qualification_runtime_sha256"] != baseline["qualification_runtime"]["sha256"]:
        raise VerificationError("Identity qualification runtime hash disagrees with baseline")
    if receipt["qualification_runtime_hash"] != baseline["qualification_runtime"]["sha256"]:
        raise VerificationError("Receipt qualification runtime hash disagrees with baseline")

    # 6. Regeneration actually passed.
    if tests["returncode"] != 0:
        raise VerificationError(f"Regeneration recorded non-zero exit: {tests['returncode']}")
    if receipt["final_status"] != "QUALIFIED_FROZEN_BENCHMARK":
        raise VerificationError(f"Receipt final_status is {receipt['final_status']!r}")

    # 7. Deterministic bundle hash: recompute over all present files and compare.
    files = []
    for p in sorted(BUNDLE_DIR.iterdir()):
        if p.name == "BUNDLE.sha256" or not p.is_file():
            continue
        files.append({"path": p.name, "sha256": _sha256_file(p)})
    concat = "".join(f"{e['path']}\x00{e['sha256']}\n" for e in files)
    recomputed = _sha256_bytes(concat.encode())
    if recomputed != bundle_manifest["bundle_sha256"]:
        raise VerificationError("BUNDLE.sha256 does not match recomputed bundle hash")
    manifest_files = {e["path"]: e["sha256"] for e in bundle_manifest["files"]}
    for e in files:
        if manifest_files.get(e["path"]) != e["sha256"]:
            raise VerificationError(f"BUNDLE.sha256 manifest entry mismatch: {e['path']}")

    print(f"I3.3.2 qualification bundle VERIFIED.")
    print(f"  source: {commit[:12]} / {tree[:12]}")
    print(f"  final_status: {receipt['final_status']}")
    print(f"  bundle_sha256: {bundle_manifest['bundle_sha256']}")
    print(f"  files: {len(files)}")


if __name__ == "__main__":
    verify()
