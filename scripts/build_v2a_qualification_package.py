#!/usr/bin/env python3
"""Build an authenticated V2A package from an exact pushed Git commit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "background-verification-v2a"
RECEIPTS = {
    "pressure": "evidence/background_verification_v2a/pressure/pressure_full_1m.json",
    "adversarial": "evidence/background_verification_v2a/adversarial/adversarial.json",
    "network_smoke": "evidence/background_verification_v2a/network_smoke/network_smoke.json",
    "full_suite": "evidence/background_verification_v2a/tests/full_suite.json",
}
SCHEMAS = {
    "v2a_protocol": "configs/background_verification_v2a_design.json",
    "external_evidence_runtime": "hrm_adaptive_memory/external_verification/core.py",
    "claim_store": "hrm_adaptive_memory/memory_write/claim_store.py",
    "verification_event": "hrm_adaptive_memory/memory_write/verification.py",
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _remote_head() -> str:
    output = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", BRANCH],
        cwd=ROOT, text=True).strip()
    if not output:
        raise RuntimeError(f"origin/{BRANCH} does not exist")
    return output.split()[0]


def _receipt_hashes() -> dict[str, str]:
    hashes = {}
    for name, relative in RECEIPTS.items():
        path = ROOT / relative
        receipt = json.loads(path.read_text())
        if not receipt.get("run_valid"):
            raise RuntimeError(f"qualification receipt is not valid: {relative}")
        hashes[name] = _sha256(path)
    return hashes


def _certified_memory_identity() -> str:
    config = json.loads((ROOT / "configs/certified_memory_v1.json").read_text())
    return config["MEMORY_V1_CONFIG_HASH"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", default=str(ROOT / "dist"))
    args = parser.parse_args()
    if _git("status", "--porcelain"):
        raise RuntimeError("package build refuses a dirty source tree")
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH!r}, got {branch!r}")
    source_commit = _git("rev-parse", "HEAD")
    remote_head = _remote_head()
    if source_commit != remote_head:
        raise RuntimeError(
            f"local HEAD {source_commit} != origin HEAD {remote_head}")
    source_tree_hash = _git("rev-parse", "HEAD^{tree}")

    dist = Path(args.dist)
    dist.mkdir(parents=True, exist_ok=True)
    short = source_commit[:7]
    prefix = f"Daph-background-verification-v2a-{short}/"
    package = dist / f"Daph-background-verification-v2a-{short}.tar.gz"
    sidecar = package.with_suffix(package.suffix + ".sha256")
    manifest_path = ROOT / "BUILD_MANIFEST.json"
    for target in (package, sidecar, manifest_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    with tempfile.TemporaryDirectory(prefix="daph-v2a-build-") as directory:
        source_tar = Path(directory) / "source.tar"
        with source_tar.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", f"--prefix={prefix}",
                 source_commit],
                cwd=ROOT, stdout=handle, check=True)
        source_archive_sha256 = _sha256(source_tar)
        manifest = {
            "artifact": "BACKGROUND_VERIFICATION_V2A_QUALIFIED_BUILD",
            "run_valid": True,
            "mechanism_status": "QUALIFIED",
            "scientific_verdict": "PASS",
            "source_commit": source_commit,
            "branch": branch,
            "origin_head": remote_head,
            "local_head_equals_origin_head": True,
            "source_tree_hash": source_tree_hash,
            "source_archive_payload_sha256": source_archive_sha256,
            "source_archive_payload_definition": (
                "SHA-256 of deterministic git archive for source_commit before "
                "adding this generated BUILD_MANIFEST.json"),
            "final_archive_sha256_location": str(sidecar.relative_to(ROOT)),
            "build_time": datetime.now(UTC).isoformat(),
            "certified_memory_v1_identity": _certified_memory_identity(),
            "certified_memory_v1_config_sha256": _sha256(
                ROOT / "configs/certified_memory_v1.json"),
            "v2a_protocol_sha256": _sha256(
                ROOT / "configs/background_verification_v2a_design.json"),
            "schema_hashes": {
                name: {"path": relative, "sha256": _sha256(ROOT / relative)}
                for name, relative in SCHEMAS.items()
            },
            "c4_h100_lock": {
                "path": "configs/c4_requirements.lock",
                "sha256": _sha256(ROOT / "configs/c4_requirements.lock"),
                "history_modified_by_v2a": False,
            },
            "qualification_receipt_hashes": {
                name: {"path": RECEIPTS[name], "sha256": digest}
                for name, digest in _receipt_hashes().items()
            },
            "bounded_claim": (
                "DAPH can acquire immutable external evidence, preserve source "
                "lineage, deterministically verify supported exact-field claim "
                "classes, survive retries/replay/tampering, and maintain auditable "
                "current verification state."),
            "explicit_non_claims": [
                "general truth determination",
                "arbitrary scientific literature understanding",
                "verification-aware retrieval improvement",
                "answer-accuracy improvement",
            ],
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        manifest_path.write_bytes(manifest_bytes)

        with package.open("wb") as raw_output:
            with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_output, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as output_tar:
                    with tarfile.open(source_tar, mode="r") as input_tar:
                        for member in input_tar:
                            fileobj = input_tar.extractfile(member) if member.isfile() else None
                            output_tar.addfile(member, fileobj)
                    info = tarfile.TarInfo(prefix + "BUILD_MANIFEST.json")
                    info.size = len(manifest_bytes)
                    info.mode = 0o644
                    info.mtime = 0
                    output_tar.addfile(info, io.BytesIO(manifest_bytes))

    final_sha256 = _sha256(package)
    sidecar.write_text(f"{final_sha256}  {package.name}\n")
    with tarfile.open(package, mode="r:gz") as archive:
        packaged = json.load(
            archive.extractfile(prefix + "BUILD_MANIFEST.json"))
        names = archive.getnames()
    if packaged != manifest or prefix + "README.md" not in names:
        raise RuntimeError("post-build archive verification failed")
    print(json.dumps({
        "run_valid": True,
        "source_commit": source_commit,
        "source_tree_hash": source_tree_hash,
        "source_archive_payload_sha256": source_archive_sha256,
        "final_archive_sha256": final_sha256,
        "manifest": str(manifest_path),
        "package": str(package),
        "sidecar": str(sidecar),
        "archive_entries": len(names),
        "archive_bytes": package.stat().st_size,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
