"""Fail-closed identity binding for BACKGROUND_VERIFICATION_V2A receipts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Mapping


IDENTITY_VERSION = "BACKGROUND_VERIFICATION_V2A_QUALIFICATION_IDENTITY_V1"
COMPONENT_PATHS = {
    "protocol": "configs/background_verification_v2a_design.json",
    "external_evidence_runtime": "hrm_adaptive_memory/external_verification/core.py",
    "qualification_identity_runtime": (
        "hrm_adaptive_memory/external_verification/qualification.py"),
    "claim_store": "hrm_adaptive_memory/memory_write/claim_store.py",
    "verification_event": "hrm_adaptive_memory/memory_write/verification.py",
    "certified_memory_v1": "configs/certified_memory_v1.json",
    "c4_h100_lock": "configs/c4_requirements.lock",
}
TEST_CORPUS_PATHS = (
    "tests/unit/test_background_verification_v2a.py",
    "tests/unit/test_v2a_qualification.py",
    "tests/fixtures/external_verification_v2a",
    "scripts/pressure_test_external_verification_v2a.py",
    "scripts/record_v2a_adversarial_regressions.py",
    "scripts/run_v2a_network_smoke.py",
    "scripts/record_v2a_test_receipt.py",
)
QUALIFICATION_LOCK = "configs/v2a_qualification_requirements.lock"
QUALIFICATION_DISTRIBUTIONS = (
    "numpy", "torch", "pytest", "transformers", "huggingface-hub",
    "accelerate", "psutil", "litlogger", "pip", "setuptools", "wheel",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True).strip()


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=root)


def _tree_hash(root: Path, commit: str, path: str) -> str:
    """Hash a file or directory's committed bytes with path framing."""
    entry = _git(root, "ls-tree", commit, "--", path)
    if not entry:
        raise RuntimeError(f"qualification input is absent at {commit}: {path}")
    if entry.split()[1] == "blob":
        return _sha256(_git_bytes(root, commit, path))
    names = _git(root, "ls-tree", "-r", "--name-only", commit, "--", path).splitlines()
    hasher = hashlib.sha256()
    for name in sorted(names):
        raw = _git_bytes(root, commit, name)
        hasher.update(name.encode("utf-8") + b"\0")
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


def _combined_hash(root: Path, commit: str, paths: tuple[str, ...]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        digest = _tree_hash(root, commit, path)
        hasher.update(path.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _distribution_identity(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": "NOT_INSTALLED", "metadata_hashes": {}}
    hashes = {}
    for metadata_name in ("METADATA", "WHEEL", "RECORD", "direct_url.json"):
        value = distribution.read_text(metadata_name)
        if value is not None:
            hashes[metadata_name] = _sha256(value.encode())
    return {"version": distribution.version, "metadata_hashes": hashes}


def dependency_environment() -> dict[str, Any]:
    interpreter = Path(sys.executable).resolve()
    payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "python_cache_tag": sys.implementation.cache_tag,
        "interpreter_sha256": _file_sha256(interpreter),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform_tag": sysconfig.get_platform(),
            "libc": list(platform.libc_ver()),
        },
        "distributions": {
            name: _distribution_identity(name)
            for name in QUALIFICATION_DISTRIBUTIONS
        },
    }
    payload["sha256"] = _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return payload


def qualification_identity(root: str | Path, commit: str = "HEAD") -> dict[str, Any]:
    """Return the complete identity a V2A qualification receipt must carry."""
    root = Path(root).resolve()
    source_commit = _git(root, "rev-parse", commit)
    identity = {
        "identity_version": IDENTITY_VERSION,
        "source_commit": source_commit,
        "source_tree_hash": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "component_hashes": {
            name: {"path": path, "sha256": _tree_hash(root, source_commit, path)}
            for name, path in COMPONENT_PATHS.items()
        },
        "test_corpus_sha256": _combined_hash(root, source_commit, TEST_CORPUS_PATHS),
        "qualification_lock": {
            "path": QUALIFICATION_LOCK,
            "sha256": _tree_hash(root, source_commit, QUALIFICATION_LOCK),
        },
        "dependency_environment": dependency_environment(),
    }
    return identity


def validate_receipt_set(
    root: str | Path,
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate that every receipt qualifies one byte-identical source identity."""
    if not receipts:
        raise RuntimeError("no V2A qualification receipts were supplied")
    baseline: dict[str, Any] | None = None
    for name, receipt in receipts.items():
        if receipt.get("run_valid") is not True:
            raise RuntimeError(f"{name} receipt does not have run_valid=true")
        identity = receipt.get("qualification_identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"{name} receipt lacks qualification_identity")
        if identity.get("identity_version") != IDENTITY_VERSION:
            raise RuntimeError(f"{name} receipt has an unsupported identity version")
        for key in ("source_commit", "source_tree_hash"):
            if receipt.get(key) != identity.get(key):
                raise RuntimeError(f"{name} receipt {key} disagrees with its identity")
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise RuntimeError(
                f"{name} receipt does not match the common qualification identity")
    assert baseline is not None
    committed = qualification_identity(root, str(baseline["source_commit"]))
    if committed != baseline:
        raise RuntimeError(
            "receipt identity does not match the committed qualification inputs "
            "or the current dependency environment")
    return baseline
