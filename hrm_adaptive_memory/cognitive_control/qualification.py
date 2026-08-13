"""Fail-closed identity contract for future V2B experimental receipts.

The contract exists before V2B runs, but refuses to mint a qualification
identity until a frozen non-empty benchmark and fully pinned controller/model
configuration are supplied.  It is therefore not a source of V2B claims.
"""
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


IDENTITY_VERSION = "DAPH_COGNITIVE_CONTROL_V2B_QUALIFICATION_IDENTITY_V1"
COMPONENT_PATHS = {
    "protocol": "configs/cognitive_control_v2b_design.json",
    "policy_schema": "configs/cognitive_control_v2b_policy_schema.json",
    "authority_registry": "configs/authority_registry_v2b.json",
    "cognitive_core": "hrm_adaptive_memory/cognitive_control/core.py",
    "cognitive_datalog": "hrm_adaptive_memory/cognitive_control/datalog.py",
    "cognitive_schemas": "hrm_adaptive_memory/cognitive_control/schemas.py",
    "cognitive_actions": "hrm_adaptive_memory/cognitive_control/actions.py",
    "cognitive_checkpoints": "hrm_adaptive_memory/cognitive_control/checkpoints.py",
    "checkpoint_trusted_signers": "configs/cognitive_checkpoint_trusted_signers_v2b.json",
    "qualification_runtime": "hrm_adaptive_memory/cognitive_control/qualification.py",
    "authority_runtime": "hrm_adaptive_memory/external_verification/authority_registry.py",
    "authority_extractors": "hrm_adaptive_memory/external_verification/authority_extractors.py",
    "typed_comparator_registry": "hrm_adaptive_memory/external_verification/comparators",
    "typed_verifier": "hrm_adaptive_memory/external_verification/typed_verifier.py",
    "peer_bound_network": "hrm_adaptive_memory/external_verification/network",
}
TEST_CORPUS_PATHS = (
    "tests/unit/test_cognitive_control.py",
    "tests/unit/test_v2b_authority_and_comparators.py",
    "tests/unit/test_v2b_peer_bound_network.py",
    "tests/unit/test_v2b_cognitive_checkpoints.py",
    "tests/unit/test_v2b_schemas_and_actions.py",
    "tests/adversarial/test_v2b_infrastructure_adversarial.py",
)
QUALIFICATION_LOCK = "configs/v2b_qualification_requirements.lock"
CRITICAL_DISTRIBUTIONS = (
    "cryptography", "numpy", "torch", "pytest", "transformers", "huggingface-hub",
    "accelerate", "psutil", "litlogger", "lightning-sdk", "pip", "setuptools", "wheel",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)


def _tree_hash(root: Path, commit: str, path: str) -> str:
    entry = _git(root, "ls-tree", commit, "--", path)
    if not entry:
        raise RuntimeError(f"V2B qualification input is absent at {commit}: {path}")
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


def _distribution_identity(name: str) -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": "NOT_INSTALLED", "metadata_hashes": {}}
    hashes = {}
    for metadata_name in ("METADATA", "WHEEL", "RECORD", "direct_url.json"):
        value = distribution.read_text(metadata_name)
        if value is not None:
            hashes[metadata_name] = _sha256(value.encode("utf-8"))
    return {"version": distribution.version, "metadata_hashes": hashes}


def dependency_environment() -> dict[str, object]:
    interpreter = Path(sys.executable).resolve()
    payload: dict[str, object] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "python_cache_tag": sys.implementation.cache_tag,
        "interpreter_sha256": _file_sha256(interpreter),
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine(),
               "platform_tag": sysconfig.get_platform(), "libc": list(platform.libc_ver())},
        "distributions": {name: _distribution_identity(name) for name in CRITICAL_DISTRIBUTIONS},
    }
    payload["sha256"] = _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return payload


def validate_run_configuration(configuration: Mapping[str, Any]) -> None:
    """Reject placeholder or under-specified runs before a receipt can exist."""
    if configuration.get("schema") != "DAPH_V2B_EXPERIMENT_CONFIGURATION_V1":
        raise RuntimeError("V2B run configuration has an unsupported schema")
    if configuration.get("status") != "FROZEN_FOR_QUALIFICATION":
        raise RuntimeError("V2B run configuration is not frozen for qualification")
    controller = configuration.get("controller", {})
    model = configuration.get("model", {})
    benchmark = configuration.get("benchmark", {})
    required = (
        controller.get("id"), controller.get("revision"), model.get("id"), model.get("revision"),
        model.get("tokenizer_revision"), model.get("generation_configuration"), benchmark.get("path"),
    )
    if any(value in (None, "", "NONE", {}) for value in required):
        raise RuntimeError("V2B run configuration lacks a pinned controller/model/benchmark identity")
    if benchmark.get("status") != "FROZEN_NONEMPTY":
        raise RuntimeError("V2B benchmark is not frozen and non-empty")


def qualification_identity(root: str | Path, run_configuration: Mapping[str, Any],
                           commit: str = "HEAD") -> dict[str, object]:
    """Return the complete identity required on a future V2B receipt."""
    validate_run_configuration(run_configuration)
    root = Path(root).resolve()
    source_commit = _git(root, "rev-parse", commit)
    benchmark_path = str(run_configuration["benchmark"]["path"])
    identity: dict[str, object] = {
        "identity_version": IDENTITY_VERSION,
        "source_commit": source_commit,
        "source_tree_hash": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "component_hashes": {
            name: {"path": path, "sha256": _tree_hash(root, source_commit, path)}
            for name, path in COMPONENT_PATHS.items()
        },
        "test_corpus_sha256": _combined_hash(root, source_commit, TEST_CORPUS_PATHS),
        "benchmark": {"path": benchmark_path,
                      "sha256": _tree_hash(root, source_commit, benchmark_path)},
        "experiment_configuration": dict(run_configuration),
        "qualification_lock": {"path": QUALIFICATION_LOCK,
                               "sha256": _tree_hash(root, source_commit, QUALIFICATION_LOCK)},
        "dependency_environment": dependency_environment(),
    }
    return identity


def validate_receipt_set(root: str | Path, receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    """Fail closed unless all V2B receipts carry one exact full identity."""
    if not receipts:
        raise RuntimeError("no V2B qualification receipts were supplied")
    baseline: dict[str, object] | None = None
    for name, receipt in receipts.items():
        if receipt.get("run_valid") is not True:
            raise RuntimeError(f"{name} receipt does not have run_valid=true")
        identity = receipt.get("qualification_identity")
        if not isinstance(identity, dict) or identity.get("identity_version") != IDENTITY_VERSION:
            raise RuntimeError(f"{name} receipt lacks a supported V2B qualification identity")
        if (receipt.get("source_commit") != identity.get("source_commit")
                or receipt.get("source_tree_hash") != identity.get("source_tree_hash")):
            raise RuntimeError(f"{name} receipt conflicts with its V2B identity")
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise RuntimeError(f"{name} receipt does not match the common V2B identity")
    assert baseline is not None
    committed = qualification_identity(root, baseline["experiment_configuration"],
                                       str(baseline["source_commit"]))
    if committed != baseline:
        raise RuntimeError("V2B receipt identity does not match committed inputs or environment")
    return baseline
