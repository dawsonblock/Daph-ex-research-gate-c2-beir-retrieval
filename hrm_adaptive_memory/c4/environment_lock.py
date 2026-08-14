"""C4 environment lock — reproducibility of the runtime, not just the code.

``pip install transformers torch sentence-transformers`` under a heading that
says "INSTALL EXACT DEPENDENCIES" installs whatever the index resolves to that
day. A result produced that way is not reproducible, and the protocol's
``dependency version mismatch`` abort condition has nothing to compare against.

This module defines the comparison target. The lock records the interpreter,
the result-affecting packages, and the accelerator of the environment that
produced a certified run.

Fail-closed rules:

* A package pinned in the lock but absent from the environment is a violation.
* A version mismatch is a violation.
* A lock entry recorded as ``null`` is a violation (MUST_RECORD) — an
  unrecorded version is never "any version is fine".
* Accelerator fields are compared when the lock pins them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

LOCK_SCHEMA_VERSION = "c4-environment-lock-v1"

# Packages whose version can change C4 results. BM25 is implemented in-repo
# (hrm_adaptive_memory/backends/local.py), so rank-bm25, faiss and
# sentence-transformers are deliberately NOT pinned: the C4 pipeline never
# imports them, and pinning unused packages creates false precision.
RESULT_AFFECTING_PACKAGES = (
    "accelerate",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "numpy",
)


class EnvironmentLockViolation(AssertionError):
    """Raised when the live environment does not match the lock."""


def _package_version(name: str) -> str | None:
    """Installed version of a distribution, or None if absent."""
    from importlib import metadata
    # Distribution names use hyphens; import names use underscores.
    for candidate in (name, name.replace("_", "-")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def capture_environment() -> dict:
    """Snapshot the current environment in lock format."""
    packages = {name: _package_version(name) for name in RESULT_AFFECTING_PACKAGES}

    accelerator: dict[str, Any] = {"cuda": None, "gpu_name": None, "available": False}
    try:
        import torch
        if torch.cuda.is_available():
            accelerator = {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "available": True,
            }
    except Exception:
        pass

    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "packages": packages,
        "accelerator": accelerator,
    }


def load_lock(path: str | Path) -> dict:
    """Load a lock file. A missing or malformed lock is a violation."""
    p = Path(path)
    if not p.is_file():
        raise EnvironmentLockViolation(
            f"Environment lock not found: {p}. Generate it on the machine that "
            f"produces the run: python scripts/c4_freeze_environment.py")
    try:
        lock = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise EnvironmentLockViolation(f"Lock {p} is not valid JSON: {exc}") from exc
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise EnvironmentLockViolation(
            f"Lock schema is {lock.get('schema_version')!r}, expected "
            f"{LOCK_SCHEMA_VERSION!r}")
    return lock


def verify_environment(lock: Mapping[str, Any],
                       observed: Mapping[str, Any] | None = None,
                       ) -> tuple[bool, list[str], dict]:
    """Compare the live environment against the lock.

    Returns ``(ok, violations, observed)``. Never raises on mismatch — the
    caller decides, and certification treats any violation as fatal.
    """
    observed = dict(observed) if observed is not None else capture_environment()
    violations: list[str] = []

    locked_python = lock.get("python")
    if locked_python is None:
        violations.append("MUST_RECORD: lock.python is null")
    elif observed["python"] != locked_python:
        violations.append(
            f"python: lock={locked_python} observed={observed['python']}")

    locked_packages = lock.get("packages") or {}
    if not locked_packages:
        violations.append("MUST_RECORD: lock.packages is empty")
    for name, locked_version in sorted(locked_packages.items()):
        actual = observed["packages"].get(name)
        if locked_version is None:
            violations.append(f"MUST_RECORD: lock.packages.{name} is null")
        elif actual is None:
            violations.append(f"{name}: lock={locked_version} observed=NOT_INSTALLED")
        elif actual != locked_version:
            violations.append(f"{name}: lock={locked_version} observed={actual}")

    locked_accel = lock.get("accelerator") or {}
    obs_accel = observed.get("accelerator") or {}
    for field in ("cuda", "gpu_name"):
        locked_value = locked_accel.get(field)
        if locked_value is None:
            # An unpinned accelerator field is allowed: a CPU-only replay of the
            # pre-HRM path is legitimately device-independent. It is recorded,
            # not asserted.
            continue
        if obs_accel.get(field) != locked_value:
            violations.append(
                f"accelerator.{field}: lock={locked_value} "
                f"observed={obs_accel.get(field)}")

    return (not violations), violations, observed


def pip_requirements(lock: Mapping[str, Any]) -> str:
    """Render the lock's packages as a pip-installable requirements body."""
    lines = [
        "# Generated from the C4 environment lock. Do not edit by hand.",
        f"# python=={lock.get('python')}",
    ]
    for name, version in sorted((lock.get("packages") or {}).items()):
        if version is None:
            lines.append(f"# {name}==UNRECORDED")
        else:
            lines.append(f"{name.replace('_', '-')}=={version}")
    return "\n".join(lines) + "\n"
