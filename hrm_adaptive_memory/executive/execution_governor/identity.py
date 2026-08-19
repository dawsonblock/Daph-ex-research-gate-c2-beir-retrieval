"""Identity binding for execution assistance frames and I3.6 experiments.

Binds the exact assistance semantics (schema, templates, planner, serializer)
into a cryptographic identity. This makes the assistance semantics
scientifically immutable — changing any component changes the identity.

The identity includes:
  - schema.py SHA-256
  - planner.py SHA-256
  - serializer.py SHA-256
  - validator.py SHA-256
  - identity.py SHA-256
  - system prompt SHA-256
  - packet builder SHA-256
  - governor assessor SHA-256
  - benchmark manifest SHA-256
  - utility config SHA-256
  - policy config SHA-256
  - source commit

And per-frame:
  - assistance_frame_sha256: hash of the frame contents
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from hrm_adaptive_memory.executive.execution_governor.schema import (
    ExecutionAssistanceFrame,
)


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 of a file."""
    p = Path(path)
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assistance_frame_sha256(frame: ExecutionAssistanceFrame) -> str:
    """Compute a deterministic SHA-256 of an assistance frame."""
    d = frame.as_dict()
    encoded = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_assistance_identity(
    benchmark_manifest_path: str | Path,
    utility_config_path: str | Path,
    policy_config_path: str | Path,
    system_prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the I3.6 assistance identity.

    Binds all source files that determine assistance semantics.
    """
    eg_dir = ROOT / "hrm_adaptive_memory/executive/execution_governor"

    components = {
        "schema_sha256": file_sha256(eg_dir / "schema.py"),
        "planner_sha256": file_sha256(eg_dir / "planner.py"),
        "serializer_sha256": file_sha256(eg_dir / "serializer.py"),
        "validator_sha256": file_sha256(eg_dir / "validator.py"),
        "identity_sha256": file_sha256(eg_dir / "identity.py"),
        "packet_builder_sha256": file_sha256(
            ROOT / "hrm_adaptive_memory/executive/i3_5_1/packet_builder.py"),
        "governor_assessor_sha256": file_sha256(
            ROOT / "hrm_adaptive_memory/executive/governor/assessor.py"),
        "governor_bottlenecks_sha256": file_sha256(
            ROOT / "hrm_adaptive_memory/executive/governor/bottlenecks.py"),
        "governor_state_sha256": file_sha256(
            ROOT / "hrm_adaptive_memory/executive/governor/state.py"),
        "benchmark_manifest_sha256": file_sha256(benchmark_manifest_path),
        "utility_config_sha256": file_sha256(utility_config_path),
        "policy_config_sha256": file_sha256(policy_config_path),
    }

    if system_prompt_path is not None:
        components["system_prompt_sha256"] = file_sha256(system_prompt_path)

    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        source_commit = "UNKNOWN"
    components["source_commit"] = source_commit

    # Combined identity
    binding = json.dumps(components, sort_keys=True)
    combined_sha = hashlib.sha256(binding.encode()).hexdigest()

    return {
        "schema": "DAPH_V2B_I3_6_ASSISTANCE_IDENTITY_V1",
        "components": components,
        "assistance_identity_sha256": combined_sha,
    }
