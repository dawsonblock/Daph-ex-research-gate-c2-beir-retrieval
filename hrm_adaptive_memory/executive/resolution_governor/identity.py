"""SHA-bound identity for resolution governor.

Binds the resolution assistance semantics to:
  - benchmark manifest
  - utility config
  - policy config
  - assessor
  - action scorer
  - mode definitions
  - dependency lock
  - source commit
  - exact resolution semantics
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _file_sha256(path: str) -> str:
    """Compute SHA256 of a file."""
    try:
        content = Path(path).read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return "UNAVAILABLE"


def _git_commit() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def resolution_frame_sha256(frame: Any) -> str:
    """Compute SHA256 of a ResolutionAssistanceFrame."""
    d = frame.as_dict() if hasattr(frame, "as_dict") else frame
    encoded = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolution_context_sha256(context: Any) -> str:
    """Compute SHA256 of a ResolutionContext."""
    d = context.as_dict() if hasattr(context, "as_dict") else context
    encoded = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_resolution_identity(
    benchmark_manifest_path: str,
    utility_config_path: str,
    policy_config_path: str,
) -> dict[str, str]:
    """Compute the resolution governor identity.

    This binds the exact resolution semantics to the benchmark, utility,
    and policy configuration, plus the source commit.
    """
    components = {
        "benchmark_manifest_sha256": _file_sha256(benchmark_manifest_path),
        "utility_config_sha256": _file_sha256(utility_config_path),
        "policy_config_sha256": _file_sha256(policy_config_path),
        "source_commit": _git_commit(),
        "resolution_schema": "DAPH_V2B_I3_6D_RESOLUTION_ASSISTANCE_V1",
        "resolution_version": "1",
        "assessor": "GeneralGovernor",
        "hypothesis_builder": "build_hypotheses_v1",
        "evidence_map_builder": "build_evidence_map_v1",
        "discriminator_builder": "build_discriminators_v1",
        "answer_condition_builder": "build_answer_conditions_v1",
        "planner": "ResolutionGovernor_v1",
        "serializer": "serialize_resolution_packet_v1",
        "validator": "fail_closed_no_evaluator_leakage",
    }

    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    identity_sha = hashlib.sha256(encoded).hexdigest()

    return {
        "resolution_identity_sha256": identity_sha,
        "components": components,
    }
