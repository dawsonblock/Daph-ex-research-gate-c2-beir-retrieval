"""Identity and configuration hashing for selective governor gate.

I3.5.2c: Expanded to include the full scientific treatment identity, not just
the gate code. The experiment identity binds:
  - Gate identity (features, model, intervention_gate, thresholds, predictor)
  - System prompt hash
  - Packet builder hash
  - Governor assessor hash
  - Governor serializer hash
  - Utility hash
  - Benchmark manifest hash
  - Runner hash
  - Source commit
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

GATE_IDENTITY_SCHEMA = "DAPH_V2B_I3_5_2_GATE_IDENTITY_V1"
GATE_IDENTITY_VERSION = 1

EXPERIMENT_IDENTITY_SCHEMA = "DAPH_V2B_I3_5_2_EXPERIMENT_IDENTITY_V1"
EXPERIMENT_IDENTITY_VERSION = 1


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file, returning empty string if file doesn't exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (FileNotFoundError, OSError):
        return ""


def _git_commit_sha(repo_root: Path) -> str:
    """Get the current git commit SHA, or empty string if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def compute_gate_identity(
    *,
    delta_u_threshold: float = 5.0,
    max_harm_probability: float = 0.15,
    min_confidence: float = 0.60,
    predictor_name: str = "RuleBasedInterventionPredictor",
) -> dict[str, str]:
    """Compute deterministic SHA-256 identity for the selective gate."""
    pkg_dir = Path(__file__).resolve().parent

    h_features = _file_sha256(pkg_dir / "features.py")
    h_model = _file_sha256(pkg_dir / "model.py")
    h_gate = _file_sha256(pkg_dir / "intervention_gate.py")

    config_dict = {
        "schema": GATE_IDENTITY_SCHEMA,
        "schema_version": GATE_IDENTITY_VERSION,
        "predictor_name": predictor_name,
        "delta_u_threshold": delta_u_threshold,
        "max_harm_probability": max_harm_probability,
        "min_confidence": min_confidence,
        "features_sha256": h_features,
        "model_sha256": h_model,
        "gate_sha256": h_gate,
    }

    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    h_identity = hashlib.sha256(canonical.encode()).hexdigest()

    return {
        "gate_identity_sha256": h_identity,
        "features_sha256": h_features,
        "model_sha256": h_model,
        "gate_sha256": h_gate,
    }


def compute_experiment_identity(
    *,
    gate_identity: dict[str, str] | None = None,
    repo_root: str | Path | None = None,
    benchmark_manifest_path: str | Path | None = None,
    utility_path: str | Path | None = None,
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the full experiment identity binding all scientific components.

    This goes beyond the gate identity to include:
      - System prompt hash
      - Packet builder hash
      - Governor assessor + serializer hash
      - Utility hash
      - Benchmark manifest hash
      - Runner hash
      - Source commit
    """
    if gate_identity is None:
        gate_identity = compute_gate_identity()

    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]

    # Component file hashes
    exec_dir = repo / "hrm_adaptive_memory" / "executive"
    h_system_prompt = _file_sha256(exec_dir / "i3_5_1" / "model_prompt.py")
    h_packet_builder = _file_sha256(exec_dir / "i3_5_1" / "packet_builder.py")
    h_governor_assessor = _file_sha256(exec_dir / "governor" / "assessor.py")
    h_governor_serializer = _file_sha256(exec_dir / "governor" / "serializer.py")
    h_runner = _file_sha256(exec_dir / "i3_5_2" / "trajectory_runner.py")
    h_modes = _file_sha256(exec_dir / "i3_5_2" / "modes.py")
    h_utility = _file_sha256(Path(utility_path)) if utility_path else ""
    h_benchmark_manifest = _file_sha256(Path(benchmark_manifest_path)) if benchmark_manifest_path else ""
    h_policy = _file_sha256(Path(policy_path)) if policy_path else ""

    # Source commit
    commit_sha = _git_commit_sha(repo)

    identity_payload = {
        "schema": EXPERIMENT_IDENTITY_SCHEMA,
        "schema_version": EXPERIMENT_IDENTITY_VERSION,
        # Gate identity
        "predictor": gate_identity.get("predictor_name", "RuleBasedInterventionPredictor"),
        "delta_q_threshold": 5.0,
        "max_harm_probability": 0.15,
        "min_confidence": 0.60,
        "features_sha256": gate_identity["features_sha256"],
        "model_sha256": gate_identity["model_sha256"],
        "intervention_gate_sha256": gate_identity["gate_sha256"],
        "gate_identity_sha256": gate_identity["gate_identity_sha256"],
        # Component hashes
        "system_prompt_sha256": h_system_prompt,
        "packet_builder_sha256": h_packet_builder,
        "governor_assessor_sha256": h_governor_assessor,
        "governor_serializer_sha256": h_governor_serializer,
        "runner_sha256": h_runner,
        "modes_sha256": h_modes,
        "utility_sha256": h_utility,
        "benchmark_manifest_sha256": h_benchmark_manifest,
        "policy_sha256": h_policy,
        # Source
        "source_commit": commit_sha,
    }

    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    h_experiment = hashlib.sha256(canonical.encode()).hexdigest()
    identity_payload["experiment_identity_sha256"] = h_experiment

    return identity_payload


def save_gate_identity(
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    benchmark_manifest_path: str | Path | None = None,
    utility_path: str | Path | None = None,
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute and save the complete experiment identity to a JSON file."""
    identity = compute_experiment_identity(
        repo_root=repo_root,
        benchmark_manifest_path=benchmark_manifest_path,
        utility_path=utility_path,
        policy_path=policy_path,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity
