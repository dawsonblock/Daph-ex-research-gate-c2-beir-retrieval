"""Canonical experiment identity for I3.5.1.

This is the single root identity. Every receipt, result, score,
statistic, and report must bind to this identity via
experiment_identity_sha256.

No scientific artifact may be loaded together with another artifact
whose experiment identity SHA differs.

Schema identity: DAPH_V2B_I3_5_1_EXPERIMENT_IDENTITY_V1 (frozen).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IDENTITY_SCHEMA = "DAPH_V2B_I3_5_1_EXPERIMENT_IDENTITY_V1"
IDENTITY_VERSION = 1


def _file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_sha256(path: str | Path) -> str:
    """SHA-256 of a Python module's source code."""
    return _file_sha256(path)


def _git_commit() -> str:
    """Current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _git_tree_sha() -> str:
    """Current git tree hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


@dataclass(frozen=True)
class ExperimentIdentity:
    """Single canonical experiment identity binding all frozen components."""

    experiment_id: str
    schema: str
    schema_version: int

    # Source
    source_commit: str
    source_tree_sha256: str

    # Benchmark
    benchmark_identity: str
    split_identity: str
    task_corpus_sha256: str

    # Scientific criteria
    scientific_criteria_sha256: str

    # Generation config
    generation_config_sha256: str

    # Model policy
    model_policy_sha256: str

    # Prompt
    prompt_sha256: str

    # Decoder
    decoder_sha256: str

    # Runner
    runner_sha256: str

    # Condition scheduler
    condition_scheduler_sha256: str

    # Packet serializer
    packet_serializer_sha256: str

    # Governor
    governor_sha256: str
    governor_config_sha256: str
    action_semantics_sha256: str

    # Executor
    executor_sha256: str

    # Scoring
    scoring_sha256: str

    # Statistics
    statistics_sha256: str

    # Oracle
    oracle_manifest_sha256: str

    # Observable oracle views
    observable_oracle_views_sha256: str

    # Runtime environment
    runtime_environment_identity: str
    dependency_lock_sha256: str

    # Artifact schema versions
    artifact_schema_versions: dict[str, str] = field(default_factory=dict)

    # Timestamp (excluded from identity hash)
    created_before_first_model_call: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "benchmark_identity": self.benchmark_identity,
            "split_identity": self.split_identity,
            "task_corpus_sha256": self.task_corpus_sha256,
            "scientific_criteria_sha256": self.scientific_criteria_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "model_policy_sha256": self.model_policy_sha256,
            "prompt_sha256": self.prompt_sha256,
            "decoder_sha256": self.decoder_sha256,
            "runner_sha256": self.runner_sha256,
            "condition_scheduler_sha256": self.condition_scheduler_sha256,
            "packet_serializer_sha256": self.packet_serializer_sha256,
            "governor_sha256": self.governor_sha256,
            "governor_config_sha256": self.governor_config_sha256,
            "action_semantics_sha256": self.action_semantics_sha256,
            "executor_sha256": self.executor_sha256,
            "scoring_sha256": self.scoring_sha256,
            "statistics_sha256": self.statistics_sha256,
            "oracle_manifest_sha256": self.oracle_manifest_sha256,
            "observable_oracle_views_sha256": self.observable_oracle_views_sha256,
            "runtime_environment_identity": self.runtime_environment_identity,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "artifact_schema_versions": dict(self.artifact_schema_versions),
            "created_before_first_model_call": self.created_before_first_model_call,
        }

    def sha256(self) -> str:
        """Canonical SHA-256 of the experiment identity (excluding timestamp)."""
        d = self.as_dict()
        d["created_before_first_model_call"] = ""  # Exclude from hash
        encoded = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_experiment_identity(
    *,
    root: str | Path = ".",
    benchmark_identity: str,
    split_identity: str,
    task_corpus_sha256: str,
    scientific_criteria_sha256: str,
    oracle_manifest_sha256: str,
    observable_oracle_views_sha256: str,
) -> ExperimentIdentity:
    """Build the canonical experiment identity from all frozen components.

    This function reads module source hashes and config files to produce
    a single identity object. It must be called before the first model call.
    """
    root = Path(root)
    exec_dir = root / "hrm_adaptive_memory" / "executive"
    i351_dir = exec_dir / "i3_5_1"
    gov_dir = exec_dir / "governor"

    # Compute all module hashes
    module_hashes = {
        "runner": _module_sha256(i351_dir / "trajectory_runner.py"),
        "condition_scheduler": _module_sha256(i351_dir / "factorial_scheduler.py"),
        "packet_serializer": _module_sha256(i351_dir / "packet_builder.py"),
        "executor": _module_sha256(exec_dir / "executor.py"),
        "scoring": _module_sha256(i351_dir / "scoring.py"),
        "statistics": _module_sha256(i351_dir / "statistics.py"),
        "decoder": _module_sha256(exec_dir / "model_decoder.py"),
        "prompt": _module_sha256(i351_dir / "model_prompt.py"),
    }

    # Governor hashes
    governor_identity = _compute_governor_identity(gov_dir)
    governor_scoring_config_sha = _file_sha256(
        root / "experiments/v2b_i3_5_1/configs/governor_scoring_v1.json"
    )

    # Generation config
    gen_config_sha = _file_sha256(
        root / "experiments/v2b_i3_5_1/configs/generation_config_v1.json"
    )

    # Model policy
    model_policy_sha = _file_sha256(
        root / "experiments/v2b_i3_5_1/configs/model_policy_v1.json"
    )

    # Dependency lock
    dep_lock_sha = _file_sha256(root / "poetry.lock")
    if not dep_lock_sha:
        dep_lock_sha = _file_sha256(root / "requirements.txt")
    if not dep_lock_sha:
        dep_lock_sha = "NO_LOCK_FILE"

    identity = ExperimentIdentity(
        experiment_id="v2b_i3_5_1_experiment_v1",
        schema=IDENTITY_SCHEMA,
        schema_version=IDENTITY_VERSION,
        source_commit=_git_commit(),
        source_tree_sha256=_git_tree_sha(),
        benchmark_identity=benchmark_identity,
        split_identity=split_identity,
        task_corpus_sha256=task_corpus_sha256,
        scientific_criteria_sha256=scientific_criteria_sha256,
        generation_config_sha256=gen_config_sha,
        model_policy_sha256=model_policy_sha,
        prompt_sha256=module_hashes["prompt"],
        decoder_sha256=module_hashes["decoder"],
        runner_sha256=module_hashes["runner"],
        condition_scheduler_sha256=module_hashes["condition_scheduler"],
        packet_serializer_sha256=module_hashes["packet_serializer"],
        governor_sha256=governor_identity["governor_sha256"],
        governor_config_sha256=governor_scoring_config_sha,
        action_semantics_sha256=governor_identity["action_semantics_sha256"],
        executor_sha256=module_hashes["executor"],
        scoring_sha256=module_hashes["scoring"],
        statistics_sha256=module_hashes["statistics"],
        oracle_manifest_sha256=oracle_manifest_sha256,
        observable_oracle_views_sha256=observable_oracle_views_sha256,
        runtime_environment_identity=f"{platform.platform()}/{sys.version}",
        dependency_lock_sha256=dep_lock_sha,
        artifact_schema_versions={
            "conditions": "DAPH_V2B_I3_5_1_CONDITIONS_V1",
            "receipts": "DAPH_V2B_I3_5_1_RECEIPT_V1",
            "results": "DAPH_V2B_I3_5_1_RESULTS_V1",
            "scores": "DAPH_V2B_I3_5_1_SCORES_V1",
            "stats": "DAPH_V2B_I3_5_1_STATS_V1",
            "report": "DAPH_V2B_I3_5_1_REPORT_V1",
            "base_packet": "DAPH_V2B_I3_5_1_BASE_PACKET_V1",
            "governor_packet": "DAPH_V2B_I3_5_1_GOVERNOR_PACKET_V1",
        },
        created_before_first_model_call=datetime.now(timezone.utc).isoformat(),
    )
    return identity


def _compute_governor_identity(gov_dir: Path) -> dict[str, str]:
    """Compute governor identity from source files."""
    from hrm_adaptive_memory.executive.governor.identity import compute_governor_identity
    return compute_governor_identity()


def save_experiment_identity(
    identity: ExperimentIdentity,
    path: str | Path,
) -> str:
    """Save the experiment identity to a JSON file and return its file SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.as_dict()
    payload["experiment_identity_sha256"] = identity.sha256()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _file_sha256(path)


def assert_same_experiment_identity(*artifacts: dict[str, Any]) -> None:
    """Assert that all artifacts share the same experiment identity SHA.

    Every scientific artifact (receipts, results, scores, stats, report)
    must contain experiment_identity_sha256. If any two differ, raise.
    """
    shas: set[str] = set()
    for i, a in enumerate(artifacts):
        sha = a.get("experiment_identity_sha256")
        if sha is None:
            raise ValueError(
                f"Artifact {i} missing experiment_identity_sha256")
        shas.add(sha)
    if len(shas) > 1:
        raise ValueError(
            f"Experiment identity mismatch: artifacts have different SHAs: {shas}")
