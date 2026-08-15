"""Full I3.4 experiment identity.

Binds together all frozen components that define the experiment:

- qualified I3.3.2 benchmark
- Scientific Criteria V2
- evaluation subset
- observable oracle views
- controller
- provider/model policy
- generation config
- retry policy
- paired scheduler
- statistical implementation
- runtime environment

This is the single object that identifies the experiment.  It must exist
before development results are treated as experiment evidence.

Schema identity: ``DAPH_V2B_I3_4_EXPERIMENT_IDENTITY_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IDENTITY_SCHEMA = "DAPH_V2B_I3_4_EXPERIMENT_IDENTITY_V1"
IDENTITY_VERSION = 1


@dataclass(frozen=True)
class ExperimentIdentity:
    """Full experiment identity binding all frozen components."""

    experiment_id: str
    schema: str
    schema_version: int

    # Benchmark
    benchmark_id: str
    benchmark_closure_sha256: str
    benchmark_status: str

    # Scientific criteria
    scientific_criteria_version: str  # "V2"
    scientific_criteria_sha256: str

    # Evaluation subset
    evaluation_splits: tuple[str, ...]
    evaluation_task_count: int

    # Observable oracle views
    observable_oracle_views_sha256: str
    observable_oracle_view_count: int

    # Controller
    controller_id: str
    controller_identity_sha256: str

    # Provider/model policy
    model_identity_policy_sha256: str
    frozen_model: str
    frozen_provider: str

    # Generation config
    generation_config_sha256: str

    # Retry policy
    retry_policy_id: str
    retry_policy_sha256: str

    # Paired scheduler
    scheduler_id: str
    scheduler_sha256: str

    # Statistical implementation
    statistical_module_sha256: str

    # Runtime environment
    python_version: str
    platform: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "benchmark": {
                "benchmark_id": self.benchmark_id,
                "benchmark_closure_sha256": self.benchmark_closure_sha256,
                "benchmark_status": self.benchmark_status,
            },
            "scientific_criteria": {
                "version": self.scientific_criteria_version,
                "sha256": self.scientific_criteria_sha256,
            },
            "evaluation_subset": {
                "splits": list(self.evaluation_splits),
                "task_count": self.evaluation_task_count,
            },
            "observable_oracle_views": {
                "sha256": self.observable_oracle_views_sha256,
                "view_count": self.observable_oracle_view_count,
            },
            "controller": {
                "controller_id": self.controller_id,
                "identity_sha256": self.controller_identity_sha256,
            },
            "provider_model_policy": {
                "identity_policy_sha256": self.model_identity_policy_sha256,
                "frozen_model": self.frozen_model,
                "frozen_provider": self.frozen_provider,
            },
            "generation_config": {
                "sha256": self.generation_config_sha256,
            },
            "retry_policy": {
                "id": self.retry_policy_id,
                "sha256": self.retry_policy_sha256,
            },
            "paired_scheduler": {
                "scheduler_id": self.scheduler_id,
                "sha256": self.scheduler_sha256,
            },
            "statistical_implementation": {
                "module_sha256": self.statistical_module_sha256,
            },
            "runtime_environment": {
                "python_version": self.python_version,
                "platform": self.platform,
                "created_at": self.created_at,
            },
        }

    def sha256(self) -> str:
        """Canonical SHA-256 of the experiment identity (excluding timestamp)."""
        # Hash everything except created_at (which is set at creation time).
        d = self.as_dict()
        d["runtime_environment"]["created_at"] = ""  # Exclude from hash
        encoded = json.dumps(d, sort_keys=True,
                             separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _module_sha256(path: str) -> str:
    """Compute SHA-256 of a Python module's source code."""
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_experiment_identity(
    *,
    observable_oracle_views_sha256: str,
    observable_oracle_view_count: int,
    root: str | Path = ".",
) -> ExperimentIdentity:
    """Build the full experiment identity from all frozen components.

    This function reads the frozen config files and module source hashes
    to produce a single identity object.
    """
    root = Path(root)

    # Import frozen components
    from .i3_4_generation_config import FROZEN_CONFIG
    from .i3_4_model_identity_policy import FROZEN_IDENTITY_POLICY
    from .i3_4_retry_policy import FROZEN_RETRY_POLICY
    from .pinned_model_controller import CONTROLLER_ID

    # Load benchmark baseline
    baseline_path = root / "configs/v2b_i3_3_3_baseline.json"
    baseline = json.loads(baseline_path.read_text())

    # Load scientific criteria V2
    criteria_path = root / "experiments/v2b_i3_4/configs/v2b_i3_4_scientific_criteria_v2.json"
    criteria = json.loads(criteria_path.read_text())

    # Load controller identity
    controller_identity_path = root / "experiments/v2b_i3_4/manifests/v2b_i3_4_controller_identity_v1.json"
    controller_identity = json.loads(controller_identity_path.read_text())

    # Compute module hashes
    exec_dir = root / "hrm_adaptive_memory/executive"
    statistical_sha = _module_sha256(str(exec_dir / "i3_4_statistical_analysis.py"))
    scheduler_sha = _module_sha256(str(exec_dir / "i3_4_pair_scheduler.py"))
    retry_sha = _module_sha256(str(exec_dir / "i3_4_retry_policy.py"))

    # Evaluation splits
    eval_splits = ("development", "validation", "held_out_instance",
                   "held_out_surface", "held_out_structure")
    eval_task_count = 300 + 150 + 100 + 50 + 150  # 750

    identity = ExperimentIdentity(
        experiment_id="v2b_i3_4_experiment_v1",
        schema=IDENTITY_SCHEMA,
        schema_version=IDENTITY_VERSION,
        benchmark_id=baseline["benchmark_id"],
        benchmark_closure_sha256=baseline["benchmark_closure_sha256"],
        benchmark_status=baseline.get("status", "QUALIFIED_FROZEN_BENCHMARK"),
        scientific_criteria_version="V2",
        scientific_criteria_sha256=criteria.get("criteria_sha256", ""),
        evaluation_splits=eval_splits,
        evaluation_task_count=eval_task_count,
        observable_oracle_views_sha256=observable_oracle_views_sha256,
        observable_oracle_view_count=observable_oracle_view_count,
        controller_id=CONTROLLER_ID,
        controller_identity_sha256=controller_identity.get("controller_identity_sha256", ""),
        model_identity_policy_sha256=FROZEN_IDENTITY_POLICY.sha256(),
        frozen_model=FROZEN_IDENTITY_POLICY.frozen_model,
        frozen_provider=FROZEN_IDENTITY_POLICY.frozen_provider,
        generation_config_sha256=FROZEN_CONFIG.sha256(),
        retry_policy_id=FROZEN_RETRY_POLICY.policy_id,
        retry_policy_sha256=retry_sha,
        scheduler_id="v2b_i3_4_pair_scheduler_v1",
        scheduler_sha256=scheduler_sha,
        statistical_module_sha256=statistical_sha,
        python_version=sys.version,
        platform=platform.platform(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return identity


def save_experiment_identity(
    identity: ExperimentIdentity,
    path: str | Path,
) -> str:
    """Save the experiment identity to a JSON file and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.as_dict()
    payload["identity_sha256"] = identity.sha256()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
