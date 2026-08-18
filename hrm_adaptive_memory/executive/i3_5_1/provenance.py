"""Provenance DAG for I3.5.1 derived artifacts.

Every derived artifact embeds its input hashes, forming a DAG:

  receipts
     ↓
  trajectory replay
     ↓
  canonical results
     ↓
  scientific scores
     ↓
  statistics
     ↓
  analysis
     ↓
  report

No report code recalculates scientific metrics independently.
It renders canonical analysis.

Schema identity: DAPH_V2B_I3_5_1_PROVENANCE_V1 (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA = "DAPH_V2B_I3_5_1_PROVENANCE_V1"
PROVENANCE_VERSION = 1


def _file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProvenanceNode:
    """One node in the provenance DAG."""
    artifact_type: str  # receipts, results, scores, stats, analysis, report
    artifact_sha256: str
    source_artifacts: dict[str, str]  # {artifact_type: sha256}
    experiment_identity_sha256: str
    schema: str = PROVENANCE_SCHEMA
    schema_version: int = PROVENANCE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "artifact_sha256": self.artifact_sha256,
            "source_artifacts": dict(self.source_artifacts),
            "experiment_identity_sha256": self.experiment_identity_sha256,
        }


def build_results_provenance(
    *,
    results_sha256: str,
    source_receipts_sha256: str,
    receipt_chain_root: str,
    experiment_identity_sha256: str,
) -> ProvenanceNode:
    """Provenance for canonical results."""
    return ProvenanceNode(
        artifact_type="results",
        artifact_sha256=results_sha256,
        source_artifacts={
            "receipts": source_receipts_sha256,
            "receipt_chain_root": receipt_chain_root,
        },
        experiment_identity_sha256=experiment_identity_sha256,
    )


def build_scores_provenance(
    *,
    scores_sha256: str,
    source_results_sha256: str,
    experiment_identity_sha256: str,
) -> ProvenanceNode:
    """Provenance for scientific scores."""
    return ProvenanceNode(
        artifact_type="scores",
        artifact_sha256=scores_sha256,
        source_artifacts={"results": source_results_sha256},
        experiment_identity_sha256=experiment_identity_sha256,
    )


def build_stats_provenance(
    *,
    stats_sha256: str,
    source_results_sha256: str,
    source_scores_sha256: str,
    statistics_implementation_sha256: str,
    experiment_identity_sha256: str,
) -> ProvenanceNode:
    """Provenance for statistics."""
    return ProvenanceNode(
        artifact_type="stats",
        artifact_sha256=stats_sha256,
        source_artifacts={
            "results": source_results_sha256,
            "scores": source_scores_sha256,
            "statistics_implementation": statistics_implementation_sha256,
        },
        experiment_identity_sha256=experiment_identity_sha256,
    )


def build_report_provenance(
    *,
    report_sha256: str,
    source_stats_sha256: str,
    source_analysis_sha256: str,
    source_run_id: str,
    experiment_identity_sha256: str,
) -> ProvenanceNode:
    """Provenance for the final report."""
    return ProvenanceNode(
        artifact_type="report",
        artifact_sha256=report_sha256,
        source_artifacts={
            "stats": source_stats_sha256,
            "analysis": source_analysis_sha256,
            "run_id": source_run_id,
        },
        experiment_identity_sha256=experiment_identity_sha256,
    )


def verify_provenance_chain(nodes: list[ProvenanceNode]) -> bool:
    """Verify that a chain of provenance nodes is internally consistent.

    Each node's source_artifacts must match the artifact_sha256 of the
    preceding node (by artifact_type).
    """
    sha_by_type: dict[str, str] = {}
    for node in nodes:
        for src_type, src_sha in node.source_artifacts.items():
            if src_type in sha_by_type:
                if sha_by_type[src_type] != src_sha:
                    return False
        sha_by_type[node.artifact_type] = node.artifact_sha256
    return True


def assert_same_experiment_identity(*artifacts: dict[str, Any]) -> None:
    """Assert that all artifacts share the same experiment identity SHA."""
    from .experiment_identity import assert_same_experiment_identity as _assert
    _assert(*artifacts)
