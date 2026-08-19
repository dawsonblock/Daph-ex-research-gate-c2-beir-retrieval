"""I3.7 — Evidence-Bearing Benchmark.

Extends the aggregate-state benchmark with proposition-level evidence
items that can be retrieved, searched, and verified by the controller.

This package provides:
  - EvidenceItem: proposition-level evidence with hypothesis relationships
  - EvidenceTask: benchmark task with hidden evidence items
  - EvidenceRuntime: runtime tracking available/retrieved/verified evidence
  - EvidenceExecutor: targeted RETRIEVE/SEARCH/VERIFY semantics
  - EvidenceSnapshot: cognitive state with proposition-level evidence

The existing aggregate-state benchmark (C0) is preserved as a control.
This new benchmark (C1) provides the semantic substrate the resolution
governor needs to actually test hypothesis discrimination.
"""
from __future__ import annotations

from .schema import (
    EvidenceItem, EvidenceTask, EvidenceRuntime, EvidenceActionExecution,
    EvidenceSnapshot, EvidenceHypothesis, EvidenceRelation,
    EVIDENCE_SCHEMA, EVIDENCE_VERSION,
    initial_evidence_runtime,
)
from .executor import EvidenceExecutor, build_evidence_snapshot
from .generator import EvidenceTaskGenerator, generate_evidence_tasks
from .structural_ood_generator import StructuralOODGenerator, generate_structural_ood_tasks
from .serializer import serialize_evidence_snapshot, assert_no_evidence_leakage
from .loader import load_evidence_benchmark, save_evidence_benchmark, EvidenceBenchmark

__all__ = [
    "EvidenceItem", "EvidenceTask", "EvidenceRuntime", "EvidenceActionExecution",
    "EvidenceSnapshot", "EvidenceHypothesis", "EvidenceRelation",
    "EVIDENCE_SCHEMA", "EVIDENCE_VERSION",
    "initial_evidence_runtime",
    "EvidenceExecutor",
    "EvidenceTaskGenerator", "generate_evidence_tasks",
    "StructuralOODGenerator", "generate_structural_ood_tasks",
    "serialize_evidence_snapshot",
    "load_evidence_benchmark", "save_evidence_benchmark",
]
