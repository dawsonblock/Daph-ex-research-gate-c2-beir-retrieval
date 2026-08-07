"""C4 stage contracts: data structures that flow between pipeline stages.

Every arm is a configuration over the same stages. The data structures here
are the contracts between stages, so parity between arms can be checked by
comparing stage outputs rather than re-reading code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# --- Frozen constants -------------------------------------------------------

C4_PRIMARY_PACKET_BUDGET = 6
C4_CANDIDATE_BUDGET = 50
C4_RRF_K = 10


# --- Arm configuration ------------------------------------------------------

@dataclass(frozen=True)
class C4Arm:
    """Declarative arm configuration. Differing in exactly one field = one
    mechanism change, which is what makes parity trivial to test."""

    arm_id: str
    description: str
    query_policy: str          # "original" | "subject_preserving"
    retrieval_policy: str      # "bm25_only" | "bm25_bge_fusion"
    identity_policy: str       # "none" | "i3_explicit_identity"
    selector_policy: str       # "s0" | "s2c_with_s0_fallback" | "oracle" | "oracle_evidence"
    evidence_policy: str       # "bounded_packet"
    packet_budget: int = C4_PRIMARY_PACKET_BUDGET
    classification: str = "PRIMARY"  # "PRIMARY" | "DIAGNOSTIC" | "CEILING"


# --- Stage results ----------------------------------------------------------

@dataclass(frozen=True)
class QueryResult:
    """Output of the query stage."""
    original_question: str
    rendered_query: str
    query_hash: str
    query_policy: str
    query_policy_version: str


@dataclass(frozen=True)
class RetrievalResult:
    """Output of the retrieval stage."""
    bm25_ranked: tuple[tuple[str, float], ...]
    bge_ranked: tuple[tuple[str, float], ...]
    fusion_ranked: tuple[tuple[str, float], ...]
    candidate_ids: tuple[str, ...]
    candidate_budget: int
    retrieval_policy: str
    bm25_backend: str
    bge_model_id: str
    bge_revision: str
    rrf_k: int


@dataclass(frozen=True)
class IdentityResolution:
    """Output of the identity stage.

    Reuses the qualified I3 parser. No reimplementation of identity parsing.
    """
    status: str                # "EXACT" | "RESOLVED" | "AMBIGUOUS" | "UNRESOLVED"
    surface: str | None
    canonical: str | None
    evidence_ids: tuple[str, ...]
    candidate_mappings: tuple[Mapping[str, Any], ...]
    resolution_needed: bool
    resolution_attempted: bool
    resolution_changed_state: bool


@dataclass(frozen=True)
class SelectionResult:
    """Output of the selection stage."""
    selector: str              # "s0" | "s2c" | "oracle"
    selected_ids: tuple[str, ...]
    selector_policy: str
    identity_status: str       # carried forward for fallback logic


@dataclass(frozen=True)
class PacketResult:
    """Output of the packet stage."""
    packet_ids: tuple[str, ...]
    packet_contents: tuple[str, ...]
    packet_token_count: int
    packet_hash: str
    packet_budget: int


@dataclass(frozen=True)
class PreHRMResult:
    """Complete pre-HRM state for one task under one arm.

    This is the CPU-only dry-run artifact. It contains everything needed to
    verify parity and no-leakage before HRM is loaded.
    """
    task_id: str
    arm_id: str
    split: str
    query: QueryResult
    retrieval: RetrievalResult
    identity: IdentityResolution
    selection: SelectionResult
    packet: PacketResult
    information_state_before: Mapping[str, Any]
    information_state_after: Mapping[str, Any]


@dataclass(frozen=True)
class HRMResult:
    """Output of the HRM generation stage."""
    output: str
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int
    model_id: str
    model_revision: str
    latency_seconds: float


@dataclass(frozen=True)
class TaskReceipt:
    """Complete task-level receipt with runtime/evaluator separation."""
    task_id: str
    arm_id: str
    split: str
    # Runtime payload (no oracle keys)
    runtime_payload: Mapping[str, Any]
    # Evaluator annotation (oracle metadata allowed here)
    evaluator_annotation: Mapping[str, Any]
