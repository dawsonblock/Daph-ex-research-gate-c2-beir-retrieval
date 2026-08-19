"""Resolution governor: structured deliberation state for I3.6d.

The resolution governor moves away from generic action recommendations
toward explicit hypothesis management, evidence relationships, and
decision-conversion structure.

Architecture:

    Controller-visible state
              |
              v
     Resolution Governor
              |
    +---------+---------+
    |         |         |
    v         v         v
  Hypotheses  Evidence  Discriminators
    |         Map       |
    +---------+---------+
              |
              v
     Resolution Context
              |
              v
        Base Model
              |
    +---------+---------+
    |         |         |
    v         v         v
  retrieve   verify   reason
    |         |         |
    +---------+---------+
              |
              v
      Context Update
              |
      answer condition?
       /          \
     yes           no
     |              |
   ANSWER    next discriminator
"""
from __future__ import annotations

from .schema import (
    RESOLUTION_SCHEMA,
    RESOLUTION_VERSION,
    Hypothesis,
    EvidenceAssessment,
    Discriminator,
    ResolutionStep,
    AnswerCondition,
    SearchSpecification,
    ResolutionAssistanceFrame,
    ResolutionContext,
    HypothesisUpdate,
    ResolutionReceipt,
)
from .planner import ResolutionGovernor
from .serializer import (
    serialize_resolution_packet,
    assert_no_evaluator_leakage,
)
from .identity import (
    compute_resolution_identity,
    resolution_frame_sha256,
    resolution_context_sha256,
)

__all__ = [
    "RESOLUTION_SCHEMA",
    "RESOLUTION_VERSION",
    "Hypothesis",
    "EvidenceAssessment",
    "Discriminator",
    "ResolutionStep",
    "AnswerCondition",
    "SearchSpecification",
    "ResolutionAssistanceFrame",
    "ResolutionContext",
    "HypothesisUpdate",
    "ResolutionReceipt",
    "ResolutionGovernor",
    "serialize_resolution_packet",
    "assert_no_evaluator_leakage",
    "compute_resolution_identity",
    "resolution_frame_sha256",
    "resolution_context_sha256",
]
