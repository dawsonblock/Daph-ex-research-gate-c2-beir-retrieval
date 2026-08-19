"""Execution-assisting governor: structured execution scaffolds.

I3.6 changes the governor intervention from an action recommendation into
an execution scaffold. Instead of telling the model WHICH action to take,
the governor tells the model WHAT must be accomplished, HOW success is
recognized, and WHEN to stop.

Core principle:
    Don't tell the model merely what to do.
    Tell it what must be accomplished, how success is recognized,
    and when to stop.
"""
from __future__ import annotations

from hrm_adaptive_memory.executive.execution_governor.schema import (
    ExecutionAssistanceFrame,
    ExecutionStep,
    ASSISTANCE_SCHEMA,
    ASSISTANCE_VERSION,
)
from hrm_adaptive_memory.executive.execution_governor.planner import (
    ExecutionGovernor,
    plan_assistance,
)
from hrm_adaptive_memory.executive.execution_governor.serializer import (
    serialize_assistance_packet,
)
from hrm_adaptive_memory.executive.execution_governor.validator import (
    assert_no_evaluator_leakage,
    validate_assistance_frame,
)
from hrm_adaptive_memory.executive.execution_governor.identity import (
    assistance_frame_sha256,
    compute_assistance_identity,
)

__all__ = [
    "ExecutionAssistanceFrame",
    "ExecutionStep",
    "ASSISTANCE_SCHEMA",
    "ASSISTANCE_VERSION",
    "ExecutionGovernor",
    "plan_assistance",
    "serialize_assistance_packet",
    "assert_no_evaluator_leakage",
    "validate_assistance_frame",
    "assistance_frame_sha256",
    "compute_assistance_identity",
]
