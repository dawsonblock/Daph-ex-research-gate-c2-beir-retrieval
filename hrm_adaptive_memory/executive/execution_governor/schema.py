"""Execution assistance schema: the core data structures for I3.6.

The ExecutionAssistanceFrame is a deterministic, machine-verifiable scaffold
that tells the model WHAT to accomplish, HOW success is recognized, and WHEN
to stop — not just WHICH action to take.

Design constraints:
  - Frozen dataclass: deterministic and hashable
  - No evaluator leakage: no oracle Q-values, gold labels, or future outcomes
  - Bounded: max_assisted_steps enforces termination
  - Machine-verifiable: success/failure conditions are concrete strings
  - No essay-length hidden reasoning: structured fields only
"""
from __future__ import annotations

from dataclasses import dataclass, field


ASSISTANCE_SCHEMA = "DAPH_V2B_I3_6_EXECUTION_ASSISTANCE_V1"
ASSISTANCE_VERSION = 1


@dataclass(frozen=True)
class ExecutionStep:
    """A single bounded execution step within an assistance frame.

    Each step specifies:
      - operation: what to do (e.g., "verify_top_evidence")
      - target: what to operate on (e.g., "highest-confidence unverified evidence")
      - purpose: why this step matters (e.g., "determine if evidence supports the answer")
      - stop_condition: when this step is complete (e.g., "verification_status changes")
    """
    operation: str
    target: str
    purpose: str
    stop_condition: str

    def as_dict(self) -> dict:
        return {
            "operation": self.operation,
            "target": self.target,
            "purpose": self.purpose,
            "stop_condition": self.stop_condition,
        }


@dataclass(frozen=True)
class ExecutionAssistanceFrame:
    """A structured execution scaffold produced by the execution governor.

    This is the core I3.6 object. It provides:
      - recommended_action: which metareasoning action to take
      - bottleneck_type: what prevents task completion
      - bottleneck_description: human-readable explanation
      - objective: what this assistance aims to accomplish
      - target_type: what kind of target to operate on
      - target_description: specific target description
      - known_evidence: what evidence is currently available
      - missing_information: what information is needed
      - execution_steps: bounded sequence of operations
      - success_conditions: when the objective is met
      - failure_conditions: when the objective cannot be met
      - next_action_on_success: what to do if successful
      - next_action_on_failure: what to do if failed
      - max_assisted_steps: hard budget for assisted execution
      - governor_reason_code: compact reason code
      - source_state_sha256: hash of the state that produced this frame
    """
    schema: str
    version: int
    recommended_action: str
    bottleneck_type: str
    bottleneck_description: str
    objective: str
    target_type: str
    target_description: str
    known_evidence: tuple[str, ...]
    missing_information: tuple[str, ...]
    execution_steps: tuple[ExecutionStep, ...]
    success_conditions: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    next_action_on_success: str | None
    next_action_on_failure: str | None
    max_assisted_steps: int
    governor_reason_code: str
    source_state_sha256: str

    def __post_init__(self) -> None:
        """Validate the frame at construction time."""
        if self.schema != ASSISTANCE_SCHEMA:
            raise ValueError(f"Schema mismatch: {self.schema} != {ASSISTANCE_SCHEMA}")
        if self.version != ASSISTANCE_VERSION:
            raise ValueError(f"Version mismatch: {self.version} != {ASSISTANCE_VERSION}")
        if not self.recommended_action:
            raise ValueError("recommended_action is required")
        if not self.bottleneck_type:
            raise ValueError("bottleneck_type is required")
        if not self.objective:
            raise ValueError("objective is required")
        if not self.success_conditions:
            raise ValueError("at least one success_condition is required")
        if not self.failure_conditions:
            raise ValueError("at least one failure_condition is required")
        if self.max_assisted_steps < 1:
            raise ValueError("max_assisted_steps must be >= 1")
        if self.max_assisted_steps > 3:
            raise ValueError("max_assisted_steps must be <= 3 (bounded)")
        if len(self.execution_steps) > self.max_assisted_steps:
            raise ValueError(
                f"execution_steps ({len(self.execution_steps)}) exceeds "
                f"max_assisted_steps ({self.max_assisted_steps})")
        if len(self.known_evidence) > 8:
            raise ValueError("known_evidence must have at most 8 items")
        if len(self.missing_information) > 8:
            raise ValueError("missing_information must have at most 8 items")
        if len(self.success_conditions) > 4:
            raise ValueError("success_conditions must have at most 4 items")
        if len(self.failure_conditions) > 4:
            raise ValueError("failure_conditions must have at most 4 items")

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "version": self.version,
            "recommended_action": self.recommended_action,
            "bottleneck_type": self.bottleneck_type,
            "bottleneck_description": self.bottleneck_description,
            "objective": self.objective,
            "target_type": self.target_type,
            "target_description": self.target_description,
            "known_evidence": list(self.known_evidence),
            "missing_information": list(self.missing_information),
            "execution_steps": [s.as_dict() for s in self.execution_steps],
            "success_conditions": list(self.success_conditions),
            "failure_conditions": list(self.failure_conditions),
            "next_action_on_success": self.next_action_on_success,
            "next_action_on_failure": self.next_action_on_failure,
            "max_assisted_steps": self.max_assisted_steps,
            "governor_reason_code": self.governor_reason_code,
            "source_state_sha256": self.source_state_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionAssistanceFrame:
        """Reconstruct from a dict, rejecting unknown fields."""
        allowed = {
            "schema", "version", "recommended_action", "bottleneck_type",
            "bottleneck_description", "objective", "target_type",
            "target_description", "known_evidence", "missing_information",
            "execution_steps", "success_conditions", "failure_conditions",
            "next_action_on_success", "next_action_on_failure",
            "max_assisted_steps", "governor_reason_code", "source_state_sha256",
        }
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in assistance frame: {unknown}")
        steps = tuple(
            ExecutionStep(
                operation=s["operation"],
                target=s["target"],
                purpose=s["purpose"],
                stop_condition=s["stop_condition"],
            )
            for s in d.get("execution_steps", [])
        )
        return cls(
            schema=d["schema"],
            version=d["version"],
            recommended_action=d["recommended_action"],
            bottleneck_type=d["bottleneck_type"],
            bottleneck_description=d["bottleneck_description"],
            objective=d["objective"],
            target_type=d["target_type"],
            target_description=d["target_description"],
            known_evidence=tuple(d.get("known_evidence", [])),
            missing_information=tuple(d.get("missing_information", [])),
            execution_steps=steps,
            success_conditions=tuple(d["success_conditions"]),
            failure_conditions=tuple(d["failure_conditions"]),
            next_action_on_success=d.get("next_action_on_success"),
            next_action_on_failure=d.get("next_action_on_failure"),
            max_assisted_steps=d["max_assisted_steps"],
            governor_reason_code=d["governor_reason_code"],
            source_state_sha256=d["source_state_sha256"],
        )
