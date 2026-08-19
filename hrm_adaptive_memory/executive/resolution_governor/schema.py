"""Resolution assistance schema for I3.6d.

Core data structures:
  - Hypothesis: competing explanations with evidence relationships
  - EvidenceAssessment: evidence with hypothesis links and verification state
  - Discriminator: what information would change the decision
  - ResolutionStep: bounded execution step with decision consequences
  - AnswerCondition: explicit hypothesis -> answer mapping
  - SearchSpecification: executable query construction
  - ResolutionAssistanceFrame: the full resolution scaffold
  - ResolutionContext: persistent deliberation state across steps
  - HypothesisUpdate: hypothesis elimination/keep/downweight
  - ResolutionReceipt: information-conversion instrumentation

Design constraints:
  - Frozen dataclasses: deterministic and hashable
  - No evaluator leakage: derived strictly from controller-visible state
  - Bounded: max_additional_actions enforces termination
  - Machine-verifiable: conditions are concrete strings
  - Unknown-field rejection on deserialization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RESOLUTION_SCHEMA = "DAPH_V2B_I3_6D_RESOLUTION_ASSISTANCE_V1"
RESOLUTION_VERSION = 1

# Hypothesis status values (derived from controller-visible evidence only)
HYPOTHESIS_SUPPORTED = "SUPPORTED"
HYPOTHESIS_WEAK = "WEAK"
HYPOTHESIS_CONTRADICTED = "CONTRADICTED"
HYPOTHESIS_UNRESOLVED = "UNRESOLVED"
HYPOTHESIS_ELIMINATED = "ELIMINATED"

# Verification states
VERIFICATION_SUFFICIENT = "SUFFICIENT"
VERIFICATION_MISSING = "MISSING"
VERIFICATION_UNVERIFIED = "UNVERIFIED"
VERIFICATION_FALSIFIED = "FALSIFIED"
VERIFICATION_STALE = "STALE"

# Temporal states
TEMPORAL_CURRENT = "CURRENT"
TEMPORAL_STALE = "STALE"
TEMPORAL_UNKNOWN = "UNKNOWN"

# Hypothesis update operations
UPDATE_KEEP = "KEEP"
UPDATE_DOWNWEIGHT = "DOWNWEIGHT"
UPDATE_ELIMINATE = "ELIMINATE"


@dataclass(frozen=True)
class Hypothesis:
    """A competing explanation for the task answer.

    Status is derived strictly from controller-visible evidence:
      SUPPORTED: has verified current supporting evidence, no contradictions
      WEAK: has some support but unverified or insufficient
      CONTRADICTED: has verified contradicting evidence
      UNRESOLVED: mixed or insufficient evidence
      ELIMINATED: falsified by verified evidence
    """
    hypothesis_id: str
    proposition: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    current_status: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        if not self.proposition:
            raise ValueError("proposition is required")
        if self.current_status not in (
            HYPOTHESIS_SUPPORTED, HYPOTHESIS_WEAK, HYPOTHESIS_CONTRADICTED,
            HYPOTHESIS_UNRESOLVED, HYPOTHESIS_ELIMINATED,
        ):
            raise ValueError(f"Invalid hypothesis status: {self.current_status}")
        if len(self.proposition) > 200:
            raise ValueError("proposition must be <= 200 chars")
        if len(self.supporting_evidence_ids) > 8:
            raise ValueError("at most 8 supporting evidence ids")
        if len(self.contradicting_evidence_ids) > 8:
            raise ValueError("at most 8 contradicting evidence ids")

    def as_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "proposition": self.proposition,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "current_status": self.current_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        allowed = {"hypothesis_id", "proposition", "supporting_evidence_ids",
                   "contradicting_evidence_ids", "current_status"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in Hypothesis: {unknown}")
        return cls(
            hypothesis_id=d["hypothesis_id"],
            proposition=d["proposition"],
            supporting_evidence_ids=tuple(d.get("supporting_evidence_ids", [])),
            contradicting_evidence_ids=tuple(d.get("contradicting_evidence_ids", [])),
            current_status=d["current_status"],
        )


@dataclass(frozen=True)
class EvidenceAssessment:
    """Evidence with explicit hypothesis relationships.

    Unlike the old aggregate counts, this maps each evidence item to
    which hypotheses it supports or contradicts.
    """
    evidence_id: str
    claim: str
    source_type: str
    supports: tuple[str, ...]      # hypothesis_ids this supports
    contradicts: tuple[str, ...]   # hypothesis_ids this contradicts
    verification_state: str
    temporal_state: str
    relevance: str

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if not self.claim:
            raise ValueError("claim is required")
        if self.verification_state not in (
            VERIFICATION_SUFFICIENT, VERIFICATION_MISSING, VERIFICATION_UNVERIFIED,
            VERIFICATION_FALSIFIED, VERIFICATION_STALE,
        ):
            raise ValueError(f"Invalid verification_state: {self.verification_state}")
        if self.temporal_state not in (
            TEMPORAL_CURRENT, TEMPORAL_STALE, TEMPORAL_UNKNOWN,
        ):
            raise ValueError(f"Invalid temporal_state: {self.temporal_state}")
        if len(self.claim) > 200:
            raise ValueError("claim must be <= 200 chars")
        if len(self.supports) > 8:
            raise ValueError("at most 8 supports")
        if len(self.contradicts) > 8:
            raise ValueError("at most 8 contradicts")

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "claim": self.claim,
            "source_type": self.source_type,
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
            "verification_state": self.verification_state,
            "temporal_state": self.temporal_state,
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceAssessment:
        allowed = {"evidence_id", "claim", "source_type", "supports",
                   "contradicts", "verification_state", "temporal_state", "relevance"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in EvidenceAssessment: {unknown}")
        return cls(
            evidence_id=d["evidence_id"],
            claim=d["claim"],
            source_type=d["source_type"],
            supports=tuple(d.get("supports", [])),
            contradicts=tuple(d.get("contradicts", [])),
            verification_state=d["verification_state"],
            temporal_state=d["temporal_state"],
            relevance=d["relevance"],
        )


@dataclass(frozen=True)
class Discriminator:
    """A question whose answer would change the decision.

    This is the key innovation: instead of 'find more evidence',
    the governor specifies what information would actually discriminate
    between competing hypotheses.
    """
    question: str
    if_true_supports: str       # hypothesis_id
    if_false_supports: str      # hypothesis_id
    evidence_target: str
    verification_required: bool

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("question is required")
        if not self.if_true_supports:
            raise ValueError("if_true_supports is required")
        if not self.if_false_supports:
            raise ValueError("if_false_supports is required")
        if not self.evidence_target:
            raise ValueError("evidence_target is required")
        if len(self.question) > 200:
            raise ValueError("question must be <= 200 chars")
        if len(self.evidence_target) > 200:
            raise ValueError("evidence_target must be <= 200 chars")

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "if_true_supports": self.if_true_supports,
            "if_false_supports": self.if_false_supports,
            "evidence_target": self.evidence_target,
            "verification_required": self.verification_required,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Discriminator:
        allowed = {"question", "if_true_supports", "if_false_supports",
                   "evidence_target", "verification_required"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in Discriminator: {unknown}")
        return cls(
            question=d["question"],
            if_true_supports=d["if_true_supports"],
            if_false_supports=d["if_false_supports"],
            evidence_target=d["evidence_target"],
            verification_required=d["verification_required"],
        )


@dataclass(frozen=True)
class SearchSpecification:
    """Executable query construction for SEARCH_MORE.

    Instead of 'find evidence for the claim', this specifies:
      - subject: what to search about
      - required_property: what property must be established
      - temporal_constraint: must be current/stale/any
      - source_constraint: what type of source
      - must_confirm: what the result must confirm
      - must_disambiguate: what hypotheses it must distinguish
      - reject_if: what to reject
    """
    subject: str
    required_property: str
    temporal_constraint: str | None
    source_constraint: str | None
    must_confirm: tuple[str, ...]
    must_disambiguate: tuple[str, ...]
    reject_if: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("subject is required")
        if not self.required_property:
            raise ValueError("required_property is required")
        if len(self.subject) > 200:
            raise ValueError("subject must be <= 200 chars")
        if len(self.must_confirm) > 4:
            raise ValueError("at most 4 must_confirm items")
        if len(self.must_disambiguate) > 4:
            raise ValueError("at most 4 must_disambiguate items")
        if len(self.reject_if) > 4:
            raise ValueError("at most 4 reject_if items")

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "required_property": self.required_property,
            "temporal_constraint": self.temporal_constraint,
            "source_constraint": self.source_constraint,
            "must_confirm": list(self.must_confirm),
            "must_disambiguate": list(self.must_disambiguate),
            "reject_if": list(self.reject_if),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SearchSpecification:
        allowed = {"subject", "required_property", "temporal_constraint",
                   "source_constraint", "must_confirm", "must_disambiguate", "reject_if"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in SearchSpecification: {unknown}")
        return cls(
            subject=d["subject"],
            required_property=d["required_property"],
            temporal_constraint=d.get("temporal_constraint"),
            source_constraint=d.get("source_constraint"),
            must_confirm=tuple(d.get("must_confirm", [])),
            must_disambiguate=tuple(d.get("must_disambiguate", [])),
            reject_if=tuple(d.get("reject_if", [])),
        )


@dataclass(frozen=True)
class ResolutionStep:
    """A bounded execution step with decision consequences."""
    operation: str
    target: str
    purpose: str
    decision_consequence: str
    stop_condition: str

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("operation is required")
        if not self.target:
            raise ValueError("target is required")
        if not self.decision_consequence:
            raise ValueError("decision_consequence is required")
        if len(self.target) > 200:
            raise ValueError("target must be <= 200 chars")

    def as_dict(self) -> dict:
        return {
            "operation": self.operation,
            "target": self.target,
            "purpose": self.purpose,
            "decision_consequence": self.decision_consequence,
            "stop_condition": self.stop_condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResolutionStep:
        allowed = {"operation", "target", "purpose", "decision_consequence", "stop_condition"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in ResolutionStep: {unknown}")
        return cls(
            operation=d["operation"],
            target=d["target"],
            purpose=d.get("purpose", ""),
            decision_consequence=d["decision_consequence"],
            stop_condition=d["stop_condition"],
        )


@dataclass(frozen=True)
class AnswerCondition:
    """Explicit hypothesis -> answer mapping.

    This closes the gap between information acquired and decision changed.

    IF: hypothesis_id has verified current support
    AND: condition is met
    THEN: terminal_action with answer_payload_reference
    """
    hypothesis_id: str
    condition: str
    terminal_action: str
    answer_payload_reference: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        if not self.condition:
            raise ValueError("condition is required")
        if self.terminal_action not in ("ANSWER", "DEFER", "STOP"):
            raise ValueError(f"terminal_action must be ANSWER/DEFER/STOP, got {self.terminal_action}")
        if not self.answer_payload_reference:
            raise ValueError("answer_payload_reference is required")
        if len(self.condition) > 200:
            raise ValueError("condition must be <= 200 chars")

    def as_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "condition": self.condition,
            "terminal_action": self.terminal_action,
            "answer_payload_reference": self.answer_payload_reference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnswerCondition:
        allowed = {"hypothesis_id", "condition", "terminal_action", "answer_payload_reference"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in AnswerCondition: {unknown}")
        return cls(
            hypothesis_id=d["hypothesis_id"],
            condition=d["condition"],
            terminal_action=d["terminal_action"],
            answer_payload_reference=d["answer_payload_reference"],
        )


@dataclass(frozen=True)
class ResolutionAssistanceFrame:
    """The full resolution scaffold produced by the resolution governor.

    This is the core I3.6d object. Unlike the I3.6a/b ExecutionAssistanceFrame,
    it provides:
      - candidate_hypotheses: competing explanations
      - current_evidence: evidence with hypothesis relationships
      - unresolved_question: what must be resolved
      - discriminating_evidence: what information would change the decision
      - execution_plan: bounded steps with decision consequences
      - answer_conditions: explicit hypothesis -> answer mappings
      - defer_condition: when to give up
      - search_specification: executable query construction (for SEARCH_MORE)
      - max_additional_actions: hard budget
    """
    schema: str
    version: int
    recommended_action: str
    task_goal: str
    candidate_hypotheses: tuple[Hypothesis, ...]
    current_evidence: tuple[EvidenceAssessment, ...]
    unresolved_question: str
    discriminating_evidence: tuple[Discriminator, ...]
    execution_plan: tuple[ResolutionStep, ...]
    answer_conditions: tuple[AnswerCondition, ...]
    defer_condition: str
    search_specification: SearchSpecification | None
    max_additional_actions: int
    source_state_sha256: str

    def __post_init__(self) -> None:
        if self.schema != RESOLUTION_SCHEMA:
            raise ValueError(f"Schema mismatch: {self.schema} != {RESOLUTION_SCHEMA}")
        if self.version != RESOLUTION_VERSION:
            raise ValueError(f"Version mismatch: {self.version} != {RESOLUTION_VERSION}")
        if not self.recommended_action:
            raise ValueError("recommended_action is required")
        if not self.task_goal:
            raise ValueError("task_goal is required")
        if not self.candidate_hypotheses:
            raise ValueError("at least one candidate_hypothesis is required")
        if len(self.candidate_hypotheses) > 4:
            raise ValueError("at most 4 candidate_hypotheses")
        if len(self.current_evidence) > 8:
            raise ValueError("at most 8 current_evidence items")
        if len(self.discriminating_evidence) > 4:
            raise ValueError("at most 4 discriminating_evidence items")
        if len(self.execution_plan) > 3:
            raise ValueError("at most 3 execution_plan steps")
        if len(self.answer_conditions) > 4:
            raise ValueError("at most 4 answer_conditions")
        if self.max_additional_actions < 1:
            raise ValueError("max_additional_actions must be >= 1")
        if self.max_additional_actions > 3:
            raise ValueError("max_additional_actions must be <= 3 (bounded)")
        if len(self.execution_plan) > self.max_additional_actions:
            raise ValueError(
                f"execution_plan ({len(self.execution_plan)}) exceeds "
                f"max_additional_actions ({self.max_additional_actions})")
        if not self.defer_condition:
            raise ValueError("defer_condition is required")

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "version": self.version,
            "recommended_action": self.recommended_action,
            "task_goal": self.task_goal,
            "candidate_hypotheses": [h.as_dict() for h in self.candidate_hypotheses],
            "current_evidence": [e.as_dict() for e in self.current_evidence],
            "unresolved_question": self.unresolved_question,
            "discriminating_evidence": [d.as_dict() for d in self.discriminating_evidence],
            "execution_plan": [s.as_dict() for s in self.execution_plan],
            "answer_conditions": [a.as_dict() for a in self.answer_conditions],
            "defer_condition": self.defer_condition,
            "search_specification": self.search_specification.as_dict() if self.search_specification else None,
            "max_additional_actions": self.max_additional_actions,
            "source_state_sha256": self.source_state_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResolutionAssistanceFrame:
        allowed = {
            "schema", "version", "recommended_action", "task_goal",
            "candidate_hypotheses", "current_evidence", "unresolved_question",
            "discriminating_evidence", "execution_plan", "answer_conditions",
            "defer_condition", "search_specification", "max_additional_actions",
            "source_state_sha256",
        }
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown fields in ResolutionAssistanceFrame: {unknown}")
        hypotheses = tuple(Hypothesis.from_dict(h) for h in d.get("candidate_hypotheses", []))
        evidence = tuple(EvidenceAssessment.from_dict(e) for e in d.get("current_evidence", []))
        discriminators = tuple(Discriminator.from_dict(di) for di in d.get("discriminating_evidence", []))
        steps = tuple(ResolutionStep.from_dict(s) for s in d.get("execution_plan", []))
        conditions = tuple(AnswerCondition.from_dict(a) for a in d.get("answer_conditions", []))
        search_spec = None
        if d.get("search_specification"):
            search_spec = SearchSpecification.from_dict(d["search_specification"])
        return cls(
            schema=d["schema"],
            version=d["version"],
            recommended_action=d["recommended_action"],
            task_goal=d["task_goal"],
            candidate_hypotheses=hypotheses,
            current_evidence=evidence,
            unresolved_question=d["unresolved_question"],
            discriminating_evidence=discriminators,
            execution_plan=steps,
            answer_conditions=conditions,
            defer_condition=d["defer_condition"],
            search_specification=search_spec,
            max_additional_actions=d["max_additional_actions"],
            source_state_sha256=d["source_state_sha256"],
        )


@dataclass(frozen=True)
class HypothesisUpdate:
    """A hypothesis status update after an evidence operation."""
    hypothesis_id: str
    update: str  # KEEP / DOWNWEIGHT / ELIMINATE
    evidence_id: str
    reason_code: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        if self.update not in (UPDATE_KEEP, UPDATE_DOWNWEIGHT, UPDATE_ELIMINATE):
            raise ValueError(f"Invalid update: {self.update}")
        if not self.evidence_id:
            raise ValueError("evidence_id is required")

    def as_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "update": self.update,
            "evidence_id": self.evidence_id,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class ResolutionContext:
    """Persistent deliberation state across steps.

    Unlike the old approach of regenerating a generic frame each step,
    this context carries forward:
      - hypotheses (with updated statuses)
      - evidence map (with new evidence added)
      - active discriminator (what we're currently trying to resolve)
      - completed/pending steps
      - current best hypothesis
      - termination status
    """
    context_id: str
    hypotheses: tuple[Hypothesis, ...]
    evidence: tuple[EvidenceAssessment, ...]
    active_discriminator: Discriminator | None
    completed_steps: tuple[str, ...]
    pending_steps: tuple[str, ...]
    current_best_hypothesis: str | None
    termination_status: str  # ACTIVE / RESOLVED / DEFERRED / EXHAUSTED
    hypothesis_updates: tuple[HypothesisUpdate, ...]
    step_counter: int

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context_id is required")
        if self.termination_status not in ("ACTIVE", "RESOLVED", "DEFERRED", "EXHAUSTED"):
            raise ValueError(f"Invalid termination_status: {self.termination_status}")
        if len(self.hypotheses) > 4:
            raise ValueError("at most 4 hypotheses")
        if len(self.evidence) > 8:
            raise ValueError("at most 8 evidence items")
        if self.step_counter < 0:
            raise ValueError("step_counter must be >= 0")

    def as_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "evidence": [e.as_dict() for e in self.evidence],
            "active_discriminator": self.active_discriminator.as_dict() if self.active_discriminator else None,
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "current_best_hypothesis": self.current_best_hypothesis,
            "termination_status": self.termination_status,
            "hypothesis_updates": [u.as_dict() for u in self.hypothesis_updates],
            "step_counter": self.step_counter,
        }

    @property
    def n_viable_hypotheses(self) -> int:
        """Number of hypotheses not eliminated."""
        return sum(
            1 for h in self.hypotheses
            if h.current_status not in (HYPOTHESIS_ELIMINATED, HYPOTHESIS_CONTRADICTED)
        )

    @property
    def is_resolved(self) -> bool:
        """True when exactly one viable hypothesis remains."""
        return self.n_viable_hypotheses == 1


@dataclass(frozen=True)
class ResolutionReceipt:
    """Information-conversion instrumentation for a single assisted step.

    Records what changed in the deliberation state as a result of
    the model's action, enabling measurement of:
      P(hypothesis eliminated | intervention)
      P(discriminator resolved | intervention)
      P(answer condition reached | intervention)
      P(task rescue | intervention)
    """
    step_id: int
    action_taken: str
    hypotheses_before: int
    hypotheses_after: int
    discriminator_resolved: bool
    new_evidence_found: bool
    evidence_verified: bool
    best_hypothesis_changed: bool
    answer_condition_satisfied: bool
    hypothesis_updates: tuple[HypothesisUpdate, ...]
    terminal_result: str | None

    def as_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "action_taken": self.action_taken,
            "hypotheses_before": self.hypotheses_before,
            "hypotheses_after": self.hypotheses_after,
            "discriminator_resolved": self.discriminator_resolved,
            "new_evidence_found": self.new_evidence_found,
            "evidence_verified": self.evidence_verified,
            "best_hypothesis_changed": self.best_hypothesis_changed,
            "answer_condition_satisfied": self.answer_condition_satisfied,
            "hypothesis_updates": [u.as_dict() for u in self.hypothesis_updates],
            "terminal_result": self.terminal_result,
        }
