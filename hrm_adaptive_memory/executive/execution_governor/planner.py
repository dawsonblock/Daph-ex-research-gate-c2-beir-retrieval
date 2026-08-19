"""Deterministic execution assistance planner.

Builds ExecutionAssistanceFrames from controller-visible state using
deterministic templates — no LLM generation. This preserves
interpretability and separates planner quality from model quality.

Templates are organized by the governor's recommended action:
  - RETRIEVE: what information is missing, which class to retrieve
  - VERIFY: which claim/evidence pair to verify, what contradiction to resolve
  - SEARCH_MORE: specific missing field, search target, stop condition
  - REASON_MORE: precise unresolved inference, inputs to combine
  - ANSWER: which verified evidence supports termination
  - DEFER: which requirement failed, why more computation is uneconomic
  - STOP: not scaffolded (terminal, no continuation needed)

The planner uses only information available to the controller under the
AWARE observation regime. It must not inspect oracle Q-values, gold
labels, oracle success paths, or future trajectory outcomes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.executive.execution_governor.schema import (
    ExecutionAssistanceFrame,
    ExecutionStep,
    ASSISTANCE_SCHEMA,
    ASSISTANCE_VERSION,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.governor.bottlenecks import detect_bottlenecks
from hrm_adaptive_memory.executive.governor.state import (
    GovernorState, build_governor_state,
)
from hrm_adaptive_memory.executive.metareasoning_controller import ControllerObservation


def _state_sha256(state: GovernorState) -> str:
    """Compute a deterministic SHA-256 of the controller-visible state."""
    # Use a compact, sorted representation of the state
    cs = state.observation.cognitive_state
    state_repr = {
        "task_id": state.observation.task_id,
        "remaining_steps": state.remaining_steps,
        "prior_actions": list(state.prior_actions),
        "prior_outcomes": list(state.prior_outcomes),
        "legal_actions": list(state.legal_actions),
        "last_action": state.last_action,
        "last_outcome": state.last_outcome,
    }
    if cs is not None:
        state_repr["cognitive_state"] = {
            "verification_states": [
                {"target_id": v.target_id, "state": v.state.value,
                 "evidence_count": v.evidence_count,
                 "last_verified": v.last_verified}
                for v in cs.verification_states
            ],
            "temporal_status": cs.temporal_status.value,
            "unresolved_conflicts": [
                {"conflict_id": c.conflict_id, "relation": c.relation,
                 "status": c.status}
                for c in cs.unresolved_conflicts
            ],
            "observation_signals": list(cs.observation_signals),
            "evidence_count": sum(m.evidence_count for m in cs.relevant_memories),
        }
    return hashlib.sha256(
        json.dumps(state_repr, sort_keys=True, default=str).encode()
    ).hexdigest()


class ExecutionGovernor:
    """Deterministic execution-assistance governor.

    Wraps the existing GeneralGovernor for action selection and bottleneck
    detection, then builds a structured ExecutionAssistanceFrame from
    controller-visible state.

    The governor does NOT choose the model's action — it provides a scaffold.
    The model retains final decision authority.
    """

    def __init__(self, max_steps: int = 25):
        self._general_governor = GeneralGovernor(max_steps=max_steps)

    def plan(
        self,
        observation: ControllerObservation,
        remaining_steps: int | None = None,
        prior_actions: tuple[str, ...] | None = None,
        prior_outcomes: tuple[str, ...] | None = None,
    ) -> ExecutionAssistanceFrame | None:
        """Build an execution assistance frame for the current state.

        Returns None if no assistance is needed (e.g., READY_TO_ANSWER
        with no bottleneck, or the governor agrees with what the model
        would likely do).
        """
        if remaining_steps is None:
            remaining_steps = self._general_governor._max_steps - len(
                observation.executed_actions)

        state = build_governor_state(
            observation=observation,
            remaining_steps=remaining_steps,
            prior_actions=prior_actions,
            prior_outcomes=prior_outcomes,
        )

        # Use the existing governor for action selection and bottleneck detection
        frame = self._general_governor.assess(
            observation=observation,
            remaining_steps=remaining_steps,
            prior_actions=prior_actions,
            prior_outcomes=prior_outcomes,
        )

        bottlenecks = detect_bottlenecks(state)
        top_bottleneck = bottlenecks[0] if bottlenecks else None
        recommended_action = frame.governor_top_action
        reason_code = frame.governor_reason_code

        # Don't scaffold STOP (terminal, no execution needed)
        if recommended_action == "STOP":
            return None

        # Don't scaffold if READY_TO_ANSWER and no real bottleneck
        if (top_bottleneck is not None
                and top_bottleneck.kind == "READY_TO_ANSWER"
                and recommended_action == "ANSWER"):
            # Still scaffold ANSWER to provide termination checklist
            pass

        state_sha = _state_sha256(state)

        return plan_assistance(
            state=state,
            recommended_action=recommended_action,
            bottleneck=top_bottleneck,
            reason_code=reason_code,
            state_sha=state_sha,
        )


def plan_assistance(
    state: GovernorState,
    recommended_action: str,
    bottleneck: Any | None,
    reason_code: str,
    state_sha: str,
) -> ExecutionAssistanceFrame | None:
    """Build an assistance frame using deterministic templates.

    Dispatches to the appropriate template function based on the
    recommended action.
    """
    cs = state.observation.cognitive_state
    res = state.resources

    if recommended_action == "RETRIEVE":
        return _template_retrieve(state, cs, res, bottleneck, reason_code, state_sha)
    elif recommended_action == "VERIFY":
        return _template_verify(state, cs, res, bottleneck, reason_code, state_sha)
    elif recommended_action == "SEARCH_MORE":
        return _template_search_more(state, cs, res, bottleneck, reason_code, state_sha)
    elif recommended_action == "REASON_MORE":
        return _template_reason_more(state, cs, res, bottleneck, reason_code, state_sha)
    elif recommended_action == "ANSWER":
        return _template_answer(state, cs, res, bottleneck, reason_code, state_sha)
    elif recommended_action == "DEFER":
        return _template_defer(state, cs, res, bottleneck, reason_code, state_sha)
    elif recommended_action == "STOP":
        return None  # Terminal, no scaffold needed
    else:
        return None


def _extract_verification_info(cs: Any) -> tuple[str, str, int]:
    """Extract verification state info from cognitive state.

    Returns (verification_state_str, target_id, evidence_count).
    """
    if cs is None or not cs.verification_states:
        return ("UNKNOWN", "", 0)
    vs = cs.verification_states[0]
    state_str = vs.state.value if hasattr(vs.state, "value") else str(vs.state)
    return (state_str, vs.target_id, vs.evidence_count)


def _extract_temporal_info(cs: Any) -> str:
    """Extract temporal status from cognitive state."""
    if cs is None:
        return "UNKNOWN"
    ts = cs.temporal_status
    return ts.value if hasattr(ts, "value") else str(ts)


def _extract_conflict_info(cs: Any) -> tuple[int, bool]:
    """Extract conflict info from cognitive state.

    Returns (conflict_count, any_resolvable).
    """
    if cs is None or not cs.unresolved_conflicts:
        return (0, False)
    count = len(cs.unresolved_conflicts)
    resolvable = any(
        getattr(c, "status", "") == "RESOLVABLE"
        for c in cs.unresolved_conflicts
    )
    return (count, resolvable)


def _extract_evidence_info(cs: Any) -> tuple[int, int]:
    """Extract evidence info from cognitive state.

    Returns (total_evidence_count, verified_count).
    """
    if cs is None:
        return (0, 0)
    total = sum(m.evidence_count for m in cs.relevant_memories)
    verified = sum(
        1 for m in cs.relevant_memories
        if m.verification_state.value in ("SUFFICIENT",))
    return (total, verified)


def _template_retrieve(
    state: GovernorState, cs: Any, res: Any, bottleneck: Any,
    reason_code: str, state_sha: str,
) -> ExecutionAssistanceFrame:
    """Template for RETRIEVE: what information is missing, which class to retrieve."""
    vs_str, target_id, evidence_count = _extract_verification_info(cs)
    total_evidence, verified = _extract_evidence_info(cs)

    if vs_str == "MISSING" or evidence_count == 0:
        objective = "retrieve evidence needed to support the candidate answer"
        target_desc = "evidence class matching the task's information requirement"
        missing = ("evidence supporting the candidate answer",)
        known: tuple[str, ...] = ()
    elif vs_str == "FALSIFIED":
        objective = "retrieve alternative evidence after falsification"
        target_desc = "alternative evidence source for the falsified claim"
        missing = ("replacement evidence for the falsified claim",)
        known = (f"prior evidence falsified for target {target_id}",)
    else:
        objective = "retrieve additional evidence to strengthen the evidence base"
        target_desc = "supplementary evidence for the current claim"
        missing = ("additional supporting evidence",)
        known = (f"current evidence count: {total_evidence}",)

    return ExecutionAssistanceFrame(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="RETRIEVE",
        bottleneck_type=bottleneck.kind if bottleneck else "NO_EVIDENCE",
        bottleneck_description=f"verification_state={vs_str}, evidence_count={evidence_count}",
        objective=objective,
        target_type="evidence_class",
        target_description=target_desc,
        known_evidence=known,
        missing_information=missing,
        execution_steps=(
            ExecutionStep(
                operation="retrieve",
                target=target_desc,
                purpose="obtain evidence needed for verification",
                stop_condition="at least one new evidence item retrieved",
            ),
        ),
        success_conditions=(
            "at least one new evidence item retrieved",
            "evidence_count increases",
        ),
        failure_conditions=(
            "retrieval returns no new evidence",
            "retrieval budget exhausted",
        ),
        next_action_on_success="VERIFY",
        next_action_on_failure="SEARCH_MORE",
        max_assisted_steps=1,
        governor_reason_code=reason_code,
        source_state_sha256=state_sha,
    )


def _template_verify(
    state: GovernorState, cs: Any, res: Any, bottleneck: Any,
    reason_code: str, state_sha: str,
) -> ExecutionAssistanceFrame:
    """Template for VERIFY: which claim/evidence pair to verify, what contradiction to resolve."""
    vs_str, target_id, evidence_count = _extract_verification_info(cs)
    total_evidence, verified = _extract_evidence_info(cs)
    conflict_count, _ = _extract_conflict_info(cs)

    if vs_str == "UNVERIFIED":
        objective = "determine whether retrieved evidence supports the candidate answer"
        target_desc = f"highest-confidence unverified evidence (target: {target_id})"
        missing = ("verification status for the top evidence item",)
    elif vs_str == "STALE":
        objective = "re-verify evidence that may be outdated"
        target_desc = f"stale evidence requiring re-verification (target: {target_id})"
        missing = ("current verification status",)
    elif conflict_count > 0:
        objective = "resolve conflicting evidence through verification"
        target_desc = f"conflicting evidence pair ({conflict_count} conflicts)"
        missing = ("resolution of the conflicting claims",)
    else:
        objective = "verify the top evidence item"
        target_desc = f"top evidence item (target: {target_id})"
        missing = ("verification status",)

    known = (f"evidence_count={total_evidence}", f"verified={verified}")

    return ExecutionAssistanceFrame(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="VERIFY",
        bottleneck_type=bottleneck.kind if bottleneck else "UNVERIFIED_EVIDENCE",
        bottleneck_description=f"verification_state={vs_str}, conflicts={conflict_count}",
        objective=objective,
        target_type="evidence_item",
        target_description=target_desc,
        known_evidence=known,
        missing_information=missing,
        execution_steps=(
            ExecutionStep(
                operation="verify",
                target=target_desc,
                purpose=objective,
                stop_condition="verification_status changes to SUFFICIENT or FALSIFIED",
            ),
        ),
        success_conditions=(
            "verification_status becomes SUFFICIENT",
            "evidence supports the candidate answer",
        ),
        failure_conditions=(
            "evidence becomes FALSIFIED",
            "verification remains MISSING after verification attempt",
        ),
        next_action_on_success="ANSWER",
        next_action_on_failure="SEARCH_MORE",
        max_assisted_steps=1,
        governor_reason_code=reason_code,
        source_state_sha256=state_sha,
    )


def _template_search_more(
    state: GovernorState, cs: Any, res: Any, bottleneck: Any,
    reason_code: str, state_sha: str,
) -> ExecutionAssistanceFrame:
    """Template for SEARCH_MORE: specific missing field, search target, stop condition."""
    vs_str, target_id, evidence_count = _extract_verification_info(cs)
    temporal = _extract_temporal_info(cs)
    conflict_count, _ = _extract_conflict_info(cs)

    if temporal == "STALE":
        objective = "find current evidence to replace stale information"
        target_desc = "timestamped confirmation of the current evidence state"
        missing = ("current temporal confirmation",)
    elif conflict_count > 0:
        objective = "find disambiguating evidence to resolve the conflict"
        target_desc = "evidence that distinguishes between conflicting claims"
        missing = ("disambiguating evidence",)
    elif vs_str == "MISSING":
        objective = "find evidence supporting the candidate answer"
        target_desc = "source class containing evidence for the claim"
        missing = ("any supporting evidence",)
    else:
        objective = "find additional evidence to strengthen the evidence base"
        target_desc = "supplementary evidence source"
        missing = ("additional supporting evidence",)

    known = (f"temporal_status={temporal}", f"verification_state={vs_str}")

    return ExecutionAssistanceFrame(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="SEARCH_MORE",
        bottleneck_type=bottleneck.kind if bottleneck else "NO_EVIDENCE",
        bottleneck_description=f"temporal={temporal}, conflicts={conflict_count}",
        objective=objective,
        target_type="search_target",
        target_description=target_desc,
        known_evidence=known,
        missing_information=missing,
        execution_steps=(
            ExecutionStep(
                operation="search",
                target=target_desc,
                purpose=objective,
                stop_condition="at least one qualifying evidence item found",
            ),
        ),
        success_conditions=(
            "at least one qualifying evidence item found",
            "new evidence addresses the missing information",
        ),
        failure_conditions=(
            "no qualifying evidence after one search",
            "search budget exhausted",
        ),
        next_action_on_success="VERIFY",
        next_action_on_failure="DEFER",
        max_assisted_steps=1,
        governor_reason_code=reason_code,
        source_state_sha256=state_sha,
    )


def _template_reason_more(
    state: GovernorState, cs: Any, res: Any, bottleneck: Any,
    reason_code: str, state_sha: str,
) -> ExecutionAssistanceFrame:
    """Template for REASON_MORE: precise unresolved inference, inputs to combine."""
    signals = cs.observation_signals if cs is not None else ()
    total_evidence, verified = _extract_evidence_info(cs)

    if "COMPOSITION_INCOMPLETE" in signals:
        objective = "complete the composition chain by combining available evidence"
        target_desc = "unresolved composition step requiring evidence combination"
        missing = ("composition of evidence into a coherent answer",)
    else:
        objective = "resolve the current inference gap"
        target_desc = "unresolved inference dependency"
        missing = ("resolution of the inference dependency",)

    known = (f"evidence_count={total_evidence}", f"verified={verified}")

    return ExecutionAssistanceFrame(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="REASON_MORE",
        bottleneck_type=bottleneck.kind if bottleneck else "INSUFFICIENT_REASONING",
        bottleneck_description=f"signals={list(signals)}",
        objective=objective,
        target_type="inference_step",
        target_description=target_desc,
        known_evidence=known,
        missing_information=missing,
        execution_steps=(
            ExecutionStep(
                operation="reason",
                target=target_desc,
                purpose=objective,
                stop_condition="inference gap resolved or reasoning depth limit reached",
            ),
        ),
        success_conditions=(
            "inference gap resolved",
            "composition chain advances",
        ),
        failure_conditions=(
            "reasoning depth limit reached without resolution",
            "insufficient inputs to complete the inference",
        ),
        next_action_on_success="VERIFY",
        next_action_on_failure="DEFER",
        max_assisted_steps=2,
        governor_reason_code=reason_code,
        source_state_sha256=state_sha,
    )


def _template_answer(
    state: GovernorState, cs: Any, res: Any, bottleneck: Any,
    reason_code: str, state_sha: str,
) -> ExecutionAssistanceFrame:
    """Template for ANSWER: which verified evidence supports termination."""
    vs_str, target_id, evidence_count = _extract_verification_info(cs)
    total_evidence, verified = _extract_evidence_info(cs)
    conflict_count, _ = _extract_conflict_info(cs)

    objective = "terminate with a verified answer supported by sufficient evidence"
    target_desc = f"verified evidence supporting the answer (target: {target_id})"

    known = (
        f"verification_state={vs_str}",
        f"evidence_count={total_evidence}",
        f"verified={verified}",
        f"conflicts={conflict_count}",
    )

    return ExecutionAssistanceFrame(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="ANSWER",
        bottleneck_type=bottleneck.kind if bottleneck else "READY_TO_ANSWER",
        bottleneck_description=f"verification={vs_str}, evidence={total_evidence}",
        objective=objective,
        target_type="verified_evidence",
        target_description=target_desc,
        known_evidence=known,
        missing_information=(),
        execution_steps=(
            ExecutionStep(
                operation="answer",
                target=target_desc,
                purpose="terminate with the verified answer",
                stop_condition="answer submitted",
            ),
        ),
        success_conditions=(
            "answer submitted with sufficient verified evidence",
            "no unresolved conflicts remain",
        ),
        failure_conditions=(
            "verification_state is not SUFFICIENT",
            "unresolved conflicts exist",
        ),
        next_action_on_success=None,
        next_action_on_failure="VERIFY",
        max_assisted_steps=1,
        governor_reason_code=reason_code,
        source_state_sha256=state_sha,
    )


def _template_defer(
    state: GovernorState, cs: Any, res: Any, bottleneck: Any,
    reason_code: str, state_sha: str,
) -> ExecutionAssistanceFrame:
    """Template for DEFER: which requirement failed, why more computation is uneconomic."""
    vs_str, target_id, evidence_count = _extract_verification_info(cs)
    res_info = f"retrieval={res.has_retrieval}, verification={res.has_verification}, search={res.has_search}"

    if bottleneck and bottleneck.kind == "RESOURCE_EXHAUSTION":
        objective = "defer task because resources are exhausted"
        target_desc = "no remaining resources for useful actions"
        missing = ("additional resources to continue",)
    elif bottleneck and bottleneck.kind == "IRREDUCIBLE_UNCERTAINTY":
        objective = "defer task because the uncertainty cannot be resolved"
        target_desc = "irreducible uncertainty in the evidence"
        missing = ("information that would resolve the uncertainty",)
    else:
        objective = "defer task because further computation is not expected to improve utility"
        target_desc = "current evidence state does not support further progress"
        missing = ("information that would change the decision",)

    known = (f"verification_state={vs_str}", res_info)

    return ExecutionAssistanceFrame(
        schema=ASSISTANCE_SCHEMA,
        version=ASSISTANCE_VERSION,
        recommended_action="DEFER",
        bottleneck_type=bottleneck.kind if bottleneck else "RESOURCE_EXHAUSTION",
        bottleneck_description=f"resources={res_info}",
        objective=objective,
        target_type="requirement_gap",
        target_description=target_desc,
        known_evidence=known,
        missing_information=missing,
        execution_steps=(
            ExecutionStep(
                operation="defer",
                target=target_desc,
                purpose="terminate without answer due to unresolvable state",
                stop_condition="defer submitted",
            ),
        ),
        success_conditions=(
            "defer submitted with clear reason",
        ),
        failure_conditions=(
            "further computation could still improve utility",
        ),
        next_action_on_success=None,
        next_action_on_failure=None,
        max_assisted_steps=1,
        governor_reason_code=reason_code,
        source_state_sha256=state_sha,
    )
